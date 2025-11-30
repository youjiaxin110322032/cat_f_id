# api/index.py
import io
import os
from typing import Dict, List
import sys
import numpy as np
import httpx

from datetime import datetime
from typing import Dict, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 🔐 安全性相關引用

import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .models import ChatRequest, ChatMessage

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")  # 預設 deepseek-chat

# 🌟 檢查環境變數是否存在
if not LLM_API_KEY:
    raise RuntimeError("❌ LLM_API_KEY 未設定，請在 .env 裡加入你的 API Key")

if not LLM_ENDPOINT:
    raise RuntimeError(
        "❌ LLM_ENDPOINT 未設定。\n"
        "例如 DeepSeek:\n"
        "LLM_ENDPOINT=https://api.deepseek.com/v1/chat/completions\n"
        "或 OpenAI:\n"
        "LLM_ENDPOINT=https://api.openai.com/v1/chat/completions"
    )

# 印出設定方便 debug（正式環境建議關掉）
print("🔧 LLM 設定：")
print(" - MODEL    =", LLM_MODEL)
print(" - ENDPOINT =", LLM_ENDPOINT)
print(" - KEY 前 6 =", LLM_API_KEY[:6], "...")

# =========================
# 🧠 0. 載入辨識模組 (路徑防呆)
# =========================
try:
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD
except ImportError:
    # 若 Python 沒把專案根目錄放進 sys.path，就手動補一層
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

app = FastAPI(title="Cat Face LLM Chat", version="1.1")

# 每個 user 的聊天歷史：username -> List[ChatMessage]
user_history: Dict[str, List[ChatMessage]] = {}

# 建立 Bearer 驗證器（給 Security 用）
bearer = HTTPBearer(auto_error=False)

# 專案根目錄 / api 目錄
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
key_path = os.path.join(PROJECT_ROOT, "firebase.json")
# =========================
# 🔥 1. Firebase 初始化
# =========================
if not firebase_admin._apps:
    # firebase.json 放在「專案根目錄」
    key_path = os.path.join(PROJECT_ROOT, "firebase.json")

    env_project_id = os.environ.get("FIREBASE_PROJECT_ID")

    if env_project_id:
        # ✅ 有設定環境變數（雲端部署用）
        cred_dict = {
            "type": "service_account",
            "project_id": env_project_id,
            "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
        }
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized from environment variables")
    elif os.path.exists(key_path):
        # ✅ 沒有 env，就用本地 firebase.json
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        print(f"✅ Firebase initialized from file: {key_path}")
    else:
        # ❌ 兩邊都沒有
        print("❌ Firebase init failed: no env vars and no firebase.json")

# =========================
# 🔐 Firebase Token 驗證
# =========================
def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer Token")

    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token)
        print("✅ Auth OK:", decoded.get("email"), decoded.get("uid"))
        return decoded
    except Exception as e:
        print("❌ Auth Failed:", e)
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

# =========================
# 🔐 CORS / 靜態檔案
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://youjiaxin110322032.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端靜態檔案（frontend 在專案根目錄）
static_path = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
else:
    print(f"⚠️ Warning: 'frontend' folder not found at {static_path}")

# =========================
# 🧠 模型載入
# =========================
try:
    knn, id2name = load_model()
except RuntimeError as e:
    print("[warning] load_model 失敗：", e)
    knn, id2name = None, {}

comments_db = {}  # {"mama": [留言...], ...}

# =========================
# 🌐 路由
# =========================

@app.get("/me")
def get_me(user = Depends(verify_firebase_token)):
    return {
        "uid": user.get("uid"),
        "email": user.get("email"),
        "name": user.get("name"),
    }

@app.get("/")
def root():
    index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "frontend/index.html not found"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/labels")
def labels():
    return {
        "count": len(id2name),
        "labels": [id2name[i] for i in sorted(id2name.keys())],
    }

@app.post("/chat")
async def chat(
    req: ChatRequest,
    user = Depends(verify_firebase_token),
):
    """
    使用 DeepSeek / OpenAI 風格的 chat.completions API，
    - 維持每個使用者獨立歷史（存在記憶體 user_history）
    - 加上 system prompt（貓咪助手）
    - 做簡單長度限制避免爆 token
    """
    uid = user.get("uid") or user.get("email")
    if not uid:
        raise HTTPException(status_code=400, detail="No uid or email in token")

    # 1. 把這次 user 訊息先寫進歷史
    history = user_history.setdefault(uid, [])
    history.append(
        ChatMessage(
            role="user",
            content=req.message,
            timestamp=datetime.utcnow(),
        )
    )

    # 2. 取最近 10 則對話，防止無限變長
    last_messages = history[-10:]

    # 簡單的內容長度限制（防止單句太長炸 token）
    def truncate(text: str, max_len: int = 1000) -> str:
        text = text or ""
        if len(text) <= max_len:
            return text
        return text[-max_len:]  # 保留尾端內容即可

    # 3. DeepSeek / OpenAI 標準 messages 格式，加入 system prompt
    messages_payload = [
        {
            "role": "system",
            "content": (
                "你是一隻活潑但專業的貓咪識別與陪聊助手，"
                "說話可以可愛一點，但重點要清楚、具體，"
                "使用繁體中文回答。"
            ),
        }
    ]
    for m in last_messages:
        # m.role 是 "user" 或 "assistant"（你的 ChatMessage 模型）
        messages_payload.append(
            {
                "role": m.role,
                "content": truncate(m.content),
            }
        )

    # 4. 呼叫 LLM API
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "aKpplication/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages_payload,
        # 以下是常見參數，可依你喜好調整
        "temperature": 0.7,
        "max_tokens": 512,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(LLM_ENDPOINT, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        # 直接回傳 502 給前端，比起 500 更像「下游服務掛了」
        raise HTTPException(status_code=502, detail=f"LLM 呼叫失敗: {str(e)}")

    # DeepSeek / OpenAI 相同結構：choices[0].message.content
    try:
        assistant_reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 回傳格式異常: {str(e)}")

    # 5. 把助理回覆追加到歷史
    history.append(
        ChatMessage(
            role="assistant",
            content=assistant_reply,
            timestamp=datetime.utcnow(),
        )
    )

    if LLM_ENDPOINT.endswith("/v1"):
        LLM_ENDPOINT += "/chat/completions"
    elif LLM_ENDPOINT.endswith("/v1/"):
        LLM_ENDPOINT += "chat/completions"
    print(f"💬 LLM 回覆給 {uid}: {assistant_reply}"
          f" (via {LLM_ENDPOINT})")
    
    # 6. 回傳給前端
    return {
        "reply": assistant_reply,
        "history_len": len(history),
    }



@app.get("/history")
def get_history(user = Depends(verify_firebase_token)):
    uid = user.get("uid") or user.get("email")
    if not uid:
        raise HTTPException(status_code=400, detail="No uid or email in token")

    history = user_history.get(uid, [])
    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
        }
        for m in history
    ]


@app.post("/camera_open")
def camera_open(user = Depends(verify_firebase_token)):
    email = user.get("email")
    uid = user.get("uid")
    print(f"📷 Camera opened by {email} ({uid})")
    return {"email": email, "uid": uid}

@app.post("/reload")
def reload_model(user: dict = Depends(verify_firebase_token)):
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "by_user": user.get("email")}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user = Depends(verify_firebase_token),
):
    if knn is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(img)[:, :, ::-1]  # RGB -> BGR

        H, W = img.shape[:2]
        faces = detect_cat_faces(img)
        boxes = []

        for (x, y, w, h) in faces:
            feat = face_to_feature(img, (x, y, w, h)).reshape(1, -1)
            pred = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))

            name = id2name.get(int(pred), "Unknown")
            if proba < UNKNOWN_THRESHOLD:
                name = "Unknown"

            boxes.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "name": name,
                "proba": proba,
            })

        return {
            "user": {
                "uid": user.get("uid"),
                "email": user.get("email"),
            },
            "width": W,
            "height": H,
            "boxes": boxes,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/comments")
def get_comments(cat_name: str):
    return {"cat": cat_name, "comments": comments_db.get(cat_name, [])}

@app.post("/comment")
def post_comment(
    cat_name: str,
    payload: dict,
    user = Depends(verify_firebase_token),
):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty comment")

    author = user.get("email", "Unknown").split("@")[0]
    if cat_name not in comments_db:
        comments_db[cat_name] = []

    comments_db[cat_name].append({
        "text": text,
        "author": author,
    })

    return {"cat": cat_name, "comments": comments_db[cat_name]}
