# api/index.py
import io
import os
import sys
import numpy as np
import httpx
from datetime import datetime
from typing import Dict, List
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Response

import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # .../cat_f_id-main/api
PROJECT_ROOT = os.path.dirname(BASE_DIR)                  # .../cat_f_id-main

# -----------------------------------------------------------
# 1. 路徑與環境設定
# -----------------------------------------------------------

# 強制將專案根目錄加入 Python 搜尋路徑 (解決找不到模組問題)
current_dir = os.path.dirname(os.path.abspath(__file__)) # api 資料夾
root_dir = os.path.dirname(current_dir) # 專案根目錄
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 載入 .env
load_dotenv()

# 嘗試載入本地模組 (Models)
try:
    from .models import ChatRequest, ChatMessage
except ImportError:
    # 本地直接執行時可能需要這行
    from models import ChatRequest, ChatMessage

# -----------------------------------------------------------
# 2. 載入辨識模組 (放在最上方以免找不到)
# -----------------------------------------------------------
try:
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD
except ImportError:
    # 再次確保路徑正確 (雖然後面 sys.path 加過了，防呆用)
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

# -----------------------------------------------------------
# 3. 全域變數與生命週期 (Lifespan)
# -----------------------------------------------------------

knn = None
id2name = {}
comments_db = {}
user_history: Dict[str, List[ChatMessage]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 啟動時執行 ---
    global knn, id2name
    print("🚀 伺服器啟動中，開始載入模型...")
    try:
        # 這裡才載入模型，避免記憶體在 Import 階段就爆炸
        knn, id2name = load_model()
        print(f"✅ 模型載入成功！包含 {len(id2name)} 個類別")
    except Exception as e:
        print(f"⚠️ 模型載入失敗 (可能是記憶體不足或檔案遺失): {e}")
        knn, id2name = None, {}
    
    yield  # 應用程式開始運作
    
    # --- 關閉時執行 (清理資源) ---
    print("🛑 伺服器關閉，清理資源...")
    knn = None
    id2name = {}

# -----------------------------------------------------------
# 4. 初始化 FastAPI 與 設定
# -----------------------------------------------------------

app = FastAPI(title="Cat Face LLM Chat", version="1.1", lifespan=lifespan)

# LLM 設定檢查
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

if not LLM_API_KEY:
    print("⚠️ 警告: LLM_API_KEY 未設定，聊天功能將無法使用")
if not LLM_ENDPOINT:
    print("⚠️ 警告: LLM_ENDPOINT 未設定")

print("🔧 LLM 設定：")
print(" - MODEL    =", LLM_MODEL)
print(" - ENDPOINT =", LLM_ENDPOINT)
print(" - KEY 前 6 =", LLM_API_KEY[:6] if LLM_API_KEY else "None", "...")

# -----------------------------------------------------------
# 5. Firebase 初始化
# -----------------------------------------------------------
key_path = os.path.join(PROJECT_ROOT, "firebase.json") # 定義 key_path

if not firebase_admin._apps:
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
        try:
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized from environment variables")
        except Exception as e:
             print(f"❌ Firebase init failed (Env Vars): {e}")

    elif os.path.exists(key_path):
        # ✅ 沒有 env，就用本地 firebase.json
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        print(f"✅ Firebase initialized from file: {key_path}")
    else:
        # ❌ 兩邊都沒有
        print("❌ Firebase init failed: no env vars and no firebase.json")

# 建立 Bearer 驗證器
bearer = HTTPBearer(auto_error=False)

def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Security(bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer Token")
    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        print("❌ Auth Failed:", e)
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

# -----------------------------------------------------------
# 6. Middleware 與 靜態檔案
# -----------------------------------------------------------

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

# 前端靜態檔案
static_path = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
else:
    print(f"⚠️ Warning: 'frontend' folder not found at {static_path}")

# -----------------------------------------------------------
# 7. API路由
# -----------------------------------------------------------

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
async def chat(req: ChatRequest, user = Depends(verify_firebase_token)):
    uid = user.get("uid") or user.get("email")
    if not uid:
        raise HTTPException(status_code=400, detail="No uid or email in token")

    # 1. 寫入歷史
    history = user_history.setdefault(uid, [])
    history.append(ChatMessage(role="user", content=req.message, timestamp=datetime.utcnow()))

    # 2. 截斷歷史
    last_messages = history[-10:]
    def truncate(text: str, max_len: int = 1000) -> str:
        text = text or ""
        return text[-max_len:] if len(text) > max_len else text

    # 3. 準備 Payload
    messages_payload = [
        {"role": "system", "content": "你是一隻活潑但專業的貓咪識別與陪聊助手，說話可以可愛一點，但重點要清楚、具體，使用繁體中文回答。"}
    ]
    for m in last_messages:
        messages_payload.append({"role": m.role, "content": truncate(m.content)})

    # 4. 呼叫 API
    target_url = LLM_ENDPOINT
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": messages_payload,
        "temperature": 0.7,
        "max_tokens": 512,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(target_url, headers=headers, json=payload)
            if r.status_code != 200:
                print(f"❌ API Error: {r.status_code} - {r.text}")
            r.raise_for_status()
            data = r.json()
            assistant_reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ LLM Call Failed: {e}")
        # 若失敗，回傳錯誤訊息給前端，不要讓前端掛著
        raise HTTPException(status_code=500, detail=str(e))

    # 5. 寫回歷史
    history.append(ChatMessage(role="assistant", content=assistant_reply, timestamp=datetime.utcnow()))
    print(f"💬 LLM 回覆給 {uid}: {assistant_reply} (via {LLM_ENDPOINT})")
    
    return {"reply": assistant_reply, "history_len": len(history)}

@app.get("/history")
def get_history(user = Depends(verify_firebase_token)):
    uid = user.get("uid") or user.get("email")
    if not uid:
        raise HTTPException(status_code=400, detail="No uid")
    history = user_history.get(uid, [])
    return [{"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in history]

@app.post("/camera_open")
def camera_open(user = Depends(verify_firebase_token)):
    return {"email": user.get("email"), "uid": user.get("uid")}

@app.post("/reload")
def reload_model(user: dict = Depends(verify_firebase_token)):
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "by_user": user.get("email")}

@app.post("/predict")
async def predict(file: UploadFile = File(...), user = Depends(verify_firebase_token)):
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
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "name": name, "proba": proba})

        return {"user": {"uid": user.get("uid"), "email": user.get("email")}, "width": W, "height": H, "boxes": boxes}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/comments")
def get_comments(cat_name: str):
    return {"cat": cat_name, "comments": comments_db.get(cat_name, [])}

@app.post("/comment")
def post_comment(cat_name: str, payload: dict, user = Depends(verify_firebase_token)):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty comment")

    author = user.get("email", "Unknown").split("@")[0]
    if cat_name not in comments_db:
        comments_db[cat_name] = []

    comments_db[cat_name].append({"text": text, "author": author})
    return {"cat": cat_name, "comments": comments_db[cat_name]}

# -----------------------------------------------------------
# 8. 程式進入點 (移到最外層)
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # 這樣你可以直接用 python api/index.py 執行
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)