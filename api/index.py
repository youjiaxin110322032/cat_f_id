# api/index.py
import io
import os
import sys
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 🔐 安全性相關引用
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth

# =========================
# 🧠 0. 載入辨識模組 (路徑防呆)
# =========================
try:
    # 優先當成 api 套件
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD
except ImportError:
    # 若失敗就把上層路徑加進去，再 import
    sys.path.append("..")
    from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

app = FastAPI(title="Cat Face ID API", version="1.1")

# 建立 Bearer 驗證器（給 Security 用）
bearer = HTTPBearer(auto_error=False)

# =========================
# 🔥 1. Firebase 初始化
# =========================
if not firebase_admin._apps:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)                  # 往上一層
    key_path = os.path.join(root_dir, "firebase.json")       # 改成找根目錄

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

API_KEY = os.getenv("API_KEY")  # 目前沒用到，但保留

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
if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
    """
    回傳目前用 Bearer Token 驗證過的會員資訊
    """
    return {
        "uid": user.get("uid"),
        "email": user.get("email"),
        # 若你有在 Firebase 設 displayName，也可以順便回
        "name": user.get("name")
    }

@app.get("/")
def root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "frontend/index.html not found"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/labels")
def labels():
    """檢查目前模型的已知貓名"""
    return {
        "count": len(id2name),
        "labels": [id2name[i] for i in sorted(id2name.keys())],
    }

@app.post("/camera_open") # 打開相機的紀錄
def camera_open(user = Depends(verify_firebase_token)):
    email = user.get("email")
    uid = user.get("uid")
    print(f"📷 Camera opened by {email} ({uid})")
    return {"email": email, "uid": uid}

@app.post("/reload")
def reload_model():
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user = Depends(verify_firebase_token),  # decoded Firebase token
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

    author = user.get("email") or user.get("uid") or "匿名貓奴"

    if cat_name not in comments_db:
        comments_db[cat_name] = []

    comments_db[cat_name].append({
        "text": text,
        "author": author,
    })

    return {"cat": cat_name, "comments": comments_db[cat_name]}
