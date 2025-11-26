# api/index.py
import io
import os
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 🔐 新增：FastAPI 的 API Key 工具
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
bearer = HTTPBearer(auto_error=False)

import firebase_admin # Firebase Admin SDK
from firebase_admin import credentials, auth # 用來驗證 ID Token
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)

def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer)
):
    if not credentials:
        raise HTTPException(401, "Missing Bearer Token")

    token = credentials.credentials

    try:
        decoded = auth.verify_id_token(token)
        return decoded   # decoded['uid'], decoded['email'] 都讀得到
    except Exception:
        raise HTTPException(401, "Invalid Firebase token")
    
# 你的辨識模組（確保這些檔案在 repo 根目錄，或可被 import）
# - catfaces_demo.py
# - cat_knn.pkl
# - labels.json
from catfaces_demo import (
    load_model,
    detect_cat_faces,
    face_to_feature,
    K,
    UNKNOWN_THRESHOLD,
)

app = FastAPI(title="Cat Face ID API", version="1.1")

# =========================
# 🔐 Secure API 設定區
# =========================

# 從環境變數讀 API Key（例如在部署平台設定 API_KEY）
API_KEY = os.getenv("API_KEY")  # 例如 "super-secret-key"
API_KEY_HEADER_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Secure API 共用的驗證函式：
    - 沒帶 x-api-key → 401
    - key 跟環境變數設定不一樣 → 401
    """
    if API_KEY is None:
        # 代表你後端根本沒設定 API_KEY，算是主機端的設定問題
        raise HTTPException(
            status_code=500,
            detail="Server API_KEY is not configured."
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )
    return api_key

# CORS：把前端網域加進來
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://youjiaxin110322032.github.io",  # GitHub Pages
        # "https://你的前端其他網域",             # 之後如果有再加
        "http://localhost:5500",                 # 本機開靜態檔測試可選擇加
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === 🐾 前端靜態檔案（放在 frontend 資料夾內） ===
if not os.path.exists("frontend"):
    os.makedirs("frontend")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
# 啟動時載入模型（Serverless：函式實例冷啟動時會跑一次）

# 載入模型
try:
    knn, id2name = load_model()
except RuntimeError as e:
    print("[warning] load_model 失敗：", e)
    knn, id2name = None, {}

comments_db = {} # {"mama": ["留言1"], "tama": ["留言2"]}

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

@app.post("/reload")
def reload_model(api_key: str = Depends(verify_api_key)):
    """
    可用這個端點做熱重載、鎖 API Key
    """
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "count": len(id2name)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user = Depends(verify_firebase_token),  # 這裡其實是 decoded token
):
    if knn is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    try:
        raw = await file.read()
        # 讀圖（RGB）→ Numpy → BGR（給 OpenCV 流程使用）
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

        return {"width": W, "height": H, "boxes": boxes}
    except Exception as e:
        # 返回 400 或 500 視需求調整，這裡回傳 400 並帶上錯誤訊息
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
    
    # 從 token 拿 email/uid，組留言作者
    author = user.get("email") or user.get("uid") or "匿名貓奴"

    if cat_name not in comments_db:
        comments_db[cat_name] = []
    
    comments_db[cat_name].append({
        "text": text,
        "author": author,
    })
        
    return {"cat": cat_name, "comments": comments_db[cat_name]}
