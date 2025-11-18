# api/index.py
import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# CORS：把前端網域加進來
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://youjiaxin110322032.github.io",     # GitHub Pages
        "https://<你的-vercel-前端-網域>"               # 之後替換成你的 Vercel 網域
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
knn, id2name = load_model()
knn, id2name = load_model()

@app.get("/")
def root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "frontend/index.html not found"}
# === 🧠 模型與資料 ===
comments_db = {}  # {"mama": ["留言1"], "tama": ["留言2"]}
# 載入模型
try:
    knn, id2name = load_model()
except RuntimeError as e:
    print("[warning] load_model 失敗：", e)
    knn, id2name = None, {}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/labels")
def labels():
    """檢查目前模型的已知貓名"""
    return {"count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

@app.post("/reload")
def reload_model():
    """若你更新了 cat_knn.pkl / labels.json，可用這個端點做熱重載"""
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "count": len(id2name)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "name": name, "proba": proba
            })

        return {"width": W, "height": H, "boxes": boxes}
    except Exception as e:
        # 返回 400 或 500 視需求調整，這裡回傳 400 並帶上錯誤訊息
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/comments")
def get_comments(cat_name: str):
    return {"cat": cat_name, "comments": comments_db.get(cat_name, [])}

@app.post("/comment")
def post_comment(cat_name: str, payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty comment")
    if cat_name not in comments_db:
        comments_db[cat_name] = []
    comments_db[cat_name].append(text)
    return {"cat": cat_name, "comments": comments_db[cat_name]}

