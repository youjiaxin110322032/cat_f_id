import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from catfaces_demo import (
    load_model, detect_cat_faces, face_to_feature,
    K, UNKNOWN_THRESHOLD
)

# === 🐱 調整辨識靈敏度 ===
K = 3
UNKNOWN_THRESHOLD = 0.65

app = FastAPI(title="Cat Face ID API", version="1.2")

# === 🌐 允許前端跨域訪問 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 🐾 前端靜態檔案（放在 frontend 資料夾內） ===
if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# === 🏠 首頁（顯示 index.html） ===
@app.get("/")
def read_root():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "frontend/index.html not found"}

# === 🧠 模型與資料 ===
comments_db = {}  # {"mama": ["留言1"], "tama": ["留言2"]}
knn, id2name = load_model()

@app.get("/labels")
def labels():
    return {"count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

@app.post("/reload")
def reload_model():
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "count": len(id2name)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(img)[:, :, ::-1]  # PIL -> OpenCV BGR

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

        H, W = img.shape[:2]
        return {"width": W, "height": H, "boxes": boxes}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or server error: {e}")

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
