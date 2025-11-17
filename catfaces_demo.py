import os
import cv2
import sys
import glob
import json
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
# -----------------------
# 可調參數（簡化版本）
# -----------------------
UNKNOWN_THRESHOLD = 0.35  # 建議先 0.55~0.65，之後再依資料微調
DATA_DIR = r"C:\Users\11032\Desktop\cats"  # 資料集根目錄：cats/<cat_name>/*.jpg
MODEL_PATH = "cat_knn.pkl"        # 模型檔
LABELS_PATH = "labels.json"       # 貓名與數字 id 對照
FACE_SIZE = (128, 128)            # 取樣尺寸
K = 5  # 先試 5 或 7
knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", algorithm="brute")
CASCADE_NAME = "haarcascade_frontalcatface.xml"

def get_cascade_path():
    """盡量自動找到 cat face cascade，找不到就用當前資料夾"""
    default_path = os.path.join(cv2.data.haarcascades, CASCADE_NAME)
    if os.path.exists(default_path):
        return default_path
    local = os.path.join(os.path.dirname(__file__), CASCADE_NAME)
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        f"找不到 {CASCADE_NAME}。\n"
        f"請將檔案放在：{cv2.data.haarcascades} 或腳本同目錄。"
    )

CAT_CASCADE = cv2.CascadeClassifier(get_cascade_path())
print("Cascade loaded:", not CAT_CASCADE.empty())

# -----------------------
# 工具函數
# -----------------------
def detect_cat_faces(img_bgr, debug=False):
    """
    強化版偵測：
    - 直方圖等化（提升對比）
    - 太大圖先等比例縮小（最長邊<=800；Haar 對超大圖常失效）
    - 嘗試旋轉 0/90/180/270 度（避免 EXIF 方向問題）
    - 找不到時做一輪備援掃描（較寬鬆參數）
    - 最後把框座標轉回原圖尺度
    """
    if CAT_CASCADE.empty():
        raise RuntimeError("Cat cascade 未載入成功，請檢查 haarcascade_frontalcatface.xml 路徑。")

    H0, W0 = img_bgr.shape[:2]

    # 縮放到較合適的尺寸（提升 Haar 成功率）
    scale = 1.0
    max_side = max(H0, W0)
    if max_side > 800:
        scale = 800.0 / max_side
        img_resized = cv2.resize(img_bgr, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_bgr.copy()

    def _detect_one(gray):
        return CAT_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.08,   # 可微調：1.03~1.2
            minNeighbors=3,     # 可微調：2~6
            minSize=(40, 40)    # 如果臉很小可降到(24,24)
        )

    candidates = []  # (faces, rot_k, (Wr, Hr))
    # 嘗試四個方向
    for rot_k in [0, 1, 2, 3]:
        test = img_resized if rot_k == 0 else np.rot90(img_resized, k=rot_k)
        gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = _detect_one(gray)
        if len(faces) > 0:
            candidates.append((faces, rot_k, gray.shape[::-1]))  # (faces, rot_k, (W, H))
            break

    # 主參數沒抓到 → 備援掃描（較寬鬆）
    if not candidates:
        if debug:
            print("[debug] 主參數未偵測到，啟動備援掃描…")
        for rot_k in [0, 1, 2, 3]:
            test = img_resized if rot_k == 0 else np.rot90(img_resized, k=rot_k)
            gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            found = []
            for sf in [1.03, 1.05, 1.08, 1.1, 1.2]:
                for mn in [2, 3, 4, 5, 6]:
                    f = CAT_CASCADE.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(24, 24))
                    if len(f) > 0:
                        found = f
                        break
                if len(found) > 0:
                    break
            if len(found) > 0:
                candidates.append((found, rot_k, gray.shape[::-1]))
                break

    if not candidates:
        return []

    faces, rot_k, (Wr, Hr) = candidates[0]

    # 把方框轉回「未旋轉前的縮圖座標」
    boxes_resized = []
    for (x, y, w, h) in faces:
        if rot_k == 0:
            x0, y0, w0, h0 = x, y, w, h
        elif rot_k == 1:  # 旋轉90（np.rot90 的方向），還原
            x0, y0 = y, Wr - (x + w)
            w0, h0 = h, w
        elif rot_k == 2:  # 180
            x0, y0 = Wr - (x + w), Hr - (y + h)
            w0, h0 = w, h
        else:             # 270
            x0, y0 = Hr - (y + h), x
            w0, h0 = h, w
        boxes_resized.append((int(x0), int(y0), int(w0), int(h0)))

    # 放大回原圖尺寸
    if scale != 1.0:
        inv = 1.0 / scale
        boxes = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in boxes_resized]
    else:
        boxes = boxes_resized

    return boxes

def face_to_feature(img_bgr, box):
    """裁切貓臉 -> 灰階 -> resize -> 向量 (L2 normalize)"""
    x, y, w, h = box
    face = img_bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    vec = gray.flatten().astype(np.float32)
    # 簡單 L2 normalize
    norm = np.linalg.norm(vec) + 1e-8
    return (vec / norm)

def scan_dataset(data_dir=DATA_DIR):
    """掃描 cats/<name>/*.jpg → (X, y, label_map)"""
    X, y = [], []
    name2id, id2name = {}, {}
    next_id = 0

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"找不到資料夾：{data_dir}")

    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        if name not in name2id:
            name2id[name] = next_id
            id2name[next_id] = name
            next_id += 1
        cid = name2id[name]

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            imgs.extend(glob.glob(os.path.join(folder, ext)))
        if not imgs:
            print(f"[警告] {name} 沒有圖片，略過。")
            continue

        kept = 0
        for p in imgs:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = detect_cat_faces(img)
            if len(faces) == 0:
                continue
            # 取第一張臉
            feat = face_to_feature(img, faces[0])
            X.append(feat)
            y.append(cid)
            kept += 1

        print(f"[資料] {name} 取樣 {kept} 張可用臉部")

    if len(X) < 2 or len(set(y)) < 2:
        raise RuntimeError("樣本不足（至少需要 >=2 隻貓且每隻有偵測到臉的圖片）。")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    return X, y, {"name2id": name2id, "id2name": {int(k): v for k, v in id2name.items()}}

def train():
    X, y, labels = scan_dataset(DATA_DIR)
    print(f"[訓練] 總樣本：{len(X)}，類別數：{len(set(y))}")

    # 建議 cosine + brute（對 cosine 距離較穩定）
    knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", algorithm="brute")
    knn.fit(X, y)

    joblib.dump(knn, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"[完成] 模型已存：{MODEL_PATH}，標籤：{LABELS_PATH}")

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise RuntimeError("找不到模型或標籤，請先執行：python catfaces_demo.py train")
    knn = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    id2name = {int(k): v for k, v in labels["id2name"].items()}
    return knn, id2name

def predict_image(img_path, show=True):
    knn, id2name = load_model()
    img = cv2.imread(img_path)
    print("讀取圖片成功：", img is not None)
    print("尺寸：", img.shape if img is not None else None)
    
    if img is None:
        raise RuntimeError(f"讀不到圖片：{img_path}")
    faces = detect_cat_faces(img)
    if len(faces) == 0:
        print("沒有偵測到貓臉。")
        if show:
            cv2.imshow("cat", img); cv2.waitKey(0)
        return

    for (x, y, w, h) in faces:
        feat = face_to_feature(img, (x, y, w, h)).reshape(1, -1)
        pred = knn.predict(feat)[0]
        distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
        proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))  # 平均相似度
        name = id2name.get(int(pred), "Unknown")
        if proba < UNKNOWN_THRESHOLD:
            name = "Unknown"
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{name} ({proba:.2f})", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if show:
        cv2.imshow("predict", img); cv2.waitKey(0)
    print("預測完成。")

    # 記錄低信心或誤判樣本，方便回顧
    if name == "Unknown" or proba < 0.6:
        os.makedirs("logs_miscls", exist_ok=True)
        out = img.copy()
        tag = name.replace(" ", "_")
        cv2.imwrite(os.path.join("logs_miscls", f"{tag}_{proba:.2f}.jpg"), out)

def webcam():
    knn, id2name = load_model()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("開啟攝影機失敗。")

    print("[提示] 按 Q 離開")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_cat_faces(frame)
        for (x, y, w, h) in faces:
            feat = face_to_feature(frame, (x, y, w, h)).reshape(1, -1)
            pred = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))
            name = id2name.get(int(pred), "Unknown")
            if proba < UNKNOWN_THRESHOLD:
                name = "Unknown"
            color = (0, 200, 255) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({proba:.2f})", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("Cat Face ID (Q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

def evaluate_on_folder(val_dir):
    knn, id2name = load_model()
    name2id = {v: k for k, v in id2name.items()}

    y_true, y_pred = [], []
    for name in sorted(os.listdir(val_dir)):
        folder = os.path.join(val_dir, name)
        if not os.path.isdir(folder): continue
        true_id = name2id.get(name)
        if true_id is None: continue

        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
            imgs.extend(glob.glob(os.path.join(folder, ext)))

        for p in imgs:
            img = cv2.imread(p)
            if img is None: continue
            faces = detect_cat_faces(img)
            if len(faces) == 0: continue

            x, y0, w, h = faces[0]
            feat = face_to_feature(img, (x, y0, w, h)).reshape(1, -1)
            pred_id = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))

            # Unknown 門檻
            if proba < UNKNOWN_THRESHOLD:
                y_pred.append(-1)  # 用 -1 代表 Unknown
            else:
                y_pred.append(int(pred_id))
            y_true.append(int(true_id))

    # 僅統計已知類別
    valid_idx = [i for i,p in enumerate(y_pred) if p != -1]
    y_true_known = [y_true[i] for i in valid_idx]
    y_pred_known = [y_pred[i] for i in valid_idx]

    if y_true_known:
        print(classification_report(
            y_true_known, y_pred_known,
            target_names=[id2name[i] for i in sorted(set(y_true_known))]
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_true_known, y_pred_known))
    else:
        print("全部都被判 Unknown，請降低 UNKNOWN_THRESHOLD 或增加資料。")

# -----------------------
# CLI
# -----------------------
def main():
    prog = os.path.basename(__file__)
    if len(sys.argv) < 2:
        print("用法：")
        print(f"  訓練： python {prog} train")
        print(f"  單張： python {prog} predict <image_path>")
        print(f"  攝影機：python {prog} webcam")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print(f"請提供圖片路徑，例如：python {prog} predict test.jpg")
            sys.exit(1)
        predict_image(sys.argv[2], show=True)
    elif cmd == "webcam":
        webcam()
    else:
        print("未知指令。可用：train / predict / webcam")

if __name__ == "__main__":
    main()
