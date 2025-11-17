# libsql_db.py
import os
from libsql_client import create_client

# 從環境變數讀取 LibSQL 連線資訊（Vercel 介面填入）
LIBSQL_URL = os.getenv("LIBSQL_URL", "")
LIBSQL_AUTH_TOKEN = os.getenv("LIBSQL_AUTH_TOKEN", "")

# 建立 client（注意：這是雲端連線，無檔案）
_client = create_client(url=LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)

def query_all_cats():
    """
    回傳與本地 /cats 一樣的資料格式：
    [
      { id, name, sex, coat, ear_tip, household, caretakers: [ ... ] },
      ...
    ]
    """
    # 1) 抓 cat + household
    sql = """
    SELECT c.id, c.name, c.sex, c.coat, c.ear_tip,
           h.name AS household_name
    FROM cats c
    LEFT JOIN households h ON h.id = c.household_id
    ORDER BY c.id;
    """
    cats_rows = _client.execute(sql).rows

    # 2) 一次抓 caretakers 關聯
    care_sql = """
    SELECT ck.cat_id, t.name AS caretaker_name
    FROM cat_caretakers ck
    JOIN caretakers t ON t.id = ck.caretaker_id
    ORDER BY ck.cat_id;
    """
    care_rows = _client.execute(care_sql).rows

    # 整理成 {cat_id: [name, ...]}
    care_map = {}
    for r in care_rows:
        cat_id = r["cat_id"]
        care_map.setdefault(cat_id, []).append(r["caretaker_name"])

    # 組裝回應
    out = []
    for r in cats_rows:
        out.append({
            "id": r["id"],
            "name": r["name"],
            "sex": r["sex"],
            "coat": r["coat"],
            "ear_tip": bool(r["ear_tip"]),
            "household": r["household_name"],
            "caretakers": care_map.get(r["id"], []),
        })
    return out
