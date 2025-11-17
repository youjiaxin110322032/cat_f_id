# seed.py
import sqlite3

conn = sqlite3.connect("catvillage.db")
cur = conn.cursor()

cur.executescript("""
-- households
CREATE TABLE IF NOT EXISTS households (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

-- caretakers
CREATE TABLE IF NOT EXISTS caretakers (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

-- cats
CREATE TABLE IF NOT EXISTS cats (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  sex TEXT,
  coat TEXT,
  ear_tip INTEGER DEFAULT 0,
  household_id INTEGER REFERENCES households(id)
);

-- many-to-many
CREATE TABLE IF NOT EXISTS cat_caretakers (
  cat_id INTEGER REFERENCES cats(id),
  caretaker_id INTEGER REFERENCES caretakers(id),
  PRIMARY KEY (cat_id, caretaker_id)
);

-- 測試資料（可依需求調整）
INSERT INTO households (id, name) VALUES (1,'北巷口'),(2,'南倉庫') ON CONFLICT DO NOTHING;
INSERT INTO caretakers (id, name) VALUES (1,'阿花'),(2,'小樺') ON CONFLICT DO NOTHING;
INSERT INTO cats (id,name,sex,coat,ear_tip,household_id)
VALUES
  (1,'mama','F','tabby',1,1),
  (2,'coco','M','black',0,2)
ON CONFLICT DO NOTHING;

INSERT INTO cat_caretakers (cat_id, caretaker_id) VALUES
  (1,1),(1,2),(2,2)
ON CONFLICT DO NOTHING;
""")

conn.commit()
conn.close()

print("✅ catvillage.db 建立成功，資料已初始化！")
