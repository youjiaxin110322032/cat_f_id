-- cats.sql
CREATE TABLE IF NOT EXISTS households (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  address TEXT
);

CREATE TABLE IF NOT EXISTS caretakers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  phone TEXT
);

CREATE TABLE IF NOT EXISTS cats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  sex TEXT CHECK (sex IN ('M','F','U')) DEFAULT 'U',
  birth_year INTEGER,
  coat TEXT,
  ear_tip INTEGER DEFAULT 0,      -- 是否結紮耳剪
  household_id INTEGER REFERENCES households(id)
);

CREATE TABLE IF NOT EXISTS cat_caretaker (
  cat_id INTEGER REFERENCES cats(id),
  caretaker_id INTEGER REFERENCES caretakers(id),
  PRIMARY KEY (cat_id, caretaker_id)
);

CREATE TABLE IF NOT EXISTS feeding_spots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  lat REAL, lng REAL
);

CREATE TABLE IF NOT EXISTS feed_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cat_id INTEGER REFERENCES cats(id),
  spot_id INTEGER REFERENCES feeding_spots(id),
  fed_at TEXT NOT NULL,           -- ISO 時間字串
  food TEXT,
  note TEXT
);

CREATE TABLE IF NOT EXISTS medical_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cat_id INTEGER REFERENCES cats(id),
  visit_at TEXT NOT NULL,
  diagnosis TEXT,
  treatment TEXT,
  vet TEXT
);

CREATE TABLE IF NOT EXISTS adoptions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cat_id INTEGER REFERENCES cats(id),
  adopter TEXT,
  adopted_at TEXT
);
