from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 本地 SQLite 檔案
SQLITE_URL = "sqlite:///./catvillage.db"

engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False}  # SQLite in threads
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
