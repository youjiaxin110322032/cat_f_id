# Dockerfile
FROM python:3.12-slim

# 系統層面需要的基本套件（opencv headless 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 告訴 FastAPI 進入雲端模式
ENV ENV=vercel

# 啟動
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
