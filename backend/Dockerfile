FROM python:3.12-slim

WORKDIR /app

# LightGBM에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (레이어 캐싱 — requirements.txt 변경 없으면 스킵)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# Cloud Run은 PORT 환경변수 제공
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
