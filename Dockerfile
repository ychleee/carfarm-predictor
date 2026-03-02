FROM python:3.11-slim

WORKDIR /app

# Python 의존성 설치
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 코드 복사
COPY backend/ ./backend/

# 포트 설정
ENV PORT=8080
EXPOSE 8080

# uvicorn 실행 (Cloud Run은 PORT 환경변수 사용)
CMD ["sh", "-c", "cd backend && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
