"""
CarFarm v2 — FastAPI Backend
기준차량 기반 가격 산출 시스템
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import recommend, calculate, feedback, vehicle_info

app = FastAPI(
    title="CarFarm v2 API",
    description="기준차량 기반 자동차 경매 가격 산출 시스템",
    version="2.0.0",
)

# CORS 설정
_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
]
# 프로덕션 Firebase Hosting URL 추가
if os.environ.get("FRONTEND_URL"):
    _origins.append(os.environ["FRONTEND_URL"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(recommend.router, prefix="/api", tags=["추천"])
app.include_router(calculate.router, prefix="/api", tags=["가격산출"])
app.include_router(feedback.router, prefix="/api", tags=["피드백"])
app.include_router(vehicle_info.router, prefix="/api", tags=["차량정보"])


@app.get("/")
async def root():
    return {"message": "CarFarm v2 API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}
