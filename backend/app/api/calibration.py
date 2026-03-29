"""
캘리브레이션 피드백 API
목표가 입력 → 세그먼트별 보정계수 산출
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.calibration_engine import store_feedback_and_recalculate

logger = logging.getLogger(__name__)

router = APIRouter()


class CalibrationFeedbackRequest(BaseModel):
    vehicle_id: str | None = None
    maker: str
    model: str
    trim: str
    year: int
    mileage: int
    price_type: str  # "auction" | "retail" | "export"
    predicted_price: float
    target_price: float
    user_note: str | None = None


class CalibrationFeedbackResponse(BaseModel):
    status: str
    exclusion_pct: float
    direction: str
    scale_factor: float
    price_bias: float
    mileage_slope: float
    confidence: float
    feedback_count: int
    segment_key: str


@router.post("/calibration-feedback", response_model=CalibrationFeedbackResponse)
async def submit_calibration_feedback(request: CalibrationFeedbackRequest):
    """목표가 피드백 저장 및 보정계수 재계산"""
    if request.price_type not in ("auction", "retail", "export"):
        raise HTTPException(status_code=400, detail="price_type은 auction/retail/export 중 하나여야 합니다")
    if request.predicted_price <= 0:
        raise HTTPException(status_code=400, detail="predicted_price는 0보다 커야 합니다")
    if request.target_price <= 0:
        raise HTTPException(status_code=400, detail="target_price는 0보다 커야 합니다")

    try:
        result = store_feedback_and_recalculate(
            vehicle_id=request.vehicle_id,
            maker=request.maker,
            model=request.model,
            trim=request.trim,
            year=request.year,
            mileage=request.mileage,
            price_type=request.price_type,
            predicted_price=request.predicted_price,
            target_price=request.target_price,
            user_note=request.user_note,
        )
        return CalibrationFeedbackResponse(
            status="saved",
            exclusion_pct=result.exclusion_pct,
            direction=result.direction,
            scale_factor=result.scale_factor,
            price_bias=result.price_bias,
            mileage_slope=result.mileage_slope,
            confidence=result.confidence,
            feedback_count=result.feedback_count,
            segment_key=result.segment_key,
        )
    except Exception as e:
        logger.exception("캘리브레이션 피드백 처리 실패")
        raise HTTPException(status_code=500, detail=f"피드백 처리 실패: {str(e)}")
