"""
룰 엔진 가격 산출 API

기준차량 선택 후 룰 엔진이 단계별 보정을 수행.
각 단계가 투명하게 기록되어 "왜 이 가격인지" 설명 가능.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.rule_engine import calculate_price as engine_calculate
from app.services.firestore_db import get_vehicle_detail

router = APIRouter()


class CalculateRequest(BaseModel):
    """가격 산출 요청"""
    target_vehicle: TargetVehicleSchema  # 대상차량 정보 (snake_case & camelCase 모두 수용)
    reference_auction_id: str            # 선택된 기준차량 ID
    reference_auction_price: float       # 기준차량 낙찰가 (만원)


class AdjustmentStepResponse(BaseModel):
    """보정 단계"""
    rule_name: str        # 룰 이름
    rule_id: str          # 룰 ID
    description: str      # 설명
    amount: float         # 보정액 (만원)
    details: str          # 상세 계산 과정
    data_source: str      # 데이터 근거


class CalculateResponse(BaseModel):
    """가격 산출 응답"""
    reference_price: float                      # 기준차량 낙찰가
    adjustments: list[AdjustmentStepResponse]   # 보정 단계들
    total_adjustment: float                     # 총 보정액
    estimated_retail: float                     # 추정 소매가
    estimated_auction: float                    # 예상 낙찰가
    confidence: str                             # 신뢰도
    summary: str                                # 한 줄 요약


@router.post("/calculate", response_model=CalculateResponse)
async def calculate_price(request: CalculateRequest):
    """
    기준차량 기반 가격 산출.
    룰 엔진이 단계별 보정 수행, 각 단계 투명하게 표시.
    """
    # 기준차량 정보 조회
    ref_detail = get_vehicle_detail(request.reference_auction_id)
    reference = {
        "auction_price": request.reference_auction_price,
    }
    if ref_detail:
        reference.update({
            "maker": ref_detail.get("maker", ""),
            "model": ref_detail.get("model_name", ""),
            "year": ref_detail.get("연식", 0),
            "mileage": ref_detail.get("주행거리", 0),
            "color": ref_detail.get("색상", ""),
            "color_group": ref_detail.get("color_group", ""),
            "trim": ref_detail.get("trim", ""),
            "usage": ref_detail.get("usage_type", "personal"),
            "exchange_count": ref_detail.get("exchange_count", 0),
            "bodywork_count": ref_detail.get("bodywork_count", 0),
            "segment": ref_detail.get("segment", ""),
        })

    result = engine_calculate(request.target_vehicle.model_dump(), reference)
    return result
