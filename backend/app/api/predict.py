"""
LLM 기반 가격 예측 API — 유사차량 자동 분석

대상차량을 입력하면 유사 차량을 자동 수집하여
Claude Sonnet이 데이터 기반으로 적정 가격을 추론합니다.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.llm_price_predictor import predict_price

logger = logging.getLogger(__name__)

router = APIRouter()


class PriceFactorResponse(BaseModel):
    factor: str
    impact: float
    description: str


class StatsResponse(BaseModel):
    count: int = 0
    mean: float = 0
    median: float = 0
    min: float = 0
    max: float = 0


class PredictPriceResponse(BaseModel):
    estimated_auction: float
    estimated_retail: float
    confidence: str
    reasoning: str = ""
    factors: list[PriceFactorResponse] = []
    auction_reasoning: str = ""
    retail_reasoning: str = ""
    auction_factors: list[PriceFactorResponse] = []
    retail_factors: list[PriceFactorResponse] = []
    comparable_summary: str = ""
    key_comparables: list[str] = []
    vehicles_analyzed: int = 0
    auction_stats: StatsResponse = StatsResponse()
    retail_stats: StatsResponse = StatsResponse()


@router.post("/predict-price", response_model=PredictPriceResponse)
async def predict_price_endpoint(target: TargetVehicleSchema):
    """
    LLM 기반 가격 예측.

    유사 차량을 자동 수집 → Claude Sonnet이 데이터를 분석하여 적정 가격 추론.
    """
    try:
        target_dict = {
            "maker": target.maker,
            "model": target.model,
            "year": target.year,
            "mileage": target.mileage,
            "fuel": target.fuel,
            "trim": target.trim,
            "color": target.color,
            "options": target.options,
            "exchange_count": target.exchange_count,
            "bodywork_count": target.bodywork_count,
            "generation": target.generation,
            "part_damages": [
                {"part": pd.part, "damage_type": pd.damage_type}
                for pd in target.part_damages
            ],
        }

        result = await asyncio.to_thread(predict_price, target_dict)

        logger.info(
            "가격 예측 완료 — 소매: %.0f만, 낙찰: %.0f만, 신뢰도: %s, 분석: %d건 (tokens: %d+%d)",
            result.estimated_retail,
            result.estimated_auction,
            result.confidence,
            result.vehicles_analyzed,
            result.input_tokens,
            result.output_tokens,
        )

        # stats dict → StatsResponse 변환
        def _to_stats(d: dict) -> StatsResponse:
            return StatsResponse(
                count=d.get("count", 0),
                mean=d.get("mean", 0),
                median=d.get("median", 0),
                min=d.get("min", 0),
                max=d.get("max", 0),
            )

        return PredictPriceResponse(
            estimated_auction=result.estimated_auction,
            estimated_retail=result.estimated_retail,
            confidence=result.confidence,
            reasoning=result.reasoning,
            factors=[
                PriceFactorResponse(**f) for f in result.factors
            ],
            auction_reasoning=result.auction_reasoning,
            retail_reasoning=result.retail_reasoning,
            auction_factors=[
                PriceFactorResponse(**f) for f in result.auction_factors
            ],
            retail_factors=[
                PriceFactorResponse(**f) for f in result.retail_factors
            ],
            comparable_summary=result.comparable_summary,
            key_comparables=result.key_comparables,
            vehicles_analyzed=result.vehicles_analyzed,
            auction_stats=_to_stats(result.auction_stats),
            retail_stats=_to_stats(result.retail_stats),
        )

    except Exception as e:
        logger.exception("가격 예측 오류")
        raise HTTPException(
            status_code=500,
            detail=f"가격 예측 오류: {type(e).__name__}: {str(e)}",
        )
