"""
LLM 기반 가격 예측 API — 유사차량 자동 분석

대상차량을 입력하면 유사 차량을 자동 수집하여
Claude Sonnet이 데이터 기반으로 적정 가격을 추론합니다.
"""

import asyncio
import logging
import math

from fastapi import APIRouter, HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.firestore_client import get_firestore_db
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
    estimated_auction_export: float = 0
    last_export_date: str = ""
    estimated_retail: float
    confidence: str
    reasoning: str = ""
    factors: list[PriceFactorResponse] = []
    auction_reasoning: str = ""
    retail_reasoning: str = ""
    export_reasoning: str = ""
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

        # stats dict → StatsResponse 변환 (NaN-safe)
        def _to_stats(d: dict) -> StatsResponse:
            return StatsResponse(**_safe_stats(d))

        return PredictPriceResponse(
            estimated_auction=result.estimated_auction,
            estimated_auction_export=result.estimated_auction_export,
            last_export_date=result.last_export_date,
            estimated_retail=result.estimated_retail,
            confidence=result.confidence,
            reasoning=result.reasoning,
            factors=[
                PriceFactorResponse(**f) for f in result.factors
            ],
            auction_reasoning=result.auction_reasoning,
            retail_reasoning=result.retail_reasoning,
            export_reasoning=result.export_reasoning,
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


# ── 비동기 예측 (Firestore 저장) ──


@router.post("/predict-price-async")
async def predict_price_async_endpoint(
    target: TargetVehicleSchema,
):
    """
    AI 가격 예측 — Firestore에 결과 저장.

    Firestore에 processing 마킹 후 동기 실행.
    Cloud Run BackgroundTasks는 응답 후 CPU 스로틀링으로 작업이 완료되지 않으므로,
    요청 내에서 동기적으로 처리한다. 클라이언트는 Firestore 리스너로 상태를 추적.
    """
    vehicle_id = target.vehicle_id
    if not vehicle_id:
        raise HTTPException(status_code=400, detail="vehicle_id 필수")

    db = get_firestore_db()
    doc_ref = db.collection("aiPricePredictions").document(vehicle_id)

    # 즉시 processing 상태 기록 (앱의 Firestore 리스너가 바로 감지)
    doc_ref.set({
        "status": "processing",
        "createdAt": SERVER_TIMESTAMP,
        "updatedAt": SERVER_TIMESTAMP,
        "error": None,
    })

    # 동기 실행 (Cloud Run 요청 컨텍스트 내에서 CPU 보장)
    _run_prediction_sync(target, vehicle_id, doc_ref)

    return {"status": "done"}


def _safe_float(val, default=0) -> float:
    """NaN/inf → 0 변환 (Firestore에 NaN 저장되면 Flutter .round()에서 크래시)"""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _safe_stats(stats: dict) -> dict:
    """stats dict의 숫자 값을 NaN-safe하게 변환"""
    return {
        "count": int(_safe_float(stats.get("count", 0))),
        "mean": _safe_float(stats.get("mean", 0)),
        "median": _safe_float(stats.get("median", 0)),
        "min": _safe_float(stats.get("min", 0)),
        "max": _safe_float(stats.get("max", 0)),
    }


def _run_prediction_sync(target: TargetVehicleSchema, vehicle_id: str, doc_ref):
    """백그라운드에서 예측 실행 후 Firestore 업데이트 (동기 함수)."""
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

        result = predict_price(target_dict)

        logger.info(
            "비동기 가격 예측 완료 — vehicle_id=%s, 소매: %.0f만, 낙찰: %.0f만",
            vehicle_id,
            result.estimated_retail,
            result.estimated_auction,
        )

        # Firestore에 결과 저장
        doc_ref.update({
            "status": "done",
            "estimatedRetail": result.estimated_retail,
            "estimatedAuction": result.estimated_auction,
            "estimatedAuctionExport": result.estimated_auction_export,
            "lastExportDate": result.last_export_date,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "auctionReasoning": result.auction_reasoning,
            "retailReasoning": result.retail_reasoning,
            "exportReasoning": result.export_reasoning,
            "auctionFactors": [
                {"factor": f["factor"], "impact": f["impact"], "description": f["description"]}
                for f in result.auction_factors
            ],
            "retailFactors": [
                {"factor": f["factor"], "impact": f["impact"], "description": f["description"]}
                for f in result.retail_factors
            ],
            "comparableSummary": result.comparable_summary,
            "keyComparables": result.key_comparables,
            "vehiclesAnalyzed": result.vehicles_analyzed,
            "auctionStats": _safe_stats(result.auction_stats),
            "retailStats": _safe_stats(result.retail_stats),
            "updatedAt": SERVER_TIMESTAMP,
            "error": None,
        })

    except Exception as e:
        logger.exception("비동기 가격 예측 오류 — vehicle_id=%s", vehicle_id)
        try:
            doc_ref.update({
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}",
                "updatedAt": SERVER_TIMESTAMP,
            })
        except Exception:
            logger.exception("Firestore 에러 업데이트 실패")
