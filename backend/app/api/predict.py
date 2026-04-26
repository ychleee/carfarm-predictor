"""
LLM 기반 가격 예측 API — 유사차량 자동 분석

대상차량을 입력하면 유사 차량을 자동 수집하여
Claude Sonnet이 데이터 기반으로 적정 가격을 추론합니다.
"""

import asyncio
import logging
import math
import time

from fastapi import APIRouter, HTTPException
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.firestore_client import get_firestore_db
from app.services.llm_price_predictor import predict_price
from app.services.llm_price_predictor_gemini import predict_price_gemini, build_i2_result
from app.services.llm_price_predictor_gemini_raw import predict_price_gemini_raw, build_i3_result
from app.services.gemini_shared import collect_data, call_gemini_llm
from app.services.ml_price_predictor import predict_price_ml

logger = logging.getLogger(__name__)

router = APIRouter()


class PriceFactorResponse(BaseModel):
    factor: str
    impact: float
    description: str


class PriceStatsResponse(BaseModel):
    count: int = 0
    mean: float = 0
    median: float = 0
    min: float = 0
    max: float = 0
    std: float = 0


class PredictPriceResponse(BaseModel):
    estimated_auction: float
    estimated_auction_export: float = 0
    last_export_date: str = ""
    estimated_retail: float
    confidence: str
    reasoning: str = ""
    factors: list[PriceFactorResponse] = []
    auction_factors: list[PriceFactorResponse] = []
    retail_factors: list[PriceFactorResponse] = []
    auction_reasoning: str = ""
    retail_reasoning: str = ""
    export_reasoning: str = ""
    comparable_summary: str = ""
    key_comparables: list[str] = []
    vehicles_analyzed: int = 0
    auction_stats: PriceStatsResponse | None = None
    retail_stats: PriceStatsResponse | None = None
    comparable_auction_vehicles: list[dict] = []
    comparable_retail_vehicles: list[dict] = []
    retail_brackets: list[dict] = []
    auction_brackets: list[dict] = []
    export_brackets: list[dict] = []


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
            "factory_price": target.factory_price,
            "base_price": target.base_price,
            "part_damages": [
                {"part": pd.part, "damage_type": pd.damage_type}
                for pd in target.part_damages
            ],
        }

        result = await asyncio.to_thread(predict_price_gemini, target_dict)

        logger.info(
            "가격 예측 완료 — 소매: %.0f만, 낙찰: %.0f만, 신뢰도: %s, 분석: %d건 (tokens: %d+%d)",
            result.estimated_retail,
            result.estimated_auction,
            result.confidence,
            result.vehicles_analyzed,
            result.input_tokens,
            result.output_tokens,
        )

        def _to_stats(s) -> PriceStatsResponse | None:
            if not s:
                return None
            return PriceStatsResponse(
                count=s.get("count", 0),
                mean=s.get("mean", 0),
                median=s.get("median", 0),
                min=s.get("min", 0),
                max=s.get("max", 0),
                std=s.get("std", 0),
            )

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
            auction_factors=[
                PriceFactorResponse(**f) for f in result.auction_factors
            ],
            retail_factors=[
                PriceFactorResponse(**f) for f in result.retail_factors
            ],
            auction_reasoning=result.auction_reasoning,
            retail_reasoning=result.retail_reasoning,
            export_reasoning=result.export_reasoning,
            comparable_summary=result.comparable_summary,
            key_comparables=result.key_comparables,
            vehicles_analyzed=result.vehicles_analyzed,
            auction_stats=_to_stats(result.auction_stats),
            retail_stats=_to_stats(result.retail_stats),
            comparable_auction_vehicles=result.comparable_auction_vehicles,
            comparable_retail_vehicles=result.comparable_retail_vehicles,
            retail_brackets=result.retail_brackets,
            auction_brackets=result.auction_brackets,
            export_brackets=result.export_brackets,
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

    # 스레드 풀에서 실행 — 이벤트 루프 블로킹 방지로 동시 다중 예측 가능
    # Cloud Run 요청 컨텍스트 내이므로 CPU 스로틀링 없이 완료까지 보장
    await asyncio.to_thread(_run_prediction_sync, target, vehicle_id, doc_ref)

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
            "factory_price": target.factory_price,
            "base_price": target.base_price,
            "part_damages": [
                {"part": pd.part, "damage_type": pd.damage_type}
                for pd in target.part_damages
            ],
        }

        result = predict_price_gemini(target_dict)

        logger.info(
            "비동기 가격 예측 완료 — vehicle_id=%s, 소매: %.0f만, 낙찰: %.0f만",
            vehicle_id,
            result.estimated_retail,
            result.estimated_auction,
        )

        # i3(ML) 모델도 함께 실행
        i3_data = {}
        try:
            ml_result = predict_price_ml(target_dict)
            i3_data = {
                "i3EstimatedAuction": _safe_float(ml_result.estimated_auction),
                "i3EstimatedAuctionExport": _safe_float(ml_result.estimated_auction_export),
                "i3EstimatedRetail": _safe_float(ml_result.estimated_retail),
                "i3Confidence": ml_result.confidence,
                "i3AuctionReasoning": ml_result.auction_reasoning,
                "i3RetailReasoning": ml_result.retail_reasoning,
                "i3ExportReasoning": ml_result.export_reasoning,
                "i3AuctionBrackets": ml_result.auction_brackets,
                "i3ExportBrackets": ml_result.export_brackets,
                "i3RetailBrackets": ml_result.retail_brackets,
            }
            logger.info(
                "i3(ML) 예측 완료 — vehicle_id=%s, 소매: %.0f만, 낙찰: %.0f만",
                vehicle_id,
                ml_result.estimated_retail,
                ml_result.estimated_auction,
            )
        except Exception as ml_err:
            logger.warning("i3(ML) 예측 실패 — vehicle_id=%s: %s", vehicle_id, ml_err)

        # Firestore에 결과 저장
        update_data = {
            "status": "done",
            "color": target.color or "",
            "factoryPrice": _safe_float(target.factory_price),
            "basePrice": _safe_float(target.base_price),
            "estimatedRetail": result.estimated_retail,
            "estimatedAuction": result.estimated_auction,
            "estimatedAuctionExport": result.estimated_auction_export,
            "lastExportDate": result.last_export_date,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "auctionReasoning": result.auction_reasoning,
            "retailReasoning": result.retail_reasoning,
            "exportReasoning": result.export_reasoning,
            "auctionFactors": result.auction_factors,
            "retailFactors": result.retail_factors,
            "comparableSummary": result.comparable_summary,
            "keyComparables": result.key_comparables,
            "vehiclesAnalyzed": result.vehicles_analyzed,
            "auctionStats": result.auction_stats or None,
            "retailStats": result.retail_stats or None,
            "comparableAuctionVehicles": result.comparable_auction_vehicles,
            "comparableRetailVehicles": result.comparable_retail_vehicles,
            "retailBrackets": result.retail_brackets,
            "auctionBrackets": result.auction_brackets,
            "exportBrackets": result.export_brackets,
            "updatedAt": SERVER_TIMESTAMP,
            "error": None,
        }
        update_data.update(i3_data)
        doc_ref.update(update_data)

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


# ── 멀티 모델 동시 예측 (모델 개발용) ──


class CommonDataResponse(BaseModel):
    """모델 공통 데이터 — 수집된 유사차량 + 통계 + 가격추이"""
    comparable_auction_vehicles: list[dict] = []
    comparable_retail_vehicles: list[dict] = []
    auction_stats: PriceStatsResponse | None = None
    retail_stats: PriceStatsResponse | None = None
    vehicles_analyzed: int = 0
    last_export_date: str = ""
    auction_brackets: list[dict] = []
    export_brackets: list[dict] = []
    retail_brackets: list[dict] = []


class ModelResult(BaseModel):
    """모델별 결과 — 산출방식 + 추정비율 + 가격"""
    model_id: str
    model_name: str
    elapsed_ms: int = 0
    error: str | None = None
    estimated_auction: float = 0
    estimated_auction_export: float = 0
    estimated_retail: float = 0
    confidence: str = ""
    reasoning: str = ""
    auction_reasoning: str = ""
    retail_reasoning: str = ""
    export_reasoning: str = ""
    factors: list[PriceFactorResponse] = []
    auction_factors: list[PriceFactorResponse] = []
    retail_factors: list[PriceFactorResponse] = []
    comparable_summary: str = ""
    key_comparables: list[str] = []
    retail_brackets: list[dict] = []
    auction_brackets: list[dict] = []
    export_brackets: list[dict] = []


class MultiModelResponse(BaseModel):
    common_data: CommonDataResponse
    results: list[ModelResult]


def _result_to_model_result(
    model_id: str, model_name: str, result, elapsed_ms: int,
) -> ModelResult:
    return ModelResult(
        model_id=model_id,
        model_name=model_name,
        elapsed_ms=elapsed_ms,
        estimated_auction=result.estimated_auction,
        estimated_auction_export=result.estimated_auction_export,
        estimated_retail=result.estimated_retail,
        confidence=result.confidence,
        reasoning=result.reasoning,
        auction_reasoning=result.auction_reasoning,
        retail_reasoning=result.retail_reasoning,
        export_reasoning=result.export_reasoning,
        factors=[PriceFactorResponse(**f) for f in result.factors],
        auction_factors=[PriceFactorResponse(**f) for f in result.auction_factors],
        retail_factors=[PriceFactorResponse(**f) for f in result.retail_factors],
        comparable_summary=result.comparable_summary,
        key_comparables=result.key_comparables,
        retail_brackets=result.retail_brackets,
        auction_brackets=result.auction_brackets,
        export_brackets=result.export_brackets,
    )


@router.post("/predict-price-multi", response_model=MultiModelResponse)
async def predict_price_multi_endpoint(target: TargetVehicleSchema):
    """
    멀티 모델 동시 가격 예측 (모델 개발용).

    1) 공통 데이터 수집 (Firestore 유사차량) → common_data
    2) i2: Gemini LLM + 시장데이터 보정 → 산출방식/가격
    3) i3: ML 모델 (LightGBM) → 산출방식/가격
    """
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
        "factory_price": target.factory_price,
        "base_price": target.base_price,
        "part_damages": [
            {"part": pd.part, "damage_type": pd.damage_type}
            for pd in target.part_damages
        ],
    }

    # 1) 공통 데이터 수집
    from app.services.llm_price_predictor import (
        _compact_auction_vehicle,
        _compact_retail_vehicle,
    )

    data = await asyncio.to_thread(collect_data, target_dict)

    compact_auction = [_compact_auction_vehicle(v) for v in data.auction_vehicles[:20]]
    compact_retail = [_compact_retail_vehicle(v) for v in data.retail_vehicles[:15]]

    export_dates = [
        v.get("개최일", "") for v in data.auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    def _to_stats(s) -> PriceStatsResponse | None:
        if not s or s.get("count", 0) == 0:
            return None
        return PriceStatsResponse(
            count=s.get("count", 0),
            mean=s.get("mean", 0),
            median=s.get("median", 0),
            min=s.get("min", 0),
            max=s.get("max", 0),
            std=s.get("std", 0),
        )

    common_data = CommonDataResponse(
        comparable_auction_vehicles=compact_auction,
        comparable_retail_vehicles=compact_retail,
        auction_stats=_to_stats(data.auction_stats),
        retail_stats=_to_stats(data.retail_stats),
        vehicles_analyzed=data.total_vehicles,
        last_export_date=last_export_date,
    )

    # 2) i2: Gemini + 시장데이터 (수집된 data 재사용)
    # 3) i3: ML 모델 (data 불필요, 독립 실행)

    async def _run_i2():
        t_start = time.monotonic()
        try:
            if data.total_vehicles == 0:
                return ModelResult(
                    model_id="i2", model_name="Gemini + 시장데이터",
                    error="유사 차량 데이터를 찾을 수 없어 가격 예측이 불가합니다.",
                )
            llm_result = await asyncio.to_thread(call_gemini_llm, data.user_message)
            result = await asyncio.to_thread(build_i2_result, target_dict, data, llm_result)
            elapsed = int((time.monotonic() - t_start) * 1000)
            return _result_to_model_result("i2", "Gemini + 시장데이터", result, elapsed)
        except Exception as e:
            elapsed = int((time.monotonic() - t_start) * 1000)
            logger.exception("[i2] 오류")
            return ModelResult(
                model_id="i2", model_name="Gemini + 시장데이터",
                elapsed_ms=elapsed, error=str(e),
            )

    async def _run_i3():
        t_start = time.monotonic()
        try:
            result = await asyncio.to_thread(predict_price_ml, target_dict)
            elapsed = int((time.monotonic() - t_start) * 1000)
            return _result_to_model_result("i3", "ML 모델 (LightGBM)", result, elapsed)
        except Exception as e:
            elapsed = int((time.monotonic() - t_start) * 1000)
            logger.exception("[i3-ML] 오류")
            return ModelResult(
                model_id="i3", model_name="ML 모델 (LightGBM)",
                elapsed_ms=elapsed, error=str(e),
            )

    i2_result, i3_result = await asyncio.gather(_run_i2(), _run_i3())

    # i2의 시장데이터 brackets을 공통 가격추이로 사용
    if not i2_result.error:
        common_data = common_data.model_copy(update={
            "auction_brackets": i2_result.auction_brackets,
            "export_brackets": i2_result.export_brackets,
            "retail_brackets": i2_result.retail_brackets,
        })

    logger.info(
        "[multi] i2: 낙찰=%s, 소매=%s / i3: 낙찰=%s, 소매=%s / 공통 %d건",
        i2_result.estimated_auction, i2_result.estimated_retail,
        i3_result.estimated_auction, i3_result.estimated_retail,
        common_data.vehicles_analyzed,
    )

    return MultiModelResponse(common_data=common_data, results=[i2_result, i3_result])
