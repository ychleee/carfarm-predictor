"""
가격 산출 API

기준차량 선택 후 룰 엔진 또는 비율 기반 엔진이 단계별 보정을 수행.
criteria가 제공되면 비율 기반 엔진, 없으면 기존 룰 엔진 사용.
각 단계가 투명하게 기록되어 "왜 이 가격인지" 설명 가능.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.rule_engine import calculate_price as engine_calculate
from app.services.ratio_calculator import calculate_with_criteria
from app.services.firestore_db import get_vehicle_detail
from app.services.llm_criteria_analyzer import analyze_criteria

logger = logging.getLogger(__name__)

router = APIRouter()


# =========================================================================
# 공통 스키마
# =========================================================================


class PricingCriteriaInput(BaseModel):
    """보정 기준 (LLM 분석 or 사용자 수정)"""
    mileage_rate_per_10k: float = 1.5
    mileage_ceiling_km: int = 200000
    year_rate_per_year: float = 2.5


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


# =========================================================================
# /api/calculate — 가격 산출
# =========================================================================


class ReferenceInspection(BaseModel):
    """기준차량 검차 상태 (프론트엔드에서 전달 — 엔카 API 보강 데이터 반영)"""
    frame_exchange: int = 0
    frame_bodywork: int = 0
    frame_corrosion: int = 0
    exterior_exchange: int = 0
    exterior_bodywork: int = 0
    exterior_corrosion: int = 0


class CalculateRequest(BaseModel):
    """가격 산출 요청"""
    target_vehicle: TargetVehicleSchema  # 대상차량 정보 (snake_case & camelCase 모두 수용)
    reference_auction_id: str            # 선택된 기준차량 ID
    reference_auction_price: float       # 기준차량 낙찰가 (만원)
    criteria: Optional[PricingCriteriaInput] = None  # 제공 시 비율 기반 엔진 사용
    reference_inspection: Optional[ReferenceInspection] = None  # 검차 상태 (엔카 보강)
    reference_factory_price: Optional[float] = None  # 기준차량 출고가 (만원, 프론트엔드 보강)
    reference_base_price: Optional[float] = None     # 기준차량 기본가 (만원, 프론트엔드 보강)


@router.post("/calculate", response_model=CalculateResponse)
async def calculate_price(request: CalculateRequest):
    """
    기준차량 기반 가격 산출.

    - criteria 제공 시: 비율 기반 엔진 (ratio_calculator)
    - criteria 미제공 시: 기존 룰 엔진 (rule_engine) — 하위 호환
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
            "part_damages": ref_detail.get("part_damages", []),
            "segment": ref_detail.get("segment", ""),
            "factory_price": ref_detail.get("factory_price", 0),
            "base_price": ref_detail.get("base_price", 0),
            "frame_exchange": ref_detail.get("frame_exchange", 0),
            "frame_bodywork": ref_detail.get("frame_bodywork", 0),
            "frame_corrosion": ref_detail.get("frame_corrosion", 0),
            "exterior_exchange": ref_detail.get("exterior_exchange", 0),
            "exterior_bodywork": ref_detail.get("exterior_bodywork", 0),
            "exterior_corrosion": ref_detail.get("exterior_corrosion", 0),
        })

    # 프론트엔드에서 전달된 출고가/기본가로 보강 (엔카 API enrichment 반영)
    # 프론트엔드 데이터가 Firestore 폴백보다 정확하므로 항상 우선 사용
    has_frontend_price = (
        (request.reference_factory_price and request.reference_factory_price > 0)
        or (request.reference_base_price and request.reference_base_price > 0)
    )
    if has_frontend_price:
        fp = request.reference_factory_price or 0
        bp = request.reference_base_price or 0
        reference["factory_price"] = fp
        reference["base_price"] = bp
        logger.info(
            "출고가/기본가 보강 (프론트엔드) — 출고가: %.0f만, 기본가: %.0f만",
            fp, bp,
        )

    # 프론트엔드에서 전달된 검차 데이터로 보강 (엔카 API enrichment 반영)
    if request.reference_inspection:
        insp = request.reference_inspection
        # Firestore에 검차 데이터가 없을 때 프론트엔드 데이터 사용
        has_firestore_damage = (
            reference.get("frame_exchange", 0) > 0
            or reference.get("frame_bodywork", 0) > 0
            or reference.get("exterior_exchange", 0) > 0
            or reference.get("exterior_bodywork", 0) > 0
        )
        has_frontend_damage = (
            insp.frame_exchange > 0 or insp.frame_bodywork > 0
            or insp.exterior_exchange > 0 or insp.exterior_bodywork > 0
        )
        if not has_firestore_damage and has_frontend_damage:
            reference["frame_exchange"] = insp.frame_exchange
            reference["frame_bodywork"] = insp.frame_bodywork
            reference["frame_corrosion"] = insp.frame_corrosion
            reference["exterior_exchange"] = insp.exterior_exchange
            reference["exterior_bodywork"] = insp.exterior_bodywork
            reference["exterior_corrosion"] = insp.exterior_corrosion
            reference["exchange_count"] = (
                insp.frame_exchange + insp.exterior_exchange
            )
            reference["bodywork_count"] = (
                insp.frame_bodywork + insp.exterior_bodywork
            )
            # part_damages가 비어있으면 삭제하여 fallback 로직 활성화
            if not reference.get("part_damages"):
                reference["part_damages"] = []
            logger.info(
                "검차 데이터 보강 — 프레임: 교환%d 판금%d / 외판: 교환%d 판금%d",
                insp.frame_exchange, insp.frame_bodywork,
                insp.exterior_exchange, insp.exterior_bodywork,
            )

    target_dict = request.target_vehicle.model_dump()

    # criteria가 제공되면 비율 기반 엔진 사용
    if request.criteria is not None:
        criteria_dict = request.criteria.model_dump()
        logger.info(
            "비율 기반 엔진 사용 — criteria: %s, ref_factory: %s, tgt_factory: %s",
            criteria_dict,
            reference.get("factory_price"),
            target_dict.get("factory_price"),
        )
        result = calculate_with_criteria(target_dict, reference, criteria_dict)
    else:
        logger.info("룰 엔진 사용 (criteria 미제공)")
        result = engine_calculate(target_dict, reference)

    return result


# =========================================================================
# /api/analyze-criteria — LLM 보정 기준 분석
# =========================================================================


class AnalyzeCriteriaRequest(BaseModel):
    """보정 기준 분석 요청"""
    target_vehicle: TargetVehicleSchema
    reference_auction_id: str
    reference_auction_price: float


class AnalyzeCriteriaResponse(BaseModel):
    """보정 기준 분석 응답"""
    criteria: PricingCriteriaInput
    analysis_summary: str = ""
    vehicles_analyzed: int = 0
    confidence: str = "보통"


@router.post("/analyze-criteria", response_model=AnalyzeCriteriaResponse)
async def analyze_criteria_endpoint(request: AnalyzeCriteriaRequest):
    """
    LLM 기반 보정 기준 분석.

    유사차량 데이터를 Claude Sonnet에 전달하여
    주행거리/연식/트림 보정 비율(rate)을 도출한다.
    """
    try:
        # 기준차량 정보 조회
        ref_detail = get_vehicle_detail(request.reference_auction_id)
        reference = {
            "auction_price": request.reference_auction_price,
        }
        if ref_detail:
            reference.update({
                "year": ref_detail.get("연식", 0),
                "mileage": ref_detail.get("주행거리", 0),
                "trim": ref_detail.get("trim", ""),
                "fuel": ref_detail.get("연료", ""),
                "factory_price": ref_detail.get("factory_price", 0),
            })

        target_dict = {
            "maker": request.target_vehicle.maker,
            "model": request.target_vehicle.model,
            "year": request.target_vehicle.year,
            "mileage": request.target_vehicle.mileage,
            "fuel": request.target_vehicle.fuel,
            "trim": request.target_vehicle.trim,
            "factory_price": request.target_vehicle.factory_price,
        }

        # LLM 호출 (블로킹 → asyncio.to_thread)
        result = await asyncio.to_thread(
            analyze_criteria, target_dict, reference
        )

        logger.info(
            "보정 기준 분석 완료 — 신뢰도: %s, 분석: %d건 (tokens: %d+%d)",
            result.confidence,
            result.vehicles_analyzed,
            result.input_tokens,
            result.output_tokens,
        )

        return AnalyzeCriteriaResponse(
            criteria=PricingCriteriaInput(
                mileage_rate_per_10k=result.criteria.mileage_rate_per_10k,
                mileage_ceiling_km=result.criteria.mileage_ceiling_km,
                year_rate_per_year=result.criteria.year_rate_per_year,
            ),
            analysis_summary=result.analysis_summary,
            vehicles_analyzed=result.vehicles_analyzed,
            confidence=result.confidence,
        )

    except Exception as e:
        logger.exception("보정 기준 분석 오류")
        raise HTTPException(
            status_code=500,
            detail=f"보정 기준 분석 오류: {type(e).__name__}: {str(e)}",
        )
