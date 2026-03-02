"""
기준차량 추천 API — LLM 에이전트 기반 (소매가/낙찰가 병렬 추천)

소매가(엔카)와 낙찰가 각각에 대해 LLM(Claude Sonnet) 에이전트가
독립적으로 DB 검색 → 분석 → 기준차량 3건 선별.
두 추천을 asyncio로 병렬 실행하여 응답 시간 단축.
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.firestore_db import get_vehicle_detail, get_retail_detail
from app.services.llm_recommender import (
    recommend_references,
    recommend_retail_references,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class RetailVehicle(BaseModel):
    """엔카 소매가 차량"""
    auction_id: str
    vehicle_name: str | None = None
    year: int | None = None
    mileage: int | None = None
    retail_price: float | None = None   # 소매가 (만원)
    color: str | None = None
    trim: str | None = None
    source_url: str | None = None
    factory_price: float | None = None  # 출고가 (만원)
    options: str | None = None          # 옵션
    fuel: str | None = None             # 연료
    # 프레임 검차
    frame_exchange: int = 0
    frame_bodywork: int = 0
    frame_corrosion: int = 0
    # 외부패널 검차
    exterior_exchange: int = 0
    exterior_bodywork: int = 0
    exterior_corrosion: int = 0
    # LLM 추천 이유
    reason: str | None = None


class AuctionVehicle(BaseModel):
    """낙찰가 차량"""
    auction_id: str
    vehicle_name: str | None = None
    year: int | None = None
    mileage: int | None = None
    auction_price: float | None = None  # 낙찰가 (만원)
    auction_date: str | None = None
    color: str | None = None
    trim: str | None = None
    options: str | None = None
    factory_price: float | None = None  # 출고가 (만원)
    inspection_grade: str | None = None # 검차등급
    is_export: bool = False             # 수출여부
    fuel: str | None = None             # 연료
    # 프레임 검차
    frame_exchange: int = 0
    frame_bodywork: int = 0
    frame_corrosion: int = 0
    # 외부패널 검차
    exterior_exchange: int = 0
    exterior_bodywork: int = 0
    exterior_corrosion: int = 0
    # LLM 추천 이유
    reason: str | None = None


class RecommendResponse(BaseModel):
    """추천 응답 — 소매가/낙찰가 분리"""
    target: TargetVehicleSchema
    retail_vehicles: list[RetailVehicle]    # 엔카 소매가 (최대 3)
    auction_vehicles: list[AuctionVehicle]  # 낙찰가 (최대 3)
    reasoning: str | None = None            # LLM 전체 선별 근거


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_reference_vehicles(target: TargetVehicleSchema):
    """
    대상차량에 대한 소매가/낙찰가 기준차량 추천.
    LLM 에이전트(Claude Sonnet)가 DB를 검색·분석하여 각각 최대 3건 선별.
    소매가와 낙찰가 추천을 병렬로 실행.
    """
    try:
        target_dict = {
            "maker": target.maker,
            "model": target.model,
            "year": target.year,
            "mileage": target.mileage,
            "trim": target.trim,
            "generation": target.generation,
            "fuel": target.fuel,
            "options": target.options,
            "exchange_count": target.exchange_count,
            "bodywork_count": target.bodywork_count,
        }

        # ── 소매가 + 낙찰가 LLM 추천 병렬 실행 ──
        retail_result, auction_result = await asyncio.gather(
            asyncio.to_thread(recommend_retail_references, target_dict),
            asyncio.to_thread(recommend_references, target_dict),
        )

        logger.info(
            "LLM 추천 완료 — 소매: %d건 (tokens: %d+%d), 낙찰: %d건 (tokens: %d+%d)",
            len(retail_result.recommendations),
            retail_result.total_input_tokens,
            retail_result.total_output_tokens,
            len(auction_result.recommendations),
            auction_result.total_input_tokens,
            auction_result.total_output_tokens,
        )

        # ── 소매가 결과 구성 ──
        retail_vehicles = []
        for rec in retail_result.recommendations:
            vid = rec.get("auction_id")
            if not vid:
                continue
            detail = get_retail_detail(vid)
            if not detail:
                logger.warning("소매 차량 상세 조회 실패: %s", vid)
                continue
            retail_vehicles.append(RetailVehicle(
                auction_id=vid,
                vehicle_name=detail.get("차명"),
                year=detail.get("연식"),
                mileage=detail.get("주행거리"),
                retail_price=detail.get("소매가"),
                color=detail.get("색상"),
                trim=detail.get("trim"),
                source_url=detail.get("source_url"),
                factory_price=detail.get("factory_price") or None,
                options=detail.get("옵션") or None,
                fuel=detail.get("연료") or None,
                frame_exchange=detail.get("frame_exchange", 0),
                frame_bodywork=detail.get("frame_bodywork", 0),
                frame_corrosion=detail.get("frame_corrosion", 0),
                exterior_exchange=detail.get("exterior_exchange", 0),
                exterior_bodywork=detail.get("exterior_bodywork", 0),
                exterior_corrosion=detail.get("exterior_corrosion", 0),
                reason=rec.get("reason"),
            ))

        # ── 낙찰가 결과 구성 ──
        auction_vehicles = []
        for rec in auction_result.recommendations:
            vid = rec.get("auction_id")
            if not vid:
                continue
            detail = get_vehicle_detail(vid)
            if not detail:
                logger.warning("낙찰 차량 상세 조회 실패: %s", vid)
                continue
            auction_vehicles.append(AuctionVehicle(
                auction_id=vid,
                vehicle_name=detail.get("차명"),
                year=detail.get("연식"),
                mileage=detail.get("주행거리"),
                auction_price=detail.get("낙찰가"),
                auction_date=detail.get("개최일"),
                color=detail.get("색상"),
                trim=detail.get("trim"),
                options=detail.get("옵션"),
                factory_price=detail.get("factory_price") or None,
                inspection_grade=detail.get("grade_score") or None,
                is_export=bool(detail.get("is_export", 0)),
                fuel=detail.get("연료") or None,
                frame_exchange=detail.get("frame_exchange", 0),
                frame_bodywork=detail.get("frame_bodywork", 0),
                frame_corrosion=detail.get("frame_corrosion", 0),
                exterior_exchange=detail.get("exterior_exchange", 0),
                exterior_bodywork=detail.get("exterior_bodywork", 0),
                exterior_corrosion=detail.get("exterior_corrosion", 0),
                reason=rec.get("reason"),
            ))

        # ── reasoning 합치기 ──
        reasoning_parts = []
        if retail_result.reasoning:
            reasoning_parts.append(f"[소매가] {retail_result.reasoning}")
        if auction_result.reasoning:
            reasoning_parts.append(f"[낙찰가] {auction_result.reasoning}")
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        return RecommendResponse(
            target=target,
            retail_vehicles=retail_vehicles,
            auction_vehicles=auction_vehicles,
            reasoning=reasoning,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 검색 오류: {type(e).__name__}: {str(e)}",
        )
