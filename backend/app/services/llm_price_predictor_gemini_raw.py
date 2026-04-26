"""
CarFarm i3 모델 — Gemini 순수 LLM 가격 예측 (시장 데이터 보정 없음)

공통 데이터 수집(gemini_shared)을 사용하고,
시장 데이터 보정 없이 LLM 순수 결과만 반환.
"""

from __future__ import annotations

import logging

from app.services.llm_price_predictor import (
    PricePrediction,
    _compact_auction_vehicle,
    _compact_retail_vehicle,
)
from app.services.gemini_shared import (
    CollectedData,
    GeminiLLMResult,
    collect_data,
    call_gemini_llm,
)

logger = logging.getLogger(__name__)


def predict_price_gemini_raw(target: dict) -> PricePrediction:
    """
    Gemini 순수 LLM 가격 예측 (i3 모델) — 단독 호출용.

    데이터 수집 + Gemini 호출만 수행, 시장 데이터 보정 없음.
    """
    data = collect_data(target)
    if data.total_vehicles == 0:
        return _empty_result(data, "[i3]")

    llm_result = call_gemini_llm(data.user_message)
    return build_i3_result(data, llm_result)


def build_i3_result(
    data: CollectedData,
    llm_result: GeminiLLMResult,
) -> PricePrediction:
    """
    i3 후처리 — LLM 결과를 그대로 반환 (시장 데이터 보정 없음).

    멀티 모델 엔드포인트에서 공유된 데이터/LLM 결과를 받아 처리할 때도 사용.
    """
    parsed = llm_result.parsed
    if not parsed:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"[i3] Gemini 응답 파싱 실패. 원본: {llm_result.raw_text[:500]}",
            vehicles_analyzed=data.total_vehicles,
            auction_stats=data.auction_stats,
            retail_stats=data.retail_stats,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
        )

    # 최근 수출 낙찰일
    export_dates = [
        v.get("개최일", "") for v in data.auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    # compact 직렬화
    compact_auction = [_compact_auction_vehicle(v) for v in data.auction_vehicles[:20]]
    compact_retail = [_compact_retail_vehicle(v) for v in data.retail_vehicles[:15]]

    return PricePrediction(
        estimated_auction=parsed.get("estimated_auction", 0),
        estimated_auction_export=parsed.get("estimated_auction_export", 0),
        last_export_date=last_export_date,
        estimated_retail=parsed.get("estimated_retail", 0),
        confidence=parsed.get("confidence", "보통"),
        reasoning=parsed.get("reasoning", ""),
        factors=parsed.get("factors", []) or parsed.get("auction_factors", []),
        auction_reasoning=parsed.get("auction_reasoning", ""),
        retail_reasoning=parsed.get("retail_reasoning", ""),
        export_reasoning=parsed.get("export_reasoning", ""),
        auction_factors=parsed.get("auction_factors", []),
        retail_factors=parsed.get("retail_factors", []),
        comparable_summary=parsed.get("comparable_summary", ""),
        key_comparables=parsed.get("key_comparables", []),
        vehicles_analyzed=data.total_vehicles,
        auction_stats=data.auction_stats,
        retail_stats=data.retail_stats,
        comparable_auction_vehicles=compact_auction,
        comparable_retail_vehicles=compact_retail,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
    )


def _empty_result(data: CollectedData, prefix: str) -> PricePrediction:
    return PricePrediction(
        estimated_auction=0,
        estimated_retail=0,
        confidence="낮음",
        reasoning=f"{prefix} 유사 차량 데이터를 찾을 수 없어 가격 예측이 불가합니다.",
        vehicles_analyzed=0,
        auction_stats=data.auction_stats,
        retail_stats=data.retail_stats,
    )
