"""
CarFarm i2 모델 — Gemini + 시장 데이터 보정

공통 데이터 수집(gemini_shared)을 사용하고,
시장 데이터 보정(retail_estimator)을 적용하여 최종 가격을 산출.
"""

from __future__ import annotations

import logging

from app.services.llm_price_predictor import (
    PricePrediction,
    _compact_auction_vehicle,
    _compact_retail_vehicle,
    _estimate_export_from_domestic,
)
from app.services.gemini_shared import (
    CollectedData,
    GeminiLLMResult,
    collect_data,
    call_gemini_llm,
)

logger = logging.getLogger(__name__)


def predict_price_gemini(target: dict) -> PricePrediction:
    """
    Gemini API 기반 가격 예측 (i2 모델) — 단독 호출용.

    데이터 수집 + Gemini 호출 + 시장 데이터 보정을 모두 수행.
    """
    data = collect_data(target)
    if data.total_vehicles == 0:
        return _empty_result(data, "[i2]")

    llm_result = call_gemini_llm(data.user_message)
    return build_i2_result(target, data, llm_result)


def build_i2_result(
    target: dict,
    data: CollectedData,
    llm_result: GeminiLLMResult,
) -> PricePrediction:
    """
    i2 후처리 — LLM 결과에 시장 데이터 보정을 적용.

    멀티 모델 엔드포인트에서 공유된 데이터/LLM 결과를 받아 처리할 때도 사용.
    """
    parsed = llm_result.parsed
    if not parsed:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"[i2] Gemini 응답 파싱 실패. 원본: {llm_result.raw_text[:500]}",
            vehicles_analyzed=data.total_vehicles,
            auction_stats=data.auction_stats,
            retail_stats=data.retail_stats,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens,
        )

    # LLM 결과 보관
    auction_reasoning = parsed.get("auction_reasoning", "")
    retail_reasoning = parsed.get("retail_reasoning", "")
    export_reasoning = parsed.get("export_reasoning", "")
    legacy_reasoning = parsed.get("reasoning", "")
    auction_factors = parsed.get("auction_factors", [])
    retail_factors = parsed.get("retail_factors", [])

    # 최근 수출 낙찰일
    export_dates = [
        v.get("개최일", "") for v in data.auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    # ── 시장 데이터 보정 ──
    from app.services.retail_estimator import (
        estimate_retail_by_market,
        estimate_auction_by_market,
        estimate_export_auction_by_market,
    )

    target_fuel = target.get("fuel", "") or ""
    market_auction_result = estimate_auction_by_market(
        maker=target.get("maker", ""),
        model=target.get("model", ""),
        trim=target.get("trim", ""),
        year=target.get("year", 0),
        mileage=target.get("mileage", 0),
        factory_price=target.get("factory_price", 0) or 0,
        base_price=target.get("base_price", 0) or 0,
        fuel=target_fuel,
    )

    if market_auction_result.success:
        final_auction = market_auction_result.estimated_auction
        auction_reasoning_final = f"[시장 데이터] {market_auction_result.details}"
    else:
        final_auction = parsed.get("estimated_auction", 0)
        auction_reasoning_final = (
            f"[시장 데이터 부족 — LLM 폴백] {market_auction_result.details}\n"
            f"{auction_reasoning or legacy_reasoning}"
        )

    market_retail_result = estimate_retail_by_market(
        maker=target.get("maker", ""),
        model=target.get("model", ""),
        trim=target.get("trim", ""),
        year=target.get("year", 0),
        mileage=target.get("mileage", 0),
        factory_price=target.get("factory_price", 0) or 0,
        base_price=target.get("base_price", 0) or 0,
        fuel=target_fuel,
        auction_brackets=market_auction_result.brackets if market_auction_result.success else None,
    )

    if market_retail_result.success:
        final_retail = market_retail_result.estimated_retail
        retail_reasoning_final = f"[시장 데이터] {market_retail_result.details}"
    else:
        final_retail = parsed.get("estimated_retail", 0)
        retail_reasoning_final = (
            f"[시장 데이터 부족 — LLM 폴백] {market_retail_result.details}\n"
            f"{retail_reasoning or legacy_reasoning}"
        )

    # ── 수출 낙찰가: 시장 데이터 ──
    market_export_result = estimate_export_auction_by_market(
        maker=target.get("maker", ""),
        model=target.get("model", ""),
        trim=target.get("trim", ""),
        year=target.get("year", 0),
        mileage=target.get("mileage", 0),
        factory_price=target.get("factory_price", 0) or 0,
        base_price=target.get("base_price", 0) or 0,
        fuel=target_fuel,
    )

    if market_export_result.success:
        export_price = market_export_result.estimated_auction
        export_reasoning_final = f"[시장 데이터] {market_export_result.details}"
    else:
        export_price = 0
        export_reasoning_final = ""
        target_mileage = target.get("mileage", 0)
        export_vehicles = market_export_result.vehicles or []
        domestic_brackets = (
            sorted(market_auction_result.brackets, key=lambda b: b.bracket_start)
            if market_auction_result.success and market_auction_result.brackets
            else []
        )
        if export_vehicles and domestic_brackets:
            export_price, export_reasoning_final = _estimate_export_from_domestic(
                export_vehicles=export_vehicles,
                domestic_brackets=domestic_brackets,
                target_mileage=target_mileage,
                export_details=market_export_result.details,
                domestic_estimate=market_auction_result.estimated_auction,
            )
        if export_price <= 0:
            export_price = parsed.get("estimated_auction_export", 0) or 0
            export_reasoning_final = (
                f"[수출 시장 데이터 부족 — LLM 폴백] {market_export_result.details}\n"
                f"{export_reasoning}"
            )

    # ── 낙찰가 < 소매가 정합성 체크 ──
    if final_auction > 0 and final_retail > 0 and final_auction >= final_retail * 0.97:
        capped = round(final_retail * 0.90, 1)
        auction_reasoning_final += (
            f"\n\n── 정합성 보정 ──\n"
            f"낙찰가({final_auction:.0f}만) ≥ 소매가({final_retail:.0f}만)의 97%: "
            f"소매가×90%={capped:.0f}만으로 캡"
        )
        final_auction = capped

    # ── bracket 직렬화 ──
    def _serialize_brackets(brackets: list) -> list[dict]:
        return [
            {
                "s": b.bracket_start,
                "e": b.bracket_end,
                "n": b.count,
                "r": round(b.effective_ratio * 100, 1),
                "mn": round(min(b.prices)) if b.prices else 0,
                "mx": round(max(b.prices)) if b.prices else 0,
            }
            for b in brackets if b.count > 0
        ]

    retail_brackets = _serialize_brackets(market_retail_result.brackets) if market_retail_result.brackets else []
    auction_brackets = _serialize_brackets(market_auction_result.brackets) if market_auction_result.brackets else []
    export_brackets = _serialize_brackets(market_export_result.brackets) if market_export_result.brackets else []

    # ── 유사차량 compact 직렬화 ──
    market_auction_all = market_auction_result.vehicles + market_export_result.vehicles
    if market_auction_all:
        compact_auction = [_compact_auction_vehicle(v) for v in market_auction_all]
    else:
        compact_auction = [_compact_auction_vehicle(v) for v in data.auction_vehicles[:20]]

    if market_retail_result.vehicles:
        compact_retail = [_compact_retail_vehicle(v) for v in market_retail_result.vehicles]
    else:
        compact_retail = [_compact_retail_vehicle(v) for v in data.retail_vehicles[:15]]

    market_total = len(compact_auction) + len(compact_retail)

    return PricePrediction(
        estimated_auction=final_auction,
        estimated_auction_export=export_price,
        last_export_date=last_export_date,
        estimated_retail=final_retail,
        confidence=parsed.get("confidence", "보통"),
        reasoning=legacy_reasoning or auction_reasoning,
        factors=parsed.get("factors", []) or auction_factors,
        auction_reasoning=auction_reasoning_final,
        retail_reasoning=retail_reasoning_final,
        export_reasoning=export_reasoning_final,
        auction_factors=auction_factors,
        retail_factors=retail_factors,
        comparable_summary=parsed.get("comparable_summary", ""),
        key_comparables=parsed.get("key_comparables", []),
        vehicles_analyzed=market_total if market_total > 0 else data.total_vehicles,
        auction_stats=data.auction_stats,
        retail_stats=data.retail_stats,
        comparable_auction_vehicles=compact_auction,
        comparable_retail_vehicles=compact_retail,
        retail_brackets=retail_brackets,
        auction_brackets=auction_brackets,
        export_brackets=export_brackets,
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
