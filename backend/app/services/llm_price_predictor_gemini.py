"""
CarFarm i2 모델 — Gemini API 기반 가격 예측 서비스

i1 모델(Claude)과 동일한 데이터 수집/프롬프트를 사용하되 LLM만 Gemini로 교체.
Isaac 프로젝트의 기존 Gemini API 키를 사용.
"""

from __future__ import annotations

import json
import logging
import os
import re

from google import genai

from app.services.llm_price_predictor import (
    SYSTEM_PROMPT,
    PricePrediction,
    _fetch_comparable_vehicles,
    _build_user_message,
    _to_man_won,
    _compact_auction_vehicle,
    _compact_retail_vehicle,
)

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-flash"
_gemini_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Gemini API 클라이언트 (싱글톤)"""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    _gemini_client = genai.Client(api_key=api_key)
    logger.info("[i2/Gemini] 클라이언트 초기화 완료 — model=%s", _GEMINI_MODEL)
    return _gemini_client


def _parse_prediction(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)
    return json.loads(text)


def predict_price_gemini(target: dict) -> PricePrediction:
    """
    Gemini API 기반 가격 예측 (i2 모델).

    i1과 동일한 데이터 수집 + 프롬프트 → Gemini API 호출.
    시장 데이터 보정(retail_estimator)은 적용하지 않고 LLM 순수 결과만 반환.
    """
    # 1) 데이터 수집 (i1과 동일)
    auction_vehicles, retail_vehicles, auction_stats_raw, retail_stats_raw = (
        _fetch_comparable_vehicles(target)
    )

    def _normalize_stats(s: dict) -> dict:
        if s.get("count", 0) == 0:
            return s
        out = dict(s)
        for k in ("mean", "median", "min", "max", "std"):
            if k in out:
                out[k] = _to_man_won(out[k])
        return out

    auction_stats = _normalize_stats(auction_stats_raw)
    retail_stats = _normalize_stats(retail_stats_raw)

    total_vehicles = len(auction_vehicles) + len(retail_vehicles)
    logger.info(
        "[i2/Gemini] 데이터 수집 — 낙찰: %d건, 소매: %d건",
        len(auction_vehicles), len(retail_vehicles),
    )

    if total_vehicles == 0:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="[i2] 유사 차량 데이터를 찾을 수 없어 가격 예측이 불가합니다.",
            vehicles_analyzed=0,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
        )

    # 2) 프롬프트 생성 (i1과 동일)
    user_message = _build_user_message(
        target, auction_vehicles, retail_vehicles,
        auction_stats, retail_stats,
    )

    # 3) Gemini API 호출
    client = _get_client()
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=8192,
            response_mime_type="application/json",
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        ),
    )

    input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    logger.info("[i2/Gemini] API — tokens: %d+%d", input_tokens, output_tokens)

    # 4) 응답 파싱
    try:
        raw_text = response.text
    except Exception as e:
        logger.warning("[i2/Gemini] response.text 접근 실패: %s", e)
        raw_text = ""

    if not raw_text:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="[i2] Gemini 응답이 비어있습니다.",
            vehicles_analyzed=total_vehicles,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    try:
        # response_mime_type=application/json → 순수 JSON
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # 폴백: 마크다운 코드펜스 등에서 JSON 추출
        try:
            parsed = _parse_prediction(raw_text)
        except Exception:
            parsed = None

    if not parsed or not isinstance(parsed, dict):
        logger.warning("[i2/Gemini] JSON 파싱 실패. 원본(앞 300자): %s", raw_text[:300])
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"[i2] Gemini 응답 파싱 실패. 원본: {raw_text[:500]}",
            vehicles_analyzed=total_vehicles,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        v.get("개최일", "") for v in auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    # ── 시장 데이터 보정 (i1과 동일) ──
    from app.services.retail_estimator import (
        estimate_retail_by_market,
        estimate_auction_by_market,
        estimate_export_auction_by_market,
    )
    from app.services.llm_price_predictor import _estimate_export_from_domestic

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
        compact_auction = [_compact_auction_vehicle(v) for v in auction_vehicles[:20]]

    if market_retail_result.vehicles:
        compact_retail = [_compact_retail_vehicle(v) for v in market_retail_result.vehicles]
    else:
        compact_retail = [_compact_retail_vehicle(v) for v in retail_vehicles[:15]]

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
        vehicles_analyzed=market_total if market_total > 0 else total_vehicles,
        auction_stats=auction_stats,
        retail_stats=retail_stats,
        comparable_auction_vehicles=compact_auction,
        comparable_retail_vehicles=compact_retail,
        retail_brackets=retail_brackets,
        auction_brackets=auction_brackets,
        export_brackets=export_brackets,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
