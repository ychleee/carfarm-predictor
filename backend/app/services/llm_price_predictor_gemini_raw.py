"""
CarFarm i3 모델 — Gemini API 순수 LLM 가격 예측 (시장 데이터 보정 없음)

i2와 동일한 데이터 수집/프롬프트/Gemini 호출을 하되,
시장 데이터 보정(retail_estimator)을 적용하지 않고 LLM 순수 결과만 반환.
"""

from __future__ import annotations

import json
import logging

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
from app.services.llm_price_predictor_gemini import (
    _get_client,
    _parse_prediction,
    _GEMINI_MODEL,
)

logger = logging.getLogger(__name__)


def predict_price_gemini_raw(target: dict) -> PricePrediction:
    """
    Gemini API 순수 LLM 가격 예측 (i3 모델).

    i2와 동일한 데이터 수집 + Gemini 호출이지만
    시장 데이터 보정(retail_estimator) 없이 LLM 결과를 그대로 반환.
    """
    # 1) 데이터 수집
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
        "[i3/Gemini-Raw] 데이터 수집 — 낙찰: %d건, 소매: %d건",
        len(auction_vehicles), len(retail_vehicles),
    )

    if total_vehicles == 0:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="[i3] 유사 차량 데이터를 찾을 수 없어 가격 예측이 불가합니다.",
            vehicles_analyzed=0,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
        )

    # 2) 프롬프트 생성
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
    logger.info("[i3/Gemini-Raw] API — tokens: %d+%d", input_tokens, output_tokens)

    # 4) 응답 파싱
    try:
        raw_text = response.text
    except Exception as e:
        logger.warning("[i3/Gemini-Raw] response.text 접근 실패: %s", e)
        raw_text = ""

    if not raw_text:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="[i3] Gemini 응답이 비어있습니다.",
            vehicles_analyzed=total_vehicles,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            parsed = _parse_prediction(raw_text)
        except Exception:
            parsed = None

    if not parsed or not isinstance(parsed, dict):
        logger.warning("[i3/Gemini-Raw] JSON 파싱 실패. 원본(앞 300자): %s", raw_text[:300])
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"[i3] Gemini 응답 파싱 실패. 원본: {raw_text[:500]}",
            vehicles_analyzed=total_vehicles,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # 최근 수출 낙찰일
    export_dates = [
        v.get("개최일", "") for v in auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    # compact 직렬화
    compact_auction = [_compact_auction_vehicle(v) for v in auction_vehicles[:20]]
    compact_retail = [_compact_retail_vehicle(v) for v in retail_vehicles[:15]]

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
        vehicles_analyzed=total_vehicles,
        auction_stats=auction_stats,
        retail_stats=retail_stats,
        comparable_auction_vehicles=compact_auction,
        comparable_retail_vehicles=compact_retail,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
