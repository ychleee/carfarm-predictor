"""
Gemini 공통 모듈 — 데이터 수집 + Gemini LLM 호출

i2(시장 데이터 보정)와 i3(순수 LLM) 모델이 공유하는 데이터 수집 및 LLM 호출 로직.
데이터 수집은 1회만 수행하고, 이후 각 모델이 후처리만 분기하여 처리.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from google import genai

from app.services.llm_price_predictor import (
    SYSTEM_PROMPT,
    PricePrediction,
    _fetch_comparable_vehicles,
    _build_user_message,
    _to_man_won,
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
    logger.info("[Gemini] 클라이언트 초기화 완료 — model=%s", _GEMINI_MODEL)
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


@dataclass
class CollectedData:
    """공통 데이터 수집 결과"""
    auction_vehicles: list = field(default_factory=list)
    retail_vehicles: list = field(default_factory=list)
    auction_stats: dict = field(default_factory=dict)
    retail_stats: dict = field(default_factory=dict)
    total_vehicles: int = 0
    user_message: str = ""


@dataclass
class GeminiLLMResult:
    """Gemini LLM 호출 결과"""
    parsed: dict | None = None
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def collect_data(target: dict) -> CollectedData:
    """
    공통 데이터 수집 — Firestore에서 유사 차량 조회 + 통계 정규화 + 프롬프트 생성.

    i2와 i3 모두 동일한 데이터를 사용하므로 1회만 수행.
    """
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
        "[Gemini] 데이터 수집 — 낙찰: %d건, 소매: %d건",
        len(auction_vehicles), len(retail_vehicles),
    )

    user_message = ""
    if total_vehicles > 0:
        user_message = _build_user_message(
            target, auction_vehicles, retail_vehicles,
            auction_stats, retail_stats,
        )

    return CollectedData(
        auction_vehicles=auction_vehicles,
        retail_vehicles=retail_vehicles,
        auction_stats=auction_stats,
        retail_stats=retail_stats,
        total_vehicles=total_vehicles,
        user_message=user_message,
    )


def call_gemini_llm(user_message: str) -> GeminiLLMResult:
    """
    Gemini API 호출 + 응답 파싱.

    동일한 프롬프트에 대해 1회만 호출하고 결과를 공유할 수 있도록 분리.
    """
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
    logger.info("[Gemini] API — tokens: %d+%d", input_tokens, output_tokens)

    try:
        raw_text = response.text
    except Exception as e:
        logger.warning("[Gemini] response.text 접근 실패: %s", e)
        raw_text = ""

    parsed = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                parsed = _parse_prediction(raw_text)
            except Exception:
                parsed = None

        if parsed and not isinstance(parsed, dict):
            logger.warning("[Gemini] JSON 파싱 결과가 dict가 아님: %s", type(parsed))
            parsed = None

        if not parsed:
            logger.warning("[Gemini] JSON 파싱 실패. 원본(앞 300자): %s", raw_text[:300])

    return GeminiLLMResult(
        parsed=parsed,
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
