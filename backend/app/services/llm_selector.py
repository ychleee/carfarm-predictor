"""
LLM 기반 기준차량 선별 — Stage 2

후보 차량 리스트에서 대상차량에 가장 적합한 15대를 LLM이 선별.
Claude Haiku로 빠르게 처리 (1~2초).
"""

from __future__ import annotations

import json
import os
import re

import anthropic


SELECTOR_PROMPT = """당신은 자동차 경매 전문 프라이싱 매니저입니다.
대상차량에 대해 후보 차량 리스트에서 **기준차량으로 가장 적합한 최대 {limit}대**를 선별하세요.

## 선별 기준 (중요도 순)

1. **트림 동일** — 트림이 같은 차량 우선
2. **연식 근접** — 연식 차이가 작을수록 좋음
3. **세대(generation) 동일** — 같은 세대 우선
4. **연료 동일** — 같은 연료 타입 우선
5. **주행거리 근접** — 비슷한 주행거리 우선
6. **옵션/출고가 유사** — 유사 사양 우선
7. **판매일 최신** — 최근 거래 우선
8. **진단 상태 유사** — 대상차량과 비슷한 사고이력 차량 우선

## 진단 관련 규칙
- **골격(프레임) 교환이 있는 차량은 강하게 후순위 처리** — 프레임 교환은 중대 사고를 의미
- 프레임 판금/부식도 감점 요소
- 외판 손상은 프레임 손상보다 덜 심각하지만, 많을수록 후순위
- 대상차량과 진단 상태가 비슷한 차량이 가격 비교에 더 적합

## 추가 규칙
- 조건에 맞는 차량이 {limit}대 미만이면 있는 만큼만 선택
- 후보 리스트 전체를 검토한 뒤 최적의 조합을 선택

## 응답 형식

반드시 아래 JSON만 반환하세요 (마크다운 코드블록 없이):
{{"selected": [{{"id": "차량ID", "reason": "추천 이유 한 줄 요약"}}, ...], "reasoning": "전체 선별 근거 요약 (2~3문장)"}}
"""


class SelectionResult:
    """LLM 선별 결과"""
    def __init__(
        self,
        ids: list[str],
        reasons: dict[str, str] | None = None,
        reasoning: str | None = None,
    ):
        self.ids = ids
        self.reasons = reasons or {}
        self.reasoning = reasoning


def select_best_vehicles(
    target: dict,
    candidates: list[dict],
    limit: int = 15,
    model: str = "claude-haiku-4-5-20251001",
) -> SelectionResult:
    """
    LLM을 사용하여 후보 중 최적의 차량 ID를 선별.

    Args:
        target: 대상차량 정보 dict (maker, model, year, trim 등)
        candidates: 후보 차량 list[dict] (auction_id 포함)
        limit: 선별할 최대 수
        model: 사용할 Claude 모델

    Returns:
        SelectionResult (ids, reasons, reasoning)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return SelectionResult(ids=[])

    if not candidates:
        return SelectionResult(ids=[])

    # 후보가 limit 이하면 LLM 호출 불필요
    if len(candidates) <= limit:
        return SelectionResult(ids=[c["auction_id"] for c in candidates])

    client = anthropic.Anthropic(api_key=api_key)

    # 대상차량 요약
    target_summary = _build_target_summary(target)

    # 후보 차량 리스트 (필수 정보만 추려서 토큰 절약)
    candidates_summary = _build_candidates_summary(candidates)

    user_message = f"""## 대상차량
{target_summary}

## 후보 차량 ({len(candidates)}대)
{candidates_summary}

위 후보 중에서 대상차량의 기준차량으로 가장 적합한 최대 {limit}대를 선별해주세요."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SELECTOR_PROMPT.format(limit=limit),
            messages=[{"role": "user", "content": user_message}],
        )

        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text = block.text
                break

        result = _parse_selection(result_text)
        # 유효한 ID만 필터
        valid_ids = {c["auction_id"] for c in candidates}
        filtered_ids = [sid for sid in result.ids if sid in valid_ids][:limit]
        return SelectionResult(
            ids=filtered_ids,
            reasons={k: v for k, v in result.reasons.items() if k in valid_ids},
            reasoning=result.reasoning,
        )

    except Exception:
        # LLM 실패 시 빈 리스트 (호출자가 fallback 처리)
        return SelectionResult(ids=[])


def _build_target_summary(target: dict) -> str:
    """대상차량 정보 요약"""
    lines = []
    field_map = {
        "maker": "제작사",
        "model": "모델",
        "generation": "세대",
        "year": "연식",
        "mileage": "주행거리(km)",
        "fuel": "연료",
        "trim": "트림",
        "color": "색상",
        "options": "옵션",
    }
    for key, label in field_map.items():
        val = target.get(key)
        if val is not None and val != "" and val != [] and val != 0:
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            lines.append(f"- {label}: {val}")

    # 대상차량은 항상 AA등급(무사고)으로 가정 — 실제 검차 상태 전달하지 않음
    lines.append("- 검차상태: AA등급 (무사고 가정)")

    return "\n".join(lines) if lines else "정보 없음"


def _build_candidates_summary(candidates: list[dict]) -> str:
    """후보 차량 리스트를 간결한 텍스트로 변환"""
    lines = []
    for c in candidates:
        parts = [f"ID:{c['auction_id']}"]
        if c.get("차명"):
            parts.append(c["차명"])
        if c.get("trim"):
            parts.append(f"트림:{c['trim']}")
        if c.get("연식"):
            parts.append(f"{c['연식']}년")
        if c.get("주행거리"):
            parts.append(f"{c['주행거리']}km")

        # 가격 + 출고가 대비 비율
        price = c.get("낙찰가") or c.get("소매가")
        price_label = "낙찰" if c.get("낙찰가") else "소매"
        if price:
            parts.append(f"{price_label}:{price}만")
        fp = c.get("factory_price")
        if fp and fp > 0:
            parts.append(f"출고:{fp}만")
            if price and price > 0:
                ratio = round(price / fp * 100)
                parts.append(f"출고대비:{ratio}%")

        if c.get("개최일"):
            parts.append(f"날짜:{c['개최일']}")

        # 프레임/외판 진단 정보
        fe = c.get("frame_exchange", 0)
        fb = c.get("frame_bodywork", 0)
        fc = c.get("frame_corrosion", 0)
        if fe or fb or fc:
            parts.append(f"프레임(교{fe}/판{fb}/부{fc})")
        ee = c.get("exterior_exchange", 0)
        eb = c.get("exterior_bodywork", 0)
        ec = c.get("exterior_corrosion", 0)
        if ee or eb or ec:
            parts.append(f"외판(교{ee}/판{eb}/부{ec})")

        if c.get("옵션"):
            parts.append(f"옵션:{c['옵션'][:50]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _parse_selection(text: str) -> SelectionResult:
    """LLM 응답에서 selected + reasoning 파싱"""
    parsed = None

    # JSON 블록 추출 시도 (새 형식: "selected" 배열)
    json_match = re.search(r'\{.*"selected"\s*:\s*\[.*?\].*?\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # 직접 JSON 파싱
    if parsed is None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

    if parsed is None:
        # 구 형식 호환: selected_ids
        old_match = re.search(r'\{.*"selected_ids"\s*:\s*\[.*?\].*?\}', text, re.DOTALL)
        if old_match:
            try:
                old_parsed = json.loads(old_match.group(0))
                return SelectionResult(ids=old_parsed.get("selected_ids", []))
            except json.JSONDecodeError:
                pass
        return SelectionResult(ids=[])

    # 새 형식 파싱
    selected = parsed.get("selected", [])
    ids = []
    reasons = {}
    for item in selected:
        if isinstance(item, dict):
            vid = item.get("id", "")
            ids.append(vid)
            reason = item.get("reason")
            if reason:
                reasons[vid] = reason
        elif isinstance(item, str):
            ids.append(item)

    reasoning = parsed.get("reasoning")

    return SelectionResult(ids=ids, reasons=reasons, reasoning=reasoning)
