"""
LLM 기반 보정 기준 분석 서비스

유사차량 데이터를 LLM에 전달하여 보정 기준(rate)을 도출한다.
최종 가격이 아닌, 보정 기준(비율)만 분석하는 역할.

v2: 유사차량을 대상차량과 동일 조건(연식/출고가/AA급)으로 환산하여 LLM에 전달.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import yaml

from app.services.firestore_db import fetch_comparable_vehicles
from app.services.ratio_calculator import _calc_inspection_adj

logger = logging.getLogger(__name__)

# =========================================================================
# YAML 룰 로드
# =========================================================================

_RULES_PATH = Path(__file__).parent.parent.parent / "rules" / "pricing_rules.yaml"

def _load_pricing_rules() -> dict:
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_PRICING_RULES = _load_pricing_rules()


# =========================================================================
# 데이터 클래스
# =========================================================================


@dataclass
class PricingCriteria:
    """LLM이 도출한 보정 기준"""
    mileage_rate_per_10k: float = 1.0    # %/만km
    mileage_age_band: str = "4~6년"
    mileage_ceiling_km: int = 200000
    year_rate_per_year: float = 1.4      # %/년


@dataclass
class AnalyzeCriteriaResult:
    """분석 결과"""
    criteria: PricingCriteria
    analysis_summary: str = ""
    vehicles_analyzed: int = 0
    confidence: str = "보통"
    input_tokens: int = 0
    output_tokens: int = 0


# =========================================================================
# 시스템 프롬프트
# =========================================================================

CRITERIA_SYSTEM_PROMPT = """당신은 한국 중고차 시장의 가격 보정 기준 분석 전문가입니다.
제공된 유사 차량 데이터를 분석하여 **보정 기준(rate)**을 도출하는 것이 역할입니다.
최종 가격을 산출하는 것이 아니라, 가격 보정에 사용할 **비율/기준**만 도출합니다.

## 환산 개념
모든 유사차량은 대상차량과 **같은 연식, 같은 출고가, AA급 무사고** 조건으로 환산되었습니다.
따라서 "환산가(만)" 열은 연식·출고가·검차 차이가 모두 제거된 금액입니다.
당신은 **환산가와 주행거리의 관계**에만 집중하여 주행거리 감가율을 도출하세요.

## 연식대별 주행거리 감가율 (업계 기준, 반드시 준수)
차량 연식(경과연수)에 따라 주행거리 감가율의 허용 범위가 다릅니다.
이 범위를 벗어나는 값을 도출하지 마세요.

| 연식대 | 기본값 | 허용 범위 |
|--------|--------|-----------|
| 2~3년  | 1.4%/만km | 1.0 ~ 2.0% |
| 4~6년  | 1.0%/만km | 0.7 ~ 1.5% |
| 7~9년  | 0.7%/만km | 0.5 ~ 1.0% |
| 10년+  | 0.7%/만km | 0.3 ~ 1.0% |

- 연식 보정율: 1.4%/년 (허용 범위: 1.0 ~ 2.0%)

## 분석 방법
1. 대상차량의 연식대를 먼저 확인하고, 해당 연식대의 허용 범위 내에서 감가율을 도출
2. 환산가(만)와 주행거리의 상관관계에서 주행거리 감가율 추정 (%p/만km)
3. 데이터에서 도출된 값이 허용 범위를 벗어나면, 범위 내 가장 가까운 값으로 조정
4. 연식 보정율은 1.4%/년을 기본으로 유지

## 출력 형식
반드시 아래 JSON 형식으로만 응답하세요.
{
  "mileage_rate_per_10k": 1.0,
  "mileage_age_band": "4~6년",
  "mileage_ceiling_km": 200000,
  "year_rate_per_year": 1.4,
  "analysis_summary": "분석된 25대 차량 기준, ...",
  "confidence": "보통"
}"""


# =========================================================================
# 기본값 (LLM 실패 시 폴백)
# =========================================================================

def _get_default_criteria(target_year: int) -> PricingCriteria:
    """연식 기반 업계 기본 보정 기준 반환 (LLM 실패 시 폴백, YAML 기준)"""
    age = 2026 - target_year

    if age <= 3:
        return PricingCriteria(
            mileage_rate_per_10k=1.4,
            mileage_age_band="2~3년",
            year_rate_per_year=1.4,
        )
    elif age <= 6:
        return PricingCriteria(
            mileage_rate_per_10k=1.0,
            mileage_age_band="4~6년",
            year_rate_per_year=1.4,
        )
    elif age <= 9:
        return PricingCriteria(
            mileage_rate_per_10k=0.7,
            mileage_age_band="7~9년",
            year_rate_per_year=1.4,
        )
    else:
        return PricingCriteria(
            mileage_rate_per_10k=0.7,
            mileage_age_band="10년+",
            year_rate_per_year=1.4,
        )


# =========================================================================
# 유사차량 정규화(환산)
# =========================================================================

def _pick_vehicle_price(v: dict) -> float:
    """유사차량에서 출고가 또는 기본가 추출 (만원). 없으면 0."""
    fp = v.get("factory_price", 0) or 0
    if fp > 0:
        return float(fp)
    bp = v.get("base_price", 0) or 0
    return float(bp)


def _normalize_vehicle(v: dict, target: dict, rules: dict) -> float | None:
    """
    유사차량 1건을 대상차량과 동일 조건으로 환산.

    환산 공식:
      1) 연식 보정: (target_year - v_year) × year_rate × v_factory
      2) 검차 보정: _calc_inspection_adj → %p → v_factory 기준 금액
      3) 중간가: v_price + year_adj + inspect_adj
      4) 출고가 스케일링: mid × (target_factory / v_factory)

    반환: 환산가 (만원) 또는 None (환산 불가)
    """
    v_price = v.get("낙찰가", 0) or 0
    if v_price <= 0:
        return None

    v_factory = _pick_vehicle_price(v)
    target_factory = _pick_vehicle_price(target)

    if v_factory <= 0 or target_factory <= 0:
        return None

    target_year = target.get("year", 0) or 0
    v_year = v.get("연식", 0) or 0

    # 1) 연식 환산
    year_rate = rules.get("year_adjustment", {}).get("rate_per_year", 0.014)
    year_adj = (target_year - v_year) * year_rate * v_factory

    # 2) 검차 환산 (AA급 무사고 기준)
    inspect_adj_pct, _ = _calc_inspection_adj(v)
    inspect_adj = inspect_adj_pct / 100 * v_factory

    # 3) 중간가
    mid = v_price + year_adj + inspect_adj

    # 4) 출고가 스케일링
    normalized = mid * (target_factory / v_factory)

    return round(normalized, 1)


# =========================================================================
# 유사차량 테이블 포맷
# =========================================================================

def _format_comparable_table(vehicles: list[dict], target: dict) -> str:
    """유사차량을 환산가 포함 compact 테이블로 포맷"""
    if not vehicles:
        return "(데이터 없음)"

    lines = [
        "연식 | 주행(km) | 낙찰가(만) | 출고가(만) | 비율 | 환산가(만) | 트림 | 연료 | 판매일"
    ]
    for v in vehicles:
        auction_price = v.get("낙찰가", 0) or 0
        factory_price = v.get("factory_price", 0) or 0

        # 낙찰가/출고가 비율 계산
        if factory_price > 0 and auction_price > 0:
            ratio = f"{auction_price / factory_price:.3f}"
        else:
            ratio = "-"

        # 환산가
        norm = v.get("환산가")
        norm_str = f"{norm:,.0f}" if norm is not None else "-"

        sale_date = v.get("개최일", "")
        if sale_date and len(str(sale_date)) > 7:
            sale_date = str(sale_date)[:7]  # YYYY-MM

        lines.append(
            f"{v.get('연식', '')} | "
            f"{v.get('주행거리', 0):,} | "
            f"{auction_price:,.0f} | "
            f"{factory_price or '-'} | "
            f"{ratio} | "
            f"{norm_str} | "
            f"{v.get('trim', '')} | "
            f"{v.get('연료', '')} | "
            f"{sale_date}"
        )
    return "\n".join(lines)


def _build_criteria_user_message(
    target: dict,
    reference: dict,
    vehicles: list[dict],
) -> str:
    """LLM에 전달할 유저 메시지 생성"""

    target_factory = _pick_vehicle_price(target)

    target_info = (
        f"## 대상차량\n"
        f"- 제작사: {target.get('maker', '')}\n"
        f"- 모델: {target.get('model', '')}\n"
        f"- 연식: {target.get('year', '')}년\n"
        f"- 주행거리: {target.get('mileage', 0):,}km\n"
        f"- 연료: {target.get('fuel', '')}\n"
        f"- 트림: {target.get('trim', '')}\n"
        f"- 출고가/기본가: {target_factory:,.0f}만원\n"
    )

    ref_info = (
        f"## 기준차량 (선택된 비교 차량)\n"
        f"- 연식: {reference.get('year', '')}년\n"
        f"- 주행거리: {reference.get('mileage', 0):,}km\n"
        f"- 낙찰가: {reference.get('auction_price', 0):,.0f}만원\n"
        f"- 출고가: {reference.get('factory_price', 0)}만원\n"
        f"- 트림: {reference.get('trim', '')}\n"
        f"- 연료: {reference.get('fuel', '')}\n"
    )

    norm_info = (
        f"## 환산 기준\n"
        f"모든 유사차량은 대상차량과 동일 조건으로 환산되었습니다:\n"
        f"- 연식: {target.get('year', '')}년 기준 (연당 1.4% 보정)\n"
        f"- 출고가: {target_factory:,.0f}만원 기준 (비례 스케일링)\n"
        f"- 검차: AA급 무사고 기준 (사고이력 보정 제거)\n"
        f"\"환산가(만)\" 열은 위 조건으로 정규화된 금액입니다.\n"
    )

    return (
        f"{target_info}\n"
        f"{ref_info}\n"
        f"{norm_info}\n"
        f"## 유사차량 데이터 ({len(vehicles)}건)\n"
        f"{_format_comparable_table(vehicles, target)}\n\n"
        f"환산가와 주행거리의 관계를 분석하여 주행거리 감가율(%p/만km)을 도출해주세요.\n"
        f"연식 보정율은 환산 과정에서 이미 반영되었으므로, 데이터에서 잔차가 보이지 않는 한 1.4%/년을 유지하세요."
    )


# =========================================================================
# 연식대별 클램핑
# =========================================================================

# (기본값, 최소, 최대)
_MILEAGE_RATE_BANDS: list[tuple[int, float, float, float]] = [
    # max_age, default, min, max
    (3,  1.4, 1.0, 2.0),
    (6,  1.0, 0.7, 1.5),
    (9,  0.7, 0.5, 1.0),
    (99, 0.7, 0.3, 1.0),
]

_YEAR_RATE_MIN = 1.0
_YEAR_RATE_MAX = 2.0


def _clamp_criteria(criteria: PricingCriteria, target_year: int) -> PricingCriteria:
    """LLM 출력을 연식대별 허용 범위로 클램핑."""
    age = 2026 - target_year

    for max_age, default, rate_min, rate_max in _MILEAGE_RATE_BANDS:
        if age <= max_age:
            break

    original = criteria.mileage_rate_per_10k
    clamped = max(rate_min, min(rate_max, original))
    if clamped != original:
        logger.warning(
            "주행거리 감가율 클램핑: %.2f → %.2f (%d년식, %d년차, 범위 %.1f~%.1f)",
            original, clamped, target_year, age, rate_min, rate_max,
        )
    criteria.mileage_rate_per_10k = clamped

    original_yr = criteria.year_rate_per_year
    clamped_yr = max(_YEAR_RATE_MIN, min(_YEAR_RATE_MAX, original_yr))
    if clamped_yr != original_yr:
        logger.warning(
            "연식 보정율 클램핑: %.2f → %.2f", original_yr, clamped_yr,
        )
    criteria.year_rate_per_year = clamped_yr

    return criteria


# =========================================================================
# JSON 파싱
# =========================================================================

def _parse_criteria_response(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    # ```json ... ``` 블록 추출
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # { ... } 블록 추출
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    return json.loads(text)


# =========================================================================
# 메인 분석 함수
# =========================================================================

def analyze_criteria(
    target: dict,
    reference: dict,
    model: str = "claude-sonnet-4-20250514",
) -> AnalyzeCriteriaResult:
    """
    LLM 기반 보정 기준 분석 — 메인 진입점.

    1. 유사 차량 자동 수집 (Firestore)
    2. compact 테이블로 프롬프트 구성 (낙찰가/출고가 비율 포함)
    3. Claude Sonnet 1회 호출
    4. JSON 파싱 → AnalyzeCriteriaResult 반환
    5. 실패 시 연식 기반 업계 기본값 폴백

    Args:
        target: 대상차량 정보 dict (maker, model, year, mileage, trim, ...)
        reference: 기준차량 정보 dict (auction_price, year, mileage, trim, factory_price, ...)
        model: Claude 모델 ID

    Returns:
        AnalyzeCriteriaResult: 보정 기준 + 분석 요약
    """
    maker = target.get("maker", "")
    model_name = target.get("model", "")
    year = target.get("year", 2024)
    fuel = target.get("fuel")
    trim = target.get("trim")

    # 1) 유사 차량 수집
    vehicles = fetch_comparable_vehicles(
        maker=maker,
        model=model_name,
        year=year,
        fuel=fuel,
        trim=trim,
        limit=100,
    )

    total_vehicles = len(vehicles)
    logger.info("보정 기준 분석 데이터 수집 — %d건", total_vehicles)

    # 각 유사차량에 환산가 주입
    normalized_count = 0
    for v in vehicles:
        norm = _normalize_vehicle(v, target, _PRICING_RULES)
        v["환산가"] = norm
        if norm is not None:
            normalized_count += 1
    logger.info("환산 가능 차량: %d / %d건", normalized_count, total_vehicles)

    # 데이터 부족 시 기본값 반환
    if total_vehicles == 0:
        criteria = _get_default_criteria(year)
        return AnalyzeCriteriaResult(
            criteria=criteria,
            analysis_summary="유사 차량 데이터가 없어 업계 기본값을 사용합니다.",
            vehicles_analyzed=0,
            confidence="낮음",
        )

    # 2) 프롬프트 생성
    user_message = _build_criteria_user_message(target, reference, vehicles)

    # 3) LLM 호출
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=CRITERIA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        logger.info("LLM 보정 기준 분석 — tokens: %d+%d", input_tokens, output_tokens)

        # 4) 응답 파싱
        raw_text = response.content[0].text
        parsed = _parse_criteria_response(raw_text)

        criteria = PricingCriteria(
            mileage_rate_per_10k=float(parsed.get("mileage_rate_per_10k", 1.0)),
            mileage_age_band=parsed.get("mileage_age_band", ""),
            mileage_ceiling_km=int(parsed.get("mileage_ceiling_km", 200000)),
            year_rate_per_year=float(parsed.get("year_rate_per_year", 1.4)),
        )

        # 연식대별 허용 범위로 클램핑
        criteria = _clamp_criteria(criteria, year)

        return AnalyzeCriteriaResult(
            criteria=criteria,
            analysis_summary=parsed.get("analysis_summary", ""),
            vehicles_analyzed=total_vehicles,
            confidence=parsed.get("confidence", "보통"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning("LLM 보정 기준 응답 파싱 실패: %s", e)
        criteria = _get_default_criteria(year)
        return AnalyzeCriteriaResult(
            criteria=criteria,
            analysis_summary=f"LLM 응답 파싱 실패 — 업계 기본값 사용 ({e})",
            vehicles_analyzed=total_vehicles,
            confidence="낮음",
        )

    except Exception as e:
        logger.exception("LLM 보정 기준 분석 오류: %s", e)
        criteria = _get_default_criteria(year)
        return AnalyzeCriteriaResult(
            criteria=criteria,
            analysis_summary=f"LLM 호출 실패 — 업계 기본값 사용 ({type(e).__name__})",
            vehicles_analyzed=total_vehicles,
            confidence="낮음",
        )
