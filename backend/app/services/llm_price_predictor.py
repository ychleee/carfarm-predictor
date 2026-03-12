"""
CarFarm v3 — LLM 기반 가격 예측 서비스

유사 차량을 자동 수집 → Claude Sonnet에 single-shot 전달 → 가격 추론.
기존 rule_engine(8규칙 보정)을 대체하는 데이터 기반 접근.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import anthropic

from app.services.firestore_db import (
    search_auction_db,
    search_retail_db,
    get_price_stats,
)

logger = logging.getLogger(__name__)

# =========================================================================
# 데이터 클래스
# =========================================================================


@dataclass
class PriceFactor:
    factor: str
    impact: float
    description: str


@dataclass
class PricePrediction:
    estimated_auction: float
    estimated_retail: float
    confidence: str  # 높음 / 보통 / 낮음
    reasoning: str
    factors: list[dict] = field(default_factory=list)
    comparable_summary: str = ""
    key_comparables: list[str] = field(default_factory=list)
    vehicles_analyzed: int = 0
    auction_stats: dict = field(default_factory=dict)
    retail_stats: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


# =========================================================================
# 시스템 프롬프트 — 도메인 지식 포함
# =========================================================================

SYSTEM_PROMPT = """당신은 한국 중고차 시장의 전문 프라이싱 분석가입니다.
제공된 유사 차량 데이터를 분석하여 대상차량의 적정 가격을 추론하는 것이 역할입니다.

## 분석 방법

1. **유사차량 데이터 분석**: 제공된 낙찰가/소매가 유사차량 테이블을 꼼꼼히 분석하세요.
2. **시세 통계 참조**: 평균, 중앙값, 최소, 최대 통계를 기준점으로 활용하세요.
3. **차이 요인 반영**: 대상차량과 유사차량 간 차이(주행거리, 색상, 옵션, 사고이력 등)를 가격에 반영하세요.

## 업계 가격 보정 기준 (참고)

### 주행거리 감가
- 2~3년차: 소매가의 2% / 1만km
- 4~6년차: 1.5% / 1만km
- 7년 이상: 1% / 1만km
- 20만km 초과: 증감 미적용 (천장 효과)

### 교환/판금 (사고이력)
- 외판 교환(X): 부위당 출고가의 3% 감가
- 판금(W): 연식/가격대에 따라 1~2% 감가
- 골격사고 (프론트패널, 플로어패널, 사이드멤버 등):
  - C등급(1부위): 15% 감가
  - D등급(2부위): 17% 감가
  - E등급(3부위): 18% 감가
  - F등급(5부위+): 20% 감가

### 색상 선호도 (20만원/단계)
- 대형/SUV: 흰색=검정 > 메탈(회색) > 실버 > 원색
- 중형: 흰색 > 검정 > 메탈 > 실버 > 원색
- 경차: 흰색 > 미색 > 메탈=실버 > 검정 > 원색

### 옵션
- 선루프: 50만원
- 일반 추가옵션: 약 20만원/개
- 기본옵션(에어백, 에어컨 등)은 가격 차이 없음

### 연식 차이
- 연당 2% (낙찰가 기준)

### 소매가-낙찰가 관계
- 소매가 ≈ 낙찰가 / 0.90 (국산차), 0.88 (수입차)
- 최소 마진: 150만원

## 출력 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{
  "estimated_auction": 1500,
  "estimated_retail": 1700,
  "confidence": "높음",
  "reasoning": "분석 근거를 3~5문장으로 설명",
  "factors": [
    {"factor": "주행거리", "impact": -50, "description": "평균 대비 2만km 초과로 약 50만원 감가"},
    {"factor": "색상", "impact": 20, "description": "흰색 선호도 프리미엄"}
  ],
  "comparable_summary": "낙찰 20건(평균 1,480만), 소매 10건(평균 1,720만) 분석",
  "key_comparables": ["가장_유사한_차량_ID_1", "ID_2", "ID_3"]
}
```

- estimated_auction, estimated_retail: 만원 단위 정수
- confidence: "높음"(데이터 충분, 조건 유사), "보통"(데이터 적거나 조건 차이), "낮음"(데이터 부족)
- factors: 가격에 영향을 미치는 요인 목록. impact는 만원 단위 (+는 가산, -는 감가)
- key_comparables: 가장 유사한 차량 3건의 auction_id
"""


# =========================================================================
# 유사도 점수 계산
# =========================================================================

def _similarity_score(target: dict, vehicle: dict) -> float:
    """대상차량과 후보차량 간 유사도 점수 (높을수록 유사)"""
    score = 0.0

    # 연료 일치 (필수 — 불일치 시 -999)
    t_fuel = (target.get("fuel") or "").strip()
    v_fuel = (vehicle.get("연료") or vehicle.get("fuel") or "").strip()
    if t_fuel and v_fuel and not _fuel_match(t_fuel, v_fuel):
        return -999

    # 트림 일치
    t_trim = (target.get("trim") or "").strip().lower()
    v_trim = (vehicle.get("trim") or "").strip().lower()
    if t_trim and v_trim:
        if t_trim == v_trim:
            score += 30
        elif t_trim in v_trim or v_trim in t_trim:
            score += 20

    # 연식 차이
    t_year = target.get("year", 0)
    v_year = vehicle.get("연식", 0) or 0
    if t_year and v_year:
        diff = abs(t_year - v_year)
        if diff == 0:
            score += 20
        elif diff == 1:
            score += 15
        elif diff == 2:
            score += 10
        elif diff == 3:
            score += 5

    # 주행거리 차이
    t_mil = target.get("mileage", 0) or 0
    v_mil = vehicle.get("주행거리", 0) or 0
    if t_mil and v_mil:
        diff_km = abs(t_mil - v_mil)
        if diff_km <= 10000:
            score += 15
        elif diff_km <= 30000:
            score += 10
        elif diff_km <= 50000:
            score += 5

    # 색상 일치
    t_color = (target.get("color") or "").strip()
    v_color = (vehicle.get("색상") or "").strip()
    if t_color and v_color and t_color == v_color:
        score += 5

    return score


def _fuel_match(a: str, b: str) -> bool:
    """연료 동의어 매칭"""
    synonyms = [
        {"가솔린", "휘발유", "gasoline"},
        {"디젤", "경유", "diesel"},
        {"하이브리드", "hybrid", "HEV"},
        {"전기", "EV", "electric"},
        {"LPG", "lpg"},
    ]
    a_lower = a.lower()
    b_lower = b.lower()
    if a_lower == b_lower:
        return True
    for group in synonyms:
        a_in = any(s.lower() in a_lower for s in group)
        b_in = any(s.lower() in b_lower for s in group)
        if a_in and b_in:
            return True
    return False


# =========================================================================
# 데이터 수집
# =========================================================================

def _fetch_comparable_vehicles(target: dict) -> tuple[list[dict], list[dict], dict, dict]:
    """
    유사 차량 자동 수집.

    Returns:
        (auction_vehicles, retail_vehicles, auction_stats, retail_stats)
    """
    maker = target.get("maker", "")
    model = target.get("model", "")
    year = target.get("year", 2024)
    fuel = target.get("fuel")
    trim = target.get("trim")

    year_min = year - 3
    year_max = year + 3

    # 1) 낙찰가 데이터
    auction_raw = search_auction_db(
        model=model, maker=maker, fuel=fuel, trim=trim,
        year_min=year_min, year_max=year_max,
        limit=200, sort_by="날짜",
    )

    # 트림 매칭이 부족하면 완화 재검색
    if len(auction_raw) < 10 and trim:
        auction_raw_relaxed = search_auction_db(
            model=model, maker=maker, fuel=fuel, trim=None,
            year_min=year_min, year_max=year_max,
            limit=200, sort_by="날짜",
        )
        # 기존 결과에 없는 것만 추가
        seen = {v.get("auction_id") for v in auction_raw}
        for v in auction_raw_relaxed:
            if v.get("auction_id") not in seen:
                auction_raw.append(v)

    # 2) 소매가 데이터
    retail_raw = search_retail_db(
        model=model, maker=maker, fuel=fuel, trim=trim,
        year_min=year_min, year_max=year_max,
        limit=200,
    )

    if len(retail_raw) < 10 and trim:
        retail_raw_relaxed = search_retail_db(
            model=model, maker=maker, fuel=fuel, trim=None,
            year_min=year_min, year_max=year_max,
            limit=200,
        )
        seen = {v.get("auction_id") for v in retail_raw}
        for v in retail_raw_relaxed:
            if v.get("auction_id") not in seen:
                retail_raw.append(v)

    # 3) 유사도 점수로 정렬 + 상위 선별
    for v in auction_raw:
        v["_score"] = _similarity_score(target, v)
    for v in retail_raw:
        v["_score"] = _similarity_score(target, v)

    auction_raw = [v for v in auction_raw if v["_score"] > -900]
    retail_raw = [v for v in retail_raw if v["_score"] > -900]

    auction_raw.sort(key=lambda x: x["_score"], reverse=True)
    retail_raw.sort(key=lambda x: x["_score"], reverse=True)

    auction_top = auction_raw[:20]
    retail_top = retail_raw[:10]

    # 4) 시세 통계
    auction_stats = get_price_stats(maker, model, year=year)
    retail_stats_raw = get_price_stats(maker, model)  # 전체 기간

    return auction_top, retail_top, auction_stats, retail_stats_raw


# =========================================================================
# 프롬프트 생성
# =========================================================================

def _format_auction_table(vehicles: list[dict]) -> str:
    """낙찰가 유사차량을 compact 테이블로 포맷"""
    if not vehicles:
        return "(데이터 없음)"

    lines = ["ID | 연식 | 주행(km) | 낙찰가(만) | 트림 | 색상 | 연료 | 옵션수 | 교환 | 판금 | 골격부위 | 출고가(만) | 판매일"]
    for v in vehicles:
        options_str = v.get("옵션", "")
        n_opts = len(options_str.split(",")) if options_str and options_str.strip() else 0

        # 골격 부위 수
        part_damages = v.get("part_damages", [])
        structural_parts = {"FRONT_PANEL", "FRONT_CROSS_MEMBER", "FLOOR_PANEL",
                          "SIDE_MEMBER", "REAR_CROSS_MEMBER", "TRUNK_FLOOR_PANEL", "REAR_PANEL"}
        structural_count = sum(1 for pd in part_damages
                              if pd.get("part") in structural_parts
                              and pd.get("damage_type") in ("EXCHANGE", "PANEL_WELDING", "BENT"))

        sale_date = v.get("개최일", "")
        if sale_date and len(str(sale_date)) > 7:
            sale_date = str(sale_date)[:7]  # YYYY-MM

        lines.append(
            f"{v.get('auction_id', '')[:12]} | "
            f"{v.get('연식', '')} | "
            f"{v.get('주행거리', 0):,} | "
            f"{v.get('낙찰가', 0):,.0f} | "
            f"{v.get('trim', '')} | "
            f"{v.get('색상', '')} | "
            f"{v.get('연료', '')} | "
            f"{n_opts} | "
            f"{v.get('exchange_count', 0)} | "
            f"{v.get('bodywork_count', 0)} | "
            f"{structural_count} | "
            f"{v.get('factory_price', 0) or ''} | "
            f"{sale_date}"
        )
    return "\n".join(lines)


def _format_retail_table(vehicles: list[dict]) -> str:
    """소매가 유사차량을 compact 테이블로 포맷"""
    if not vehicles:
        return "(데이터 없음)"

    lines = ["ID | 연식 | 주행(km) | 소매가(만) | 트림 | 색상 | 연료 | 옵션수 | 출고가(만)"]
    for v in vehicles:
        options_str = v.get("옵션", "")
        n_opts = len(options_str.split(",")) if options_str and options_str.strip() else 0

        lines.append(
            f"{v.get('auction_id', '')[:12]} | "
            f"{v.get('연식', '')} | "
            f"{v.get('주행거리', 0):,} | "
            f"{v.get('소매가', 0):,.0f} | "
            f"{v.get('trim', '')} | "
            f"{v.get('색상', '')} | "
            f"{v.get('연료', '')} | "
            f"{n_opts} | "
            f"{v.get('factory_price', 0) or ''}"
        )
    return "\n".join(lines)


def _format_stats(stats: dict, label: str) -> str:
    """시세 통계를 한 줄로 포맷"""
    if stats.get("count", 0) == 0:
        return f"{label}: 데이터 없음"
    return (
        f"{label}: 건수={stats['count']}, "
        f"평균={stats.get('mean', 0):,.0f}만, "
        f"중앙값={stats.get('median', 0):,.0f}만, "
        f"최소={stats.get('min', 0):,.0f}만, "
        f"최대={stats.get('max', 0):,.0f}만"
    )


def _build_user_message(
    target: dict,
    auction_vehicles: list[dict],
    retail_vehicles: list[dict],
    auction_stats: dict,
    retail_stats: dict,
) -> str:
    """LLM에 전달할 유저 메시지 생성"""

    # 대상차량 정보
    target_info = (
        f"## 대상차량\n"
        f"- 제작사: {target.get('maker', '')}\n"
        f"- 모델: {target.get('model', '')}\n"
        f"- 연식: {target.get('year', '')}년\n"
        f"- 주행거리: {target.get('mileage', 0):,}km\n"
        f"- 연료: {target.get('fuel', '')}\n"
        f"- 트림: {target.get('trim', '')}\n"
        f"- 색상: {target.get('color', '')}\n"
        f"- 옵션: {', '.join(target.get('options', [])) or '정보없음'}\n"
        f"- 교환 부위 수: {target.get('exchange_count', 0)}\n"
        f"- 판금 부위 수: {target.get('bodywork_count', 0)}\n"
    )

    # 파트 손상 정보
    part_damages = target.get("part_damages", [])
    if part_damages:
        damage_strs = [f"  - {pd.get('part', '')}: {pd.get('damage_type', '')}" for pd in part_damages]
        target_info += f"- 부위별 손상:\n" + "\n".join(damage_strs) + "\n"

    return (
        f"{target_info}\n"
        f"## 시세 통계 (최근 3개월)\n"
        f"{_format_stats(auction_stats, '낙찰가')}\n"
        f"{_format_stats(retail_stats, '소매가')}\n\n"
        f"## 낙찰가 유사차량 ({len(auction_vehicles)}건)\n"
        f"{_format_auction_table(auction_vehicles)}\n\n"
        f"## 소매가 유사차량 ({len(retail_vehicles)}건)\n"
        f"{_format_retail_table(retail_vehicles)}\n\n"
        f"위 데이터를 분석하여 대상차량의 적정 낙찰가와 소매가를 추론해주세요."
    )


# =========================================================================
# LLM 호출 + 파싱
# =========================================================================

def _parse_prediction(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    # ```json ... ``` 블록 추출
    import re
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # { ... } 블록 추출
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    return json.loads(text)


def predict_price(
    target: dict,
    model: str = "claude-sonnet-4-20250514",
) -> PricePrediction:
    """
    LLM 기반 가격 예측 — 메인 진입점.

    1. 유사 차량 자동 수집 (Firestore)
    2. compact 테이블 + 통계로 프롬프트 구성
    3. Claude Sonnet 1회 호출
    4. JSON 파싱 → PricePrediction 반환
    """
    # 1) 데이터 수집
    auction_vehicles, retail_vehicles, auction_stats, retail_stats = (
        _fetch_comparable_vehicles(target)
    )

    total_vehicles = len(auction_vehicles) + len(retail_vehicles)
    logger.info(
        "가격 예측 데이터 수집 — 낙찰: %d건, 소매: %d건",
        len(auction_vehicles), len(retail_vehicles),
    )

    # 데이터 부족 시 경고
    if total_vehicles == 0:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="유사 차량 데이터를 찾을 수 없어 가격 예측이 불가합니다.",
            vehicles_analyzed=0,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
        )

    # 2) 프롬프트 생성
    user_message = _build_user_message(
        target, auction_vehicles, retail_vehicles,
        auction_stats, retail_stats,
    )

    # 3) LLM 호출
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # 토큰 사용량
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    logger.info("LLM 가격 예측 — tokens: %d+%d", input_tokens, output_tokens)

    # 4) 응답 파싱
    raw_text = response.content[0].text

    try:
        parsed = _parse_prediction(raw_text)
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning("LLM 응답 JSON 파싱 실패: %s", e)
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"LLM 응답 파싱 실패. 원본: {raw_text[:500]}",
            vehicles_analyzed=total_vehicles,
            auction_stats=auction_stats,
            retail_stats=retail_stats,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return PricePrediction(
        estimated_auction=parsed.get("estimated_auction", 0),
        estimated_retail=parsed.get("estimated_retail", 0),
        confidence=parsed.get("confidence", "보통"),
        reasoning=parsed.get("reasoning", ""),
        factors=parsed.get("factors", []),
        comparable_summary=parsed.get("comparable_summary", ""),
        key_comparables=parsed.get("key_comparables", []),
        vehicles_analyzed=total_vehicles,
        auction_stats=auction_stats,
        retail_stats=retail_stats,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
