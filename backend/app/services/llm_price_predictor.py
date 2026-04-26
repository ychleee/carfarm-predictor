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
from datetime import datetime, timezone


import anthropic

from app.services.firestore_db import (
    search_auction_db,
    search_retail_db,
    get_price_stats,
)
from app.services.taxonomy_search import resolve_base_model
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
    estimated_auction_export: float = 0
    last_export_date: str = ""
    reasoning: str = ""
    factors: list[dict] = field(default_factory=list)
    auction_reasoning: str = ""
    retail_reasoning: str = ""
    export_reasoning: str = ""
    auction_factors: list[dict] = field(default_factory=list)
    retail_factors: list[dict] = field(default_factory=list)
    comparable_summary: str = ""
    key_comparables: list[str] = field(default_factory=list)
    vehicles_analyzed: int = 0
    auction_stats: dict = field(default_factory=dict)
    retail_stats: dict = field(default_factory=dict)
    comparable_auction_vehicles: list[dict] = field(default_factory=list)
    comparable_retail_vehicles: list[dict] = field(default_factory=list)
    retail_brackets: list[dict] = field(default_factory=list)
    auction_brackets: list[dict] = field(default_factory=list)
    export_brackets: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# =========================================================================
# 시스템 프롬프트 — 도메인 지식 포함
# =========================================================================

SYSTEM_PROMPT = """당신은 한국 중고차 시장의 전문 프라이싱 분석가입니다.
제공된 유사 차량 데이터를 분석하여 대상차량의 적정 가격을 추론하는 것이 역할입니다.

## ★ 가격 산정 원칙 (최우선) ★
- **유사차량 데이터를 분석하여 합리적인 시장 가격을 산정하세요.**
- 무사고 차량 데이터 10건 이상: 하위 25%~중앙값 부근을 기준점으로 잡으세요.
- 무사고 차량 데이터 10건 미만: 중앙값 부근을 기준점으로 잡으세요. 데이터가 적을수록 보수적 하한 산정은 피하세요.
- 사고차(교환·판금 있는 차량)의 가격은 무사고 가격으로 역보정한 후 분석하세요 (아래 '사고차 역보정' 참조).
- 명백한 이상치(비정상적으로 낮거나 높은 가격)는 제외하세요.
- 낙찰가·소매가 모두 이 원칙을 적용합니다.

## 분석 방법

1. **최근 데이터 우선**: 유사차량 테이블의 판매일/등록일을 반드시 확인하세요. 최근 1~2개월 내 데이터를 가장 중요한 기준으로 삼고, 3개월 이상 지난 데이터는 보조 참고 자료로만 활용하세요. 시세는 시간에 따라 변동하므로 오래된 데이터로 가격을 산정하면 현재 시세와 괴리가 생깁니다.
2. **출고가/기본가 기반 분석**: 대상차량의 출고가(또는 기본가)를 기준점으로 잡고, 옵션 차이를 보정하세요.
   - 출고가 - 기본가 = 옵션 총 가치. 유사차량과의 옵션 차이를 이 기준으로 보정합니다.
   - 출고가가 없으면 유사차량의 출고가/기본가 데이터에서 유추하세요.
2. **유사차량 데이터 분석**: 제공된 낙찰가/소매가 유사차량 테이블을 꼼꼼히 분석하세요.
3. **시세 통계 참조**: 평균, 중앙값, 최소, 최대 통계에서 **하위권(1사분위~중앙값)**을 기준점으로 활용하세요.
4. **차이 요인 반영**: 대상차량과 유사차량 간 차이(주행거리, 색상, 옵션 등)를 가격에 반영하세요. 대상차량은 AA등급(무사고)으로 간주하므로 사고이력/검차상태 감가는 적용하지 마세요.
5. **수출/내수 구분 주의**: 수출 차량은 내수 대비 가격이 크게 다를 수 있습니다. 반드시 대상차량의 내수/수출 여부를 확인하고, 비교 대상도 같은 유형으로 비교하세요.

## 업계 가격 보정 기준 (실측 데이터 반영)

### 주행거리 감가 (소매가 기준 %/만km)
- 2~3년차: 1.4% / 만km
- 4~6년차: 1.0% / 만km
- 7~9년차: 0.7% / 만km
- 10년 이상: 2,000만원 이상 차량은 0.7%/만km, 미만은 7만원/만km 정액
- 20만km 초과: 증감 미적용 (천장 효과)

### 대상차량 AA등급 가정 (★ 중요 ★)
- **대상차량은 무사고 AA등급으로 간주합니다.** 사고이력·교환·판금·골격사고 감가를 적용하지 마세요.
- 유사차량의 사고이력은 참고하되, 대상차량 가격에서 검차 관련 감가를 적용하지 마세요.

### 사고차 가격 역보정 (★ 매우 중요 ★)
- 유사차량이 사고차(교환/판금 기록 있음)인 경우, 해당 차량의 실거래가에 사고 감가분을 **다시 더하여** 무사고 기준가를 역산하세요.
- **무사고 AA등급 예상가는 동일 조건 사고차의 실거래가보다 반드시 높아야 합니다.**
- 사고 감가 역보정 기준 (1건당):
  - 외판 교환: +30~50만원
  - 외판 판금: +15~25만원
  - 프레임/골격 교환: +100~200만원
  - 프레임/골격 판금: +50~80만원
- 예시: 외판교환1 + 외판판금7인 차량이 420만에 거래 → 무사고 추정가 = 420 + (교환 40 + 판금 7×20) ≈ 600만
- 사고차만 있고 무사고 차량이 없는 경우, 역보정된 가격을 핵심 기준으로 사용하세요.

### 색상 선호도 (20만원/단계, 연식에 따라 감소)
- 대형/SUV: 흰색=검정 > 메탈(회색) > 실버 > 원색
- 중형: 흰색 > 검정 > 메탈 > 실버 > 원색
- 경차: 흰색 > 미색 > 메탈=실버 > 검정 > 원색
- 연식 가중치: 3년이하 100%, 7년이하 70%, 10년이하 50%, 이상 30%

### 옵션 보정 (출고가-기본가 차이 기반)
- 선루프: 50만원 (업계 합의)
- 기본옵션(에어백, 에어컨, ABS, ESC, 파워윈도우 등)은 가격 차이 없음
- 추가옵션: 출고가-기본가 차이에서 옵션 수로 나눈 단가 참고 (보통 약 20만원/개)

### 연식 차이
- 연당 1.4% (업계 2% → 실측 반영 축소)

### 수출/내수 구분 (★ 매우 중요 ★)
- estimated_auction은 반드시 내수(수출=빈칸) 유사차량만으로 산정하세요
- estimated_auction_export는 반드시 수출(수출=Y) 유사차량만으로 산정하세요
- **수출가 산정 시에도 내수와 동일하게 주행거리/연식/트림/색상 차이를 반드시 보정하세요.** 수출 유사차량의 낙찰가를 단순 평균하지 말고, 대상차량과의 조건 차이를 감안하여 보정된 가격을 산출하세요.
- 테이블에 수출=Y인 차량이 1건이라도 있으면 estimated_auction_export를 반드시 산출하세요 (0으로 두지 마세요)
- 수출 데이터가 없으면 0으로 두세요
- 테이블의 "수출" 열을 반드시 확인하세요
- 수출가와 내수가의 크기 관계는 시장 데이터가 보여주는 그대로 반영하세요. 인위적으로 보정하지 마세요.

### 소매가 산정 방법 (★ 실제 소매 매물 기반 ★)
- **소매가는 반드시 소매가 유사차량 테이블의 실제 매물 가격을 기반으로 산정하세요.**
- 무사고 + 유사 조건(연식/주행/트림) 차량의 실제 매물가를 핵심 기준점으로 사용합니다.
- 대상차량과의 주행거리/연식/옵션 차이를 반영하여 보정하세요.
- 소매가 유사차량 데이터가 충분하면 낙찰가와의 비율 공식은 참고하지 마세요.
- 소매가 데이터가 부족할 때만 보조적으로 아래 공식 참고:
  - 1,500만원 초과: 소매가 ≈ 낙찰가 / 0.90 (국산), 0.88 (수입)
  - 1,500만원 이하: 비율 공식이 맞지 않으므로 사용 금지. 실제 소매 매물가 또는 낙찰가 + 고정 마진(150~200만원)으로 산정
- **★ 필수: 낙찰가 1,500만원 이하인 경우, 소매가는 낙찰가보다 최소 150만원 이상 높아야 합니다.** 이는 딜러 마진·정비·이전비 등 실비를 반영한 업계 최소 기준입니다. 소매가 산정 결과가 낙찰가+150만원 미만이면 반드시 상향 조정하세요.

## 출력 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
낙찰가와 소매가 각각에 대해 별도의 요인 분석과 근거를 제공하세요.

```json
{
  "estimated_auction": 1500,
  "estimated_auction_export": 1650,
  "estimated_retail": 1700,
  "confidence": "높음",
  "auction_reasoning": "내수 낙찰가 분석 근거를 3~5문장으로 설명",
  "retail_reasoning": "소매가 분석 근거를 3~5문장으로 설명",
  "export_reasoning": "수출 낙찰가 분석 근거",
  "auction_factors": [
    {"factor": "주행거리", "impact": -50, "description": "낙찰 평균 대비 2만km 초과로 약 50만원 감가"},
    {"factor": "연식", "impact": -30, "description": "기준 대비 1년 구형으로 약 30만원 감가"}
  ],
  "retail_factors": [
    {"factor": "주행거리", "impact": -60, "description": "소매 평균 대비 2만km 초과로 약 60만원 감가"},
    {"factor": "색상", "impact": 20, "description": "흰색 선호도 프리미엄"}
  ],
  "comparable_summary": "낙찰 20건(평균 1,480만), 소매 20건(평균 1,720만) 분석",
  "key_comparables": ["가장_유사한_차량_ID_1", "ID_2", "ID_3"]
}
```

- estimated_auction: 내수 낙찰 예상가 (만원 단위 정수)
- estimated_auction_export: 수출 낙찰 예상가 (만원 단위 정수, 수출 데이터가 없으면 0)
- estimated_retail: 소매 예상가 (만원 단위 정수)
- confidence: "높음"(데이터 충분, 조건 유사), "보통"(데이터 적거나 조건 차이), "낮음"(데이터 부족)
- auction_factors: 내수 낙찰가에 영향을 미치는 요인. impact는 만원 단위
- retail_factors: 소매가에 영향을 미치는 요인. impact는 만원 단위
- auction_reasoning: 내수 낙찰가 산정 근거
- retail_reasoning: 소매가 산정 근거
- export_reasoning: 수출 낙찰가 산정 근거
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

    # 트림 일치 ("기본형"/"(세부등급 없음)" 정규화, 공백 차이 허용)
    from app.services.firestore_db import _normalize_trim
    t_trim = _normalize_trim(target.get("trim") or "")
    v_trim = _normalize_trim(vehicle.get("trim") or "")
    if t_trim and v_trim:
        t_compact = t_trim.replace(" ", "")
        v_compact = v_trim.replace(" ", "")
        if t_trim == v_trim or t_compact == v_compact:
            score += 30
        elif (t_trim in v_trim or v_trim in t_trim
              or (t_compact and v_compact and (t_compact in v_compact or v_compact in t_compact))):
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

    # 영업용/택시/렌터카 매칭 — 용도 유사 차량 우선
    _commercial_keywords = ("영업", "택시", "렌터카", "렌트", "법인")
    t_trim_lower = (target.get("trim") or "").lower()
    v_trim_lower = (vehicle.get("trim") or vehicle.get("트림") or "").lower()
    t_is_commercial = any(kw in t_trim_lower for kw in _commercial_keywords)
    v_is_commercial = any(kw in v_trim_lower for kw in _commercial_keywords)
    if t_is_commercial and v_is_commercial:
        score += 25  # 같은 용도 차량 우선
    elif t_is_commercial != v_is_commercial:
        score -= 10  # 용도 불일치 페널티

    # 색상 일치
    t_color = (target.get("color") or "").strip()
    v_color = (vehicle.get("색상") or "").strip()
    if t_color and v_color and t_color == v_color:
        score += 5

    # 출고가(없으면 기본가) 유사도
    t_fp = target.get("factory_price") or target.get("base_price") or 0
    v_fp = vehicle.get("factory_price") or vehicle.get("base_price") or 0
    if t_fp > 0 and v_fp > 0:
        ratio = min(t_fp, v_fp) / max(t_fp, v_fp)
        if ratio >= 0.98:       # ±2% 이내 — 사실상 동일
            score += 20
        elif ratio >= 0.95:     # ±5% 이내
            score += 15
        elif ratio >= 0.90:     # ±10% 이내
            score += 10
        elif ratio >= 0.80:     # ±20% 이내
            score += 5

    # 최근 데이터 가산점 (개최일 또는 매물등록일)
    date_str = vehicle.get("개최일") or vehicle.get("매물등록일") or ""
    if date_str:
        try:
            sale_date = datetime.fromisoformat(date_str.replace("Z", "+00:00") if "T" in date_str else date_str + "T00:00:00+00:00")
            days_ago = (datetime.now(timezone.utc) - sale_date).days
            if days_ago <= 30:
                score += 25      # 1개월 이내
            elif days_ago <= 60:
                score += 20      # 2개월 이내
            elif days_ago <= 90:
                score += 15      # 3개월 이내
            elif days_ago <= 180:
                score += 8       # 6개월 이내
            # 6개월 초과: 가산 없음
        except (ValueError, TypeError):
            pass

    return score


def _is_hybrid(fuel: str) -> bool:
    """하이브리드/PHEV/전기복합 여부 판별"""
    f = fuel.lower()
    return ("하이브리드" in f or "hybrid" in f or "hev" in f
            or "전기" in f and "가솔린" in f   # 가솔린+전기 = PHEV
            or "전기" in f and ("디젤" in f or "경유" in f))


def _fuel_match(a: str, b: str) -> bool:
    """연료 동의어 매칭 — 하이브리드/비하이브리드 엄격 구분 (공백 무시)"""
    synonyms = [
        {"가솔린", "휘발유", "gasoline"},
        {"디젤", "경유", "diesel"},
        {"하이브리드", "hybrid", "HEV"},
        {"전기", "EV", "electric"},
        {"LPG", "lpg", "LPi", "lpi", "엘피지"},
        {"수소", "수소+전기", "hydrogen", "FCEV"},
    ]
    a_lower = a.lower()
    b_lower = b.lower()
    if a_lower == b_lower:
        return True
    # 하이브리드 구분: 양쪽이 일치해야 함
    if _is_hybrid(a) != _is_hybrid(b):
        return False
    # 공백 제거 후 비교
    a_compact = a_lower.replace(" ", "")
    b_compact = b_lower.replace(" ", "")
    if a_compact == b_compact:
        return True
    for group in synonyms:
        a_in = any(s.lower().replace(" ", "") in a_compact for s in group)
        b_in = any(s.lower().replace(" ", "") in b_compact for s in group)
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

    # 모델↔트림 필드 오매핑 보정: model이 maker와 같으면 trim에서 모델 추출
    # 예: model="현대", trim="아반떼MD M16 GDi 럭셔리" → model="아반떼", trim="M16 GDi 럭셔리"
    if model and trim and model.strip() == maker.strip():
        resolved = resolve_base_model(trim, maker)
        if resolved and resolved.lower() != maker.lower():
            # trim에서 모델명 부분 제거하여 실제 트림만 남김
            remaining = trim
            # 정규화된 모델명(예: "아반떼")을 원본 trim에서 찾아 제거
            import re as _re
            # "아반떼MD" 처럼 세대접두사+모델이 붙어있을 수 있으므로 전체 앞부분에서 제거
            pattern = _re.compile(
                r"^(?:더\s*올?\s*뉴\s*|올\s*뉴\s*|뉴\s*)?.*?" + _re.escape(resolved),
                _re.IGNORECASE,
            )
            m = pattern.search(remaining)
            if m:
                remaining = remaining[m.end():].strip()
            model = resolved
            trim = remaining or None
            logger.info("모델↔트림 보정: model=%s, trim=%s", model, trim)

    # 같은 연식 → ±1 → ±2 → ±3 (고주행 시 ±5) 자동 확대
    target_mileage = target.get("mileage", 0) or 0
    max_year_delta = 5 if target_mileage > 200000 else 3
    auction_raw: list[dict] = []
    retail_raw: list[dict] = []
    for y_delta in range(0, max_year_delta + 1):
        y_min = year - y_delta
        y_max = year + y_delta
        auction_raw = search_auction_db(
            model=model, maker=maker, fuel=fuel, trim=trim,
            year_min=y_min, year_max=y_max,
            limit=200, sort_by="날짜",
            domestic_only=False,
        )
        retail_raw = search_retail_db(
            model=model, maker=maker, fuel=fuel, trim=trim,
            year_min=y_min, year_max=y_max,
            limit=200,
        )
        if len(auction_raw) + len(retail_raw) >= 3:
            break

    # 트림 필터로 결과 부족 시 트림 없이 재검색 (유사도 점수로 정렬)
    if trim and len(auction_raw) + len(retail_raw) < 3:
        logger.info("트림 '%s' 매칭 결과 부족 (%d건) → 트림 없이 재검색",
                     trim, len(auction_raw) + len(retail_raw))
        for y_delta in range(0, max_year_delta + 1):
            y_min = year - y_delta
            y_max = year + y_delta
            auction_raw = search_auction_db(
                model=model, maker=maker, fuel=fuel, trim=None,
                year_min=y_min, year_max=y_max,
                limit=200, sort_by="날짜",
                domestic_only=False,
            )
            retail_raw = search_retail_db(
                model=model, maker=maker, fuel=fuel, trim=None,
                year_min=y_min, year_max=y_max,
                limit=200,
            )
            if len(auction_raw) + len(retail_raw) >= 3:
                break

    # 3) 유사도 점수로 정렬 + 상위 선별 (내수/수출 분리 확보)
    for v in auction_raw:
        v["_score"] = _similarity_score(target, v)
    for v in retail_raw:
        v["_score"] = _similarity_score(target, v)

    auction_raw = [v for v in auction_raw if v["_score"] > -900]
    retail_raw = [v for v in retail_raw if v["_score"] > -900]

    auction_raw.sort(key=lambda x: x["_score"], reverse=True)
    retail_raw.sort(key=lambda x: x["_score"], reverse=True)

    # 낙찰: 내수 top 15 + 수출 top 5 (수출 데이터가 밀리지 않도록 별도 슬롯)
    auction_domestic = [v for v in auction_raw if not v.get("is_export")][:15]
    auction_export = [v for v in auction_raw if v.get("is_export")][:5]
    seen_auction_ids = {v.get("auction_id") for v in auction_domestic + auction_export}
    # 남은 슬롯을 점수순으로 채움
    remaining = 20 - len(auction_domestic) - len(auction_export)
    if remaining > 0:
        for v in auction_raw:
            if v.get("auction_id") not in seen_auction_ids:
                auction_domestic.append(v)
                seen_auction_ids.add(v.get("auction_id"))
                remaining -= 1
                if remaining <= 0:
                    break
    auction_top = auction_domestic + auction_export

    retail_top = retail_raw[:50]

    # 유사도 점수 내림차순 정렬
    retail_top.sort(key=lambda v: -v.get("_score", 0))

    # 4) 시세 통계 (낙찰/소매 분리, 동일 연식 기준)
    auction_stats = get_price_stats(maker, model, year=year, price_type="auction")
    retail_stats_raw = get_price_stats(maker, model, year=year, price_type="retail")

    return auction_top, retail_top, auction_stats, retail_stats_raw



# =========================================================================
# 프롬프트 생성
# =========================================================================

def _to_man_won(value) -> float:
    """원 → 만원 변환 (이미 만원 단위면 그대로)"""
    v = float(value or 0)
    if v > 100000:  # 10만원 이상이면 원 단위로 판단
        return round(v / 10000, 1)
    return v


def _format_auction_table(vehicles: list[dict]) -> str:
    """낙찰가 유사차량을 compact 테이블로 포맷"""
    if not vehicles:
        return "(데이터 없음)"

    lines = ["ID | 연식 | 주행(km) | 낙찰가(만) | 트림 | 색상 | 연료 | 옵션수 | 교환 | 판금 | 골격부위 | 프레임교환 | 프레임판금 | 외판교환 | 외판판금 | 출고가(만) | 기본가(만) | 수출 | 판매일"]
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
            sale_date = str(sale_date)[:10]  # YYYY-MM

        is_export = "Y" if v.get("is_export", 0) else ""

        auction_price = _to_man_won(v.get('낙찰가', 0))
        factory_price = _to_man_won(v.get('factory_price', 0)) if v.get('factory_price') else ''
        base_price = _to_man_won(v.get('base_price', 0)) if v.get('base_price') else ''

        lines.append(
            f"{v.get('auction_id', '')[:12]} | "
            f"{v.get('연식', '')} | "
            f"{v.get('주행거리', 0):,} | "
            f"{auction_price:,.0f} | "
            f"{v.get('trim', '')} | "
            f"{v.get('색상', '')} | "
            f"{v.get('연료', '')} | "
            f"{n_opts} | "
            f"{v.get('exchange_count', 0)} | "
            f"{v.get('bodywork_count', 0)} | "
            f"{structural_count} | "
            f"{v.get('frame_exchange', 0)} | "
            f"{v.get('frame_bodywork', 0)} | "
            f"{v.get('exterior_exchange', 0)} | "
            f"{v.get('exterior_bodywork', 0)} | "
            f"{factory_price} | "
            f"{base_price} | "
            f"{is_export} | "
            f"{sale_date}"
        )
    return "\n".join(lines)


def _format_retail_table(vehicles: list[dict]) -> str:
    """소매가 유사차량을 compact 테이블로 포맷 (검차·진단 포함)"""
    if not vehicles:
        return "(데이터 없음)"

    lines = ["ID | 연식 | 주행(km) | 소매가(만) | 트림 | 색상 | 연료 | 옵션수 | 프레임교환 | 프레임판금 | 외판교환 | 외판판금 | 사고이력 | 출고가(만) | 기본가(만) | 등록일"]
    for v in vehicles:
        options_str = v.get("옵션", "") or v.get("options", "")
        n_opts = len(options_str.split(",")) if options_str and options_str.strip() else 0

        factory_price = _to_man_won(v.get('factory_price', 0)) if v.get('factory_price') else ''
        base_price = _to_man_won(v.get('base_price', 0)) if v.get('base_price') else ''

        accident = v.get("accident_summary", "")

        listing_date = v.get("매물등록일", "") or v.get("listing_date", "")
        if listing_date and len(str(listing_date)) > 7:
            listing_date = str(listing_date)[:10]

        lines.append(
            f"{v.get('auction_id', '')[:12]} | "
            f"{v.get('연식', '')} | "
            f"{v.get('주행거리', 0):,} | "
            f"{v.get('소매가', 0):,.0f} | "
            f"{v.get('trim', '')} | "
            f"{v.get('색상', '')} | "
            f"{v.get('연료', '')} | "
            f"{n_opts} | "
            f"{v.get('frame_exchange', 0)} | "
            f"{v.get('frame_bodywork', 0)} | "
            f"{v.get('exterior_exchange', 0)} | "
            f"{v.get('exterior_bodywork', 0)} | "
            f"{accident} | "
            f"{factory_price} | "
            f"{base_price} | "
            f"{listing_date}"
        )
    return "\n".join(lines)


def _format_stats(stats: dict, label: str) -> str:
    """시세 통계를 한 줄로 포맷 (만원 단위)"""
    if stats.get("count", 0) == 0:
        return f"{label}: 데이터 없음"
    return (
        f"{label}: 건수={stats['count']}, "
        f"평균={_to_man_won(stats.get('mean', 0)):,.0f}만, "
        f"중앙값={_to_man_won(stats.get('median', 0)):,.0f}만, "
        f"최소={_to_man_won(stats.get('min', 0)):,.0f}만, "
        f"최대={_to_man_won(stats.get('max', 0)):,.0f}만"
    )


def _build_user_message(
    target: dict,
    auction_vehicles: list[dict],
    retail_vehicles: list[dict],
    auction_stats: dict,
    retail_stats: dict,
) -> str:
    """LLM에 전달할 유저 메시지 생성"""

    # 출고가/기본가
    factory_price = target.get('factory_price', 0) or 0
    base_price = target.get('base_price', 0) or 0
    domestic = target.get('domestic', True)

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
        f"- 출고가: {factory_price:,.0f}만원\n"
        f"- 기본가: {base_price:,.0f}만원\n"
    )

    # 옵션 총 가치 (출고가 - 기본가)
    if factory_price > 0 and base_price > 0 and factory_price > base_price:
        option_value = factory_price - base_price
        target_info += f"- 옵션 총 가치 (출고가-기본가): {option_value:,.0f}만원\n"

    # 내수/수출 구분
    target_info += f"- 내수/수출: {'내수' if domestic else '수출'}\n"

    # 대상차량은 AA등급(무사고) 가정
    target_info += "- 검차상태: AA등급 (무사고 가정)\n"

    # 데이터 범위 경고 (고주행 차량)
    target_mileage = target.get("mileage", 0) or 0
    data_gap_warning = ""
    if auction_vehicles or retail_vehicles:
        all_mileages = []
        for v in auction_vehicles:
            m = v.get("주행거리", 0)
            if m:
                all_mileages.append(m)
        for v in retail_vehicles:
            m = v.get("주행거리", 0)
            if m:
                all_mileages.append(m)
        if all_mileages:
            max_data_km = max(all_mileages)
            gap_km = target_mileage - max_data_km
            if gap_km > 50000:
                data_gap_warning = (
                    f"\n\n⚠️ **주의: 대상차량 주행거리({target_mileage:,}km)가 "
                    f"비교 데이터 범위(최대 {max_data_km:,}km)를 {gap_km:,}km 초과합니다.**\n"
                    f"주행거리 차이만큼 추가 감가를 반드시 적용하세요. "
                    f"고주행(20만km+) 차량은 시장 수요가 급감하여 "
                    f"km당 감가율이 일반 구간보다 1.5~2배 가속됩니다.\n"
                )

    return (
        f"{target_info}\n"
        f"## 시세 통계 (최근 3개월)\n"
        f"{_format_stats(auction_stats, '낙찰가')}\n"
        f"{_format_stats(retail_stats, '소매가')}\n\n"
        f"## 낙찰가 유사차량 ({len(auction_vehicles)}건)\n"
        f"{_format_auction_table(auction_vehicles)}\n\n"
        f"## 소매가 유사차량 ({len(retail_vehicles)}건)\n"
        f"{_format_retail_table(retail_vehicles)}\n\n"
        f"{data_gap_warning}"
        f"위 데이터를 분석하여 대상차량의 적정 낙찰가와 소매가를 추론해주세요.\n"
        f"출고가/기본가 정보를 반드시 반영하세요. 대상차량은 AA등급(무사고)으로 간주하므로 사고이력 감가는 적용하지 마세요.\n"
        f"**소매가는 반드시 소매가 유사차량의 실제 매물가를 기반으로 산정하세요.**\n"
        f"estimated_auction은 내수(수출=빈칸) 차량 데이터로, estimated_auction_export는 수출(수출=Y) 차량 데이터로 각각 산출하세요."
    )


# =========================================================================
# 색상 보정
# =========================================================================

def _calc_color_adjustment(target: dict, ref_vehicles: list[dict]) -> tuple[float, str]:
    """
    대상차량 색상에 따른 가격 보정 (기준: 흰색 = 0).

    Returns:
        (adjustment_amount_만원, description_text)
    """
    import yaml
    from pathlib import Path
    from app.services.rule_engine import normalize_color

    target_color = target.get("color", "")
    color_group = normalize_color(target_color)

    # 색상 보정 테이블 로드
    rules_path = Path(__file__).parent.parent.parent / "rules" / "pricing_rules.yaml"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
        color_rules = rules.get("color_adjustment", {})
    except Exception as e:
        logger.warning("색상 보정 룰 로드 실패: %s", e)
        return 0, ""

    # 세그먼트 추정 (기준차량에서 가장 많은 segment 사용)
    segments = [v.get("segment", "") for v in ref_vehicles if v.get("segment")]
    if segments:
        from collections import Counter
        segment = Counter(segments).most_common(1)[0][0]
    else:
        segment = ""

    if '대형' in segment or '프리미엄' in segment:
        table_key = 'large'
    elif 'SUV' in segment.upper():
        table_key = 'suv'
    elif '경차' in segment:
        table_key = 'compact'
    else:
        table_key = 'medium'

    table = color_rules.get(table_key, color_rules.get('medium', {}))
    target_adj = table.get(color_group, table.get('other', 0))

    # 연식 가중치
    year = int(target.get("year", 0) or 0)
    age = 2026 - year if year > 0 else 5
    weight = 0.3  # default
    for w in color_rules.get('age_weight', []):
        if age <= w['max_age']:
            weight = w['weight']
            break

    amount = round(target_adj * weight, 1)

    if amount == 0:
        desc = f"대상 색상: {target_color} ({color_group}) — 선호 색상, 보정 없음"
    else:
        desc = (
            f"대상 색상: {target_color} ({color_group})\n"
            f"차급: {table_key} (세그먼트: {segment or '일반'})\n"
            f"색상 보정: {target_adj:+.0f}만원 × 연식가중치 {weight} = {amount:+.1f}만원"
        )

    return amount, desc


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


def _get_domestic_ratio_at_mileage(
    mileage: int,
    brackets: list,
) -> float:
    """내수 bracket에서 해당 주행거리의 effective_ratio를 보간하여 반환."""
    if not brackets:
        return 0.0
    bucket_km = (mileage // 10000) * 10000
    # 정확히 일치하는 bracket 찾기
    for b in brackets:
        if b.bracket_start <= bucket_km < b.bracket_end:
            return b.effective_ratio
    # 범위 밖이면 가장 가까운 bracket
    if bucket_km < brackets[0].bracket_start:
        return brackets[0].effective_ratio
    return brackets[-1].effective_ratio


def _estimate_export_from_domestic(
    export_vehicles: list[dict],
    domestic_brackets: list,
    target_mileage: int,
    export_details: str,
    domestic_estimate: float,
) -> tuple[float, str]:
    """
    수출 데이터 부족 시, 내수 bracket 비율로 주행거리 보정하여 수출가 추정.

    각 수출 차량의 가격을 내수 bracket 비율로 대상 주행거리에 맞게 환산한 뒤
    평균을 반환.
    """
    target_domestic_ratio = _get_domestic_ratio_at_mileage(
        target_mileage, domestic_brackets,
    )
    if target_domestic_ratio <= 0:
        return 0, ""

    estimates = []
    detail_lines = []
    for ev in export_vehicles:
        ev_mileage = int(ev.get("주행거리", 0) or 0)
        ev_price = _to_man_won(float(ev.get("낙찰가", 0) or 0))
        if ev_price <= 0 or ev_mileage <= 0:
            continue
        ev_domestic_ratio = _get_domestic_ratio_at_mileage(
            ev_mileage, domestic_brackets,
        )
        if ev_domestic_ratio <= 0:
            continue
        # 주행거리 보정: 수출가 × (대상 내수비율 / 수출차량 내수비율)
        scale = target_domestic_ratio / ev_domestic_ratio
        adjusted = ev_price * scale
        estimates.append(adjusted)
        detail_lines.append(
            f"  수출차량 {ev_mileage:,}km {ev_price:,.0f}만 "
            f"× 내수비율({target_domestic_ratio*100:.1f}%/{ev_domestic_ratio*100:.1f}%) "
            f"= {adjusted:,.0f}만"
        )

    if not estimates:
        return 0, ""

    avg_price = round(sum(estimates) / len(estimates))
    reasoning = (
        f"[수출 데이터 부족 — 내수 비율 보정] {export_details}\n"
        f"내수 주행거리별 감가 비율을 적용하여 대상 {target_mileage:,}km 기준으로 환산:\n"
        + "\n".join(detail_lines)
        + f"\n= 추정 수출 낙찰가: {avg_price:,}만원"
    )
    return avg_price, reasoning


def _safe_thumbnail(url: str | None) -> str:
    """CORS 차단되는 외부 도메인 썸네일 제거 (encar 등)"""
    if not url:
        return ""
    blocked = ("ci.encar.com", "encar.com", "img.encar.com")
    if any(d in url for d in blocked):
        return ""
    return url


def _compact_auction_vehicle(v: dict) -> dict:
    """낙찰 유사차량 → compact dict (Firestore 용량 절약)"""
    sale_date = v.get("개최일", "")
    if sale_date and len(str(sale_date)) > 7:
        sale_date = str(sale_date)[:10]
    return {
        "id": v.get("auction_id", ""),
        "nm": v.get("차명", "") or v.get("vehicle_name", ""),
        "yr": v.get("연식", 0) or 0,
        "ml": v.get("주행거리", 0) or 0,
        "pr": round(_to_man_won(v.get("낙찰가", 0))),
        "dt": sale_date,
        "cl": v.get("색상", ""),
        "tr": v.get("trim", ""),
        "fl": v.get("연료", ""),
        "fp": round(_to_man_won(v.get("factory_price", 0))) if v.get("factory_price") else 0,
        "bp": round(_to_man_won(v.get("base_price", 0))) if v.get("base_price") else 0,
        "ex": 1 if v.get("is_export") else 0,
        "fe": v.get("frame_exchange", 0) or 0,
        "fb": v.get("frame_bodywork", 0) or 0,
        "ee": v.get("exterior_exchange", 0) or 0,
        "eb": v.get("exterior_bodywork", 0) or 0,
        "dg": 1 if v.get("has_diagnosis") else 0,
        "th": _safe_thumbnail(v.get("thumbnail")),
    }


def _compact_retail_vehicle(v: dict) -> dict:
    """소매 유사차량 → compact dict"""
    sale_date = v.get("개최일", "") or v.get("매물등록일", "") or v.get("listing_date", "")
    if sale_date and len(str(sale_date)) > 7:
        sale_date = str(sale_date)[:10]
    return {
        "id": v.get("auction_id", ""),
        "nm": v.get("차명", "") or v.get("vehicle_name", ""),
        "yr": v.get("연식", 0) or 0,
        "ml": v.get("주행거리", 0) or 0,
        "pr": round(v.get("소매가", 0) or 0),
        "dt": sale_date,
        "cl": v.get("색상", ""),
        "tr": v.get("trim", ""),
        "fl": v.get("연료", ""),
        "fp": round(_to_man_won(v.get("factory_price", 0))) if v.get("factory_price") else 0,
        "bp": round(_to_man_won(v.get("base_price", 0))) if v.get("base_price") else 0,
        "fe": v.get("frame_exchange", 0) or 0,
        "fb": v.get("frame_bodywork", 0) or 0,
        "ee": v.get("exterior_exchange", 0) or 0,
        "eb": v.get("exterior_bodywork", 0) or 0,
        "st": v.get("status", ""),
        "dg": 1 if v.get("has_diagnosis") else 0,
        "th": _safe_thumbnail(v.get("thumbnail")),
    }


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
    auction_vehicles, retail_vehicles, auction_stats_raw, retail_stats_raw = (
        _fetch_comparable_vehicles(target)
    )

    # 통계 단위 정규화 (원 → 만원)
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
        temperature=0,
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

    # 하위 호환: 기존 reasoning/factors도 유지
    auction_reasoning = parsed.get("auction_reasoning", "")
    retail_reasoning = parsed.get("retail_reasoning", "")
    auction_factors = parsed.get("auction_factors", [])
    retail_factors = parsed.get("retail_factors", [])
    legacy_reasoning = parsed.get("reasoning", "")
    legacy_factors = parsed.get("factors", [])

    # 최근 수출 낙찰일 계산 (LLM 아닌 데이터 직접 계산)
    export_dates = [
        v.get("개최일", "") for v in auction_vehicles if v.get("is_export")
    ]
    last_export_date = max(export_dates) if export_dates else ""

    export_reasoning = parsed.get("export_reasoning", "")

    # ── 낙찰가/소매가: 시장 데이터 단독 ──
    from app.services.retail_estimator import (
        estimate_retail_by_market, estimate_auction_by_market,
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
        final_auction = parsed.get("estimated_auction", 0)  # 시장 데이터 실패 시 LLM 폴백
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
        final_retail = parsed.get("estimated_retail", 0)  # 시장 데이터 실패 시 LLM 폴백
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
        # 수출 데이터 부족 시: 내수 bracket 비율로 주행거리 보정 시도
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
            # 내수 보정도 불가 → LLM 폴백
            export_price = parsed.get("estimated_auction_export", 0) or 0
            export_reasoning_final = (
                f"[수출 시장 데이터 부족 — LLM 폴백] {market_export_result.details}\n"
                f"{export_reasoning}"
            )

    # ── 캘리브레이션 보정 reasoning ──
    for _result, _label in [
        (market_auction_result, "auction"),
        (market_retail_result, "retail"),
        (market_export_result, "export"),
    ]:
        _params = _result.blended_params
        if _params and (_result.learned_correction_applied or _result.feedback_excluded > 0):
            _desc = "\n\n── 캘리브레이션 보정 ──\n"
            if _params.auto_cal_count > 0:
                _desc += f"자동 보정 (LOO {_params.auto_cal_count}건)\n"
            if _params.manual_feedback_count > 0:
                _desc += f"수동 피드백 {_params.manual_feedback_count}건\n"
            if _result.feedback_excluded > 0:
                _desc += f"이상치 {_result.feedback_excluded}건 제외\n"
            if _params.scale_factor != 1.0:
                _desc += f"스케일 보정: ×{_params.scale_factor:.3f}\n"
            if _params.price_bias != 0:
                _desc += f"바이어스 보정: {_params.price_bias:+.1f}만원\n"
            if _params.mileage_slope != 0:
                _desc += f"주행거리 보정: {_params.mileage_slope:+.4f}/만km\n"
            _desc += f"(신뢰도 {_params.confidence:.0%})"
            if _label == "auction":
                auction_reasoning_final += _desc
            elif _label == "retail":
                retail_reasoning_final += _desc
            else:
                export_reasoning_final += _desc

    # ── 낙찰가 < 소매가 정합성 체크 ──
    if final_auction > 0 and final_retail > 0 and final_auction >= final_retail * 0.97:
        capped = round(final_retail * 0.90, 1)
        logger.warning(
            "낙찰가(%.0f) ≥ 소매가(%.0f)×0.97 — 소매가×0.90=%.0f으로 캡",
            final_auction, final_retail, capped,
        )
        auction_reasoning_final += (
            f"\n\n── 정합성 보정 ──\n"
            f"낙찰가({final_auction:.0f}만) ≥ 소매가({final_retail:.0f}만)의 97%: "
            f"소매가×90%={capped:.0f}만으로 캡"
        )
        final_auction = capped

    # ── bracket 직렬화 (IQR 이상치 제거 + 스무딩된 effective_ratio) ──
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

    # ── 유사차량 compact 직렬화 (시장 추정에 사용된 전체 차량) ──
    # 낙찰: 내수 + 수출 통합
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
    result = PricePrediction(
        estimated_auction=final_auction,
        estimated_auction_export=export_price,
        last_export_date=last_export_date,
        estimated_retail=final_retail,
        confidence=parsed.get("confidence", "보통"),
        reasoning=legacy_reasoning or auction_reasoning,
        factors=legacy_factors or auction_factors,
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

    return result
