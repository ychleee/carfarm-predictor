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

import asyncio

import anthropic

from app.services.firestore_db import (
    search_auction_db,
    search_retail_db,
    get_price_stats,
)
from app.services.encar_api import enrich_with_details

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
    input_tokens: int = 0
    output_tokens: int = 0


# =========================================================================
# 시스템 프롬프트 — 도메인 지식 포함
# =========================================================================

SYSTEM_PROMPT = """당신은 한국 중고차 시장의 전문 프라이싱 분석가입니다.
제공된 유사 차량 데이터를 분석하여 대상차량의 적정 가격을 추론하는 것이 역할입니다.

## 분석 방법

1. **최근 데이터 우선**: 유사차량 테이블의 판매일/등록일을 반드시 확인하세요. 최근 1~2개월 내 데이터를 가장 중요한 기준으로 삼고, 3개월 이상 지난 데이터는 보조 참고 자료로만 활용하세요. 시세는 시간에 따라 변동하므로 오래된 데이터로 가격을 산정하면 현재 시세와 괴리가 생깁니다.
2. **출고가/기본가 기반 분석**: 대상차량의 출고가(또는 기본가)를 기준점으로 잡고, 옵션 차이를 보정하세요.
   - 출고가 - 기본가 = 옵션 총 가치. 유사차량과의 옵션 차이를 이 기준으로 보정합니다.
   - 출고가가 없으면 유사차량의 출고가/기본가 데이터에서 유추하세요.
2. **유사차량 데이터 분석**: 제공된 낙찰가/소매가 유사차량 테이블을 꼼꼼히 분석하세요.
3. **시세 통계 참조**: 평균, 중앙값, 최소, 최대 통계를 기준점으로 활용하세요.
4. **차이 요인 반영**: 대상차량과 유사차량 간 차이(주행거리, 색상, 옵션, 사고이력, 검차상태 등)를 가격에 반영하세요.
5. **수출/내수 구분 주의**: 수출 차량은 내수 대비 가격이 크게 다를 수 있습니다. 반드시 대상차량의 내수/수출 여부를 확인하고, 비교 대상도 같은 유형으로 비교하세요.

## 업계 가격 보정 기준 (실측 데이터 반영)

### 주행거리 감가 (소매가 기준 %/만km)
- 2~3년차: 1.4% / 만km
- 4~6년차: 1.0% / 만km
- 7~9년차: 0.7% / 만km
- 10년 이상: 2,000만원 이상 차량은 0.7%/만km, 미만은 7만원/만km 정액
- 20만km 초과: 증감 미적용 (천장 효과)

### 교환/판금 (사고이력 — 출고가 기준)
- 외판 교환(X): 부위당 2% 감가 (업계 3% → 실측 1.7% 반영)
- 판금(W): 연식/가격대에 따라 차등
  - 3년 미만 또는 2,500만원 이상: 부위당 1.4%
  - 3~7년 또는 1,500~2,500만원: 부위당 1.0%
  - 500만원 미만: 미적용
  - 기타: 부위당 0.7%
- 부위별 가중치:
  - 주요 외판(후드, 루프, 트렁크, 펜더, 도어, 쿼터패널): 1.5배
  - 소형 부품(범퍼, 램프, 미러, 유리, 휠): 0.3배
  - 기타: 1.0배

### 골격사고 (교환/판금과 별도로 추가 적용)
- 골격부위: 프론트패널, 크로스멤버, 플로어패널, 사이드멤버, 리어패널, 트렁크플로어
- 필러(A/B/C): 교환 시 골격사고로 분류
- C등급(1부위): 15% 감가
- D등급(2부위): 17% 감가
- E등급(3부위): 18% 감가
- F등급(5부위+): 20% 감가

### 검차상태 반영
유사차량 테이블의 검차 정보(프레임교환/판금/부식, 외판교환/판금/부식)를 꼭 확인하세요.
- 대상차량의 검차상태가 양호하면 유사차량 중 사고이력이 없는 차량과 비교
- 대상차량의 검차상태가 나쁘면 해당 감가를 반영

### 엔카진단 데이터 활용 (★ 소매가 산정의 핵심 ★)
- 소매가 테이블에 "엔카진단" 열이 있습니다. "Y"이면 엔카 공식 진단을 받은 차량입니다.
- 엔카진단 차량은 검증된 상태 정보가 있으므로 가격 신뢰도가 가장 높습니다.
- **소매가 산정 시 엔카진단 차량의 실제 매물 가격을 최우선 기준으로 사용하세요.**
- 엔카진단 + 무사고 차량의 가격대가 가장 신뢰도 높은 소매가 기준점입니다.
- "사고이력" 열도 확인하세요: "무사고"는 프리미엄, 사고이력이 있으면 그만큼 감가합니다.
- 엔카진단이 아닌 차량은 보조 참고로만 활용하세요.

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
- 수출 낙찰가는 일반적으로 내수 낙찰가보다 높습니다 (해외 수요로 인한 프리미엄).
- estimated_auction은 반드시 내수(수출=빈칸) 유사차량만으로 산정하세요
- estimated_auction_export는 반드시 수출(수출=Y) 유사차량만으로 산정하세요
- **수출가 산정 시에도 내수와 동일하게 주행거리/연식/트림/색상/사고이력 차이를 반드시 보정하세요.** 수출 유사차량의 낙찰가를 단순 평균하지 말고, 대상차량과의 조건 차이를 감안하여 보정된 가격을 산출하세요.
- 테이블에 수출=Y인 차량이 1건이라도 있으면 estimated_auction_export를 반드시 산출하세요 (0으로 두지 마세요)
- 수출 데이터가 없으면 0으로 두세요
- 테이블의 "수출" 열을 반드시 확인하세요
- 수출가가 내수보다 낮게 나올 경우, export_reasoning에 왜 낮은지 반드시 설명하세요 (예: 수출 유사차량이 고주행/구형 위주라 데이터 한계)

### 소매가 산정 방법 (★ 실제 엔카 매물 기반 ★)
- **소매가는 반드시 소매가 유사차량 테이블의 실제 엔카 매물 가격을 기반으로 산정하세요.**
- 엔카진단(Y) + 무사고 + 유사 조건(연식/주행/트림) 차량의 실제 매물가를 핵심 기준점으로 사용합니다.
- 대상차량과의 주행거리/연식/옵션/사고이력 차이를 반영하여 보정하세요.
- 소매가 유사차량 데이터가 충분하면 낙찰가와의 비율 공식은 참고하지 마세요.
- 소매가 데이터가 부족할 때만 보조적으로 아래 공식 참고:
  - 1,500만원 초과: 소매가 ≈ 낙찰가 / 0.90 (국산), 0.88 (수입)
  - 1,500만원 이하: 비율 공식이 맞지 않으므로 사용 금지. 실제 엔카 매물가 또는 낙찰가 + 고정 마진(150~200만원)으로 산정
- **★ 필수: 낙찰가 1,500만원 이하인 경우, 소매가는 낙찰가보다 최소 150만원 이상 높아야 합니다.** 이는 딜러 마진·정비·이전비 등 실비를 반영한 업계 최소 기준입니다. 소매가 산정 결과가 낙찰가+150만원 미만이면 반드시 상향 조정하세요.

## 출력 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
낙찰가와 소매가 각각에 대해 별도의 요인 분석과 근거를 제공하세요.

```json
{
  "estimated_auction": 1500,
  "estimated_auction_export": 1200,
  "estimated_retail": 1700,
  "confidence": "높음",
  "auction_reasoning": "내수 낙찰가 분석 근거를 3~5문장으로 설명",
  "retail_reasoning": "소매가 분석 근거를 3~5문장으로 설명",
  "export_reasoning": "수출 낙찰가 분석 근거. 특히 내수보다 낮은 경우 왜 낮은지 반드시 설명",
  "auction_factors": [
    {"factor": "주행거리", "impact": -50, "description": "낙찰 평균 대비 2만km 초과로 약 50만원 감가"},
    {"factor": "검차상태", "impact": -30, "description": "외판 교환 1부위로 약 30만원 감가"}
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
- export_reasoning: 수출 낙찰가 산정 근거 (내수보다 낮을 경우 이유 필수 설명)
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

    # 1) 낙찰가 데이터 (내수+수출 모두 수집 → LLM이 분리 분석)
    auction_raw = search_auction_db(
        model=model, maker=maker, fuel=fuel, trim=trim,
        year_min=year_min, year_max=year_max,
        limit=200, sort_by="날짜",
        domestic_only=False,
    )

    # 트림 매칭이 부족하면 완화 재검색
    if len(auction_raw) < 10 and trim:
        auction_raw_relaxed = search_auction_db(
            model=model, maker=maker, fuel=fuel, trim=None,
            year_min=year_min, year_max=year_max,
            limit=200, sort_by="날짜",
            domestic_only=False,
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

    retail_top = retail_raw[:20]

    # 3.5) 엔카 소매 데이터: API 보강 (진단·사고이력·옵션·기본가)
    if retail_top:
        _enrich_retail_encar(retail_top)

    # 3.6) 엔카진단 차량 우선 정렬 — LLM이 진단 차량을 먼저 참고하도록
    retail_top.sort(key=lambda v: (
        0 if v.get("has_diagnosis") else 1,                    # 엔카진단 우선
        0 if v.get("accident_summary") == "무사고" else 1,     # 무사고 우선
        -v.get("_score", 0),                                   # 유사도 점수 내림차순
    ))

    # 4) 시세 통계 (낙찰/소매 분리, 동일 연식 기준)
    auction_stats = get_price_stats(maker, model, year=year, price_type="auction")
    retail_stats_raw = get_price_stats(maker, model, year=year, price_type="retail")

    return auction_top, retail_top, auction_stats, retail_stats_raw


def _enrich_retail_encar(vehicles: list[dict]) -> None:
    """소매(엔카) 차량에 엔카 API 보강 — 진단·사고이력·옵션·기본가."""
    # auction_id → encar ID 변환 (encar_ 접두사 제거)
    encar_vehicles = []
    original_ids = {}
    for v in vehicles:
        aid = v.get("auction_id", "")
        if aid.startswith("encar_"):
            stripped = aid[len("encar_"):]
            original_ids[stripped] = aid
            v["auction_id"] = stripped
            encar_vehicles.append(v)
        elif aid.isdigit():
            encar_vehicles.append(v)

    if not encar_vehicles:
        return

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(enrich_with_details(encar_vehicles, max_concurrent=5))
        loop.close()
    except Exception as e:
        logger.warning("엔카 API 보강 실패 (무시): %s", e)

    # auction_id 복원
    for v in vehicles:
        aid = v.get("auction_id", "")
        if aid in original_ids:
            v["auction_id"] = original_ids[aid]


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
            sale_date = str(sale_date)[:7]  # YYYY-MM

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

    lines = ["ID | 연식 | 주행(km) | 소매가(만) | 트림 | 색상 | 연료 | 옵션수 | 프레임교환 | 프레임판금 | 외판교환 | 외판판금 | 사고이력 | 엔카진단 | 출고가(만) | 기본가(만) | 등록일"]
    for v in vehicles:
        options_str = v.get("옵션", "") or v.get("options", "")
        n_opts = len(options_str.split(",")) if options_str and options_str.strip() else 0

        factory_price = _to_man_won(v.get('factory_price', 0)) if v.get('factory_price') else ''
        base_price = _to_man_won(v.get('base_price', 0)) if v.get('base_price') else ''

        accident = v.get("accident_summary", "")
        diagnosis = "Y" if v.get("has_diagnosis") else ""

        listing_date = v.get("매물등록일", "") or v.get("listing_date", "")
        if listing_date and len(str(listing_date)) > 7:
            listing_date = str(listing_date)[:7]

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
            f"{diagnosis} | "
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

    # 교환/판금
    target_info += (
        f"- 교환 부위 수: {target.get('exchange_count', 0)}\n"
        f"- 판금 부위 수: {target.get('bodywork_count', 0)}\n"
    )

    # 파트 손상 정보
    part_damages = target.get("part_damages", [])
    if part_damages:
        damage_strs = [f"  - {pd.get('part', '')}: {pd.get('damage_type', '')}" for pd in part_damages]
        target_info += f"- 부위별 손상:\n" + "\n".join(damage_strs) + "\n"

    # 엔카진단 통계 요약
    diag_vehicles = [v for v in retail_vehicles if v.get("has_diagnosis")]
    diag_noaccident = [v for v in diag_vehicles if v.get("accident_summary") == "무사고"]
    diag_summary = ""
    if diag_vehicles:
        diag_prices = [v.get("소매가", 0) for v in diag_vehicles if v.get("소매가", 0) > 0]
        diag_na_prices = [v.get("소매가", 0) for v in diag_noaccident if v.get("소매가", 0) > 0]
        diag_summary = f"\n### 엔카진단 차량 요약\n"
        diag_summary += f"- 엔카진단 차량: {len(diag_vehicles)}건"
        if diag_prices:
            diag_summary += f" (가격 범위: {min(diag_prices):,.0f}~{max(diag_prices):,.0f}만원, 평균: {sum(diag_prices)/len(diag_prices):,.0f}만원)"
        diag_summary += f"\n- 엔카진단+무사고: {len(diag_noaccident)}건"
        if diag_na_prices:
            diag_summary += f" (가격 범위: {min(diag_na_prices):,.0f}~{max(diag_na_prices):,.0f}만원, 평균: {sum(diag_na_prices)/len(diag_na_prices):,.0f}만원)"
        diag_summary += f"\n- ★ 소매가 산정 시 위 엔카진단 차량 가격을 핵심 기준으로 사용하세요.\n"

    return (
        f"{target_info}\n"
        f"## 시세 통계 (최근 3개월)\n"
        f"{_format_stats(auction_stats, '낙찰가')}\n"
        f"{_format_stats(retail_stats, '소매가')}\n\n"
        f"## 낙찰가 유사차량 ({len(auction_vehicles)}건)\n"
        f"{_format_auction_table(auction_vehicles)}\n\n"
        f"## 소매가 유사차량 ({len(retail_vehicles)}건)\n"
        f"{_format_retail_table(retail_vehicles)}\n"
        f"{diag_summary}\n"
        f"위 데이터를 분석하여 대상차량의 적정 낙찰가와 소매가를 추론해주세요.\n"
        f"출고가/기본가 정보와 검차상태(교환/판금/골격)를 반드시 반영하세요.\n"
        f"**소매가는 반드시 소매가 유사차량(특히 엔카진단 차량)의 실제 매물가를 기반으로 산정하세요. 낙찰가÷비율 공식이 아닌 실제 엔카 매물 데이터를 기준으로 하세요.**\n"
        f"estimated_auction은 내수(수출=빈칸) 차량 데이터로, estimated_auction_export는 수출(수출=Y) 차량 데이터로 각각 산출하세요."
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

    result = PricePrediction(
        estimated_auction=parsed.get("estimated_auction", 0),
        estimated_auction_export=parsed.get("estimated_auction_export", 0),
        last_export_date=last_export_date,
        estimated_retail=parsed.get("estimated_retail", 0),
        confidence=parsed.get("confidence", "보통"),
        reasoning=legacy_reasoning or auction_reasoning,
        factors=legacy_factors or auction_factors,
        auction_reasoning=auction_reasoning or legacy_reasoning,
        retail_reasoning=retail_reasoning or legacy_reasoning,
        export_reasoning=export_reasoning,
        auction_factors=auction_factors or legacy_factors,
        retail_factors=retail_factors or legacy_factors,
        comparable_summary=parsed.get("comparable_summary", ""),
        key_comparables=parsed.get("key_comparables", []),
        vehicles_analyzed=total_vehicles,
        auction_stats=auction_stats,
        retail_stats=retail_stats,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return result
