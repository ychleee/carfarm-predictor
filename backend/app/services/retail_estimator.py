"""
시장 데이터 기반 소매가 추정 엔진

알고리즘:
  1. 같은 트림 + 같은 연식(부족하면 ±1년)의 소매 차량 검색
  2. 10,000km 구간별 소매가/출고가(기본가) 비율 산출
  3. 구간 내 비율 분산이 크고 소매가 분산이 작으면 → 소매가÷대상출고가로 보정 비율 산출
  4. 비율 변화 추이로 대상 주행거리의 감가 비율 보간/외삽
  5. 대상 출고가(기본가) × 비율 = 추정 소매가

낙찰가 대비 소매가(할인율) 방식은 사용하지 않음.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =========================================================================
# 데이터 모델
# =========================================================================

@dataclass
class MileageBracket:
    """10,000km 구간 데이터"""
    bracket_start: int = 0        # 구간 시작 km (0, 10000, 20000, ...)
    bracket_end: int = 0          # 구간 끝 km
    ratios: list[float] = field(default_factory=list)   # 소매가/출고가 비율 (개별)
    prices: list[float] = field(default_factory=list)   # 소매가 (만원)
    mileages: list[int] = field(default_factory=list)   # 개별 주행거리
    effective_ratio: float = 0.0  # 최종 사용 비율 (보정 포함)
    median_ratio: float = 0.0
    median_price: float = 0.0
    ratio_cv: float = 0.0        # 비율 변동계수 (CV)
    price_cv: float = 0.0        # 소매가 변동계수 (CV)
    smoothed: bool = False        # 소매가 기준 보정 여부
    count: int = 0


@dataclass
class RetailEstimateResult:
    """소매가 추정 결과"""
    estimated_retail: float = 0.0
    estimated_ratio: float = 0.0       # 추정 비율 (%, 예: 52.3)
    reference_price_used: float = 0.0  # 사용한 출고가/기본가 (만원)
    reference_price_label: str = ""    # "출고가" or "기본가"
    method: str = ""                   # ratio_direct / ratio_interpolation / ratio_extrapolation
    brackets: list[MileageBracket] = field(default_factory=list)
    vehicles_found: int = 0
    year_range_used: str = ""
    details: str = ""
    confidence: str = "보통"
    quality_filter: str = ""
    success: bool = False


# =========================================================================
# 설정값
# =========================================================================

RATIO_CV_THRESHOLD = 0.15     # 비율 CV > 15% → 비율 변동 큼
PRICE_CV_THRESHOLD = 0.10     # 소매가 CV < 10% → 소매가 안정적
MIN_VEHICLES_PER_BRACKET = 2  # 구간당 최소 차량 수
MIN_TOTAL_VEHICLES = 3        # 전체 최소 차량 수 (이 이하면 추정 불가)
MIN_VEHICLES_DESIRED = 6      # 소매 목표 차량 수 (이하이면 연식 확대)
MIN_AUCTION_VEHICLES_DESIRED = 6   # 낙찰 목표 차량 수 (이하이면 연식 확대)


# =========================================================================
# 엔카 API 보강 + 품질 필터
# =========================================================================

def _extract_encar_id(vehicle: dict) -> str | None:
    """차량 dict에서 엔카 car ID 추출 (auction_id 또는 source_url)"""
    aid = vehicle.get("auction_id", "")
    if isinstance(aid, str) and aid.startswith("encar_"):
        return aid[len("encar_"):]
    if str(aid).isdigit() and len(str(aid)) >= 6:
        return str(aid)
    # source_url에서 추출 시도
    source_url = vehicle.get("source_url", "")
    if "encar.com" in source_url:
        import re
        m = re.search(r'/(\d{6,})', source_url)
        if m:
            return m.group(1)
    return None


def _enrich_vehicles(vehicles: list[dict]) -> int:
    """
    엔카 API로 진단·사고이력 정보 보강 (in-place).
    Returns: 보강 성공 차량 수.
    """
    from app.services.encar_api import enrich_with_details

    encar_list: list[dict] = []
    orig_aids: dict[int, str] = {}  # encar_list index → original auction_id

    for v in vehicles:
        eid = _extract_encar_id(v)
        if eid:
            orig_aids[len(encar_list)] = v.get("auction_id", "")
            v["auction_id"] = eid
            encar_list.append(v)

    if not encar_list:
        logger.info("엔카 ID 식별 불가 — API 보강 스킵 (%d건)", len(vehicles))
        return 0

    logger.info("엔카 API 보강: %d/%d건", len(encar_list), len(vehicles))
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(enrich_with_details(encar_list, max_concurrent=10))
        loop.close()
    except Exception as e:
        logger.warning("엔카 API 보강 실패: %s", e)

    # auction_id 복원
    for idx, orig in orig_aids.items():
        encar_list[idx]["auction_id"] = orig

    return sum(1 for v in encar_list if "has_diagnosis" in v)


def _filter_quality_vehicles(vehicles: list[dict]) -> tuple[list[dict], str]:
    """
    품질 기준으로 차량 필터링 (단계적 완화).

    1. 엔카진단 + 무사고
    2. 무사고 (진단 무관)
    3. 전체 사용
    """
    enriched = [v for v in vehicles if "has_diagnosis" in v]

    if not enriched:
        return vehicles, f"전체 {len(vehicles)}건 (API 보강 불가)"

    # 1순위: 엔카진단 + 무사고
    diagnosed_clean = [
        v for v in enriched
        if v.get("has_diagnosis") and v.get("accident_summary") == "무사고"
    ]
    if len(diagnosed_clean) >= MIN_TOTAL_VEHICLES:
        return diagnosed_clean, f"엔카진단+무사고 {len(diagnosed_clean)}건"

    # 2순위: 무사고 (보강된 차량 중)
    no_accident = [v for v in enriched if v.get("accident_summary") == "무사고"]
    if len(no_accident) >= MIN_TOTAL_VEHICLES:
        diag = sum(1 for v in no_accident if v.get("has_diagnosis"))
        return no_accident, f"무사고 {len(no_accident)}건 (엔카진단 {diag}건)"

    # 3순위: 보강된 차량 전체
    if len(enriched) >= MIN_TOTAL_VEHICLES:
        return enriched, f"보강차량 {len(enriched)}건 (진단+무사고 부족)"

    # 4순위: 전체
    return vehicles, f"전체 {len(vehicles)}건 (품질 필터 차량 부족)"


# =========================================================================
# 핵심 로직
# =========================================================================

def _to_man_won(value) -> float:
    """원 → 만원 변환 (이미 만원 단위면 그대로)"""
    v = float(value or 0)
    if v > 100000:  # 10만원 이상이면 원 단위로 판단
        return round(v / 10000, 1)
    return v


# ── 기준차량 AA등급 정규화 ──
_STRUCTURAL_GRADE_PCT = {1: 0.15, 2: 0.17, 3: 0.18, 4: 0.19}
_STRUCTURAL_DEFAULT_PCT = 0.20


def _calc_aa_adjustment(vehicle: dict) -> float:
    """
    기준차량(유사차량)의 사고이력 → AA등급 보정 비율 (출고가 대비 소수).

    사고 차량의 가격을 무사고 수준으로 상향 정규화할 때 사용.
    예: return 0.04 → 가격에 × (1 + 0.04) 적용.
    """
    frame_exchange = int(vehicle.get("frame_exchange", 0) or 0)
    exterior_exchange = int(vehicle.get("exterior_exchange", 0) or 0)
    frame_bodywork = int(vehicle.get("frame_bodywork", 0) or 0)
    exterior_bodywork = int(vehicle.get("exterior_bodywork", 0) or 0)

    raw = 0.0

    # 골격사고 (프레임 교환)
    if frame_exchange > 0:
        raw += _STRUCTURAL_GRADE_PCT.get(frame_exchange, _STRUCTURAL_DEFAULT_PCT)

    # 외판 교환: 부위당 2%
    raw += exterior_exchange * 0.02

    # 판금: 부위당 1%
    raw += (frame_bodywork + exterior_bodywork) * 0.01

    if raw == 0:
        return 0.0

    # 연식 가중치 (오래될수록 사고 영향 감소)
    v_year = int(vehicle.get("연식", 0) or 0)
    if v_year > 0:
        age = datetime.now().year - v_year
        if age <= 3:
            weight = 1.0
        elif age <= 6:
            weight = 0.5
        elif age <= 9:
            weight = 0.3
        else:
            weight = 0.15
        raw *= weight

    return raw


def _resolve_target_ref_price(
    maker: str, model: str, trim: str,
    vehicles: list[dict],
) -> tuple[float, str]:
    """
    대상차량 출고가/기본가를 확보하는 폴백 로직.

    1순위: DB에서 같은 maker+model 출고가 조회
    2순위: 같은 트림 기준차량 출고가 중앙값
    """
    # 1순위: DB 조회
    try:
        from app.services.firestore_db import get_firestore_db, _lookup_factory_price
        db = get_firestore_db()
        found = _lookup_factory_price(db, maker, model)
        if found:
            fp = found.get("factory_price", 0) or 0
            bp = found.get("base_price", 0) or 0
            if fp > 0:
                logger.info("대상 출고가 DB 조회: %s %s → %.0f만원", maker, model, fp)
                return fp, "출고가"
            if bp > 0:
                logger.info("대상 기본가 DB 조회: %s %s → %.0f만원", maker, model, bp)
                return bp, "기본가"
    except Exception as e:
        logger.warning("출고가 DB 조회 실패: %s", e)

    # 2순위: 기준차량 중앙값
    ref_prices = [_pick_ref_price(v)[0] for v in vehicles if _pick_ref_price(v)[0] > 0]
    if ref_prices:
        median_price = statistics.median(ref_prices)
        logger.info("대상 출고가 기준차량 중앙값: %.0f만원 (%d건)", median_price, len(ref_prices))
        return median_price, "출고가"

    return 0.0, ""


def _pick_ref_price(vehicle: dict) -> tuple[float, str]:
    """출고가 우선, 없으면 기본가 반환 (만원 단위)"""
    fp = vehicle.get("factory_price", 0) or 0
    if fp > 0:
        return float(fp), "출고가"
    bp = vehicle.get("base_price", 0) or 0
    if bp > 0:
        return float(bp), "기본가"
    return 0.0, ""


def _is_clean_vehicle(v: dict) -> bool:
    """사고이력 없고 원색이 아닌 '정상' 차량 여부"""
    # 사고 체크
    if (int(v.get("frame_exchange", 0) or 0) > 0
            or int(v.get("exterior_exchange", 0) or 0) > 0
            or int(v.get("frame_bodywork", 0) or 0) > 0
            or int(v.get("exterior_bodywork", 0) or 0) > 0):
        return False
    # 원색 체크
    from app.services.rule_engine import normalize_color
    color = v.get("색상", "") or v.get("color", "") or ""
    if normalize_color(color) == "other":
        return False
    return True


def _lower_half_mean(values: list[float]) -> float:
    """하위 50% 평균 — 고가 이상치 영향 자연 차단."""
    if not values:
        return 0.0
    s = sorted(values)
    half = max(len(s) // 2, 1)  # 최소 1건
    return statistics.mean(s[:half])


def _filter_outliers(values: list[float]) -> list[float]:
    """IQR 기반 이상치 제거. 3건 미만이면 그대로 반환."""
    if len(values) < 3:
        return values
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [v for v in values if lower <= v <= upper]
    return filtered if filtered else values  # 전부 제거되면 원본 유지


def _bracket_key(mileage: int) -> int:
    """주행거리 → 10,000km 구간 시작값"""
    return (mileage // 10000) * 10000


def _build_brackets(
    vehicles: list[dict],
    tgt_ref_price: float,
) -> dict[int, MileageBracket]:
    """
    소매 차량 → 10,000km 구간별 분류 및 비율 산출.

    effective_ratio = min(clean_ratios) — 사고/원색 차량 제외 후 최저비율.
    """
    brackets: dict[int, MileageBracket] = {}
    # 구간별 정상차량 비율 (사고·원색 제외)
    clean_ratios_map: dict[int, list[float]] = {}

    for v in vehicles:
        v_mileage = int(v.get("주행거리", 0) or 0)
        v_retail = float(v.get("소매가", 0) or 0)
        if v_retail <= 0:
            continue

        # 출고가/기본가 둘 다 없는 차량 제외
        ref_price, _ = _pick_ref_price(v)
        if ref_price <= 0:
            continue

        key = _bracket_key(v_mileage)
        if key not in brackets:
            brackets[key] = MileageBracket(
                bracket_start=key,
                bracket_end=key + 10000,
            )
            clean_ratios_map[key] = []

        b = brackets[key]

        # AA등급 정규화: 사고차량 가격을 무사고 수준으로 상향
        aa_adj = _calc_aa_adjustment(v)
        if aa_adj > 0 and ref_price > 0:
            v_retail_aa = v_retail + aa_adj * ref_price
        elif aa_adj > 0:
            v_retail_aa = v_retail * (1 + aa_adj)
        else:
            v_retail_aa = v_retail

        b.prices.append(v_retail_aa)
        b.mileages.append(v_mileage)

        if ref_price > 0:
            ratio = v_retail_aa / ref_price
            if ratio < 1.0:  # 비율 >= 100% 는 비정상 데이터 제외
                b.ratios.append(ratio)
                if _is_clean_vehicle(v):
                    clean_ratios_map[key].append(ratio)

    # 이상치 필터링 + effective_ratio 결정 (보수적: min(mean, median) of clean)
    for b in brackets.values():
        b.count = len(b.prices)
        key = b.bracket_start

        # 가격 이상치 필터링 → CV 기반 조건부 적용
        filtered_prices = _filter_outliers(b.prices) if b.prices else []
        if filtered_prices:
            mean_price = statistics.mean(filtered_prices)
            med_price = statistics.median(filtered_prices)
            if len(filtered_prices) > 1:
                b.price_cv = statistics.stdev(filtered_prices) / mean_price if mean_price > 0 else 0
            # CV > 12%: 하위 50% 평균 (이상치 차단), 아니면 min(mean, median)
            if b.price_cv > 0.12:
                b.median_price = _lower_half_mean(filtered_prices)
            else:
                b.median_price = min(mean_price, med_price)

        # 비율 이상치 필터링 → CV 기반 조건부 적용
        b.ratios = _filter_outliers(b.ratios) if b.ratios else []
        clean_raw = clean_ratios_map.get(key, [])
        clean = _filter_outliers(clean_raw) if clean_raw else []

        use_ratios = clean if clean else b.ratios
        if use_ratios:
            r_mean = statistics.mean(use_ratios)
            r_median = statistics.median(use_ratios)
            if len(b.ratios) > 1:
                b.ratio_cv = statistics.stdev(b.ratios) / statistics.mean(b.ratios) if statistics.mean(b.ratios) > 0 else 0
            # 가격 또는 비율 CV > 12%: 하위 50% 평균, 아니면 min(mean, median)
            if max(b.price_cv, b.ratio_cv) > 0.12:
                b.median_ratio = _lower_half_mean(use_ratios)
            else:
                b.median_ratio = min(r_mean, r_median)

        # effective_ratio
        if use_ratios:
            b.effective_ratio = b.median_ratio
        elif tgt_ref_price > 0 and b.median_price > 0:
            b.effective_ratio = b.median_price / tgt_ref_price
            b.smoothed = True

    return brackets


def _interpolate_ratio(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
) -> tuple[float, str]:
    """
    비율 추이를 기반으로 대상 주행거리의 감가 비율 산출.

    Returns:
        (ratio, method)  — ratio는 소수 (예: 0.723)
    """
    # effective_ratio가 유효한 구간만 사용
    usable = [b for b in sorted_brackets if b.effective_ratio > 0]
    if not usable:
        return 0, "no_data"

    target_key = _bracket_key(target_mileage)
    target_mid = target_key + 5000

    # 1) 정확한 구간 존재 → 추세선과 평균하여 안정화
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        # 인접 구간 이동평균으로 평활화
        ratio = _moving_avg(target_key, usable, lambda b: b.effective_ratio)
        if ratio <= 0:
            ratio = exact.effective_ratio
        # 건수 가중 추세선 80% 가중 평균
        if len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.effective_ratio for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            trend_ratio = y_mean + slope * (target_mid / 10000 - x_mean)
            if trend_ratio > 0:
                ratio = ratio * 0.2 + trend_ratio * 0.8
        return ratio, "ratio_direct"

    # 2) lower/upper 구간 분리 (중심점 기준)
    lower = [b for b in usable if (b.bracket_start + 5000) <= target_mid]
    upper = [b for b in usable if (b.bracket_start + 5000) > target_mid]

    # 3) 양쪽 존재 → 선형 보간
    if lower and upper:
        lb = lower[-1]
        ub = upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        ratio = lb.effective_ratio + t * (ub.effective_ratio - lb.effective_ratio)
        return ratio, "ratio_interpolation"

    # 4) 한쪽만 존재 → 건수 가중 추세선에서 감쇠 외삽
    if len(usable) >= 2:
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.effective_ratio for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)

        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        # 추세선 값에서 시작 (원값 대신)
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.effective_ratio
        distance = target_mid_10k - nearest_mid_10k
        ratio = _dampened_extrapolation(trend_at_nearest, slope, distance)
        ratio = max(ratio, 0.05)
        return ratio, "ratio_trend_extrapolation"

    # 5) 구간 1개만 존재 → 그 비율 직접 사용
    return usable[0].effective_ratio, "ratio_nearest"


def _interpolate_price(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
) -> tuple[float, str]:
    """
    출고가 없을 때: 구간별 소매가 중앙값 추이로 대상 주행거리의 소매가 직접 추정.

    Returns:
        (price, method)  — price는 만원 단위
    """
    # 가격 보간: 1건이라도 실제 시장 데이터이므로 모든 구간 사용
    usable = [b for b in sorted_brackets if b.median_price > 0]
    if not usable:
        return 0, "no_data"

    target_key = _bracket_key(target_mileage)
    target_mid = target_key + 5000

    # 1) 정확한 구간 → 추세선과 평균하여 안정화
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        # 인접 구간 이동평균으로 평활화
        price = _moving_avg(target_key, usable, lambda b: b.median_price)
        if price <= 0:
            price = exact.median_price
        # 건수 가중 추세선 80% 가중 평균
        if len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.median_price for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            trend_price = y_mean + slope * (target_mid / 10000 - x_mean)
            if trend_price > 0:
                price = price * 0.2 + trend_price * 0.8
        return price, "price_direct"

    # 2) lower/upper 분리
    lower = [b for b in usable if (b.bracket_start + 5000) <= target_mid]
    upper = [b for b in usable if (b.bracket_start + 5000) > target_mid]

    # 3) 양쪽 → 선형 보간
    if lower and upper:
        lb, ub = lower[-1], upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        price = lb.median_price + t * (ub.median_price - lb.median_price)
        return max(price, 0), "price_interpolation"

    # 4) 한쪽만 → 건수 가중 추세선에서 감쇠 외삽
    if len(usable) >= 2:
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.median_price for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.median_price
        distance = target_mid_10k - nearest_mid_10k
        price = _dampened_extrapolation(trend_at_nearest, slope, distance)
        return max(price, 0), "price_trend_extrapolation"

    # 5) 1개 구간
    return usable[0].median_price, "price_nearest"


def _moving_avg(
    target_key: int,
    usable: list[MileageBracket],
    value_fn,
) -> float:
    """
    인접 구간 가중 이동평균.

    양쪽 인접 구간 존재: prev×0.25 + current×0.5 + next×0.25
    한쪽만 존재:         neighbor×0.3 + current×0.7
    인접 없음:           current 그대로
    """
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if not exact:
        return 0.0

    current_val = value_fn(exact)
    if current_val <= 0:
        return 0.0

    # 인접 구간 찾기 (10,000km 단위)
    exact_idx = next(i for i, b in enumerate(usable) if b.bracket_start == target_key)
    prev_val = value_fn(usable[exact_idx - 1]) if exact_idx > 0 else 0.0
    next_val = value_fn(usable[exact_idx + 1]) if exact_idx < len(usable) - 1 else 0.0

    has_prev = prev_val > 0
    has_next = next_val > 0

    if has_prev and has_next:
        return prev_val * 0.25 + current_val * 0.5 + next_val * 0.25
    elif has_prev:
        return prev_val * 0.3 + current_val * 0.7
    elif has_next:
        return next_val * 0.3 + current_val * 0.7
    return current_val


def _dampened_extrapolation(base_value: float, slope: float, distance: float) -> float:
    """
    지수 감쇠 외삽: 데이터 범위를 벗어나는 구간마다 기울기를 절반씩 감쇠.

    base_value: 가장 가까운 구간의 값 (비율 또는 가격)
    slope: 만km당 기울기
    distance: 외삽 거리 (만km 단위, 양수)
    """
    if abs(distance) < 0.01:
        return base_value

    # 1만km 단위로 감쇠 적용 (매 구간 기울기 × 0.5)
    decay = 0.5
    result = base_value
    remaining = abs(distance)
    current_slope = slope

    while remaining > 0:
        step = min(remaining, 1.0)  # 1만km씩 적용
        result += current_slope * step * (1 if distance > 0 else -1)
        remaining -= step
        current_slope *= decay  # 다음 구간은 기울기 절반

    return result


def _calc_slope(xs: list[float], ys: list[float]) -> float:
    """단순 선형회귀 기울기 (최소자승법)"""
    n = len(xs)
    if n < 2:
        return 0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0
    return numerator / denominator


def _calc_weighted_slope(
    xs: list[float], ys: list[float], weights: list[float],
) -> tuple[float, float, float]:
    """
    건수 가중 선형회귀.

    Returns: (slope, x_mean_w, y_mean_w) — 가중 평균 x, y도 함께 반환.
    """
    n = len(xs)
    if n < 2:
        return 0, 0, 0
    w_sum = sum(weights)
    if w_sum == 0:
        return 0, 0, 0
    x_mean = sum(w * x for w, x in zip(weights, xs)) / w_sum
    y_mean = sum(w * y for w, y in zip(weights, ys)) / w_sum
    numerator = sum(w * (x - x_mean) * (y - y_mean) for w, x, y in zip(weights, xs, ys))
    denominator = sum(w * (x - x_mean) ** 2 for w, x in zip(weights, xs))
    if denominator == 0:
        return 0, x_mean, y_mean
    return numerator / denominator, x_mean, y_mean



def _build_details(
    result: RetailEstimateResult,
    maker: str, model: str, trim: str,
    year: int, mileage: int,
) -> str:
    """상세 분석 텍스트"""
    lines = [
        f"검색: {maker} {model} / 트림: {trim} / 연식: {result.year_range_used}",
        f"소매 차량 {result.vehicles_found}건 수집, {len(result.brackets)}개 구간 분석",
        f"대상 주행거리: {mileage:,}km",
        "",
        "── 구간별 비율 추이 ──",
    ]

    for b in result.brackets:
        km_label = f"{b.bracket_start // 10000}~{b.bracket_end // 10000}만km"
        ratio_pct = f"{b.effective_ratio * 100:.1f}%"
        if b.prices:
            p_min, p_max = min(b.prices), max(b.prices)
            price_range = f"{p_min:,.0f}만" if p_min == p_max else f"{p_min:,.0f}~{p_max:,.0f}만"
        else:
            price_range = "-"
        lines.append(f"  {km_label} ({b.count}건): {ratio_pct} [소매가 {price_range}]")

    lines.append("")

    method_labels = {
        "ratio_direct": "해당 구간 비율 직접 적용",
        "ratio_interpolation": "인접 구간 비율 선형 보간",
        "ratio_trend_extrapolation": "비율 추세(기울기) 기반 외삽",
        "ratio_nearest": "가장 가까운 구간 비율 적용",
        "price_direct": "해당 구간 소매가 중앙값 적용 (출고가 없음)",
        "price_interpolation": "인접 구간 소매가 선형 보간 (출고가 없음)",
        "price_trend_extrapolation": "소매가 추세 기반 외삽 (출고가 없음)",
        "price_nearest": "가장 가까운 구간 소매가 적용 (출고가 없음)",
        "no_data": "유효 데이터 없음",
    }
    lines.append(f"산출 방식: {method_labels.get(result.method, result.method)}")

    if result.estimated_ratio > 0 and result.reference_price_used > 0:
        lines.append(
            f"추정 비율: {result.estimated_ratio:.1f}% "
            f"× 대상 {result.reference_price_label} {result.reference_price_used:,.0f}만원"
        )
    elif result.estimated_retail > 0 and result.reference_price_used <= 0:
        lines.append("출고가/기본가 없음 → 같은 트림 유사 차량 소매가 추이로 직접 추정")

    if result.quality_filter:
        lines.append(f"품질 필터: {result.quality_filter}")
    lines.append(f"= 추정 소매가: {result.estimated_retail:,.0f}만원")

    return "\n".join(lines)


# =========================================================================
# 공개 API
# =========================================================================

def estimate_retail_by_market(
    maker: str,
    model: str,
    trim: str,
    year: int,
    mileage: int,
    factory_price: float = 0,
    base_price: float = 0,
) -> RetailEstimateResult:
    """
    시장 데이터 기반 소매가 추정 (비율 추이 방식).

    같은 트림+연식의 소매 차량에서 10,000km 구간별 소매가/출고가 비율을 산출하고,
    비율 변화 추이로 대상 차량 주행거리의 비율을 보간/외삽한 뒤
    대상 출고가(기본가)에 적용하여 소매가를 산출한다.

    Returns:
        RetailEstimateResult: success=False이면 데이터 부족 (소매가 산출 불가)
    """
    from app.services.firestore_db import search_retail_db

    result = RetailEstimateResult()

    # 대상 차량 기준가
    if factory_price > 0:
        tgt_ref_price = factory_price
        tgt_ref_label = "출고가"
    elif base_price > 0:
        tgt_ref_price = base_price
        tgt_ref_label = "기본가"
    else:
        tgt_ref_price = 0
        tgt_ref_label = ""
    result.reference_price_used = tgt_ref_price
    result.reference_price_label = tgt_ref_label

    if not trim:
        result.method = "no_data"
        result.details = "트림 정보 없음 — 소매가 추정 불가"
        logger.info("소매가 추정 스킵: 트림 없음")
        return result

    # 1단계: 같은 트림 + 같은 연식 (부족하면 연식 점진 확대)
    vehicles = search_retail_db(
        model=model, maker=maker, trim=trim,
        year_min=year, year_max=year, limit=500,
    )
    result.year_range_used = f"{year}년"

    if len(vehicles) < MIN_VEHICLES_DESIRED:
        vehicles = search_retail_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 1, year_max=year + 1, limit=500,
        )
        result.year_range_used = f"{year - 1}~{year + 1}년"

    if len(vehicles) < MIN_VEHICLES_DESIRED:
        vehicles = search_retail_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 2, year_max=year + 2, limit=500,
        )
        result.year_range_used = f"{year - 2}~{year + 2}년"

    result.vehicles_found = len(vehicles)
    logger.info(
        "소매가 추정: %s %s %s %s — %d건",
        maker, model, trim, result.year_range_used, len(vehicles),
    )

    if len(vehicles) < MIN_TOTAL_VEHICLES:
        result.method = "no_data"
        result.details = (
            f"같은 트림({trim}) 연식({result.year_range_used}) "
            f"소매 차량 {len(vehicles)}건 — 데이터 부족"
        )
        return result

    # 1.5단계: 엔카 API 보강 + 품질 필터링
    enriched_count = _enrich_vehicles(vehicles)
    vehicles, quality_desc = _filter_quality_vehicles(vehicles)
    result.vehicles_found = len(vehicles)
    result.quality_filter = quality_desc
    logger.info("품질 필터: %s (API 보강: %d건)", quality_desc, enriched_count)

    # 1.7단계: 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 2단계: 구간별 비율 산출
    bracket_map = _build_brackets(vehicles, tgt_ref_price)
    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 추정 (비율 × 출고가 → 실제 소매가로 보정)
    if tgt_ref_price > 0:
        ratio, ratio_method = _interpolate_ratio(mileage, sorted_brackets)
        price, price_method = _interpolate_price(mileage, sorted_brackets)

        if ratio > 0:
            ratio_est = ratio * tgt_ref_price
            if price > 0:
                # 비율 추정값과 소매가 보간값 가중 평균 (가격 60% 우선)
                result.estimated_retail = round(ratio_est * 0.4 + price * 0.6, 1)
                result.method = ratio_method + "+price_adj"
            else:
                result.estimated_retail = round(ratio_est, 1)
                result.method = ratio_method
            result.estimated_ratio = round(ratio * 100, 1)
            result.success = True
        elif price > 0:
            result.estimated_retail = round(price, 1)
            result.method = price_method
            result.success = True
        else:
            result.method = "no_data"
    else:
        # 출고가 없음 → 유사 차량 소매가 추이로 직접 추정
        price, method = _interpolate_price(mileage, sorted_brackets)
        result.method = method
        if price > 0:
            result.estimated_retail = round(price, 1)
            result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    usable_count = sum(1 for b in sorted_brackets if b.effective_ratio > 0 or b.median_price > 0)
    if (result.vehicles_found >= 10
            and result.method in ("ratio_direct", "price_direct")
            and target_bracket
            and target_bracket.count >= 3):
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and usable_count >= 2:
        result.confidence = "보통"
    else:
        result.confidence = "낮음"

    result.details = _build_details(result, maker, model, trim, year, mileage)
    return result


# =========================================================================
# 낙찰가 구간별 보수적 추정 (min(mean, median) + 추세선 비교)
# =========================================================================

@dataclass
class AuctionEstimateResult:
    """낙찰가 추정 결과"""
    estimated_auction: float = 0.0
    estimated_ratio: float = 0.0       # 추정 비율 (%, 예: 52.3)
    reference_price_used: float = 0.0
    reference_price_label: str = ""
    method: str = ""
    brackets: list[MileageBracket] = field(default_factory=list)
    vehicles_found: int = 0
    year_range_used: str = ""
    details: str = ""
    confidence: str = "보통"
    success: bool = False


def _build_auction_brackets(
    vehicles: list[dict],
    tgt_ref_price: float,
) -> dict[int, MileageBracket]:
    """
    낙찰 차량 → 10,000km 구간별 분류 및 비율 산출.

    effective_ratio = min(clean_ratios) — 사고/원색 차량 제외 후 최저비율.
    """
    brackets: dict[int, MileageBracket] = {}
    clean_ratios_map: dict[int, list[float]] = {}

    for v in vehicles:
        v_mileage = int(v.get("주행거리", 0) or 0)
        v_auction = _to_man_won(v.get("낙찰가", 0) or 0)
        if v_auction <= 0:
            continue

        # 출고가/기본가 둘 다 없는 차량 제외
        ref_price, _ = _pick_ref_price(v)
        if ref_price <= 0:
            continue

        key = _bracket_key(v_mileage)
        if key not in brackets:
            brackets[key] = MileageBracket(
                bracket_start=key,
                bracket_end=key + 10000,
            )
            clean_ratios_map[key] = []

        b = brackets[key]

        # AA등급 정규화: 사고차량 가격을 무사고 수준으로 상향
        aa_adj = _calc_aa_adjustment(v)
        if aa_adj > 0 and ref_price > 0:
            v_auction_aa = v_auction + aa_adj * ref_price
        elif aa_adj > 0:
            v_auction_aa = v_auction * (1 + aa_adj)
        else:
            v_auction_aa = v_auction

        b.prices.append(v_auction_aa)
        b.mileages.append(v_mileage)

        if ref_price > 0:
            ratio = v_auction_aa / ref_price
            if ratio < 1.0:  # 비율 >= 100% 는 비정상 데이터 제외
                b.ratios.append(ratio)
                if _is_clean_vehicle(v):
                    clean_ratios_map[key].append(ratio)

    # 이상치 필터링 + effective_ratio 결정 (보수적: min(mean, median) of clean)
    for b in brackets.values():
        b.count = len(b.prices)
        key = b.bracket_start

        # 가격 이상치 필터링 → CV 기반 조건부 적용
        filtered_prices = _filter_outliers(b.prices) if b.prices else []
        if filtered_prices:
            mean_price = statistics.mean(filtered_prices)
            med_price = statistics.median(filtered_prices)
            if len(filtered_prices) > 1:
                b.price_cv = statistics.stdev(filtered_prices) / mean_price if mean_price > 0 else 0
            # CV > 12%: 하위 50% 평균 (이상치 차단), 아니면 min(mean, median)
            if b.price_cv > 0.12:
                b.median_price = _lower_half_mean(filtered_prices)
            else:
                b.median_price = min(mean_price, med_price)

        # 비율 이상치 필터링 → CV 기반 조건부 적용
        b.ratios = _filter_outliers(b.ratios) if b.ratios else []
        clean_raw = clean_ratios_map.get(key, [])
        clean = _filter_outliers(clean_raw) if clean_raw else []

        use_ratios = clean if clean else b.ratios
        if use_ratios:
            r_mean = statistics.mean(use_ratios)
            r_median = statistics.median(use_ratios)
            if len(b.ratios) > 1:
                b.ratio_cv = statistics.stdev(b.ratios) / statistics.mean(b.ratios) if statistics.mean(b.ratios) > 0 else 0
            # 가격 또는 비율 CV > 12%: 하위 50% 평균, 아니면 min(mean, median)
            if max(b.price_cv, b.ratio_cv) > 0.12:
                b.median_ratio = _lower_half_mean(use_ratios)
            else:
                b.median_ratio = min(r_mean, r_median)

        # effective_ratio
        if use_ratios:
            b.effective_ratio = b.median_ratio
        elif tgt_ref_price > 0 and b.median_price > 0:
            b.effective_ratio = b.median_price / tgt_ref_price
            b.smoothed = True

    return brackets


def estimate_auction_by_market(
    maker: str,
    model: str,
    trim: str,
    year: int,
    mileage: int,
    factory_price: float = 0,
    base_price: float = 0,
) -> AuctionEstimateResult:
    """
    시장 데이터 기반 낙찰가 추정 (구간별 평균 방식).

    소매가 추정과 동일 구조이나, 구간별 평균(mean) 비율/가격 사용.
    """
    from app.services.firestore_db import search_auction_db

    result = AuctionEstimateResult()

    # 대상 차량 기준가
    if factory_price > 0:
        tgt_ref_price = factory_price
        tgt_ref_label = "출고가"
    elif base_price > 0:
        tgt_ref_price = base_price
        tgt_ref_label = "기본가"
    else:
        tgt_ref_price = 0
        tgt_ref_label = ""
    result.reference_price_used = tgt_ref_price
    result.reference_price_label = tgt_ref_label

    if not trim:
        result.method = "no_data"
        result.details = "트림 정보 없음 — 낙찰가 추정 불가"
        logger.info("낙찰가 추정 스킵: 트림 없음")
        return result

    # 1단계: 같은 트림 + 같은 연식 (부족하면 연식 점진 확대)
    vehicles = search_auction_db(
        model=model, maker=maker, trim=trim,
        year_min=year, year_max=year, limit=500,
        domestic_only=True,
    )
    result.year_range_used = f"{year}년"

    if len(vehicles) < MIN_AUCTION_VEHICLES_DESIRED:
        vehicles = search_auction_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 1, year_max=year + 1, limit=500,
            domestic_only=True,
        )
        result.year_range_used = f"{year - 1}~{year + 1}년"

    if len(vehicles) < MIN_AUCTION_VEHICLES_DESIRED:
        vehicles = search_auction_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 2, year_max=year + 2, limit=500,
            domestic_only=True,
        )
        result.year_range_used = f"{year - 2}~{year + 2}년"

    result.vehicles_found = len(vehicles)
    logger.info(
        "낙찰가 추정: %s %s %s %s — %d건",
        maker, model, trim, result.year_range_used, len(vehicles),
    )

    if len(vehicles) < MIN_TOTAL_VEHICLES:
        result.method = "no_data"
        result.details = (
            f"같은 트림({trim}) 연식({result.year_range_used}) "
            f"낙찰 차량 {len(vehicles)}건 — 데이터 부족"
        )
        return result

    # 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 2단계: 구간별 비율 산출 (평균 기반)
    bracket_map = _build_auction_brackets(vehicles, tgt_ref_price)
    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 추정 (비율 × 출고가 → 실제 낙찰가로 보정)
    if tgt_ref_price > 0:
        ratio, ratio_method = _interpolate_auction_ratio(mileage, sorted_brackets)
        price, price_method = _interpolate_auction_price(mileage, sorted_brackets)

        if ratio > 0:
            ratio_est = ratio * tgt_ref_price
            if price > 0:
                # 비율 추정값과 낙찰가 보간값 가중 평균 (가격 60% 우선)
                result.estimated_auction = round(ratio_est * 0.4 + price * 0.6, 1)
                result.method = ratio_method + "+price_adj"
            else:
                result.estimated_auction = round(ratio_est, 1)
                result.method = ratio_method
            result.estimated_ratio = round(ratio * 100, 1)
            result.success = True
        elif price > 0:
            result.estimated_auction = round(price, 1)
            result.method = price_method
            result.success = True
        else:
            result.method = "no_data"
    else:
        price, method = _interpolate_auction_price(mileage, sorted_brackets)
        result.method = method
        if price > 0:
            result.estimated_auction = round(price, 1)
            result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    usable_count = sum(1 for b in sorted_brackets if b.effective_ratio > 0 or b.median_price > 0)
    if (result.vehicles_found >= 10
            and result.method in ("ratio_direct", "price_direct")
            and target_bracket
            and target_bracket.count >= 3):
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and usable_count >= 2:
        result.confidence = "보통"
    else:
        result.confidence = "낮음"

    # 상세 텍스트
    lines = [
        f"검색: {maker} {model} / 트림: {trim} / 연식: {result.year_range_used}",
        f"낙찰 차량 {result.vehicles_found}건 수집, {len(result.brackets)}개 구간 분석",
        f"대상 주행거리: {mileage:,}km",
        "",
        "── 구간별 비율 추이 (평균) ──",
    ]
    for b in result.brackets:
        km_label = f"{b.bracket_start // 10000}~{b.bracket_end // 10000}만km"
        ratio_pct = f"{b.effective_ratio * 100:.1f}%"
        if b.prices:
            p_min, p_max = min(b.prices), max(b.prices)
            price_range = f"{p_min:,.0f}만" if p_min == p_max else f"{p_min:,.0f}~{p_max:,.0f}만"
        else:
            price_range = "-"
        lines.append(f"  {km_label} ({b.count}건): {ratio_pct} [낙찰가 {price_range}]")
    lines.append("")
    lines.append(f"산출 방식: {result.method}")
    if result.estimated_ratio > 0 and result.reference_price_used > 0:
        lines.append(
            f"추정 비율: {result.estimated_ratio:.1f}% "
            f"× 대상 {result.reference_price_label} {result.reference_price_used:,.0f}만원"
        )
    elif result.estimated_auction > 0 and result.reference_price_used <= 0:
        lines.append("출고가/기본가 없음 → 같은 트림 유사 차량 낙찰가 추이로 직접 추정")
    lines.append(f"= 추정 낙찰가: {result.estimated_auction:,.0f}만원")
    result.details = "\n".join(lines)

    return result


def estimate_export_auction_by_market(
    maker: str,
    model: str,
    trim: str,
    year: int,
    mileage: int,
    factory_price: float = 0,
    base_price: float = 0,
) -> AuctionEstimateResult:
    """
    시장 데이터 기반 수출 낙찰가 추정.

    내수 낙찰가 추정과 동일 구조이나, 수출 차량 데이터만 사용.
    """
    from app.services.firestore_db import search_auction_db

    result = AuctionEstimateResult()

    # 대상 차량 기준가
    if factory_price > 0:
        tgt_ref_price = factory_price
        tgt_ref_label = "출고가"
    elif base_price > 0:
        tgt_ref_price = base_price
        tgt_ref_label = "기본가"
    else:
        tgt_ref_price = 0
        tgt_ref_label = ""
    result.reference_price_used = tgt_ref_price
    result.reference_price_label = tgt_ref_label

    if not trim:
        result.method = "no_data"
        result.details = "트림 정보 없음 — 수출 낙찰가 추정 불가"
        return result

    # 1단계: 수출 차량만 검색 (연식 점진 확대)
    vehicles = search_auction_db(
        model=model, maker=maker, trim=trim,
        year_min=year, year_max=year, limit=500,
        domestic_only=False, export_only=True,
    )
    result.year_range_used = f"{year}년"

    if len(vehicles) < MIN_AUCTION_VEHICLES_DESIRED:
        vehicles = search_auction_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 1, year_max=year + 1, limit=500,
            domestic_only=False, export_only=True,
        )
        result.year_range_used = f"{year - 1}~{year + 1}년"

    if len(vehicles) < MIN_AUCTION_VEHICLES_DESIRED:
        vehicles = search_auction_db(
            model=model, maker=maker, trim=trim,
            year_min=year - 2, year_max=year + 2, limit=500,
            domestic_only=False, export_only=True,
        )
        result.year_range_used = f"{year - 2}~{year + 2}년"

    result.vehicles_found = len(vehicles)
    logger.info(
        "수출 낙찰가 추정: %s %s %s %s — %d건",
        maker, model, trim, result.year_range_used, len(vehicles),
    )

    if len(vehicles) < MIN_TOTAL_VEHICLES:
        result.method = "no_data"
        result.details = (
            f"같은 트림({trim}) 연식({result.year_range_used}) "
            f"수출 낙찰 차량 {len(vehicles)}건 — 데이터 부족"
        )
        return result

    # 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 2단계: 구간별 비율 산출
    bracket_map = _build_auction_brackets(vehicles, tgt_ref_price)
    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 추정
    if tgt_ref_price > 0:
        ratio, ratio_method = _interpolate_auction_ratio(mileage, sorted_brackets)
        price, price_method = _interpolate_auction_price(mileage, sorted_brackets)

        if ratio > 0:
            ratio_est = ratio * tgt_ref_price
            if price > 0:
                # 비율 추정값과 수출 낙찰가 보간값 가중 평균 (가격 60% 우선)
                result.estimated_auction = round(ratio_est * 0.4 + price * 0.6, 1)
                result.method = ratio_method + "+price_adj"
            else:
                result.estimated_auction = round(ratio_est, 1)
                result.method = ratio_method
            result.estimated_ratio = round(ratio * 100, 1)
            result.success = True
        elif price > 0:
            result.estimated_auction = round(price, 1)
            result.method = price_method
            result.success = True
        else:
            result.method = "no_data"
    else:
        price, method = _interpolate_auction_price(mileage, sorted_brackets)
        result.method = method
        if price > 0:
            result.estimated_auction = round(price, 1)
            result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    usable_count = sum(1 for b in sorted_brackets if b.effective_ratio > 0 or b.median_price > 0)
    if (result.vehicles_found >= 10
            and result.method in ("ratio_direct", "price_direct")
            and target_bracket
            and target_bracket.count >= 3):
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and usable_count >= 2:
        result.confidence = "보통"
    else:
        result.confidence = "낮음"

    # 상세 텍스트
    lines = [
        f"검색: {maker} {model} / 트림: {trim} / 연식: {result.year_range_used}",
        f"수출 낙찰 차량 {result.vehicles_found}건 수집, {len(result.brackets)}개 구간 분석",
        f"대상 주행거리: {mileage:,}km",
        "",
        "── 구간별 비율 추이 (수출) ──",
    ]
    for b in result.brackets:
        km_label = f"{b.bracket_start // 10000}~{b.bracket_end // 10000}만km"
        ratio_pct = f"{b.effective_ratio * 100:.1f}%"
        if b.prices:
            p_min, p_max = min(b.prices), max(b.prices)
            price_range = f"{p_min:,.0f}만" if p_min == p_max else f"{p_min:,.0f}~{p_max:,.0f}만"
        else:
            price_range = "-"
        lines.append(f"  {km_label} ({b.count}건): {ratio_pct} [수출낙찰가 {price_range}]")
    lines.append("")
    lines.append(f"산출 방식: {result.method}")
    if result.estimated_ratio > 0 and result.reference_price_used > 0:
        lines.append(
            f"추정 비율: {result.estimated_ratio:.1f}% "
            f"× 대상 {result.reference_price_label} {result.reference_price_used:,.0f}만원"
        )
    elif result.estimated_auction > 0 and result.reference_price_used <= 0:
        lines.append("출고가/기본가 없음 → 같은 트림 유사 차량 수출 낙찰가 추이로 직접 추정")
    lines.append(f"= 추정 수출 낙찰가: {result.estimated_auction:,.0f}만원")
    result.details = "\n".join(lines)

    return result


def _interpolate_auction_ratio(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
) -> tuple[float, str]:
    """낙찰가 비율 보간 (직접 구간은 추세선과 평균하여 안정화)."""
    usable = [b for b in sorted_brackets if b.effective_ratio > 0]
    if not usable:
        return 0, "no_data"

    target_key = _bracket_key(target_mileage)
    target_mid = target_key + 5000

    # 1) 정확한 구간 → 이동평균 + 건수 가중 추세선(80%)
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        ratio = _moving_avg(target_key, usable, lambda b: b.effective_ratio)
        if ratio <= 0:
            ratio = exact.effective_ratio
        if len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.effective_ratio for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            trend_ratio = y_mean + slope * (target_mid / 10000 - x_mean)
            if trend_ratio > 0:
                ratio = ratio * 0.2 + trend_ratio * 0.8
        return ratio, "ratio_direct"

    # 2) lower/upper
    lower = [b for b in usable if (b.bracket_start + 5000) <= target_mid]
    upper = [b for b in usable if (b.bracket_start + 5000) > target_mid]

    # 3) 선형 보간
    if lower and upper:
        lb, ub = lower[-1], upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        ratio = lb.effective_ratio + t * (ub.effective_ratio - lb.effective_ratio)
        return ratio, "ratio_interpolation"

    # 4) 건수 가중 추세선에서 감쇠 외삽
    if len(usable) >= 2:
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.effective_ratio for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.effective_ratio
        distance = target_mid_10k - nearest_mid_10k
        ratio = _dampened_extrapolation(trend_at_nearest, slope, distance)
        return max(ratio, 0.05), "ratio_trend_extrapolation"

    # 5) 1개 구간
    return usable[0].effective_ratio, "ratio_nearest"


def _interpolate_auction_price(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
) -> tuple[float, str]:
    """낙찰가 직접 보간 (출고가 없을 때, 추세선과 평균하여 안정화)."""
    # 가격 보간: 1건이라도 실제 시장 데이터이므로 모든 구간 사용
    usable = [b for b in sorted_brackets if b.median_price > 0]
    if not usable:
        return 0, "no_data"

    target_key = _bracket_key(target_mileage)
    target_mid = target_key + 5000

    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        price = _moving_avg(target_key, usable, lambda b: b.median_price)
        if price <= 0:
            price = exact.median_price
        if len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.median_price for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            trend_price = y_mean + slope * (target_mid / 10000 - x_mean)
            if trend_price > 0:
                price = price * 0.2 + trend_price * 0.8
        return price, "price_direct"

    lower = [b for b in usable if (b.bracket_start + 5000) <= target_mid]
    upper = [b for b in usable if (b.bracket_start + 5000) > target_mid]

    if lower and upper:
        lb, ub = lower[-1], upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        price = lb.median_price + t * (ub.median_price - lb.median_price)
        return max(price, 0), "price_interpolation"

    if len(usable) >= 2:
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.median_price for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.median_price
        distance = target_mid_10k - nearest_mid_10k
        price = _dampened_extrapolation(trend_at_nearest, slope, distance)
        return max(price, 0), "price_trend_extrapolation"

    return usable[0].median_price, "price_nearest"
