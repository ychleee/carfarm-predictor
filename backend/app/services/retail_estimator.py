"""
시장 데이터 기반 가격 추정 엔진 (소매가 + 낙찰가)

알고리즘 (연속 평활 추정):
  1. 같은 트림 + 같은 연식(부족하면 ±1년)의 차량 검색
  2. 모든 차량을 흰색·무사고로 정규화 (색상 페널티 + 사고 감가 역산)
  3. 주행거리 오름차순으로 비율(정규화가격/출고가)을 Gaussian 가중 로컬 선형회귀로 평활
  4. 대상 주행거리에서의 평활 비율 × 대상 출고가 = 추정가
  5. 10,000km 구간별 요약은 UI 표시용으로만 유지
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _round10(value: float) -> float:
    """10만원 단위 반올림 (예: 1055.2 → 1060, 938.5 → 940)"""
    return round(value / 10) * 10


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
    feedback_excluded: int = 0
    learned_correction_applied: bool = False
    blended_params: "LearnedParams | None" = None
    success: bool = False
    vehicles: list[dict] = field(default_factory=list)  # 사용된 전체 차량


# =========================================================================
# 설정값
# =========================================================================

RATIO_CV_THRESHOLD = 0.15     # 비율 CV > 15% → 비율 변동 큼
PRICE_CV_THRESHOLD = 0.10     # 소매가 CV < 10% → 소매가 안정적
MIN_VEHICLES_PER_BRACKET = 2  # 구간당 최소 차량 수
MIN_TOTAL_VEHICLES = 3        # 전체 최소 차량 수 (이 이하면 추정 불가)
MIN_VEHICLES_DESIRED = 6      # 소매 목표 차량 수 (이하이면 연식 확대)
MIN_AUCTION_VEHICLES_DESIRED = 6   # 낙찰 목표 차량 수 (이하이면 연식 확대)


def _filter_recent(
    vehicles: list[dict], months: int = 3, date_field: str = "개최일",
) -> list[dict]:
    """최근 N개월 데이터만 필터링."""
    from datetime import timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    return [v for v in vehicles if (v.get(date_field) or "") >= cutoff_str]


def _filter_recent_staged(
    vehicles: list[dict], min_count: int, date_field: str = "개최일",
) -> tuple[list[dict], int]:
    """3개월 → 6개월 → 12개월 → 전체 단계적 확장. (필터 결과, 사용 개월수) 반환."""
    for months in (3, 6, 12):
        recent = _filter_recent(vehicles, months, date_field)
        if len(recent) >= min_count:
            return recent, months
    return vehicles, 0  # 전체 사용



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

    1순위: 기준차량의 출고가 최빈값 (서브모델 혼재 시 가장 대표적인 값)
    2순위: DB에서 같은 maker+model 출고가 조회
    """
    # 1순위: 기준차량 출고가 최빈값 — 서브모델 혼재 시 가장 대표적인 출고가 사용
    fp_values = [float(v.get("factory_price", 0) or 0) for v in vehicles if float(v.get("factory_price", 0) or 0) > 0]
    if fp_values:
        # 50만원 단위로 그룹핑하여 최빈 그룹 결정
        grouped: dict[int, list[float]] = {}
        for fp in fp_values:
            key = round(fp / 50) * 50
            grouped.setdefault(key, []).append(fp)
        # 최빈 그룹 (동률 시 가장 낮은 출고가 그룹 — 기본형에 가까움)
        best_key = max(grouped.keys(), key=lambda k: (len(grouped[k]), -k))
        best_group = grouped[best_key]
        result = statistics.median(best_group)
        logger.info(
            "대상 출고가 최빈값: %.0f만원 (그룹 %d만원대, %d건/%d건)",
            result, best_key, len(best_group), len(fp_values),
        )
        return result, "출고가"

    # 2순위: 기준차량 기본가 포함 중앙값
    ref_prices = [_pick_ref_price(v)[0] for v in vehicles if _pick_ref_price(v)[0] > 0]
    if ref_prices:
        median_price = statistics.median(ref_prices)
        logger.info("대상 출고가 기준차량 중앙값: %.0f만원 (%d건)", median_price, len(ref_prices))
        return median_price, "출고가"

    # 3순위: DB 조회 (기준차량이 전혀 없을 때)
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


def _lower_pct_mean(values: list[float], pct: float = 0.4) -> float:
    """하위 pct% 평균 — 낙찰가 등 보수적 추정용."""
    if not values:
        return 0.0
    s = sorted(values)
    n = max(int(len(s) * pct), 1)
    return statistics.mean(s[:n])


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


# =========================================================================
# 연속 평활 추정 — 흰색·무사고 정규화 + Gaussian 가중 로컬 선형회귀
# =========================================================================

_COLOR_RULES_CACHE: dict | None = None


def _load_color_rules() -> dict:
    """pricing_rules.yaml 에서 색상 보정 테이블 로드 (캐시)."""
    global _COLOR_RULES_CACHE
    if _COLOR_RULES_CACHE is not None:
        return _COLOR_RULES_CACHE
    rules_path = Path(__file__).parent.parent.parent / "rules" / "pricing_rules.yaml"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
        _COLOR_RULES_CACHE = rules.get("color_adjustment", {})
    except Exception:
        _COLOR_RULES_CACHE = {}
    return _COLOR_RULES_CACHE


def _determine_vehicle_class(vehicles: list[dict]) -> str:
    """기준차량 세그먼트 다수결로 차급 결정 (large/medium/compact/suv)."""
    segments = [v.get("segment", "") for v in vehicles if v.get("segment")]
    if not segments:
        return "medium"
    segment = Counter(segments).most_common(1)[0][0]
    if "대형" in segment or "프리미엄" in segment:
        return "large"
    if "SUV" in segment.upper():
        return "suv"
    if "경차" in segment:
        return "compact"
    return "medium"


def _vehicle_color_penalty(vehicle: dict, vehicle_class: str, color_rules: dict) -> float:
    """개별 차량의 색상 페널티 (만원). 연식 가중치 없이 full 적용."""
    from app.services.rule_engine import normalize_color
    color = vehicle.get("색상", "") or vehicle.get("color", "") or ""
    color_group = normalize_color(color)

    table = color_rules.get(vehicle_class, color_rules.get("medium", {}))
    if isinstance(table, dict):
        penalty_base = table.get(color_group, table.get("other", 0))
        if isinstance(penalty_base, str):
            penalty_base = 0
    else:
        penalty_base = 0

    # 정규화 목적: 색상 영향을 완전히 제거 (연식 가중치 미적용)
    return float(penalty_base)


def _calc_aa_adjustment_full(vehicle: dict) -> float:
    """
    정규화 전용 사고 보정 — 연식 가중치 없이 full 비율 적용.
    _calc_aa_adjustment와 동일하되 연식 감쇠 없음.
    """
    frame_exchange = int(vehicle.get("frame_exchange", 0) or 0)
    exterior_exchange = int(vehicle.get("exterior_exchange", 0) or 0)
    frame_bodywork = int(vehicle.get("frame_bodywork", 0) or 0)
    exterior_bodywork = int(vehicle.get("exterior_bodywork", 0) or 0)

    raw = 0.0
    if frame_exchange > 0:
        raw += _STRUCTURAL_GRADE_PCT.get(frame_exchange, _STRUCTURAL_DEFAULT_PCT)
    raw += exterior_exchange * 0.02
    raw += (frame_bodywork + exterior_bodywork) * 0.01
    return raw


def _normalize_vehicles(
    vehicles: list[dict],
    tgt_ref_price: float,
    vehicle_class: str,
    price_field: str = "소매가",
) -> list[tuple[int, float, float]]:
    """
    모든 차량을 흰색·무사고로 정규화하여 (주행거리, 비율, 정규화가격) 반환.

    - 사고 보정: _calc_aa_adjustment로 무사고 수준 상향
    - 색상 보정: 색상 페널티 역산으로 흰색 수준 상향
    - ratio < 1.0 인 것만 유지 (비정상 데이터 제외)

    Returns: [(mileage, ratio, normalized_price), ...] 주행거리 오름차순
    """
    color_rules = _load_color_rules()
    results: list[tuple[int, float, float]] = []

    for v in vehicles:
        v_mileage = int(v.get("주행거리", 0) or 0)
        v_price = float(v.get(price_field, 0) or 0)
        if price_field == "낙찰가":
            v_price = _to_man_won(v_price)
        if v_price <= 0:
            continue

        ref_price, _ = _pick_ref_price(v)
        if ref_price <= 0:
            continue

        # 1) 사고 보정 → 무사고 수준 상향 (연식 가중치 없이 full)
        aa_adj = _calc_aa_adjustment_full(v)
        if aa_adj > 0:
            v_price += aa_adj * ref_price

        # 2) 색상 보정 → 흰색 수준 상향 (연식 가중치 없이 full)
        color_pen = _vehicle_color_penalty(v, vehicle_class, color_rules)
        if color_pen < 0:
            v_price -= color_pen

        ratio = v_price / ref_price
        if ratio >= 1.0 or ratio <= 0.05:
            continue

        results.append((v_mileage, ratio, v_price))

    results.sort(key=lambda x: x[0])
    return results


def _normalize_vehicles_price_only(
    vehicles: list[dict],
    vehicle_class: str,
    price_field: str = "소매가",
    full_normalize: bool = True,
    target_year: int = 0,
    tgt_ref_price: float = 0,
) -> list[tuple[int, float]]:
    """
    가격을 흰색·무사고·동일연식으로 정규화. Returns: [(mileage, price)] 주행거리 오름차순.

    full_normalize=True (소매): 사고/색상 보정을 연식 감쇠 없이 full 적용
    full_normalize=False (경매): 사고 보정에 연식 가중치 적용 (과보정 방지)
    target_year: (미사용, 하위 호환용)
    tgt_ref_price: 대상 출고가. > 0이면 각 차량 가격을 대상 출고가 기준으로 스케일링.
    """
    color_rules = _load_color_rules()
    results: list[tuple[int, float]] = []

    for v in vehicles:
        v_mileage = int(v.get("주행거리", 0) or 0)
        v_price = float(v.get(price_field, 0) or 0)
        if price_field == "낙찰가":
            v_price = _to_man_won(v_price)
        if v_price <= 0:
            continue

        ref_price, _ = _pick_ref_price(v)
        orig_price = v_price

        # 사고 보정
        if full_normalize:
            aa_adj = _calc_aa_adjustment_full(v)
        else:
            aa_adj = _calc_aa_adjustment(v)
        if aa_adj > 0:
            if ref_price > 0:
                v_price += aa_adj * ref_price
            else:
                v_price *= (1 + aa_adj)

        # 색상 보정 (full_normalize=False일 때도 full 색상 보정 적용)
        color_pen = _vehicle_color_penalty(v, vehicle_class, color_rules)
        if color_pen < 0:
            v_price -= color_pen

        # 출고가 정규화: 차량 출고가가 대상과 다르면 스케일링
        # fp=0 차량은 서브모델 불확실 → 스케일링 불가 → 제외
        if tgt_ref_price > 0:
            v_fp = float(v.get("factory_price", 0) or 0)
            if v_fp <= 0:
                continue  # fp 없는 차량은 정규화 불가 → 평활 추정에서 제외
            elif abs(v_fp - tgt_ref_price) / tgt_ref_price > 0.03:
                v_price = v_price * (tgt_ref_price / v_fp)

        if v_price <= 0:
            continue

        adj = v_price - orig_price
        if abs(adj) > 0.1:
            logger.debug(
                "정규화: %dkm %s %.0f만→%.0f만 (보정 %+.0f만, aa=%.3f, color=%.0f)",
                v_mileage, price_field, orig_price, v_price, adj, aa_adj, color_pen,
            )
        results.append((v_mileage, v_price))

    results.sort(key=lambda x: x[0])
    logger.info(
        "정규화 완료: %s %d건 → %d건 (class=%s, full=%s, tgt_ref=%.0f)",
        price_field, len(vehicles), len(results), vehicle_class, full_normalize, tgt_ref_price,
    )
    return results


def _normalize_vehicles_to_ratio(
    vehicles: list[dict],
    vehicle_class: str,
    price_field: str = "소매가",
    full_normalize: bool = True,
) -> list[tuple[int, float]]:
    """
    가격을 흰색·무사고로 정규화 후 출고가 대비 비율로 변환.
    출고가 없는 차량은 제외. Returns: [(mileage, ratio)] 주행거리 오름차순.
    """
    color_rules = _load_color_rules()
    results: list[tuple[int, float]] = []

    for v in vehicles:
        v_mileage = int(v.get("주행거리", 0) or 0)
        v_price = float(v.get(price_field, 0) or 0)
        if price_field == "낙찰가":
            v_price = _to_man_won(v_price)
        if v_price <= 0:
            continue

        ref_price, _ = _pick_ref_price(v)
        if ref_price <= 0:
            continue  # 출고가 없으면 비율 산출 불가 → 제외

        # 사고 보정
        if full_normalize:
            aa_adj = _calc_aa_adjustment_full(v)
        else:
            aa_adj = _calc_aa_adjustment(v)
        if aa_adj > 0:
            v_price += aa_adj * ref_price

        # 색상 보정
        color_pen = _vehicle_color_penalty(v, vehicle_class, color_rules)
        if color_pen < 0:
            v_price -= color_pen

        if v_price <= 0:
            continue

        ratio = v_price / ref_price
        results.append((v_mileage, ratio))

    results.sort(key=lambda x: x[0])
    return results


def _gaussian_weights(
    mileages: list[int],
    target_mileage: int,
    bandwidth: int = 30000,
) -> list[float]:
    """Gaussian 커널 가중치. 2×bandwidth 초과는 0 (hard cutoff)."""
    cutoff = bandwidth * 2
    return [
        math.exp(-0.5 * ((m - target_mileage) / bandwidth) ** 2)
        if abs(m - target_mileage) <= cutoff else 0.0
        for m in mileages
    ]


def _weighted_local_regression(
    xs: list[float],
    ys: list[float],
    ws: list[float],
    target_x: float,
) -> float:
    """가중 로컬 선형회귀로 target_x 에서의 y 추정."""
    w_sum = sum(ws)
    if w_sum <= 0:
        return 0.0

    wx = sum(w * x for w, x in zip(ws, xs))
    wy = sum(w * y for w, y in zip(ws, ys))
    wxx = sum(w * x * x for w, x in zip(ws, xs))
    wxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

    denom = w_sum * wxx - wx ** 2
    if abs(denom) < 1e-10:
        # 분모 0 (x 값 동일) → 가중 평균
        return wy / w_sum

    b = (w_sum * wxy - wx * wy) / denom
    a = (wy - b * wx) / w_sum
    return a + b * target_x


def _smooth_ratio_estimate(
    data: list[tuple[int, float, float]],
    target_mileage: int,
    tgt_ref_price: float,
    bandwidth: int = 30000,
) -> tuple[float, float, str]:
    """
    Gaussian 가중 로컬 선형회귀로 대상 주행거리의 비율 추정.

    Returns: (estimated_price, estimated_ratio, method)
    """
    if not data:
        return 0, 0, "no_data"
    if len(data) == 1:
        _, ratio, _ = data[0]
        return round(ratio * tgt_ref_price, 1), ratio, "smooth_single"

    mileages = [d[0] for d in data]
    ratios = [d[1] for d in data]
    weights = _gaussian_weights(mileages, target_mileage, bandwidth)

    ratio = _weighted_local_regression(
        [float(m) for m in mileages], ratios, weights, float(target_mileage),
    )
    ratio = max(ratio, 0.05)
    price = round(ratio * tgt_ref_price, 1)

    logger.info(
        "평활 비율 추정: target=%dkm, data=%d건, bandwidth=%dkm → ratio=%.3f (%.1f%%)",
        target_mileage, len(data), bandwidth, ratio, ratio * 100,
    )
    return price, ratio, "smooth_regression"


def _mileage_depreciation_rate(
    vehicle_age: int, price_manwon: float, target_mileage: int = 0,
) -> float:
    """1만km당 감가율 (가격 대비 비율) — 비즈니스 기준.

    - 2~3년: 2%
    - 4~6년: 1.5%
    - 7~9년: 1%
    - 10년+, 2000만원 이상: 1%
    - 10년+, 2000만원 미만: 10만원/1만km → 가격 대비 % 환산

    고주행거리(15만km+)에서는 시장 수요 감소·위험 증가로 감가율 가속.
    """
    if vehicle_age <= 3:
        base = 0.02
    elif vehicle_age <= 6:
        base = 0.015
    elif vehicle_age <= 9:
        base = 0.01
    else:
        if price_manwon >= 2000:
            base = 0.01
        else:
            base = 10.0 / max(price_manwon, 100) if price_manwon > 0 else 0.01

    # 고주행거리 가속 (시장 수요 감소·위험 증가)
    # 지수 감쇠와 함께 사용되므로 완만한 가속 (지수 감쇠 자체가 누적 가속 효과)
    if target_mileage > 200000:
        base *= 1.2
    elif target_mileage > 150000:
        base *= 1.15
    return base


def _apply_damage_adjustment(
    vehicles: list[dict],
    price_field: str = "소매가",
) -> list[dict]:
    """판금·도색 보정: 각 차량의 가격을 상향 보정한 복사본 리스트 반환.

    보정 기준 (1회당 신차대비율% 추가):
    - 3년 미만 or 출고가 ≥ 2500만원: 2%
    - 3~7년 or 출고가 1500~2500만원: 1.5%
    - 그 외: 1%
    - 소매가/낙찰가 500만원 미만: 미적용
    """
    to_man_won = price_field == "낙찰가"
    now_year = datetime.now().year
    adjusted = []
    adj_count = 0
    for v in vehicles:
        count = int(v.get("bodywork_count", 0) or 0)
        if count <= 0:
            adjusted.append(v)
            continue

        ref_price, _ = _pick_ref_price(v)
        if ref_price <= 0:
            adjusted.append(v)
            continue

        # 현재 판매가 확인 (500만 미만이면 미적용)
        sell_price = float(v.get(price_field, 0) or 0)
        if sell_price <= 0:
            adjusted.append(v)
            continue
        sell_man = _to_man_won(sell_price) if to_man_won else sell_price
        if sell_man < 500:
            adjusted.append(v)
            continue

        year = int(v.get("연식", 0) or v.get("year", 0) or 0)
        age = now_year - year if year > 0 else 99

        if age < 3 or ref_price >= 2500:
            rate = 0.02
        elif age <= 7 or ref_price >= 1500:
            rate = 0.015
        else:
            rate = 0.01

        adj_man = round(count * rate * ref_price, 1)  # 만원 단위

        new_v = dict(v)
        if to_man_won and sell_price > 100000:
            new_v[price_field] = sell_price + adj_man * 10000
        else:
            new_v[price_field] = sell_price + adj_man
        adj_count += 1
        adjusted.append(new_v)

    if adj_count > 0:
        logger.info("판금·도색 보정: %d건 가격 상향 (1회당 %.1f~2.0%%)", adj_count, 1.0)
    return adjusted


def _filter_gap_outliers(
    vehicles: list[dict],
    price_field: str = "소매가",
    gap_pct: float = 0.05,
) -> list[dict]:
    """인접 가격 간 gap이 gap_pct 이상이면 양쪽 끝에서 제거 (계산용).

    - 가장 낮은 가격 ↔ 그 다음 가격: gap ≥ 5% 이면 최저가 제외, 반복
    - 가장 높은 가격 ↔ 그 아래 가격: gap ≥ 5% 이면 최고가 제외, 반복
    - 원본 vehicle dict에 '_gap_excluded' 플래그 추가 (표시용)
    - 계산에 사용할 차량 리스트만 반환
    """
    to_man_won = price_field == "낙찰가"

    def _get_price(v: dict) -> float:
        p = float(v.get(price_field, 0) or 0)
        if to_man_won and p > 0:
            p = _to_man_won(p)
        return p

    # 가격이 있는 차량만 대상
    priced = [(i, _get_price(v)) for i, v in enumerate(vehicles) if _get_price(v) > 0]
    if len(priced) < 3:
        return vehicles  # 데이터 너무 적으면 필터 안함

    # 가격 오름차순 정렬
    priced.sort(key=lambda x: x[1])
    excluded_indices: set[int] = set()

    # 최소 유지 건수: 전체의 2/3 또는 3건 중 큰 값
    # — 데이터가 적을 때 과도한 제거 방지
    min_keep = max(math.ceil(len(priced) * 2 / 3), 3)

    # 하단 gap 제거
    while len(priced) - len([i for i, _ in priced if i in excluded_indices]) > min_keep:
        active = [(i, p) for i, p in priced if i not in excluded_indices]
        if len(active) <= min_keep:
            break
        lowest_idx, lowest_p = active[0]
        next_p = active[1][1]
        if next_p > 0 and (next_p - lowest_p) / next_p >= gap_pct:
            excluded_indices.add(lowest_idx)
        else:
            break

    # 상단 gap 제거
    while len(priced) - len([i for i, _ in priced if i in excluded_indices]) > min_keep:
        active = [(i, p) for i, p in priced if i not in excluded_indices]
        if len(active) <= min_keep:
            break
        highest_idx, highest_p = active[-1]
        prev_p = active[-2][1]
        if prev_p > 0 and (highest_p - prev_p) / highest_p >= gap_pct:
            excluded_indices.add(highest_idx)
        else:
            break

    # 플래그 설정 + 계산용 리스트 반환
    calc_vehicles = []
    for i, v in enumerate(vehicles):
        if i in excluded_indices:
            v["_gap_excluded"] = True
        else:
            v.pop("_gap_excluded", None)
            calc_vehicles.append(v)

    if excluded_indices:
        excluded_prices = [_get_price(vehicles[i]) for i in excluded_indices]
        logger.info(
            "gap 이상치 필터: %d건 제외 (가격: %s), 계산용 %d건",
            len(excluded_indices),
            [round(p, 1) for p in sorted(excluded_prices)],
            len(calc_vehicles),
        )

    return calc_vehicles


def _adaptive_bandwidth(data: list[tuple], target_mileage: int) -> int:
    """데이터 밀도에 따른 적응형 bandwidth.

    sqrt(n) 기반 최근접 이웃: 데이터가 많을수록 좁게, 적을수록 넓게.
    데이터가 희소하거나 대상이 데이터 범위 경계에 있으면 대역폭 확대.
    """
    n = len(data)
    if n <= 3:
        return 80000
    mileages = sorted(d[0] for d in data)
    distances = sorted(abs(d[0] - target_mileage) for d in data)
    k = min(max(int(math.sqrt(n)), 5), 25)  # sqrt(n), 5~25 범위
    k = min(k, n - 1)
    bw = max(distances[k], 10000)  # 최소 10,000km

    # 데이터 희소 또는 경계: 최소 3개 이상 유효 가중치(w>=0.1)를 보장
    max_bw = 40000
    if n <= 10:
        max_bw = 80000  # 소량 데이터 → 넓은 대역폭
    elif target_mileage >= mileages[-1] - 20000 or target_mileage <= mileages[0] + 20000:
        max_bw = 60000  # 경계 근처 → 대역폭 확대
    return min(bw, max_bw)


def _filter_local_outliers(
    data: list[tuple[int, float]],
    deviation_pct: float = 0.30,
    max_iterations: int = 3,
) -> list[tuple[int, float]]:
    """반복적 금액 이상치 필터: 평활 가격 대비 편차 > threshold 데이터를 반복 제거.

    1회차: 평활 가격 계산 → 편차 큰 데이터 제거
    2회차: 남은 데이터로 평활 가격 재계산 → 추가 제거
    ... 수렴 또는 max_iterations까지 반복.
    """
    if len(data) < 4:
        return data

    current = list(data)
    total_excluded = 0

    for iteration in range(max_iterations):
        if len(current) < 4:
            break

        bw = _adaptive_bandwidth(current, current[len(current) // 2][0])
        mileages = [d[0] for d in current]
        prices = [d[1] for d in current]

        filtered = []
        excluded_count = 0
        for i, (m, p) in enumerate(current):
            weights = list(_gaussian_weights(mileages, m, bw))
            weights[i] = 0
            w_sum = sum(weights)
            if w_sum < 1e-10:
                filtered.append((m, p))
                continue
            local_price = sum(w * pr for w, pr in zip(weights, prices)) / w_sum
            if local_price > 0 and abs(p - local_price) / local_price > deviation_pct:
                excluded_count += 1
            else:
                filtered.append((m, p))

        if excluded_count == 0:
            break  # 수렴

        total_excluded += excluded_count
        current = filtered
        logger.info(
            "로컬 이상치 필터(금액) %d회차: %d건 제외, 누적 %d건",
            iteration + 1, excluded_count, total_excluded,
        )

    if total_excluded > 0:
        logger.info(
            "로컬 이상치 필터(금액) 완료: 총 %d건 제외 (±%.0f%%), %d건 유지",
            total_excluded, deviation_pct * 100, len(current),
        )
    return current if len(current) >= 3 else data


def _filter_vehicles_by_local_ratio(
    vehicles: list[dict],
    tgt_ref_price: float,
    price_field: str = "소매가",
    deviation_pct: float = 0.30,
    max_iterations: int = 3,
) -> list[dict]:
    """반복적 비율 이상치 필터: 평활 비율 대비 편차 > threshold 차량을 반복 제거.

    1회차: 평활 비율 계산 → 편차 큰 차량 제거
    2회차: 남은 차량으로 평활 비율 재계산 → 추가 제거
    ... 수렴 또는 max_iterations까지 반복.

    이상치가 다수파인 경우에도 점진적으로 제거 가능.
    """
    if len(vehicles) < 4 or tgt_ref_price <= 0:
        return vehicles

    to_man_won = price_field == "낙찰가"

    # (index, mileage, ratio) 리스트 구성 — 차량 자체 출고가 기준
    entries: list[tuple[int, int, float]] = []
    for i, v in enumerate(vehicles):
        km = int(v.get("주행거리", 0) or 0)
        p = float(v.get(price_field, 0) or 0)
        if to_man_won and p > 0:
            p = _to_man_won(p)
        if p <= 0 or km <= 0:
            entries.append((i, km, -1.0))
            continue
        v_fp = float(v.get("factory_price", 0) or 0)
        ratio_ref = v_fp if v_fp > 0 else tgt_ref_price
        ratio = p / ratio_ref
        entries.append((i, km, ratio))

    excluded_indices: set[int] = set()
    total_excluded = 0

    for iteration in range(max_iterations):
        valid = [
            (idx, km, r) for idx, km, r in entries
            if r >= 0 and idx not in excluded_indices
        ]
        if len(valid) < 4:
            break

        mileages = [km for _, km, _ in valid]
        ratios = [r for _, _, r in valid]
        mid_km = mileages[len(mileages) // 2]
        bw = _adaptive_bandwidth(list(zip(mileages, ratios)), mid_km)

        new_excluded: set[int] = set()
        for j, (idx, km, ratio) in enumerate(valid):
            weights = list(_gaussian_weights(mileages, km, bw))
            weights[j] = 0
            w_sum = sum(weights)
            if w_sum < 1e-10:
                continue
            local_ratio = sum(w * r for w, r in zip(weights, ratios)) / w_sum
            if local_ratio > 0 and abs(ratio - local_ratio) / local_ratio > deviation_pct:
                new_excluded.add(idx)

        if not new_excluded:
            break  # 수렴

        excluded_indices |= new_excluded
        total_excluded += len(new_excluded)
        logger.info(
            "로컬 이상치 필터(비율) %d회차: %d건 제외, 누적 %d건 제외",
            iteration + 1, len(new_excluded), total_excluded,
        )

    if total_excluded > 0:
        logger.info(
            "로컬 이상치 필터(비율) 완료: 총 %d건 제외 (평활 비율 대비 ±%.0f%%), %d건 유지",
            total_excluded, deviation_pct * 100,
            len(vehicles) - total_excluded,
        )

    calc_vehicles = [v for i, v in enumerate(vehicles) if i not in excluded_indices]
    return calc_vehicles if len(calc_vehicles) >= 3 else vehicles


def _remove_same_as_factory(
    vehicles: list[dict],
    price_field: str = "소매가",
    tolerance: float = 0.03,
) -> list[dict]:
    """출고가(기본가)와 판매가가 거의 같은 차량 제거 (표시 + 계산 모두 제외).

    가격이 출고가의 ±3% 이내이면 비정상 데이터로 간주.
    """
    to_man_won = price_field == "낙찰가"
    filtered = []
    removed = 0
    for v in vehicles:
        ref, _ = _pick_ref_price(v)
        if ref <= 0:
            filtered.append(v)
            continue
        p = float(v.get(price_field, 0) or 0)
        if to_man_won and p > 0:
            p = _to_man_won(p)
        if p <= 0:
            filtered.append(v)
            continue
        if abs(p - ref) / ref <= tolerance:
            removed += 1
            continue  # 제외
        filtered.append(v)
    if removed > 0:
        logger.info(
            "출고가≈판매가 필터: %d건 제거 (허용 오차 ±%.0f%%), %d건 유지",
            removed, tolerance * 100, len(filtered),
        )
    return filtered


MIN_CLEAN_VEHICLES = 5  # 무사고 차량만으로 최소 이 수 이상이어야 사고차 제외


def _has_damage(v: dict) -> bool:
    """교환·판금·도색 이력이 있는 차량 여부."""
    return (
        int(v.get("frame_exchange", 0) or 0) > 0
        or int(v.get("exterior_exchange", 0) or 0) > 0
        or int(v.get("frame_bodywork", 0) or 0) > 0
        or int(v.get("exterior_bodywork", 0) or 0) > 0
    )


def _filter_damaged_vehicles(
    vehicles: list[dict],
    price_field: str = "소매가",
) -> list[dict]:
    """교환·판금·도색 차량 계산 제외.

    - 무사고 차량이 MIN_CLEAN_VEHICLES건 이상이면 사고차 제외
    - 아니면 전체 유지 (데이터 부족 방지)
    """
    clean = [v for v in vehicles if not _has_damage(v)]
    damaged_count = len(vehicles) - len(clean)
    if damaged_count == 0:
        return vehicles
    if len(clean) >= MIN_CLEAN_VEHICLES:
        logger.info(
            "사고차 필터: %d건 제외 (교환/판금/도색), 무사고 %d건 유지",
            damaged_count, len(clean),
        )
        return clean
    logger.info(
        "사고차 필터 스킵: 무사고 %d건 부족 (최소 %d건), 전체 %d건 유지",
        len(clean), MIN_CLEAN_VEHICLES, len(vehicles),
    )
    return vehicles


def _smooth_price_estimate(
    data: list[tuple[int, float]],
    target_mileage: int,
    bandwidth: int = 0,
    conservative: bool = False,
    vehicle_year: int = 0,
    conservative_pct: float = 0,
) -> tuple[float, str]:
    """
    Gaussian 가중 로컬 선형회귀로 가격 평활 추정.

    conservative=True: 경매용 — 가격이 낮은 차량에 더 높은 가중치 부여 (보수적).
    conservative=False: 소매용 — 모든 차량 동등 가중 (시장 평균).

    Returns: (estimated_price, method)
    """
    if not data:
        return 0, "no_data"
    if len(data) == 1:
        return round(data[0][1], 1), "smooth_price_single"

    # adaptive bandwidth: 데이터 밀도 기반 자동 결정
    if bandwidth <= 0:
        bandwidth = _adaptive_bandwidth(data, target_mileage)

    mileages = [d[0] for d in data]
    prices = [d[1] for d in data]
    dist_weights = list(_gaussian_weights(mileages, target_mileage, bandwidth))

    # ── 이상치 감쇄: 회귀 잔차 기반 MAD 방식 ──
    # 주행거리 추세를 제거한 후 비정상적 저가 차량 가중치 감쇄
    fmileages = [float(m) for m in mileages]
    n_active = sum(1 for w in dist_weights if w > 0)
    if n_active >= 5:
        _w_sum = sum(dist_weights)
        _wx = sum(w * x for w, x in zip(dist_weights, fmileages))
        _wy = sum(w * p for w, p in zip(dist_weights, prices))
        _wxx = sum(w * x * x for w, x in zip(dist_weights, fmileages))
        _wxy = sum(w * x * p for w, x, p in zip(dist_weights, fmileages, prices))
        _denom = _w_sum * _wxx - _wx ** 2
        if abs(_denom) > 1e-10:
            _b = (_w_sum * _wxy - _wx * _wy) / _denom
            _a = (_wy - _b * _wx) / _w_sum
            resids = [p - (_a + _b * m) for p, m in zip(prices, fmileages)]

            # 가중 잔차 중위수
            rw_active = sorted(
                ((r, w) for r, w in zip(resids, dist_weights) if w > 0),
                key=lambda x: x[0],
            )
            _tot = sum(w for _, w in rw_active)
            _cum = 0.0
            _r_med = 0.0
            for r, w in rw_active:
                _cum += w
                if _cum >= _tot * 0.5:
                    _r_med = r
                    break

            # 가중 MAD (Median Absolute Deviation)
            ad_pairs = sorted(
                ((abs(r - _r_med), w) for r, w in zip(resids, dist_weights) if w > 0),
                key=lambda x: x[0],
            )
            _cum = 0.0
            _mad = 0.0
            for ad, w in ad_pairs:
                _cum += w
                if _cum >= _tot * 0.5:
                    _mad = ad
                    break
            robust_sigma = 1.4826 * max(_mad, 1.0)

            # 하한 이상치만 감쇄 (3σ_MAD 이하)
            lower_bound = _r_med - 3.0 * robust_sigma
            dampened = 0
            for i, (r, w) in enumerate(zip(resids, dist_weights)):
                if w > 0 and r < lower_bound:
                    dist_weights[i] = w * 0.1
                    dampened += 1
            if dampened > 0:
                print(f"  [MAD이상치] {dampened}건, MAD={_mad:.0f}, rσ={robust_sigma:.0f}, 하한={lower_bound:.0f}")

    weights = dist_weights
    far_range = False  # 가중치 전부 0 → 원거리 외삽 사용 여부

    if conservative:
        # 보수적 추정: 추세선 기울기 유지 + 하위 35% 잔차 적용
        # (기존 절대가격 필터는 회귀 기울기를 왜곡하므로, 잔차 기반으로 변경)
        w_sum = sum(dist_weights)
        if w_sum > 0:
            fmileages = [float(m) for m in mileages]
            # 1) 회귀 계수 산출 (가중 선형회귀)
            wx = sum(w * x for w, x in zip(dist_weights, fmileages))
            wy = sum(w * p for w, p in zip(dist_weights, prices))
            wxx = sum(w * x * x for w, x in zip(dist_weights, fmileages))
            wxy = sum(w * x * p for w, x, p in zip(dist_weights, fmileages, prices))
            denom = w_sum * wxx - wx ** 2
            if abs(denom) < 1e-10:
                b_slope = 0.0
                a_intercept = wy / w_sum
            else:
                b_slope = (w_sum * wxy - wx * wy) / denom
                a_intercept = (wy - b_slope * wx) / w_sum
            reg_price = a_intercept + b_slope * float(target_mileage)

            # 과외삽 방지: 회귀가 가중평균을 넘으면 가중평균으로 캡
            w_mean = wy / w_sum
            base_price = min(reg_price, w_mean)

            # 2) 잔차 계산 (base_price 기준 추세선에서의 잔차)
            residuals_weights = []
            for m, p, w in zip(fmileages, prices, dist_weights):
                if w > 0:
                    pred = a_intercept + b_slope * m
                    residuals_weights.append((p - pred, w))

            # 3) 가중 하위 잔차 평균 (보수적: 저가 차량에 더 무게)
            #    기본: 10년 이내 하위 15%, 그 외 하위 25%
            #    auction(conservative_pct 지정 시) 해당 값 사용
            current_year = datetime.now().year
            age = current_year - vehicle_year if vehicle_year > 0 else 99
            if conservative_pct > 0:
                pass  # 호출자가 지정한 값 사용
            else:
                conservative_pct = 0.15 if age <= 10 else 0.25
            residuals_weights.sort(key=lambda x: x[0])
            total_w = sum(w for _, w in residuals_weights)
            target_w = total_w * conservative_pct
            cum_w = 0.0
            lower_r_sum = 0.0
            lower_w_sum = 0.0
            for r, w in residuals_weights:
                cum_w += w
                lower_r_sum += r * w
                lower_w_sum += w
                if cum_w >= target_w:
                    break

            lower_residual = lower_r_sum / lower_w_sum if lower_w_sum > 0 else 0
            price = base_price + lower_residual

            # ── 데이터 범위 초과 감가 보정 ──
            # 가중치가 있어도 target이 데이터 최대 주행거리를 크게 초과하면
            # 가중 평균만으로는 부족 → 지수 감쇠 추가 감가 적용
            max_mileage_in_data = max(fmileages)
            mileage_gap = target_mileage - max_mileage_in_data
            if mileage_gap > 50000 and vehicle_year > 0:
                current_year = datetime.now().year
                gap_age = current_year - vehicle_year
                gap_rate = _mileage_depreciation_rate(gap_age, price, target_mileage)
                # 250k+ 가속은 갭 보정에서는 적용하지 않음 (이미 rate에 200k+ 가속 포함)
                gap_units = mileage_gap / 10000
                # 지수 감쇠: (1 - rate)^units (선형보다 안정적)
                gap_adjusted = price * ((1 - gap_rate) ** gap_units)
                gap_adjusted = max(gap_adjusted, price * 0.25)  # 최저 25% 바닥
                print(f"  [보수-갭보정] price={price:.1f}, gap={mileage_gap/10000:.1f}만km, rate={gap_rate:.4f}, exp_mult={(1-gap_rate)**gap_units:.3f} → {gap_adjusted:.1f}")
                price = gap_adjusted
                far_range = True  # 갭 보정 적용 시 blend에서 smooth 우선

            print(f"  [보수] reg={reg_price:.1f}, w_mean={w_mean:.1f}, base={base_price:.1f}, lower_resid={lower_residual:.1f} → {price:.1f}")
        else:
            # 가중치 전부 0 (타겟이 데이터 범위 훨씬 밖) → 전체 데이터 균등 회귀 + 비즈니스 외삽
            fmileages = [float(m) for m in mileages]
            n = len(fmileages)
            wx = sum(fmileages)
            wy = sum(prices)
            wxx = sum(x * x for x in fmileages)
            wxy = sum(x * p for x, p in zip(fmileages, prices))
            denom = n * wxx - wx ** 2
            if abs(denom) < 1e-10:
                b_slope = 0.0
                a_intercept = wy / n
            else:
                b_slope = (n * wxy - wx * wy) / denom
                a_intercept = (wy - b_slope * wx) / n
            # 회귀 경계값 또는 eff_min 중 높은 값을 기준으로 비즈니스 룰 외삽
            eff_min = min(prices)
            max_mileage_in_data = max(mileages)
            reg_at_boundary = a_intercept + b_slope * float(max_mileage_in_data)
            # 회귀 경계값이 실제 최저가보다 높으면 채택 (더 대표적인 시작점)
            start_price = max(reg_at_boundary, eff_min)
            rate = 0.0
            if vehicle_year > 0 and target_mileage > max_mileage_in_data:
                current_year = datetime.now().year
                age = current_year - vehicle_year
                rate = _mileage_depreciation_rate(age, start_price, target_mileage)
                extra_units = (target_mileage - max_mileage_in_data) / 10000
                # 지수 감쇠: (1 - rate)^units — 선형보다 안정적, 극단 외삽에서 과도한 감가 방지
                exp_mult = (1 - rate) ** extra_units
                extrap_price = start_price * exp_mult
                price = max(extrap_price, start_price * 0.25)
            else:
                extra_units = 0
                price = max(a_intercept + b_slope * float(target_mileage), eff_min * 0.3)
            far_range = True
            print(f"  [원거리외삽] start={start_price:.1f}(eff_min={eff_min:.1f}, reg@bnd={reg_at_boundary:.1f}), max_km={max_mileage_in_data}, rate={rate:.4f}, exp_mult={(1-rate)**extra_units:.3f} → {price:.1f}")
    else:
        # 비보수적(소매) 경로: 가중 회귀 + 원거리 외삽 처리
        w_sum = sum(dist_weights)
        if w_sum > 0:
            price = _weighted_local_regression(
                [float(m) for m in mileages], prices, weights, float(target_mileage),
            )
            # 데이터 범위 초과 감가 보정 (보수적 경로와 동일 로직)
            fmileages = [float(m) for m in mileages]
            max_mileage_in_data = max(fmileages)
            mileage_gap = target_mileage - max_mileage_in_data
            if mileage_gap > 50000 and vehicle_year > 0:
                current_year = datetime.now().year
                gap_age = current_year - vehicle_year
                gap_rate = _mileage_depreciation_rate(gap_age, price if price > 0 else sum(prices)/len(prices), target_mileage)
                gap_units = mileage_gap / 10000
                w_mean = sum(w * p for w, p in zip(dist_weights, prices)) / w_sum
                gap_adjusted = w_mean * ((1 - gap_rate) ** gap_units)
                gap_adjusted = max(gap_adjusted, w_mean * 0.25)
                print(f"  [소매-갭보정] w_mean={w_mean:.1f}, gap={mileage_gap/10000:.1f}만km, rate={gap_rate:.4f} → {gap_adjusted:.1f}")
                price = gap_adjusted
                far_range = True  # 갭 보정 적용 시 blend에서 smooth 우선
        else:
            # 가중치 전부 0 → 전체 데이터 균등 회귀 + 비즈니스 외삽
            fmileages = [float(m) for m in mileages]
            n = len(fmileages)
            wx = sum(fmileages)
            wy = sum(prices)
            wxx = sum(x * x for x in fmileages)
            wxy = sum(x * p for x, p in zip(fmileages, prices))
            denom = n * wxx - wx ** 2
            if abs(denom) < 1e-10:
                a_intercept = wy / n
            else:
                b_slope = (n * wxy - wx * wy) / denom
                a_intercept = (wy - b_slope * wx) / n
            max_mileage_in_data = max(mileages)
            eff_min = min(prices)
            reg_at_boundary = a_intercept + b_slope * float(max_mileage_in_data) if abs(denom) >= 1e-10 else a_intercept
            start_price = max(reg_at_boundary, eff_min)
            if vehicle_year > 0 and target_mileage > max_mileage_in_data:
                current_year = datetime.now().year
                age = current_year - vehicle_year
                rate = _mileage_depreciation_rate(age, start_price, target_mileage)
                extra_units = (target_mileage - max_mileage_in_data) / 10000
                price = start_price * ((1 - rate) ** extra_units)
                price = max(price, start_price * 0.25)
            else:
                price = max(a_intercept + b_slope * float(target_mileage) if abs(denom) >= 1e-10 else a_intercept, eff_min * 0.3)
            far_range = True
            print(f"  [소매-원거리외삽] start={start_price:.1f}, max_km={max_mileage_in_data}, rate={rate:.4f} → {price:.1f}")

    # 과외삽 방지: 회귀 결과를 유효 데이터 범위로 제한
    max_w = max(weights) if weights else 0
    if max_w > 0:
        effective_prices = [p for p, w in zip(prices, weights) if w >= max_w * 0.1]
        if effective_prices:
            eff_max = max(effective_prices)
            eff_min = min(effective_prices)
            if price > eff_max:
                price = eff_max
            elif price < eff_min:
                max_mileage_in_data = max(mileages)
                min_mileage_in_data = min(mileages)
                mileage_range = max_mileage_in_data - min_mileage_in_data
                near_edge = target_mileage >= max_mileage_in_data - mileage_range * 0.15
                if vehicle_year > 0:
                    current_year = datetime.now().year
                    age = current_year - vehicle_year
                    rate = _mileage_depreciation_rate(age, eff_min, target_mileage)
                    if target_mileage > max_mileage_in_data:
                        # 완전 외삽: 지수 감쇠로 eff_min에서 추가 하락
                        extra_units = (target_mileage - max_mileage_in_data) / 10000
                        extrap_price = eff_min * ((1 - rate) ** extra_units)
                        if extra_units > 5:
                            # 긴 외삽(>5만km): 회귀 결과 신뢰 낮음 → 비즈니스 기준 우선
                            price = max(extrap_price, eff_min * 0.3)
                        else:
                            price = max(extrap_price, price, eff_min * 0.3)
                    elif near_edge:
                        # 데이터 경계 근처: 회귀 결과 신뢰, 비즈니스 기준 하한 적용
                        data_span_units = max(mileage_range / 10000, 2)
                        allowed_drop = eff_min * rate * data_span_units * 0.3
                        price = max(price, eff_min - allowed_drop, eff_min * 0.3)
                    else:
                        price = eff_min
                    print(f"  [감가외삽] eff_min={eff_min:.1f}, rate={rate:.3f}, max_km={max_mileage_in_data} → {price:.1f}")
                else:
                    price = eff_min

    # 디버그 로깅
    mode = "보수" if conservative else "소매"
    print(f"[평활 {mode}] target={target_mileage}km, n={len(data)}, bw={bandwidth} → {price:.1f}")

    if far_range:
        method = "smooth_price_far_range"
    elif conservative:
        method = "smooth_price_conservative"
    else:
        method = "smooth_price_regression"
    return max(round(price, 1), 0), method


def _bracket_key(mileage: int) -> int:
    """주행거리 → 10,000km 구간 시작값"""
    return (mileage // 10000) * 10000


def _build_brackets(
    vehicles: list[dict],
    tgt_ref_price: float,
    used_vehicles: list[dict] | None = None,
    target_year: int = 0,
) -> dict[int, MileageBracket]:
    """
    소매 차량 → 10,000km 구간별 분류 및 비율 산출.

    effective_ratio = min(clean_ratios) — 사고/원색 차량 제외 후 최저비율.
    used_vehicles: 전달 시 구간에 포함된 차량을 append (대수 통일용).
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

        if used_vehicles is not None:
            used_vehicles.append(v)

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

        # 비율 계산: 차량 자체 출고가 기준만 사용
        # fp=0 차량은 다른 서브모델일 수 있어 비율 계산에서 제외
        v_fp = float(v.get("factory_price", 0) or 0)
        if v_fp > 0:
            ratio = v_retail_aa / v_fp
            if ratio < 1.0:  # 비율 >= 100% 는 비정상 데이터 제외
                b.ratios.append(ratio)
                if _is_clean_vehicle(v):
                    clean_ratios_map[key].append(ratio)

    # 이상치 필터링 + effective_ratio 결정
    # 10년 이내: 항상 하위 20%, 그 외: CV 기반 조건부
    current_year = datetime.now().year
    age = current_year - target_year if target_year > 0 else 99
    force_lower = age <= 10

    for b in brackets.values():
        b.count = len(b.prices)
        key = b.bracket_start

        # 가격 이상치 필터링
        filtered_prices = _filter_outliers(b.prices) if b.prices else []
        if filtered_prices:
            mean_price = statistics.mean(filtered_prices)
            med_price = statistics.median(filtered_prices)
            if len(filtered_prices) > 1:
                b.price_cv = statistics.stdev(filtered_prices) / mean_price if mean_price > 0 else 0
            if force_lower:
                b.median_price = _lower_pct_mean(filtered_prices, 0.30)
            elif b.price_cv > 0.12:
                b.median_price = _lower_half_mean(filtered_prices)
            else:
                b.median_price = min(mean_price, med_price)

        # 비율 이상치 필터링
        b.ratios = _filter_outliers(b.ratios) if b.ratios else []
        clean_raw = clean_ratios_map.get(key, [])
        clean = _filter_outliers(clean_raw) if clean_raw else []

        use_ratios = clean if clean else b.ratios
        if use_ratios:
            r_mean = statistics.mean(use_ratios)
            r_median = statistics.median(use_ratios)
            if len(b.ratios) > 1:
                b.ratio_cv = statistics.stdev(b.ratios) / statistics.mean(b.ratios) if statistics.mean(b.ratios) > 0 else 0
            if force_lower:
                b.median_ratio = _lower_pct_mean(use_ratios, 0.30)
            elif max(b.price_cv, b.ratio_cv) > 0.12:
                b.median_ratio = _lower_half_mean(use_ratios)
            else:
                b.median_ratio = min(r_mean, r_median)

        # effective_ratio
        if use_ratios:
            b.effective_ratio = b.median_ratio
        elif tgt_ref_price > 0 and b.median_price > 0:
            b.effective_ratio = b.median_price / tgt_ref_price
            b.smoothed = True

    _smooth_cross_bin_outliers(brackets)
    return brackets


def _smooth_cross_bin_outliers(brackets: dict[int, "MileageBracket"]) -> None:
    """
    인접 구간 대비 비정상적으로 높거나 낮은 구간의 비율을 보정.

    10,000km 구간에서 인접 구간 보간값 대비 10% 이상 벗어나면 이상치로 판단.
    건수가 적을수록 강하게, 많아도 일정 수준 보정.
    """
    sorted_keys = sorted(k for k, b in brackets.items() if b.effective_ratio > 0)
    if len(sorted_keys) < 3:
        return

    for i, key in enumerate(sorted_keys):
        b = brackets[key]

        prev_key = sorted_keys[i - 1] if i > 0 else None
        next_key = sorted_keys[i + 1] if i < len(sorted_keys) - 1 else None

        expected = 0.0
        if prev_key is not None and next_key is not None:
            # 양쪽 인접 → 선형 보간
            t = (key - prev_key) / (next_key - prev_key)
            expected = brackets[prev_key].effective_ratio * (1 - t) + brackets[next_key].effective_ratio * t
        elif prev_key is not None and i >= 2:
            # 이전 2개 구간에서 추세 외삽
            pp_key = sorted_keys[i - 2]
            slope_per_key = (brackets[prev_key].effective_ratio - brackets[pp_key].effective_ratio) / (prev_key - pp_key)
            expected = brackets[prev_key].effective_ratio + slope_per_key * (key - prev_key)
        elif next_key is not None and i < len(sorted_keys) - 2:
            # 이후 2개 구간에서 추세 외삽
            nn_key = sorted_keys[i + 2]
            slope_per_key = (brackets[nn_key].effective_ratio - brackets[next_key].effective_ratio) / (nn_key - next_key)
            expected = brackets[next_key].effective_ratio - slope_per_key * (next_key - key)
        else:
            continue

        if expected <= 0:
            continue

        deviation = (b.effective_ratio - expected) / expected
        if abs(deviation) > 0.10:
            # 건수 기반 블렌딩 강도: 1건→85%, 2건→75%, 5건→45%, 10건→20%
            blend_weight = max(0.2, min(0.85, 0.95 - b.count * 0.1))
            b.effective_ratio = expected * blend_weight + b.effective_ratio * (1 - blend_weight)
            b.smoothed = True


def _enforce_auction_floor(
    retail_brackets: dict[int, MileageBracket],
    auction_brackets: list[MileageBracket],
) -> None:
    """
    소매 bracket의 effective_ratio가 낙찰가 bracket보다 낮으면 보정.

    낙찰가는 실거래가이므로 소매가(호가)보다 낮아야 정상.
    소매 비율 < 낙찰 비율인 구간은 비정상이므로,
    다른 구간의 소매/낙찰 프리미엄 비율을 적용하여 보정.
    """
    # 낙찰 bracket을 key로 매핑
    auction_map: dict[int, float] = {}
    for ab in auction_brackets:
        if ab.effective_ratio > 0:
            auction_map[ab.bracket_start] = ab.effective_ratio

    if not auction_map:
        return

    # 정상 구간에서 소매/낙찰 프리미엄 비율 수집
    premiums: list[float] = []
    for key, rb in retail_brackets.items():
        ar = auction_map.get(key, 0)
        if ar > 0 and rb.effective_ratio > ar:
            premiums.append(rb.effective_ratio / ar)

    # 프리미엄 중앙값 (정상 구간이 없으면 1.15 기본값)
    if premiums:
        median_premium = statistics.median(premiums)
    else:
        median_premium = 1.15

    # 소매 < 낙찰인 구간 보정
    for key, rb in retail_brackets.items():
        ar = auction_map.get(key, 0)
        if ar <= 0:
            continue
        floor = ar * median_premium
        if rb.effective_ratio < ar:
            # 낙찰 비율보다 낮음 → 프리미엄 적용
            rb.effective_ratio = floor
            rb.smoothed = True
        elif rb.effective_ratio < floor * 0.95:
            # 낙찰 비율보다 높지만 프리미엄 기대치의 95% 미만 → 부분 보정
            rb.effective_ratio = (rb.effective_ratio + floor) / 2
            rb.smoothed = True


def _interpolate_ratio(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
    vehicle_year: int = 0,
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

    # 1) 정확한 구간 존재
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        total_vehicles = sum(b.count for b in usable)
        # 희소 데이터: 구간 또는 차량 수 부족 → 정확 구간값 직접 사용
        if len(usable) <= 3 or total_vehicles <= 6:
            return exact.effective_ratio, "ratio_direct"
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

    # 3) 양쪽 존재 → 격차 크면 추세선, 아니면 역전 감지 + 건수 가중 보간
    if lower and upper:
        lb = lower[-1]
        ub = upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        gap = ub_mid - lb_mid
        # 구간 간 격차 > 20,000km → 전체 추세선 사용
        if gap > 20000 and len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.effective_ratio for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            ratio = y_mean + slope * (target_mid / 10000 - x_mean)
            if ratio > 0:
                return ratio, "ratio_trend_interpolation"
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        if lb.effective_ratio < ub.effective_ratio and lb.count < ub.count:
            ratio = ub.effective_ratio
        elif ub.effective_ratio > lb.effective_ratio and ub.count < lb.count:
            ratio = lb.effective_ratio
        else:
            lb_w = (1 - t) * max(lb.count, 1)
            ub_w = t * max(ub.count, 1)
            ratio = (lb.effective_ratio * lb_w + ub.effective_ratio * ub_w) / (lb_w + ub_w)
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

        # 비즈니스 기준 최소 기울기 적용 (고주행거리 외삽 시)
        biz_min_slope = None
        if vehicle_year > 0 and distance > 0:
            current_year = datetime.now().year
            age = current_year - vehicle_year
            rate = _mileage_depreciation_rate(age, trend_at_nearest * 4000, target_mileage)
            # 비율 기울기 최소값: rate × 현재 비율 (비율이 낮을수록 감가도 작아짐)
            biz_min_slope = -rate * trend_at_nearest
            if slope > biz_min_slope:  # slope은 음수, min_slope도 음수
                slope = biz_min_slope

        ratio = _dampened_extrapolation(
            trend_at_nearest, slope, distance, min_slope=biz_min_slope,
        )
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

    # 1) 정확한 구간
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        total_vehicles = sum(b.count for b in usable)
        if len(usable) <= 3 or total_vehicles <= 6:
            return exact.median_price, "price_direct"
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

    # 3) 양쪽 → 격차 크면 추세선, 아니면 역전 감지 + 건수 가중 보간
    if lower and upper:
        lb = lower[-1]
        ub = upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        gap = ub_mid - lb_mid
        if gap > 20000 and len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.median_price for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            price = y_mean + slope * (target_mid / 10000 - x_mean)
            if price > 0:
                return price, "price_trend_interpolation"
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        if lb.median_price < ub.median_price and lb.count < ub.count:
            price = ub.median_price
        elif ub.median_price > lb.median_price and ub.count < lb.count:
            price = lb.median_price
        else:
            lb_w = (1 - t) * max(lb.count, 1)
            ub_w = t * max(ub.count, 1)
            price = (lb.median_price * lb_w + ub.median_price * ub_w) / (lb_w + ub_w)
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


def _calc_cv_weights(
    sorted_brackets: list[MileageBracket],
) -> tuple[float, float]:
    """
    구간별 CV를 기반으로 비율/가격 가중치 결정.

    소량 데이터: 비율 중심 (출고가 정규화가 안정적)
    대량 데이터(≥50건): 가격 중심 (실제 시장가격이 더 신뢰도 높음)
    중간(15~49): 비율 우세 기본, CV에 따라 조정

    Returns: (ratio_weight, price_weight)
    """
    # 건수 가중 평균 CV 계산
    total_count = 0
    sum_ratio_cv = 0.0
    sum_price_cv = 0.0
    for b in sorted_brackets:
        c = max(b.count, 1)
        sum_ratio_cv += b.ratio_cv * c
        sum_price_cv += b.price_cv * c
        total_count += c

    if total_count == 0:
        return 0.7, 0.3

    # 극소량 데이터(10건 미만)는 순수 비율 (price_adj 노이즈 제거)
    if total_count < 10:
        return 1.0, 0.0

    # 소량 데이터(10~14건)는 비율 중심 고정
    if total_count < 15:
        return 0.85, 0.15

    avg_ratio_cv = sum_ratio_cv / total_count
    avg_price_cv = sum_price_cv / total_count

    # 안전 하한 (CV가 0이면 비교 불가)
    if avg_ratio_cv < 0.01 and avg_price_cv < 0.01:
        if total_count >= 50:
            return 0.3, 0.7
        return 0.7, 0.3

    # 대량 데이터(50건 이상): 가격 기반이 실제 시세를 잘 반영
    if total_count >= 50:
        if avg_ratio_cv > avg_price_cv * 2.0:
            return 0.1, 0.9
        elif avg_price_cv > avg_ratio_cv * 2.0:
            return 0.4, 0.6
        else:
            return 0.2, 0.8

    # 중간 데이터(15~49건)
    if avg_ratio_cv > avg_price_cv * 1.5:
        return 0.4, 0.6
    elif avg_price_cv > avg_ratio_cv * 1.5:
        return 0.85, 0.15
    else:
        return 0.7, 0.3


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


def _dampened_extrapolation(
    base_value: float, slope: float, distance: float,
    min_slope: float | None = None,
) -> float:
    """
    거리 적응형 감쇠 외삽: 짧은 외삽은 빠른 감쇠, 긴 외삽은 추세 유지.

    base_value: 가장 가까운 구간의 값 (비율 또는 가격)
    slope: 만km당 기울기
    distance: 외삽 거리 (만km 단위, 양수)
    min_slope: 감쇠 하한 (비즈니스 기준 최소 기울기). 감쇠가 이 이하로 내려가지 않음.
    """
    if abs(distance) < 0.01:
        return base_value

    # 외삽 거리에 따라 감쇠율 조절
    # - 짧은 거리: 빠른 감쇠 (과외삽 방지)
    # - 긴 거리: 느린 감쇠 (추세 유지 → 비즈니스 기준에 부합)
    abs_dist = abs(distance)
    if abs_dist <= 5:
        decay = 0.75
    elif abs_dist <= 10:
        decay = 0.85
    else:
        decay = 0.92

    result = base_value
    remaining = abs_dist
    current_slope = slope

    while remaining > 0:
        step = min(remaining, 1.0)  # 1만km씩 적용
        result += current_slope * step * (1 if distance > 0 else -1)
        remaining -= step
        current_slope *= decay
        # 비즈니스 최소 기울기 이하로 감쇠 방지
        if min_slope is not None and slope < 0:
            current_slope = min(current_slope, min_slope)

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
    건수 + 합의(consensus) 가중 선형회귀.

    3개 이상 구간에서는 건수 가중 중앙값 대비 편차가 큰 구간의 가중치를 감소시켜
    이상치 구간(예: 극저주행 신차급)이 추세선을 왜곡하는 것을 방지.

    Returns: (slope, x_mean_w, y_mean_w)
    """
    n = len(xs)
    if n < 2:
        return 0, 0, 0

    effective_weights = list(weights)

    # 3개 이상 구간: 합의 가중치 적용
    if n >= 3:
        # 건수 가중 중앙값 산출
        yw_sorted = sorted(zip(ys, weights), key=lambda x: x[0])
        total_w = sum(w for _, w in yw_sorted)
        cumul = 0
        w_median = yw_sorted[0][0]
        for val, w in yw_sorted:
            cumul += w
            if cumul >= total_w / 2:
                w_median = val
                break

        if w_median > 0:
            effective_weights = []
            for y, w in zip(ys, weights):
                dev = abs(y - w_median) / w_median
                cw = max(0.1, 1 - dev * 3)
                effective_weights.append(w * cw)

    w_sum = sum(effective_weights)
    if w_sum == 0:
        return 0, 0, 0
    x_mean = sum(w * x for w, x in zip(effective_weights, xs)) / w_sum
    y_mean = sum(w * y for w, y in zip(effective_weights, ys)) / w_sum
    numerator = sum(w * (x - x_mean) * (y - y_mean) for w, x, y in zip(effective_weights, xs, ys))
    denominator = sum(w * (x - x_mean) ** 2 for w, x in zip(effective_weights, xs))
    if denominator == 0:
        return 0, x_mean, y_mean
    return numerator / denominator, x_mean, y_mean



def _build_details(
    result: RetailEstimateResult,
    maker: str, model: str, trim: str,
    year: int, mileage: int,
    vehicles: list[dict] | None = None,
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

    # ── 사용된 차량 목록 ──
    if vehicles:
        lines.append("")
        lines.append("── 사용 차량 목록 ──")
        sorted_v = sorted(vehicles, key=lambda v: int(v.get("주행거리", 0) or 0))
        for v in sorted_v:
            v_name = v.get("차명", "") or v.get("vehicleName", "") or ""
            v_trim = v.get("vehicleTrim", "") or v.get("trim", "") or ""
            v_year = v.get("연식", "") or ""
            v_mileage = int(v.get("주행거리", 0) or 0)
            v_retail = float(v.get("소매가", 0) or 0)
            fp = float(v.get("factory_price", 0) or 0)
            bp = float(v.get("base_price", 0) or 0)
            ref_label = f"출고{fp:,.0f}" if fp > 0 else (f"기본{bp:,.0f}" if bp > 0 else "기준가없음")
            v_color = v.get("색상", "") or ""
            v_accident = v.get("accident_summary", "") or ""
            fe = int(v.get("frame_exchange", 0) or 0)
            ee = int(v.get("exterior_exchange", 0) or 0)
            if not v_accident:
                if fe or ee:
                    v_accident = f"교환{fe+ee}"
                else:
                    v_accident = "정보없음"
            name_parts = [v_name]
            if v_trim and v_trim not in v_name:
                name_parts.append(v_trim)
            lines.append(
                f"  {' '.join(name_parts)} {v_year}년 "
                f"{v_mileage:,}km | 소매가 {v_retail:,.0f}만 | "
                f"{ref_label}만 | {v_color} | {v_accident}"
            )

    return "\n".join(lines)


# =========================================================================
# 피드백 기반 이상치 필터
# =========================================================================

def _apply_feedback_filter(
    price_data: list[tuple[int, float]],
    exclusion_pct: float,
    direction: str,
) -> tuple[list[tuple[int, float]], int]:
    """정규화된 가격 데이터에서 피드백 기반 이상치 제거"""
    if exclusion_pct <= 0 or len(price_data) < 5:
        return price_data, 0
    n_exclude = max(1, round(len(price_data) * exclusion_pct))
    sorted_data = sorted(price_data, key=lambda x: x[1])
    if direction == "up":
        filtered = sorted_data[n_exclude:]
    else:
        filtered = sorted_data[:-n_exclude]
    filtered.sort(key=lambda x: x[0])
    return filtered, n_exclude


# =========================================================================
# 학습 보정 (Gaussian 평활 후 적용)
# =========================================================================

def _apply_learned_correction(
    raw_price: float,
    mileage: int,
    scale_factor: float,
    price_bias: float,
    mileage_slope: float,
    ref_mileage: float,
) -> float:
    """Gaussian 평활 후 학습된 보정 적용."""
    delta_km = (mileage - ref_mileage) / 10000
    corrected = raw_price * (scale_factor + mileage_slope * delta_km) + price_bias
    return max(corrected, 0)


def _needs_learned_correction(
    scale_factor: float, price_bias: float, mileage_slope: float,
) -> bool:
    """학습 보정이 필요한지 판단."""
    return scale_factor != 1.0 or price_bias != 0 or mileage_slope != 0


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
    fuel: str = "",
    auction_brackets: list["MileageBracket"] | None = None,
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

    # 같은 트림 + 같은 연식 + 같은 연료 → 부족 시 트림 없이 + 연식 확대
    year_range_used = f"{year}년"
    vehicles = search_retail_db(
        model=model, maker=maker, trim=trim, fuel=fuel,
        year_min=year, year_max=year, limit=500,
    )
    vehicles = _remove_same_as_factory(vehicles, price_field="소매가")

    # 트림 매칭 실패 시 폴백: 트림 없이 + 연식 ±1 확대
    if len(vehicles) < MIN_TOTAL_VEHICLES:
        logger.info(
            "소매가 추정: 정확 트림(%s) 부족 %d건 → 트림 없이 재검색",
            trim, len(vehicles),
        )
        for y_delta in (0, 1, 2):
            vehicles = search_retail_db(
                model=model, maker=maker, trim=None, fuel=fuel,
                year_min=year - y_delta, year_max=year + y_delta, limit=500,
            )
            vehicles = _remove_same_as_factory(vehicles, price_field="소매가")
            if len(vehicles) >= MIN_TOTAL_VEHICLES:
                year_range_used = f"{year-y_delta}~{year+y_delta}년"
                break

    result.year_range_used = year_range_used
    result.vehicles_found = len(vehicles)
    logger.info(
        "소매가 추정: %s %s %s %s fuel=%s — %d건",
        maker, model, trim, result.year_range_used, fuel or "(전체)", len(vehicles),
    )

    if len(vehicles) < MIN_TOTAL_VEHICLES:
        result.method = "no_data"
        result.details = (
            f"같은 트림({trim}) 연식({result.year_range_used}) "
            f"소매 차량 {len(vehicles)}건 — 데이터 부족"
        )
        return result

    result.vehicles_found = len(vehicles)
    result.vehicles = vehicles
    result.quality_filter = f"전체 {len(vehicles)}건"

    # 1.5단계: 최근 데이터 우선 (3개월 → 6개월 → 12개월 → 전체)
    all_count = len(vehicles)
    vehicles, used_months = _filter_recent_staged(vehicles, MIN_TOTAL_VEHICLES, "매물등록일")
    if used_months > 0:
        logger.info("소매 최근 %d개월 필터: %d → %d건", used_months, all_count, len(vehicles))
    else:
        logger.info("소매 최근 데이터 부족 — 전체 %d건 사용", len(vehicles))

    # 1.7단계: 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 1.8단계: 사고차(교환/판금/도색) 계산 제외
    calc_vehicles = _filter_damaged_vehicles(vehicles)
    # 1.9단계: gap 기반 이상치 필터 (계산용만 — 표시는 전체)
    calc_vehicles = _filter_gap_outliers(calc_vehicles, price_field="소매가")
    # 비율 기반 이상치 제외 (평활 비율 대비 ±30% 초과)
    calc_vehicles = _filter_vehicles_by_local_ratio(
        calc_vehicles, tgt_ref_price, price_field="소매가",
    )

    # 2단계: 구간별 비율 산출 (구간에 포함된 차량만 수집)
    used_vehicles: list[dict] = []
    bracket_map = _build_brackets(calc_vehicles, tgt_ref_price, used_vehicles=used_vehicles, target_year=year)

    # 낙찰가 bracket을 바닥으로 소매 bracket 보정
    if auction_brackets:
        _enforce_auction_floor(bracket_map, auction_brackets)

    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 구간별 비율 추이 + 평활 가격 추정 (blend)
    if tgt_ref_price > 0:
        ratio, method = _interpolate_ratio(mileage, sorted_brackets, vehicle_year=year)
        ratio_price = round(ratio * tgt_ref_price, 1) if ratio > 0 else 0

        # 평활 가격 추정 (소매용: 시장 평균 — conservative=False)
        vehicle_class = _determine_vehicle_class(calc_vehicles)
        price_data = _normalize_vehicles_price_only(
            calc_vehicles, vehicle_class, price_field="소매가", full_normalize=True,
            target_year=year, tgt_ref_price=tgt_ref_price,
        )
        smooth_price = 0.0
        if price_data:
            price_data = _filter_local_outliers(price_data)
            smooth_price, smooth_method = _smooth_price_estimate(
                price_data, mileage, conservative=False, vehicle_year=year,
            )

        # blend: 비율 보간과 평활 가격 (둘 다 유효할 때)
        # 서브모델 혼재(fp CV > 10%) 시 smooth에 가중치 증가 (fp 정규화 반영)
        if smooth_method == "smooth_price_far_range" and smooth_price > 0:
            # 원거리 외삽: 비즈니스 룰 기반 smooth가 bracket 외삽보다 안정적
            est_price = _round10(smooth_price)
            result.estimated_retail = est_price
            result.estimated_ratio = round(smooth_price / tgt_ref_price * 100, 1) if tgt_ref_price > 0 else 0
            result.method = "smooth_price_far_range"
            result.success = True
            print(f"  [소매 far_range] smooth={smooth_price:.1f}, ratio={ratio_price:.1f} → smooth 우선 {est_price}")
            logger.info(
                "소매 원거리외삽: smooth=%.1f, ratio=%.1f → smooth 우선 %d",
                smooth_price, ratio_price, est_price,
            )
        elif ratio_price > 0 and smooth_price > 0 and smooth_method != "smooth_price_far_range":
            # fp 변동 계수로 서브모델 혼재 정도 측정
            fp_values = [float(v.get("factory_price", 0) or 0) for v in calc_vehicles if float(v.get("factory_price", 0) or 0) > 0]
            fp_cv = (statistics.stdev(fp_values) / statistics.mean(fp_values)) if len(fp_values) >= 2 and statistics.mean(fp_values) > 0 else 0
            divergence = abs(ratio_price - smooth_price) / smooth_price if smooth_price > 0 else 0
            if fp_cv > 0.05 and divergence > 0.15:
                # 서브모델 혼재 + 큰 괴리 → smooth만 사용 (fp 정규화 반영)
                ratio_w, smooth_w = 0.0, 1.0
            elif fp_cv > 0.05:
                # 서브모델 혼재 → smooth 가중 (30:70)
                ratio_w, smooth_w = 0.30, 0.70
            else:
                ratio_w, smooth_w = 0.50, 0.50
            est_price = _round10(ratio_price * ratio_w + smooth_price * smooth_w)
            result.estimated_retail = est_price
            result.estimated_ratio = round(est_price / tgt_ref_price * 100, 1)
            result.method = f"{method}+smooth_blend"
            result.success = True
            logger.info(
                "소매 blend: ratio=%.1f, smooth=%.1f, fp_cv=%.3f, div=%.3f, w=%.0f:%.0f → %d",
                ratio_price, smooth_price, fp_cv, divergence, ratio_w * 100, smooth_w * 100, est_price,
            )
        elif ratio_price > 0:
            est_price = _round10(ratio_price)
            result.estimated_retail = est_price
            result.estimated_ratio = round(ratio * 100, 1)
            result.method = method
            result.success = True
        elif smooth_price > 0:
            result.estimated_retail = _round10(smooth_price)
            result.estimated_ratio = round(smooth_price / tgt_ref_price * 100, 1)
            result.method = "smooth_price_regression"
            result.success = True
    else:
        # 출고가 없으면 가격 추이로 직접 추정
        est_price, method = _interpolate_price(mileage, sorted_brackets)
        if est_price > 0:
            result.estimated_retail = _round10(est_price)
            result.method = method
            result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    if result.vehicles_found >= 10 and target_bracket and target_bracket.count >= 3:
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and len(sorted_brackets) >= 2:
        result.confidence = "보통"
    else:
        result.confidence = "낮음"

    result.details = _build_details(result, maker, model, trim, year, mileage, vehicles)
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
    feedback_excluded: int = 0
    learned_correction_applied: bool = False
    blended_params: "LearnedParams | None" = None
    success: bool = False
    vehicles: list[dict] = field(default_factory=list)  # 사용된 전체 차량


def _build_auction_brackets(
    vehicles: list[dict],
    tgt_ref_price: float,
    target_year: int = 0,
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

        # 비율 계산: 차량 자체 출고가 기준만 사용
        # fp=0 차량은 다른 서브모델일 수 있어 비율 계산에서 제외
        v_fp = float(v.get("factory_price", 0) or 0)
        if v_fp > 0:
            ratio = v_auction_aa / v_fp
            if ratio < 1.0:  # 비율 >= 100% 는 비정상 데이터 제외
                b.ratios.append(ratio)
                if _is_clean_vehicle(v):
                    clean_ratios_map[key].append(ratio)

    # 이상치 필터링 + effective_ratio 결정
    # 10년 이내: 하위 25%, 그 외: 하위 40%
    current_year = datetime.now().year
    age = current_year - target_year if target_year > 0 else 99
    lower_pct = 0.25 if age <= 10 else 0.40

    for b in brackets.values():
        b.count = len(b.prices)
        key = b.bracket_start

        # 가격 이상치 필터링
        filtered_prices = _filter_outliers(b.prices) if b.prices else []
        if filtered_prices:
            mean_price = statistics.mean(filtered_prices)
            if len(filtered_prices) > 1:
                b.price_cv = statistics.stdev(filtered_prices) / mean_price if mean_price > 0 else 0
            b.median_price = _lower_pct_mean(filtered_prices, lower_pct)

        # 비율 이상치 필터링
        b.ratios = _filter_outliers(b.ratios) if b.ratios else []
        clean_raw = clean_ratios_map.get(key, [])
        clean = _filter_outliers(clean_raw) if clean_raw else []

        use_ratios = clean if clean else b.ratios
        if use_ratios:
            if len(b.ratios) > 1:
                b.ratio_cv = statistics.stdev(b.ratios) / statistics.mean(b.ratios) if statistics.mean(b.ratios) > 0 else 0
            b.median_ratio = _lower_pct_mean(use_ratios, lower_pct)

        # effective_ratio
        if use_ratios:
            b.effective_ratio = b.median_ratio
        elif tgt_ref_price > 0 and b.median_price > 0:
            b.effective_ratio = b.median_price / tgt_ref_price
            b.smoothed = True

    _smooth_cross_bin_outliers(brackets)
    return brackets


def estimate_auction_by_market(
    maker: str,
    model: str,
    trim: str,
    year: int,
    mileage: int,
    factory_price: float = 0,
    base_price: float = 0,
    fuel: str = "",
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

    # 같은 트림 + 같은 연식 + 같은 연료 → 부족 시 트림 없이 + 연식 확대
    year_range_used = f"{year}년"
    vehicles = search_auction_db(
        model=model, maker=maker, trim=trim, fuel=fuel,
        year_min=year, year_max=year, limit=500,
        domestic_only=True,
    )
    vehicles = _remove_same_as_factory(vehicles, price_field="낙찰가")

    # 트림 매칭 실패 시 폴백: 트림 없이 + 연식 ±1 확대
    if len(vehicles) < MIN_TOTAL_VEHICLES:
        logger.info(
            "낙찰가 추정: 정확 트림(%s) 부족 %d건 → 트림 없이 재검색",
            trim, len(vehicles),
        )
        for y_delta in (0, 1, 2):
            vehicles = search_auction_db(
                model=model, maker=maker, trim=None, fuel=fuel,
                year_min=year - y_delta, year_max=year + y_delta, limit=500,
                domestic_only=True,
            )
            vehicles = _remove_same_as_factory(vehicles, price_field="낙찰가")
            if len(vehicles) >= MIN_TOTAL_VEHICLES:
                year_range_used = f"{year-y_delta}~{year+y_delta}년"
                break

    result.year_range_used = year_range_used
    result.vehicles_found = len(vehicles)
    result.vehicles = vehicles
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

    # 최근 데이터 우선 (3개월 → 6개월 → 12개월 → 전체)
    all_count = len(vehicles)
    vehicles, used_months = _filter_recent_staged(vehicles, MIN_TOTAL_VEHICLES, "개최일")
    if used_months > 0:
        logger.info("낙찰 최근 %d개월 필터: %d → %d건", used_months, all_count, len(vehicles))
    else:
        logger.info("낙찰 최근 데이터 부족 — 전체 %d건 사용", len(vehicles))

    # 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 사고차(교환/판금/도색) 계산 제외
    calc_vehicles = _filter_damaged_vehicles(vehicles)
    # gap 기반 이상치 필터 (계산용만 — 표시는 전체)
    calc_vehicles = _filter_gap_outliers(calc_vehicles, price_field="낙찰가")
    # 비율 기반 이상치 제외 (평활 비율 대비 ±30% 초과)
    calc_vehicles = _filter_vehicles_by_local_ratio(
        calc_vehicles, tgt_ref_price, price_field="낙찰가",
    )

    # 2단계: 구간별 비율 산출 (평균 기반)
    bracket_map = _build_auction_brackets(calc_vehicles, tgt_ref_price, target_year=year)
    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 구간별 비율 추이 + 평활 가격 추정 (소매와 동일 blend 구조)
    # 3-1) bracket 기반 비율 보간
    ratio_price = 0.0
    ratio_method = ""
    if tgt_ref_price > 0:
        ratio, ratio_method = _interpolate_auction_ratio(mileage, sorted_brackets, vehicle_year=year)
        ratio_price = round(ratio * tgt_ref_price, 1) if ratio > 0 else 0

    # 3-2) 평활 가격 추정
    vehicle_class = _determine_vehicle_class(calc_vehicles)
    price_data = _normalize_vehicles_price_only(
        calc_vehicles, vehicle_class, price_field="낙찰가", full_normalize=False,
        target_year=year, tgt_ref_price=tgt_ref_price,
    )

    smooth_price = 0.0
    smooth_method = ""
    # 경매: 하위 10%(≤10yr) / 20%(>10yr) — 소매보다 보수적
    current_year = datetime.now().year
    auction_age = current_year - year if year > 0 else 99
    auction_pct = 0.10 if auction_age <= 10 else 0.20
    if price_data:
        smooth_price, smooth_method = _smooth_price_estimate(
            price_data, mileage, conservative=True, vehicle_year=year,
            conservative_pct=auction_pct,
        )

    # 3-3) blend: 서브모델 혼재(fp CV > 5%) 시 smooth 가중
    if ratio_price > 0 and smooth_price > 0 and smooth_method != "smooth_price_far_range":
        fp_values = [float(v.get("factory_price", 0) or 0) for v in calc_vehicles if float(v.get("factory_price", 0) or 0) > 0]
        fp_cv = (statistics.stdev(fp_values) / statistics.mean(fp_values)) if len(fp_values) >= 2 and statistics.mean(fp_values) > 0 else 0
        divergence = abs(ratio_price - smooth_price) / smooth_price if smooth_price > 0 else 0
        if fp_cv > 0.05 and divergence > 0.15:
            ratio_w, smooth_w = 0.0, 1.0
        elif fp_cv > 0.05:
            ratio_w, smooth_w = 0.30, 0.70
        else:
            ratio_w, smooth_w = 0.50, 0.50
        est_price = _round10(ratio_price * ratio_w + smooth_price * smooth_w)
        result.estimated_auction = est_price
        if tgt_ref_price > 0:
            result.estimated_ratio = round(est_price / tgt_ref_price * 100, 1)
        result.method = f"{ratio_method}+smooth_blend"
        result.success = True
        logger.info(
            "낙찰 blend: ratio=%.1f, smooth=%.1f, fp_cv=%.3f, div=%.3f, w=%.0f:%.0f → %d",
            ratio_price, smooth_price, fp_cv, divergence, ratio_w * 100, smooth_w * 100, est_price,
        )
    elif smooth_method == "smooth_price_far_range" and smooth_price > 0:
        # 원거리 외삽: 비율 외삽보다 비즈니스 룰 기반 평활이 더 안정적
        result.estimated_auction = _round10(smooth_price)
        if tgt_ref_price > 0:
            result.estimated_ratio = round(smooth_price / tgt_ref_price * 100, 1)
        result.method = smooth_method
        result.success = True
        print(f"  [낙찰 far_range] smooth={smooth_price:.1f}, ratio={ratio_price:.1f} → smooth 우선 {result.estimated_auction}")
    elif ratio_price > 0:
        result.estimated_auction = _round10(ratio_price)
        if tgt_ref_price > 0:
            result.estimated_ratio = round(ratio_price / tgt_ref_price * 100, 1)
        result.method = ratio_method
        result.success = True
    elif smooth_price > 0:
        result.estimated_auction = _round10(smooth_price)
        if tgt_ref_price > 0:
            result.estimated_ratio = round(smooth_price / tgt_ref_price * 100, 1)
        result.method = smooth_method
        result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    if result.vehicles_found >= 10 and target_bracket and target_bracket.count >= 3:
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and len(sorted_brackets) >= 2:
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
        if b.prices:
            p_min, p_max = min(b.prices), max(b.prices)
            price_range = f"{p_min:,.0f}만" if p_min == p_max else f"{p_min:,.0f}~{p_max:,.0f}만"
            display_price = b.median_price if b.median_price > 0 else statistics.median(b.prices)
            if result.reference_price_used > 0:
                display_ratio = display_price / result.reference_price_used * 100
                ratio_pct = f"{display_ratio:.1f}%"
            else:
                ratio_pct = f"{b.effective_ratio * 100:.1f}%"
        else:
            price_range = "-"
            ratio_pct = f"{b.effective_ratio * 100:.1f}%"
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

    # ── 사용된 차량 목록 ──
    if vehicles:
        lines.append("")
        lines.append("── 사용 차량 목록 ──")
        sorted_v = sorted(vehicles, key=lambda v: int(v.get("주행거리", 0) or 0))
        for v in sorted_v:
            v_name = v.get("차명", "") or ""
            v_trim = v.get("trim", "") or ""
            v_year = v.get("연식", "") or ""
            v_mileage = int(v.get("주행거리", 0) or 0)
            v_price = _to_man_won(float(v.get("낙찰가", 0) or 0))
            fp = float(v.get("factory_price", 0) or 0)
            bp = float(v.get("base_price", 0) or 0)
            ref_label = f"출고{fp:,.0f}" if fp > 0 else (f"기본{bp:,.0f}" if bp > 0 else "기준가없음")
            v_color = v.get("색상", "") or ""
            v_date = v.get("개최일", "") or ""
            name_parts = [v_name]
            if v_trim and v_trim not in v_name:
                name_parts.append(v_trim)
            lines.append(
                f"  {' '.join(name_parts)} {v_year}년 "
                f"{v_mileage:,}km | 낙찰가 {v_price:,.0f}만 | "
                f"{ref_label}만 | {v_color} | {v_date}"
            )

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
    fuel: str = "",
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

    # 같은 트림 + 같은 연식 + 같은 연료 수출만 (연식 확장 없음)
    year_range_used = f"{year}년"
    vehicles = search_auction_db(
        model=model, maker=maker, trim=trim, fuel=fuel,
        year_min=year, year_max=year, limit=500,
        domestic_only=False, export_only=True,
    )
    result.year_range_used = year_range_used
    vehicles = _remove_same_as_factory(vehicles, price_field="낙찰가")

    result.vehicles_found = len(vehicles)
    result.vehicles = vehicles
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

    # 최근 데이터 우선 (3개월 → 6개월 → 12개월 → 전체)
    all_count = len(vehicles)
    vehicles, used_months = _filter_recent_staged(vehicles, MIN_TOTAL_VEHICLES, "개최일")
    if used_months > 0:
        logger.info("수출 낙찰 최근 %d개월 필터: %d → %d건", used_months, all_count, len(vehicles))
    else:
        logger.info("수출 낙찰 최근 데이터 부족 — 전체 %d건 사용", len(vehicles))

    # 대상 출고가 없으면 DB 조회 → 기준차량 중앙값 폴백
    if tgt_ref_price == 0:
        tgt_ref_price, tgt_ref_label = _resolve_target_ref_price(
            maker, model, trim, vehicles,
        )
        if tgt_ref_price > 0:
            result.reference_price_used = tgt_ref_price
            result.reference_price_label = tgt_ref_label

    # 사고차(교환/판금/도색) 계산 제외
    calc_vehicles = _filter_damaged_vehicles(vehicles)
    # gap 기반 이상치 필터 (계산용만 — 표시는 전체)
    calc_vehicles = _filter_gap_outliers(calc_vehicles, price_field="낙찰가")
    # 비율 기반 이상치 제외 (평활 비율 대비 ±30% 초과)
    calc_vehicles = _filter_vehicles_by_local_ratio(
        calc_vehicles, tgt_ref_price, price_field="낙찰가",
    )

    # 2단계: 구간별 비율 산출
    bracket_map = _build_auction_brackets(calc_vehicles, tgt_ref_price, target_year=year)
    sorted_brackets = sorted(bracket_map.values(), key=lambda x: x.bracket_start)
    result.brackets = sorted_brackets

    if not sorted_brackets:
        result.method = "no_data"
        result.details = "유효 구간 생성 실패"
        return result

    # 3단계: 연속 평활 추정 (보수적: 저가 차량 가중 + 연식 정규화)
    vehicle_class = _determine_vehicle_class(calc_vehicles)
    price_data = _normalize_vehicles_price_only(
        calc_vehicles, vehicle_class, price_field="낙찰가", full_normalize=False,
        target_year=year, tgt_ref_price=tgt_ref_price,
    )

    if price_data:
        price_data = _filter_local_outliers(price_data)
        from app.services.calibration_engine import compute_blended_params
        blended = compute_blended_params(
            price_data, mileage, maker, model, trim, year,
            price_type="export", conservative=True,
        )
        result.blended_params = blended

        # 사전 필터
        if blended.exclusion_pct > 0 and blended.direction != "none":
            price_data, result.feedback_excluded = _apply_feedback_filter(
                price_data, blended.exclusion_pct, blended.direction,
            )

        est_price, method = _smooth_price_estimate(
            price_data, mileage, conservative=True, vehicle_year=year,
        )
        if est_price > 0:
            # 사후 보정
            if _needs_learned_correction(blended.scale_factor, blended.price_bias, blended.mileage_slope):
                est_price = _apply_learned_correction(
                    est_price, mileage,
                    blended.scale_factor, blended.price_bias,
                    blended.mileage_slope, blended.ref_mileage,
                )
                result.learned_correction_applied = True
            result.estimated_auction = _round10(est_price)
            if tgt_ref_price > 0:
                result.estimated_ratio = round(est_price / tgt_ref_price * 100, 1)
            result.method = method
            result.success = True

    # 신뢰도
    target_bracket = bracket_map.get(_bracket_key(mileage))
    if result.vehicles_found >= 10 and target_bracket and target_bracket.count >= 3:
        result.confidence = "높음"
    elif result.vehicles_found >= 5 and len(sorted_brackets) >= 2:
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
        if b.prices:
            p_min, p_max = min(b.prices), max(b.prices)
            price_range = f"{p_min:,.0f}만" if p_min == p_max else f"{p_min:,.0f}~{p_max:,.0f}만"
        else:
            price_range = "-"
        ratio_pct = f"{b.effective_ratio * 100:.1f}%"
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

    # ── 사용된 차량 목록 ──
    if vehicles:
        lines.append("")
        lines.append("── 사용 차량 목록 ──")
        sorted_v = sorted(vehicles, key=lambda v: int(v.get("주행거리", 0) or 0))
        for v in sorted_v:
            v_name = v.get("차명", "") or ""
            v_trim = v.get("trim", "") or ""
            v_year = v.get("연식", "") or ""
            v_mileage = int(v.get("주행거리", 0) or 0)
            v_price = _to_man_won(float(v.get("낙찰가", 0) or 0))
            fp = float(v.get("factory_price", 0) or 0)
            bp = float(v.get("base_price", 0) or 0)
            ref_label = f"출고{fp:,.0f}" if fp > 0 else (f"기본{bp:,.0f}" if bp > 0 else "기준가없음")
            v_color = v.get("색상", "") or ""
            v_date = v.get("개최일", "") or ""
            name_parts = [v_name]
            if v_trim and v_trim not in v_name:
                name_parts.append(v_trim)
            lines.append(
                f"  {' '.join(name_parts)} {v_year}년 "
                f"{v_mileage:,}km | 수출낙찰가 {v_price:,.0f}만 | "
                f"{ref_label}만 | {v_color} | {v_date}"
            )

    result.details = "\n".join(lines)

    return result


def _interpolate_auction_ratio(
    target_mileage: int,
    sorted_brackets: list[MileageBracket],
    vehicle_year: int = 0,
) -> tuple[float, str]:
    """낙찰가 비율 보간 (직접 구간은 추세선과 평균하여 안정화)."""
    usable = [b for b in sorted_brackets if b.effective_ratio > 0]
    if not usable:
        return 0, "no_data"

    target_key = _bracket_key(target_mileage)
    target_mid = target_key + 5000

    # 1) 정확한 구간
    exact = next((b for b in usable if b.bracket_start == target_key), None)
    if exact:
        total_vehicles = sum(b.count for b in usable)
        if len(usable) <= 3 or total_vehicles <= 6:
            return exact.effective_ratio, "ratio_direct"
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

    # 3) 양쪽 → 격차 크면 추세선, 아니면 역전 감지 + 건수 가중 보간
    if lower and upper:
        lb = lower[-1]
        ub = upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        gap = ub_mid - lb_mid
        if gap > 20000 and len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.effective_ratio for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            ratio = y_mean + slope * (target_mid / 10000 - x_mean)
            if ratio > 0:
                return ratio, "ratio_trend_interpolation"
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        if lb.effective_ratio < ub.effective_ratio and lb.count < ub.count:
            ratio = ub.effective_ratio
        elif ub.effective_ratio > lb.effective_ratio and ub.count < lb.count:
            ratio = lb.effective_ratio
        else:
            lb_w = (1 - t) * max(lb.count, 1)
            ub_w = t * max(ub.count, 1)
            ratio = (lb.effective_ratio * lb_w + ub.effective_ratio * ub_w) / (lb_w + ub_w)
        return ratio, "ratio_interpolation"

    # 4) 건수 가중 추세선에서 감쇠 외삽 (소매와 동일하게 거리 제한 없음)
    if len(usable) >= 2:
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        distance = target_mid_10k - nearest_mid_10k
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.effective_ratio for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.effective_ratio
        # 비즈니스 기준 최소 기울기 적용
        biz_min_slope = None
        if vehicle_year > 0 and distance > 0:
            current_year = datetime.now().year
            age = current_year - vehicle_year
            rate = _mileage_depreciation_rate(age, trend_at_nearest * 4000, target_mileage)
            biz_min_slope = -rate * trend_at_nearest
            if slope > biz_min_slope:
                slope = biz_min_slope
        ratio = _dampened_extrapolation(trend_at_nearest, slope, distance, min_slope=biz_min_slope)
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
        total_vehicles = sum(b.count for b in usable)
        if len(usable) <= 3 or total_vehicles <= 6:
            return exact.median_price, "price_direct"
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
        lb = lower[-1]
        ub = upper[0]
        lb_mid = lb.bracket_start + 5000
        ub_mid = ub.bracket_start + 5000
        gap = ub_mid - lb_mid
        if gap > 20000 and len(usable) >= 2:
            xs = [(b.bracket_start + 5000) / 10000 for b in usable]
            ys = [b.median_price for b in usable]
            ws = [max(b.count, 1) for b in usable]
            slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
            price = y_mean + slope * (target_mid / 10000 - x_mean)
            if price > 0:
                return price, "price_trend_interpolation"
        t = (target_mid - lb_mid) / (ub_mid - lb_mid) if ub_mid != lb_mid else 0.5
        if lb.median_price < ub.median_price and lb.count < ub.count:
            price = ub.median_price
        elif ub.median_price > lb.median_price and ub.count < lb.count:
            price = lb.median_price
        else:
            lb_w = (1 - t) * max(lb.count, 1)
            ub_w = t * max(ub.count, 1)
            price = (lb.median_price * lb_w + ub.median_price * ub_w) / (lb_w + ub_w)
        return max(price, 0), "price_interpolation"

    if len(usable) >= 2:
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        distance = target_mid_10k - nearest_mid_10k
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.median_price for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.median_price
        price = _dampened_extrapolation(trend_at_nearest, slope, distance)
        return max(price, 0), "price_trend_extrapolation"

    return usable[0].median_price, "price_nearest"
