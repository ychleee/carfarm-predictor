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

import asyncio
import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

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
) -> list[tuple[int, float]]:
    """
    가격을 흰색·무사고로 정규화. Returns: [(mileage, price)] 주행거리 오름차순.

    full_normalize=True (소매): 사고/색상 보정을 연식 감쇠 없이 full 적용
    full_normalize=False (경매): 사고 보정에 연식 가중치 적용 (과보정 방지)
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
        "정규화 완료: %s %d건 → %d건 (class=%s, full=%s)",
        price_field, len(vehicles), len(results), vehicle_class, full_normalize,
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


def _adaptive_bandwidth(data: list[tuple], target_mileage: int) -> int:
    """데이터 밀도에 따른 적응형 bandwidth.

    sqrt(n) 기반 최근접 이웃: 데이터가 많을수록 좁게, 적을수록 넓게.
    """
    n = len(data)
    if n <= 3:
        return 40000
    distances = sorted(abs(d[0] - target_mileage) for d in data)
    k = min(max(int(math.sqrt(n)), 5), 25)  # sqrt(n), 5~25 범위
    k = min(k, n - 1)
    bw = max(distances[k], 10000)  # 최소 10,000km
    return min(bw, 40000)  # 최대 40,000km


def _smooth_price_estimate(
    data: list[tuple[int, float]],
    target_mileage: int,
    bandwidth: int = 0,
    conservative: bool = False,
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
    dist_weights = _gaussian_weights(mileages, target_mileage, bandwidth)

    weights = dist_weights

    if conservative:
        # 보수적 추정: 추세선 기울기 유지 + 하위 40% 잔차 적용
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

            # 3) 가중 하위 35% 잔차 평균
            residuals_weights.sort(key=lambda x: x[0])
            total_w = sum(w for _, w in residuals_weights)
            target_w = total_w * 0.35
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
            print(f"  [보수] reg={reg_price:.1f}, w_mean={w_mean:.1f}, base={base_price:.1f}, lower_resid={lower_residual:.1f} → {price:.1f}")
        else:
            price = 0
    else:
        price = _weighted_local_regression(
            [float(m) for m in mileages], prices, weights, float(target_mileage),
        )

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
                price = eff_min

    # 디버그 로깅
    mode = "보수" if conservative else "소매"
    print(f"[평활 {mode}] target={target_mileage}km, n={len(data)}, bw={bandwidth} → {price:.1f}")

    method = "smooth_price_conservative" if conservative else "smooth_price_regression"
    return max(round(price, 1), 0), method


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

    # 3단계: 연속 평활 추정 (흰색·무사고 정규화 + 가격 직접 Gaussian 평활)
    vehicle_class = _determine_vehicle_class(vehicles)
    price_data = _normalize_vehicles_price_only(
        vehicles, vehicle_class, price_field="소매가",
    )
    if price_data:
        est_price, method = _smooth_price_estimate(price_data, mileage)
        if est_price > 0:
            result.estimated_retail = est_price
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

    # 이상치 필터링 + effective_ratio 결정 (낙찰: 항상 하위 40% — 경쟁입찰 특성)
    for b in brackets.values():
        b.count = len(b.prices)
        key = b.bracket_start

        # 가격 이상치 필터링 → 낙찰: 항상 하위 40% 사용
        filtered_prices = _filter_outliers(b.prices) if b.prices else []
        if filtered_prices:
            mean_price = statistics.mean(filtered_prices)
            if len(filtered_prices) > 1:
                b.price_cv = statistics.stdev(filtered_prices) / mean_price if mean_price > 0 else 0
            b.median_price = _lower_pct_mean(filtered_prices, 0.4)

        # 비율 이상치 필터링 → 낙찰: 항상 하위 40% 사용
        b.ratios = _filter_outliers(b.ratios) if b.ratios else []
        clean_raw = clean_ratios_map.get(key, [])
        clean = _filter_outliers(clean_raw) if clean_raw else []

        use_ratios = clean if clean else b.ratios
        if use_ratios:
            if len(b.ratios) > 1:
                b.ratio_cv = statistics.stdev(b.ratios) / statistics.mean(b.ratios) if statistics.mean(b.ratios) > 0 else 0
            b.median_ratio = _lower_pct_mean(use_ratios, 0.4)

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

    # 3단계: 연속 평활 추정 (보수적: 저가 차량 가중)
    vehicle_class = _determine_vehicle_class(vehicles)
    price_data = _normalize_vehicles_price_only(
        vehicles, vehicle_class, price_field="낙찰가", full_normalize=False,
    )
    if price_data:
        est_price, method = _smooth_price_estimate(
            price_data, mileage, conservative=True,
        )
        if est_price > 0:
            result.estimated_auction = est_price
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

    # 3단계: 연속 평활 추정 (보수적: 저가 차량 가중)
    vehicle_class = _determine_vehicle_class(vehicles)
    price_data = _normalize_vehicles_price_only(
        vehicles, vehicle_class, price_field="낙찰가", full_normalize=False,
    )
    if price_data:
        est_price, method = _smooth_price_estimate(
            price_data, mileage, conservative=True,
        )
        if est_price > 0:
            result.estimated_auction = est_price
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

    # 4) 건수 가중 추세선에서 감쇠 외삽
    if len(usable) >= 2:
        nearest = lower[-1] if lower else upper[0]
        nearest_mid_10k = (nearest.bracket_start + 5000) / 10000
        target_mid_10k = target_mid / 10000
        distance = target_mid_10k - nearest_mid_10k
        # 외삽 거리 5만km 초과 시 신뢰도 부족 → 포기
        if abs(distance) > 5.0:
            return 0, "no_data"
        xs = [(b.bracket_start + 5000) / 10000 for b in usable]
        ys = [b.effective_ratio for b in usable]
        ws = [max(b.count, 1) for b in usable]
        slope, x_mean, y_mean = _calc_weighted_slope(xs, ys, ws)
        trend_at_nearest = y_mean + slope * (nearest_mid_10k - x_mean)
        if trend_at_nearest <= 0:
            trend_at_nearest = nearest.effective_ratio
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
        # 외삽 거리 5만km 초과 시 신뢰도 부족 → 포기
        if abs(distance) > 5.0:
            return 0, "no_data"
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
