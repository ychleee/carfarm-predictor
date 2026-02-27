"""
CarFarm v2.3 — Firestore 기반 차량 DB

아이작 Firestore `vehicles` 컬렉션에서 차량 검색, 상세 조회, 시세 통계 제공.
기존 auction_db.py와 동일한 함수 인터페이스를 유지하여 호출자 변경 최소화.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1.base_query import FieldFilter

from app.services.firestore_client import get_firestore_db
from app.services.rule_engine import normalize_color


# =========================================================================
# 내부 헬퍼
# =========================================================================

def _safe_number(val, default=0) -> float | int:
    """문자열/None → 숫자"""
    if val is None:
        return default
    try:
        return float(val) if isinstance(val, str) else val
    except (ValueError, TypeError):
        return default


_USAGE_MAP = {
    "자가용": "personal",
    "렌터카": "rental",
    "영업용": "commercial",
    "관용": "commercial",
    "일반": "personal",
}


def _ts_to_iso(val) -> str:
    """Firestore timestamp / datetime → ISO 날짜 문자열"""
    if val is None:
        return ""
    if hasattr(val, "isoformat"):
        return val.isoformat()[:10]
    return str(val)[:10]


def _to_legacy_dict(doc_id: str, data: dict) -> dict:
    """Firestore doc → 기존 auction_db 반환 형식 (한/영 혼합 키)"""
    # 옵션: [{name, price}] → "선루프, 네비, ..." 문자열
    options_raw = data.get("vehicleOptions") or []
    if isinstance(options_raw, list) and options_raw and isinstance(options_raw[0], dict):
        options_str = ", ".join(
            item.get("name", "") for item in options_raw if item.get("name")
        )
    elif isinstance(options_raw, list):
        options_str = ", ".join(str(o) for o in options_raw if o)
    else:
        options_str = str(options_raw) if options_raw else ""

    purpose = data.get("vehiclePurpose") or ""
    usage_type = _USAGE_MAP.get(purpose, purpose or "personal")

    sale_dest = data.get("saleDestination") or ""
    is_export = 1 if "수출" in sale_dest else 0

    color = data.get("vehicleColor") or ""

    sale_date = data.get("saleDate") or data.get("createdAt")

    # 기본가 / 출고가 (원 → 만원 변환)
    raw_base = _safe_number(data.get("vehicleBasePrice"))
    raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
    base_price = round(raw_base / 10000, 1) if raw_base > 10000 else raw_base
    factory_price = round(raw_factory / 10000, 1) if raw_factory > 10000 else raw_factory

    # partDamages 추출
    part_damages_raw = data.get("partDamages") or []
    part_damages = []
    for pd in part_damages_raw:
        if isinstance(pd, dict):
            part = pd.get("part", "")
            dt = pd.get("damageType") or pd.get("damage_type", "")
            if part and dt:
                part_damages.append({"part": part, "damage_type": dt})

    # exchange_count / bodywork_count를 part_damages에서 재계산
    exchange_types = {"EXCHANGE"}
    bodywork_types = {"PAINT_PANEL_BEATING", "PANEL_WELDING"}
    exchange_count = sum(1 for d in part_damages if d["damage_type"] in exchange_types)
    bodywork_count = sum(1 for d in part_damages if d["damage_type"] in bodywork_types)

    return {
        # 한글 키 (기존 호출자 호환)
        "auction_id": doc_id,
        "차명": data.get("vehicleName") or data.get("title") or "",
        "제작사": data.get("vehicleMaker") or "",
        "모델": data.get("vehicleModel") or "",
        "연식": int(_safe_number(data.get("vehicleYear"))),
        "주행거리": int(_safe_number(data.get("mileage"))),
        "낙찰가": _safe_number(data.get("actualBidPrice")),
        "색상": color,
        "옵션": options_str,
        "개최일": _ts_to_iso(sale_date),
        "연료": data.get("fuelType") or "",
        "변속기": data.get("transmissionType") or "",
        "배기량": data.get("engineDisplacement"),
        "평가점": data.get("inspectionGrade") or "",
        "차량경력": purpose,
        # 영문 키 (기존 호출자 호환)
        "maker": data.get("vehicleMaker") or "",
        "model_name": data.get("vehicleModel") or "",
        "generation": data.get("generation") or "",
        "fuel": data.get("fuelType") or "",
        "drive": data.get("driveType") or "",
        "trim": data.get("vehicleTrim") or "",
        "usage_type": usage_type,
        "is_export": is_export,
        "segment": data.get("vehicleCategory") or "",
        "color_group": normalize_color(color),
        "exchange_count": exchange_count,
        "bodywork_count": bodywork_count,
        "part_damages": part_damages,
        "grade_score": data.get("inspectionGrade") or "",
        "accident_severity": "unknown",
        "base_price": base_price,
        "factory_price": factory_price,
    }


def _match_contains(value: str | None, keyword: str | None) -> bool:
    """대소문자 무시 부분 문자열 매칭"""
    if not keyword:
        return True
    if not value:
        return False
    return keyword.lower() in value.lower()


# =========================================================================
# 옵션 단가 추정 — 같은 차종의 출고가-기본가 데이터에서 역산
# =========================================================================

_option_unit_cache: dict[str, float] = {}


def estimate_option_unit_price(maker: str, model: str) -> float:
    """
    같은 maker+model 차량 중 출고가/기본가 데이터가 있는 차량에서
    옵션 단가(만원/개)를 추정한다. 결과는 세션 내 캐시.

    계산: avg( (출고가 - 기본가) / 옵션 개수 )  (양수인 것만)
    """
    cache_key = f"{maker}:{model}"
    if cache_key in _option_unit_cache:
        return _option_unit_cache[cache_key]

    db = get_firestore_db()
    col = db.collection("vehicles")
    query = col.where(filter=FieldFilter("vehicleMaker", "==", maker))
    query = query.where(filter=FieldFilter("vehicleModel", "==", model))

    docs = query.limit(500).get()

    unit_prices = []
    for doc in docs:
        data = doc.to_dict()
        bp = _safe_number(data.get("vehicleBasePrice"))
        fp = _safe_number(data.get("vehicleFactoryPrice"))
        if bp <= 0 or fp <= 0:
            continue

        # 원 → 만원
        bp_man = bp / 10000 if bp > 10000 else bp
        fp_man = fp / 10000 if fp > 10000 else fp

        option_total = fp_man - bp_man
        if option_total <= 0:
            continue  # 할인(음수)은 무시

        opts = data.get("vehicleOptions") or []
        n_opts = len([o for o in opts if isinstance(o, dict) and o.get("name")])
        if n_opts == 0:
            continue

        unit_prices.append(option_total / n_opts)

    result = round(statistics.mean(unit_prices), 1) if unit_prices else 0
    _option_unit_cache[cache_key] = result
    return result


# =========================================================================
# 공개 API — auction_db.py와 동일한 시그니처
# =========================================================================

def search_auction_db(
    maker: str,
    model: str,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    drive: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    usage: str | None = None,
    domestic_only: bool = True,
    limit: int = 30,
    sort_by: str = "날짜",
) -> list[dict]:
    """
    Firestore vehicles 컬렉션에서 조건 검색.

    Firestore 쿼리로 maker + model 필터링 후,
    나머지 조건은 Python 후처리로 적용 (Firestore 복합 쿼리 제한 대응).
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    # 1차: Firestore 쿼리 (maker + model만 — 복합인덱스 불필요)
    # 정렬과 추가 필터는 Python 후처리로 수행
    query = col.where(filter=FieldFilter("vehicleMaker", "==", maker))
    query = query.where(filter=FieldFilter("vehicleModel", "==", model))

    # 넉넉히 가져와서 후처리 (최대 500건)
    fetch_limit = min(limit * 10, 500)
    docs = query.limit(fetch_limit).get()

    # 2차: Python 후처리 필터
    results = []
    for doc in docs:
        data = doc.to_dict()

        # 유효 낙찰가 필터
        price = data.get("actualBidPrice")
        try:
            price = float(price) if price else 0
        except (ValueError, TypeError):
            price = 0
        if price <= 0:
            continue

        # 연식 범위 (Firestore에서 문자열로 올 수 있으므로 int 변환)
        year = int(_safe_number(data.get("vehicleYear")))
        if year_min and (not year or year < year_min):
            continue
        if year_max and (not year or year > year_max):
            continue

        # 세대 (부분 매칭)
        if not _match_contains(data.get("generation"), generation):
            continue

        # 연료 (부분 매칭)
        if not _match_contains(data.get("fuelType"), fuel):
            continue

        # 구동방식 (부분 매칭)
        if not _match_contains(data.get("driveType"), drive):
            continue

        # 트림 (부분 매칭)
        if not _match_contains(data.get("vehicleTrim"), trim):
            continue

        # 주행거리 상한 (Firestore에서 문자열로 올 수 있으므로 숫자 변환)
        mileage = int(_safe_number(data.get("mileage")))
        if mileage_max and mileage > mileage_max:
            continue

        # 차량경력 (usage)
        if usage:
            purpose = data.get("vehiclePurpose") or ""
            mapped = _USAGE_MAP.get(purpose, purpose)
            if mapped != usage:
                continue

        # 내수 전용
        if domestic_only:
            dest = data.get("saleDestination") or ""
            if "수출" in dest:
                continue

        results.append(_to_legacy_dict(doc.id, data))

    # 옵션 단가 추정값 주입 (같은 차종 출고가-기본가 기반)
    unit_price = estimate_option_unit_price(maker, model)
    if unit_price > 0:
        for r in results:
            r["option_unit_price"] = unit_price

    # Python 정렬
    if sort_by == "가격":
        results.sort(key=lambda x: x.get("낙찰가", 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get("개최일", ""), reverse=True)

    return results[:limit]


def get_vehicle_detail(auction_id: str) -> dict | None:
    """Firestore 문서 ID로 차량 상세 조회"""
    db = get_firestore_db()
    doc = db.collection("vehicles").document(auction_id).get()
    if not doc.exists:
        return None
    result = _to_legacy_dict(doc.id, doc.to_dict())
    # 옵션 단가 추정값 주입
    maker = result.get("maker")
    model = result.get("model_name")
    if maker and model:
        unit_price = estimate_option_unit_price(maker, model)
        if unit_price > 0:
            result["option_unit_price"] = unit_price
    return result


def get_price_stats(
    maker: str,
    model: str,
    generation: str | None = None,
    year: int | None = None,
    months: int = 3,
) -> dict:
    """
    최근 N개월 시세 통계.

    Firestore에서 maker+model 조건 차량을 가져와 Python으로 통계 계산.
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    query = col.where(filter=FieldFilter("vehicleMaker", "==", maker))
    query = query.where(filter=FieldFilter("vehicleModel", "==", model))

    # 최근 N개월 날짜 기준
    cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)

    docs = query.limit(500).get()

    prices = []
    for doc in docs:
        data = doc.to_dict()

        # 유효 낙찰가
        price = _safe_number(data.get("actualBidPrice"))
        if price <= 0:
            continue

        # 내수만
        dest = data.get("saleDestination") or ""
        if "수출" in dest:
            continue

        # 세대 필터
        if generation and not _match_contains(data.get("generation"), generation):
            continue

        # 연식 필터 (타입 안전 비교)
        if year and int(_safe_number(data.get("vehicleYear"))) != year:
            continue

        # 날짜 필터
        sale_date = data.get("saleDate") or data.get("createdAt")
        if sale_date and hasattr(sale_date, "timestamp"):
            if sale_date < cutoff:
                continue

        prices.append(price)

    if not prices:
        return {
            "maker": maker,
            "model": model,
            "generation": generation,
            "year": year,
            "period_months": months,
            "count": 0,
            "message": "해당 조건의 데이터가 없습니다",
        }

    return {
        "maker": maker,
        "model": model,
        "generation": generation,
        "year": year,
        "period_months": months,
        "count": len(prices),
        "mean": round(statistics.mean(prices), 1),
        "median": round(statistics.median(prices), 1),
        "min": min(prices),
        "max": max(prices),
        "std": round(statistics.stdev(prices), 1) if len(prices) > 1 else 0,
    }
