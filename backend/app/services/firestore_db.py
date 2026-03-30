"""
CarFarm v2.3 — Firestore 기반 차량 DB

아이작 Firestore `vehicles` 컬렉션에서 차량 검색, 상세 조회, 시세 통계 제공.
기존 auction_db.py와 동일한 함수 인터페이스를 유지하여 호출자 변경 최소화.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1.base_query import FieldFilter

from app.services.firestore_client import get_firestore_db
from app.services.rule_engine import normalize_color
from app.services.taxonomy_search import resolve_base_model


def _search_token_variants(token: str) -> list[str]:
    """하이픈 포함/미포함 변형 토큰 생성 (중복 제거, 원본 우선)"""
    if not token:
        return []
    variants = [token]
    if "-" in token:
        without = token.replace("-", "")
        if without != token:
            variants.append(without)
    else:
        # 영문+한글 경계에 하이픈 삽입 (e.g. "e클래스" → "e-클래스")
        with_hyphen = re.sub(r"([a-z])([가-힣])", r"\1-\2", token)
        if with_hyphen != token:
            variants.append(with_hyphen)
    return variants


def _match_trim(target_trim: str, vehicle_trim: str) -> bool:
    """트림 유연 매칭: 양방향 포함 관계면 동일 트림으로 판단.

    예: "모던" ↔ "1.6 모던" ↔ "1.6 가솔린 모던" 모두 매칭.
    """
    if not target_trim:
        return True
    if not vehicle_trim:
        return False
    t = target_trim.strip().lower()
    v = vehicle_trim.strip().lower()
    if t == v:
        return True
    return t in v or v in t


# =========================================================================
# 내부 헬퍼
# =========================================================================

def _safe_number(val, default=0) -> float | int:
    """문자열/None → 숫자 (콤마 포함 문자열 지원, NaN/inf → default)"""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            result = float(val.replace(",", ""))
        else:
            result = val
        # NaN/inf 방지 (Firestore에 기록되면 Flutter .round()에서 크래시)
        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            return default
        return result
    except (ValueError, TypeError):
        return default


_USAGE_MAP = {
    "자가용": "personal",
    "렌터카": "rental",
    "영업용": "commercial",
    "관용": "commercial",
    "일반": "personal",
}

# ── 제작사 한글/영어 매핑 (양방향) ──

_MAKER_ALIASES: dict[str, str] = {}

def _build_maker_aliases() -> None:
    """제작사명 → canonical 매핑 (소문자 키). 같은 canonical이면 동일 제작사."""
    # 같은 그룹의 이름들은 동일한 canonical 값을 공유
    pairs = [
        ("현대", "hyundai"),
        ("기아", "kia"),
        ("제네시스", "genesis"),
        ("쌍용", "ssangyong"), ("kg모빌리티", "ssangyong"),
        ("르노코리아", "renault"), ("르노삼성", "renault"),
        ("르노코리아(삼성)", "renault"), ("르노(삼성)", "renault"),
        ("쉐보레", "chevrolet"),
        ("벤츠", "mercedes-benz"), ("메르세데스-벤츠", "mercedes-benz"),
        ("아우디", "audi"),
        ("폭스바겐", "volkswagen"),
        ("볼보", "volvo"),
        ("토요타", "toyota"), ("도요타", "toyota"),
        ("혼다", "honda"),
        ("닛산", "nissan"),
        ("렉서스", "lexus"),
        ("포르쉐", "porsche"),
        ("랜드로버", "land rover"),
        ("재규어", "jaguar"),
        ("테슬라", "tesla"),
        ("링컨", "lincoln"),
        ("캐딜락", "cadillac"),
        ("지프", "jeep"),
        ("포드", "ford"),
        ("미니", "mini"),
        ("푸조", "peugeot"),
        ("인피니티", "infiniti"),
        ("마세라티", "maserati"),
        ("벤틀리", "bentley"),
        ("페라리", "ferrari"),
        ("람보르기니", "lamborghini"),
    ]
    for name, canonical in pairs:
        _MAKER_ALIASES[name] = canonical
        _MAKER_ALIASES[canonical] = canonical  # canonical도 자기 자신으로 매핑

_build_maker_aliases()


def _match_maker(vehicle_maker: str, query_maker: str) -> bool:
    """제작사 매칭 — 직접 일치, canonical 비교, 부분 포함"""
    vm = vehicle_maker.lower().strip()
    qm = query_maker.lower().strip()
    if vm == qm:
        return True
    # canonical 비교: 둘 다 같은 canonical이면 매칭
    vm_canonical = _MAKER_ALIASES.get(vm)
    qm_canonical = _MAKER_ALIASES.get(qm)
    if vm_canonical and qm_canonical and vm_canonical == qm_canonical:
        return True
    # 단방향 별칭 비교
    if vm_canonical and vm_canonical == qm:
        return True
    if qm_canonical and qm_canonical == vm:
        return True
    # 부분 포함 매칭 (예: "메르세데스-벤츠" vs "벤츠")
    if qm in vm or vm in qm:
        return True
    return False

# ── 프레임 vs 외부패널 분류 ──

_FRAME_PARTS = {
    "FRONT_PANEL", "FRONT_CROSS_MEMBER", "FLOOR_PANEL",
    "SIDE_MEMBER", "REAR_CROSS_MEMBER", "TRUNK_FLOOR_PANEL", "REAR_PANEL",
    "LEFT_A_PILLAR", "RIGHT_A_PILLAR",
    "LEFT_B_PILLAR", "RIGHT_B_PILLAR",
    "LEFT_C_PILLAR", "RIGHT_C_PILLAR",
    "LEFT_FRONT_INSIDE_PANEL", "RIGHT_FRONT_INSIDE_PANEL",
    "LEFT_REAR_INSIDE_PANEL", "RIGHT_REAR_INSIDE_PANEL",
    "LEFT_FRONT_WHEELS_HOUSE", "RIGHT_FRONT_WHEELS_HOUSE",
    "LEFT_REAR_WHEELS_HOUSE", "RIGHT_REAR_WHEELS_HOUSE",
}

_EXTERIOR_PARTS = {
    "HOOD", "TRUNK", "ROOF",
    "FRONT_BUMPER", "REAR_BUMPER",
    "LEFT_FRONT_FENDER", "RIGHT_FRONT_FENDER",
    "LEFT_FRONT_DOOR", "RIGHT_FRONT_DOOR",
    "LEFT_REAR_DOOR", "RIGHT_REAR_DOOR",
    "LEFT_QUARTER_PANEL", "RIGHT_QUARTER_PANEL",
    "LEFT_FOOT_PANEL", "RIGHT_FOOT_PANEL",
}

_EXCHANGE_TYPES = {"EXCHANGE", "EXCHANGE_NEEDED"}
_BODYWORK_TYPES = {"PAINT_PANEL_BEATING", "PANEL_WELDING", "PANEL_BEATING",
                   "PANEL_BEATING_NEEDED", "PAINT_NEEDED", "BENT"}
_CORROSION_TYPES = {"CORROSION"}


def _calc_damage_stats(part_damages: list[dict]) -> dict:
    """partDamages → 프레임/외부패널별 교환·판금·부식 카운트"""
    stats = {
        "frame_exchange": 0,
        "frame_bodywork": 0,
        "frame_corrosion": 0,
        "exterior_exchange": 0,
        "exterior_bodywork": 0,
        "exterior_corrosion": 0,
    }
    for d in part_damages:
        part = d.get("part", "")
        dt = d.get("damage_type", "")

        if part in _FRAME_PARTS:
            prefix = "frame"
        elif part in _EXTERIOR_PARTS:
            prefix = "exterior"
        else:
            # 미분류 부위는 외부패널로 처리
            prefix = "exterior"

        if dt in _EXCHANGE_TYPES:
            stats[f"{prefix}_exchange"] += 1
        elif dt in _BODYWORK_TYPES:
            stats[f"{prefix}_bodywork"] += 1
        elif dt in _CORROSION_TYPES:
            stats[f"{prefix}_corrosion"] += 1

    return stats


def _ts_to_iso(val) -> str:
    """Firestore timestamp / datetime → ISO 날짜 문자열"""
    if val is None:
        return ""
    if hasattr(val, "isoformat"):
        return val.isoformat()[:10]
    return str(val)[:10]


def _name_with_trim(data: dict) -> str:
    """차명 + 트림 (중복 방지)"""
    name = data.get("vehicleName") or data.get("title") or ""
    trim = data.get("vehicleTrim") or ""
    if trim and trim not in name:
        return f"{name} {trim}"
    return name


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

    # partDamages 추출 (VehiclePart./DamageType. 접두사 제거)
    part_damages_raw = data.get("partDamages") or []
    part_damages = []
    for pd in part_damages_raw:
        if isinstance(pd, dict):
            part = pd.get("part") or ""
            dt = pd.get("damageType") or pd.get("damage_type") or ""
            if part.startswith("VehiclePart."):
                part = part[len("VehiclePart."):]
            if dt.startswith("DamageType."):
                dt = dt[len("DamageType."):]
            if part and dt:
                part_damages.append({"part": part, "damage_type": dt})

    # 프레임/외부패널별 교환·판금·부식 상세 카운트
    damage_stats = _calc_damage_stats(part_damages)

    # 기존 호환용 합계
    exchange_count = damage_stats["frame_exchange"] + damage_stats["exterior_exchange"]
    bodywork_count = damage_stats["frame_bodywork"] + damage_stats["exterior_bodywork"]

    return {
        # 한글 키 (기존 호출자 호환)
        "auction_id": doc_id,
        "차명": _name_with_trim(data),
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
        # 프레임/외부패널별 상세
        "frame_exchange": damage_stats["frame_exchange"],
        "frame_bodywork": damage_stats["frame_bodywork"],
        "frame_corrosion": damage_stats["frame_corrosion"],
        "exterior_exchange": damage_stats["exterior_exchange"],
        "exterior_bodywork": damage_stats["exterior_bodywork"],
        "exterior_corrosion": damage_stats["exterior_corrosion"],
        "grade_score": data.get("inspectionGrade") or "",
        "accident_severity": "unknown",
        "base_price": base_price,
        "factory_price": factory_price,
        "company_id": data.get("companyId") or "",
        "description": data.get("description") or "",
        "status": data.get("status") or "",
    }


_FUEL_SYNONYMS = {
    "가솔린": {"휘발유", "가솔린", "gasoline"},
    "휘발유": {"휘발유", "가솔린", "gasoline"},
    "디젤": {"경유", "디젤", "diesel"},
    "경유": {"경유", "디젤", "diesel"},
    "LPG": {"LPG", "lpg", "엘피지"},
    "엘피지": {"LPG", "lpg", "엘피지"},
    "하이브리드": {"하이브리드", "hybrid"},
    "전기": {"전기", "electric", "EV"},
}


def _match_contains(value: str | None, keyword: str | None) -> bool:
    """대소문자·공백 무시 부분 문자열 매칭"""
    if not keyword:
        return True
    if not value:
        return False
    # 일반 매칭
    if keyword.lower() in value.lower():
        return True
    # 공백 제거 후 재매칭 (예: "ED 에디션" vs "ED에디션")
    kw_compact = keyword.lower().replace(" ", "")
    val_compact = value.lower().replace(" ", "")
    return kw_compact in val_compact


def _is_hybrid(fuel: str) -> bool:
    """하이브리드/PHEV/전기복합 여부 판별"""
    f = fuel.lower()
    return ("하이브리드" in f or "hybrid" in f or "hev" in f
            or "전기" in f and "가솔린" in f   # 가솔린+전기 = PHEV
            or "전기" in f and ("디젤" in f or "경유" in f))


def _match_fuel(value: str | None, keyword: str | None) -> bool:
    """연료 동의어를 포함한 매칭 (가솔린↔휘발유, 디젤↔경유 등).
    하이브리드/비하이브리드를 엄격 구분: 가솔린 ≠ 가솔린 하이브리드.
    """
    if not keyword:
        return True
    if not value:
        return False
    # 하이브리드 구분: 양쪽이 일치해야 함
    if _is_hybrid(keyword) != _is_hybrid(value):
        return False
    # 일반 매칭
    if keyword.lower() in value.lower():
        return True
    # 동의어 매칭
    synonyms = _FUEL_SYNONYMS.get(keyword)
    if synonyms:
        val_lower = value.lower()
        return any(s.lower() in val_lower for s in synonyms)
    return False


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

def search_retail_db(
    model: str,
    maker: str | None = None,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    limit: int = 100,
    sort_by: str = "가격",
) -> list[dict]:
    """
    엔카 소매가 차량 검색.

    searchTokens 필드의 array-contains로 vehicleModel 검색.
    maker 제공 시 Python 후처리로 필터링.
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    resolved = resolve_base_model(model, maker) if model else model
    model_lower = resolved.lower().strip() if resolved else ""

    _ENCAR_COMPANY_ID = "KYMaGfcnzwGsvbDm6Z91"
    fetch_limit = min(limit * 10, 5000)
    # 하이픈 변형 토큰으로 모두 검색하여 합침
    seen_ids: set[str] = set()
    docs = []
    for token in _search_token_variants(model_lower) if model_lower else [""]:
        query = col.where(filter=FieldFilter("companyId", "==", _ENCAR_COMPANY_ID))
        if token:
            query = query.where(filter=FieldFilter("searchTokens", "array_contains", token))
        for doc in query.limit(fetch_limit).get():
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                docs.append(doc)

    results = []
    for doc in docs:
        data = doc.to_dict()

        # maker 후필터 (한글/영어 무관 필터링)
        if maker:
            vehicle_maker = data.get("vehicleMaker") or ""
            if not _match_maker(vehicle_maker, maker):
                continue

        # 소매가 필수 (actualBidPrice가 엔카 소매가)
        retail_price = _safe_number(data.get("actualBidPrice"))
        if retail_price <= 0:
            continue

        # 이상치 제거: 소매가 100만원 미만 또는 9,000만원 초과
        retail_manwon = retail_price / 10000 if retail_price > 100000 else retail_price
        if retail_manwon < 100 or retail_manwon > 9000:
            continue

        # 출고가와 기본가 모두 없으면 제외
        raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
        raw_base = _safe_number(data.get("vehicleBasePrice"))
        if raw_factory <= 0 and raw_base <= 0:
            continue

        # 연식 범위
        year = int(_safe_number(data.get("vehicleYear")))
        if year_min and (not year or year < year_min):
            continue
        if year_max and (not year or year > year_max):
            continue

        # 연료 (부분 매칭)
        if not _match_fuel(data.get("fuelType"), fuel):
            continue

        # 트림 (유연 매칭: 양방향 포함)
        if trim:
            v_trim_raw = (data.get("vehicleTrim") or "")
            if not _match_trim(trim, v_trim_raw):
                continue

        # 주행거리 상한
        mileage = int(_safe_number(data.get("mileage")))
        if mileage_max and mileage > mileage_max:
            continue

        results.append(_to_retail_dict(doc.id, data))

    # 중복 제거 (차명+주행거리+소매가 동일 → 출고가 있는 쪽 우선)
    dedup: dict[tuple, dict] = {}
    for r in results:
        key = (r.get("차명", ""), r.get("주행거리", 0), r.get("소매가", 0))
        existing = dedup.get(key)
        if existing is None:
            dedup[key] = r
        elif r.get("factory_price", 0) > 0 and existing.get("factory_price", 0) <= 0:
            dedup[key] = r
    results = list(dedup.values())

    # 정렬
    if sort_by == "가격":
        results.sort(key=lambda x: x.get("소매가", 0))
    else:
        results.sort(key=lambda x: x.get("연식", 0), reverse=True)

    return results[:limit]


def search_auction_db(
    model: str,
    maker: str | None = None,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    drive: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    usage: str | None = None,
    domestic_only: bool = True,
    export_only: bool = False,
    limit: int = 500,
    sort_by: str = "날짜",
    company_id: str | None = None,
) -> list[dict]:
    """
    Firestore vehicles 컬렉션에서 조건 검색.

    searchTokens 필드의 array-contains로 vehicleModel 검색.
    company_id 지정 시 해당 회사 데이터만 반환.
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    resolved = resolve_base_model(model, maker) if model else model
    model_lower = resolved.lower().strip() if resolved else ""

    # 하이픈 변형 토큰으로 모두 검색하여 합침
    use_company_filter_in_query = False
    seen_ids: set[str] = set()
    docs: list = []
    fetch_limit = min(limit * 10, 5000)
    tokens = _search_token_variants(model_lower) if model_lower else [""]
    for token in tokens:
        try:
            if company_id:
                q = col.where(filter=FieldFilter("companyId", "==", company_id))
            else:
                q = col
            if token:
                q = q.where(filter=FieldFilter("searchTokens", "array_contains", token))
            q = q.order_by("saleDate", direction="DESCENDING")
            for doc in q.limit(fetch_limit).get():
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    docs.append(doc)
            use_company_filter_in_query = True
        except Exception:
            if token:
                q = col.where(filter=FieldFilter("searchTokens", "array_contains", token))
            else:
                q = col
            q = q.order_by("saleDate", direction="DESCENDING")
            for doc in q.limit(fetch_limit).get():
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    docs.append(doc)

    # 2차: Python 후처리 필터
    results = []
    for doc in docs:
        data = doc.to_dict()

        # companyId 필터 (Firestore 쿼리에서 처리하지 못한 경우)
        if company_id and not use_company_filter_in_query:
            if (data.get("companyId") or "") != company_id:
                continue

        # maker 후필터 (Firestore 쿼리 후 Python에서 한글/영어 무관 필터링)
        if maker:
            vehicle_maker = data.get("vehicleMaker") or ""
            if not _match_maker(vehicle_maker, maker):
                continue

        # 유효 낙찰가 필터
        price = data.get("actualBidPrice")
        try:
            price = float(str(price).replace(",", "")) if price else 0
        except (ValueError, TypeError):
            price = 0
        if price <= 0:
            continue

        # 출고가와 기본가 모두 없으면 제외
        raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
        raw_base = _safe_number(data.get("vehicleBasePrice"))
        if raw_factory <= 0 and raw_base <= 0:
            continue

        # 연식 범위 (Firestore에서 문자열로 올 수 있으므로 int 변환)
        year = int(_safe_number(data.get("vehicleYear")))
        if year_min and (not year or year < year_min):
            continue
        if year_max and (not year or year > year_max):
            continue

        # 연료 (부분 매칭)
        if not _match_fuel(data.get("fuelType"), fuel):
            continue

        # 구동방식 (부분 매칭)
        if not _match_contains(data.get("driveType"), drive):
            continue

        # 트림 (유연 매칭: 양방향 포함)
        if trim:
            v_trim_raw = (data.get("vehicleTrim") or "")
            if not _match_trim(trim, v_trim_raw):
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

        # 내수/수출 필터
        dest = data.get("saleDestination") or ""
        if domestic_only:
            if "수출" in dest:
                continue
        if export_only:
            if "수출" not in dest:
                continue

        results.append(_to_legacy_dict(doc.id, data))

    # 옵션 단가 추정값 주입 (같은 차종 출고가-기본가 기반)
    if maker:
        unit_price = estimate_option_unit_price(maker, model)
        if unit_price > 0:
            for r in results:
                r["option_unit_price"] = unit_price

    # 정렬: Firestore에서 saleDate DESC로 가져왔으므로 기본 날짜순 유지
    if sort_by == "가격":
        results.sort(key=lambda x: x.get("낙찰가", 0), reverse=True)
    else:
        # 판매일 내림차순 (최신 우선)
        results.sort(key=lambda x: x.get("개최일", "") or "", reverse=True)

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

    # 출고가/기본가 없으면 같은 모델에서 조회
    if not result.get("factory_price") and not result.get("base_price"):
        if maker and model:
            fallback = _lookup_factory_price(db, maker, model)
            if fallback:
                result["factory_price"] = fallback.get("factory_price", 0)
                result["base_price"] = fallback.get("base_price", 0)

    return result


def _lookup_factory_price(db, maker: str, model: str) -> dict | None:
    """같은 모델 차량 중 출고가가 있는 문서에서 출고가/기본가 가져오기"""
    col = db.collection("vehicles")
    model_lower = model.lower().strip()
    try:
        q = col.where(
            filter=FieldFilter("searchTokens", "array_contains", model_lower)
        ).limit(50)
        for doc in q.stream():
            data = doc.to_dict()
            if (data.get("vehicleMaker") or "") != maker:
                continue
            raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
            if raw_factory > 0:
                raw_base = _safe_number(data.get("vehicleBasePrice"))
                return {
                    "factory_price": round(raw_factory / 10000, 1) if raw_factory > 10000 else raw_factory,
                    "base_price": round(raw_base / 10000, 1) if raw_base > 10000 else raw_base,
                }
    except Exception as e:
        logger.warning("출고가 조회 실패: %s", e)
    return None


def get_retail_detail(doc_id: str) -> dict | None:
    """소매(엔카) 차량 상세 조회 — _to_retail_dict 형식 반환"""
    db = get_firestore_db()
    doc = db.collection("vehicles").document(doc_id).get()
    if not doc.exists:
        return None
    return _to_retail_dict(doc.id, doc.to_dict())


def _normalize_price(price: float) -> float:
    """가격을 만원 단위로 정규화 (원 단위면 변환, 만원 단위면 그대로)."""
    if price > 100000:  # 10만원 초과면 원 단위로 판단
        return round(price / 10000, 1)
    return price


def get_price_stats(
    maker: str,
    model: str,
    generation: str | None = None,
    year: int | None = None,
    months: int = 3,
    price_type: str = "auction",
) -> dict:
    """
    최근 N개월 시세 통계.

    Firestore에서 maker+model 조건 차량을 가져와 Python으로 통계 계산.
    price_type: "auction" → actualBidPrice, "retail" → estimatedPurchasePrice
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

        # 유효 가격 (소매는 estimatedRetailPrice 우선)
        if price_type == "auction":
            price = _safe_number(data.get("actualBidPrice"))
        else:
            price = _safe_number(data.get("estimatedRetailPrice")) or _safe_number(data.get("estimatedPurchasePrice"))
        if price <= 0:
            continue

        # 만원 단위로 정규화
        price = _normalize_price(price)

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

        prices.append(float(price))

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
        "std": round((sum((p - statistics.mean(prices)) ** 2 for p in prices) / (len(prices) - 1)) ** 0.5, 1) if len(prices) > 1 else 0,
    }


# =========================================================================
# 추천 시스템 — 소매가 / 낙찰가 분리 검색
# =========================================================================

def _tokenize(text: str) -> list[str]:
    """검색어 → 토큰 리스트 (searchTokenizer.js 호환)"""
    if not text or not text.strip():
        return []
    normalized = text.lower().strip()
    tokens = set()
    tokens.add(normalized)
    for word in normalized.split():
        if word:
            tokens.add(word)
    for part in re.findall(r"[가-힣]+|[a-z]+|[0-9]+", normalized):
        tokens.add(part)
    return list(tokens)


def _to_retail_dict(doc_id: str, data: dict) -> dict:
    """소매가 차량용 dict"""
    # actualBidPrice가 엔카 소매가
    retail_price = _safe_number(data.get("actualBidPrice"))
    if retail_price > 100000:
        retail_price = round(retail_price / 10000, 0)

    raw_base = _safe_number(data.get("vehicleBasePrice"))
    base_price = round(raw_base / 10000, 1) if raw_base > 10000 else raw_base

    raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
    factory_price = round(raw_factory / 10000, 1) if raw_factory > 10000 else raw_factory

    # 옵션 문자열
    options_raw = data.get("vehicleOptions") or []
    if isinstance(options_raw, list) and options_raw and isinstance(options_raw[0], dict):
        options_str = ", ".join(
            item.get("name", "") for item in options_raw if item.get("name")
        )
    elif isinstance(options_raw, list):
        options_str = ", ".join(str(o) for o in options_raw if o)
    else:
        options_str = str(options_raw) if options_raw else ""

    # partDamages 추출 (VehiclePart./DamageType. 접두사 제거)
    part_damages_raw = data.get("partDamages") or []
    part_damages = []
    for pd in part_damages_raw:
        if isinstance(pd, dict):
            part = pd.get("part") or ""
            dt = pd.get("damageType") or pd.get("damage_type") or ""
            if part.startswith("VehiclePart."):
                part = part[len("VehiclePart."):]
            if dt.startswith("DamageType."):
                dt = dt[len("DamageType."):]
            if part and dt:
                part_damages.append({"part": part, "damage_type": dt})

    damage_stats = _calc_damage_stats(part_damages)
    exchange_count = damage_stats["frame_exchange"] + damage_stats["exterior_exchange"]
    bodywork_count = damage_stats["frame_bodywork"] + damage_stats["exterior_bodywork"]

    return {
        "auction_id": doc_id,
        "차명": data.get("vehicleName") or data.get("title") or "",
        "연식": int(_safe_number(data.get("vehicleYear"))),
        "주행거리": int(_safe_number(data.get("mileage"))),
        "소매가": retail_price,
        "색상": data.get("vehicleColor") or "",
        "trim": data.get("vehicleTrim") or "",
        "source_url": data.get("sourceUrl") or "",
        "base_price": base_price,
        "factory_price": factory_price,
        "옵션": options_str,
        "연료": data.get("fuelType") or "",
        "매물등록일": _ts_to_iso(data.get("createdAt")),
        "검차일": data.get("inspectionScheduleDate") or "",
        "source": "encar",
        "exchange_count": exchange_count,
        "bodywork_count": bodywork_count,
        "part_damages": part_damages,
        # 프레임/외부패널별 상세
        "frame_exchange": damage_stats["frame_exchange"],
        "frame_bodywork": damage_stats["frame_bodywork"],
        "frame_corrosion": damage_stats["frame_corrosion"],
        "exterior_exchange": damage_stats["exterior_exchange"],
        "exterior_bodywork": damage_stats["exterior_bodywork"],
        "exterior_corrosion": damage_stats["exterior_corrosion"],
    }


def _calc_similarity(
    data: dict,
    target_model: str | None,
    target_trim: str | None,
    target_year: int,
    target_mileage: int,
    target_options: list[str] | None,
    match_count: int,
    total_tokens: int,
) -> float:
    """차종 > 트림 > 연식 > 판매일 최신 > 주행거리·옵션 유사도 순 점수"""
    score = 0.0

    # 1순위: vehicleModel 매칭 (10000점)
    v_model = (data.get("vehicleModel") or "").lower()
    if target_model and target_model.lower() in v_model:
        score += 10000

    # 2순위: vehicleTrim 매칭 (5000점)
    v_trim = (data.get("vehicleTrim") or "").lower()
    if target_trim and target_trim.lower() in v_trim:
        score += 5000

    # 3순위: 연식 일치 (1000점), ±1년 (500점)
    if target_year > 0:
        v_year = int(_safe_number(data.get("vehicleYear")))
        if v_year > 0:
            year_diff = abs(v_year - target_year)
            if year_diff == 0:
                score += 1000
            elif year_diff == 1:
                score += 500
            elif year_diff == 2:
                score += 200

    # 4순위: 판매일 최신 (최대 300점, 최근일수록 높음)
    sale_date = data.get("saleDate") or data.get("createdAt")
    if sale_date:
        iso = _ts_to_iso(sale_date)
        if iso and len(iso) >= 10:
            try:
                days_ago = (datetime.now(timezone.utc) - datetime.fromisoformat(iso + "T00:00:00+00:00")).days
                score += max(0, 300 - days_ago * 0.5)  # 600일 지나면 0점
            except (ValueError, TypeError):
                pass

    # 5순위: 주행거리 근접도 (최대 100점)
    if target_mileage > 0:
        v_mileage = int(_safe_number(data.get("mileage")))
        if v_mileage > 0:
            diff = abs(v_mileage - target_mileage)
            score += max(0, 100 - diff / 1000)  # 10만km 차이 → 0점

    # 5순위: 옵션 유사도 (최대 100점)
    if target_options:
        v_options = data.get("vehicleOptions") or []
        if isinstance(v_options, list) and v_options:
            v_opt_names = set()
            for o in v_options:
                if isinstance(o, dict) and o.get("name"):
                    v_opt_names.add(o["name"].lower())
                elif isinstance(o, str):
                    v_opt_names.add(o.lower())
            if v_opt_names:
                target_set = {o.lower() for o in target_options}
                overlap = len(target_set & v_opt_names)
                score += (overlap / max(len(target_set), 1)) * 100

    # 토큰 매칭률 보너스 (최대 50점)
    if total_tokens > 0:
        score += (match_count / total_tokens) * 50

    return score


def search_retail_vehicles(
    maker: str,
    model: str,
    trim: str | None = None,
    year: int | None = None,
    mileage: int = 0,
    options: list[str] | None = None,
    generation: str | None = None,
    fuel: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    엔카 활성 매물 검색 (소매가) — 2단계 전략.

    1단계: vehicleModel + generation + fuelType + vehicleTrim + vehicleYear 완전 일치
    2단계: 부족하면 vehicleTrim, vehicleYear 완화하여 추가 수집
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    query = col.where(filter=FieldFilter("vehicleMaker", "==", maker))
    query = query.where(filter=FieldFilter("vehicleModel", "==", model))

    docs = query.limit(500).get()

    # 모든 소매 차량 후보 수집 (기본 필터만)
    all_candidates = []
    for doc in docs:
        data = doc.to_dict()

        # 소매가(estimatedPurchasePrice)가 있고 낙찰가(actualBidPrice)가 없는 차량 = 엔카 매물
        retail_price = _safe_number(data.get("estimatedPurchasePrice"))
        if retail_price <= 0:
            continue
        if _safe_number(data.get("actualBidPrice")) > 0:
            continue

        # 출고가와 기본가 모두 없으면 제외
        raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
        raw_base = _safe_number(data.get("vehicleBasePrice"))
        if raw_factory <= 0 and raw_base <= 0:
            continue

        # 트림 (유연 매칭: 양방향 포함)
        if trim:
            v_trim_raw = (data.get("vehicleTrim") or "")
            if not _match_trim(trim, v_trim_raw):
                continue

        all_candidates.append((doc.id, data))

    # 1단계: 엄격 매칭 (generation + fuelType + year 모두 일치)
    strict_results = []
    relaxed_pool = []

    for doc_id, data in all_candidates:
        is_strict = True

        # generation 매칭
        if generation:
            v_gen = (data.get("generation") or "").lower()
            if generation.lower() not in v_gen:
                is_strict = False

        # fuelType 매칭
        if fuel:
            v_fuel = (data.get("fuelType") or "").lower()
            if fuel.lower() not in v_fuel:
                is_strict = False

        # year 매칭 (±1년)
        if year:
            v_year = int(_safe_number(data.get("vehicleYear")))
            if v_year and abs(v_year - year) > 1:
                is_strict = False

        item = _to_retail_dict(doc_id, data)
        item["_score"] = _calc_similarity(
            data, model, trim, year or 0,
            mileage, options, 0, 0,
        )

        if is_strict:
            strict_results.append(item)
        else:
            relaxed_pool.append(item)

    # 2단계: 엄격 매칭이 부족하면 완화된 조건으로 보충
    results = strict_results
    if len(results) < limit:
        # 완화: trim/year 다르지만 연식 ±3년 이내만
        for item in relaxed_pool:
            if year:
                v_year = item.get("연식", 0)
                if v_year and abs(v_year - year) > 3:
                    continue
            results.append(item)

    results.sort(key=lambda x: x["_score"], reverse=True)

    for r in results[:limit]:
        r.pop("_score", None)

    return results[:limit]


def fetch_comparable_vehicles(
    maker: str,
    model: str,
    year: int,
    fuel: str | None = None,
    trim: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """유사차량 검색 (LLM 분석용) — 같은 연식, 출고가 있는 차량 우선"""
    vehicles = search_auction_db(
        model=model, maker=maker, fuel=fuel, trim=trim,
        year_min=year, year_max=year,
        limit=limit, sort_by="날짜",
    )
    # 출고가(factory_price)가 있는 차량을 우선 정렬
    vehicles.sort(key=lambda v: (0 if v.get("factory_price", 0) > 0 else 1))
    return vehicles


def search_auction_by_tokens(
    search_text: str,
    target_model: str | None = None,
    target_trim: str | None = None,
    target_mileage: int = 0,
    target_year: int = 0,
    target_options: list[str] | None = None,
    target_maker: str | None = None,
    target_generation: str | None = None,
    target_fuel: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    낙찰가 차량 검색 — 2단계 전략.

    1단계: vehicleModel + generation + fuelType + vehicleTrim + vehicleYear 완전 일치
    2단계: 부족하면 vehicleTrim, vehicleYear 완화하여 추가 수집
    → 수출 제외 → 유사도 순 정렬.
    """
    db = get_firestore_db()
    col = db.collection("vehicles")

    maker = target_maker
    model = target_model

    if not maker or not model:
        return []

    # 1차: Firestore 쿼리 (maker + model)
    query = col.where(filter=FieldFilter("vehicleMaker", "==", maker))
    query = query.where(filter=FieldFilter("vehicleModel", "==", model))
    docs = query.limit(500).get()

    # 모든 낙찰 차량 후보 수집 (기본 필터만)
    all_candidates = []
    for doc in docs:
        data = doc.to_dict()

        # 유효 낙찰가 필수
        price = _safe_number(data.get("actualBidPrice"))
        if price <= 0:
            continue

        # 수출 제외
        dest = data.get("saleDestination") or ""
        if "수출" in dest:
            continue

        # 출고가와 기본가 모두 없으면 제외
        raw_factory = _safe_number(data.get("vehicleFactoryPrice"))
        raw_base = _safe_number(data.get("vehicleBasePrice"))
        if raw_factory <= 0 and raw_base <= 0:
            continue

        # 트림 (유연 매칭: 양방향 포함)
        if target_trim:
            v_trim_raw = (data.get("vehicleTrim") or "")
            if not _match_trim(target_trim, v_trim_raw):
                continue

        all_candidates.append((doc.id, data))

    # 1단계: 엄격 매칭 (generation + fuelType + year 모두 일치)
    strict_results = []
    relaxed_pool = []

    for doc_id, data in all_candidates:
        is_strict = True

        # generation 매칭
        if target_generation:
            v_gen = (data.get("generation") or "").lower()
            if target_generation.lower() not in v_gen:
                is_strict = False

        # fuelType 매칭
        if target_fuel:
            v_fuel = (data.get("fuelType") or "").lower()
            if target_fuel.lower() not in v_fuel:
                is_strict = False

        # year 매칭 (±1년)
        v_year = int(_safe_number(data.get("vehicleYear")))
        if target_year > 0 and v_year > 0:
            if abs(v_year - target_year) > 1:
                is_strict = False

        item = _to_legacy_dict(doc_id, data)
        item["_score"] = _calc_similarity(
            data, model, target_trim, target_year,
            target_mileage, target_options, 0, 0,
        )

        if is_strict:
            strict_results.append(item)
        else:
            relaxed_pool.append(item)

    # 2단계: 엄격 매칭이 부족하면 완화된 조건으로 보충
    results = strict_results
    if len(results) < limit:
        # 완화: trim/year 다르지만 연식 ±3년 이내만
        for item in relaxed_pool:
            v_year = item.get("연식", 0)
            if target_year > 0 and v_year > 0:
                if abs(v_year - target_year) > 3:
                    continue
            results.append(item)

    results.sort(key=lambda x: x["_score"], reverse=True)

    for r in results[:limit]:
        r.pop("_score", None)

    return results[:limit]
