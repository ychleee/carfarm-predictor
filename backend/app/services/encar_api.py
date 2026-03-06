"""
엔카 API 직접 호출 — 소매가 실시간 검색

api.encar.com의 List API를 사용하여 실시간 매물 검색.
Detail API + 사고이력 + 성능검사 + 옵션 매핑으로 상세 정보 보강.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_ENCAR_LIST_URL = "https://api.encar.com/search/car/list/general"
_ENCAR_API_BASE = "https://api.encar.com/v1/readside"
_ENCAR_DETAIL_URL = "https://fem.encar.com/cars/detail"
_PAGE_SIZE = 500
_MAX_PAGES = 30
_ENRICH_LIMIT = 500

_LIST_MAX_RETRIES = 3
_LIST_RETRY_DELAY = 1.5  # 초
_PAGE_DELAY = 0.3  # 페이지 간 딜레이 (초)
_DETAIL_MAX_RETRIES = 2
_DETAIL_RETRY_DELAY = 0.5


# ── 연료 동의어 매핑 (엔카 FuelType 기준) ──

_FUEL_SYNONYMS = {
    "휘발유": "가솔린",
    "경유": "디젤",
    "gasoline": "가솔린",
    "diesel": "디젤",
}


def _normalize_fuel(fuel: str | None) -> str | None:
    """연료 동의어를 엔카 FuelType 기준으로 변환. 매핑 불가시 원본 반환."""
    if not fuel:
        return None
    f = fuel.strip()
    return _FUEL_SYNONYMS.get(f, f)


# ── 옵션 코드 → 이름 매핑 (싱글턴 캐시) ──

_option_map: dict[str, str] | None = None


async def _load_option_map(client: httpx.AsyncClient) -> dict[str, str]:
    """옵션 코드 매핑 로드 (최초 1회, 이후 캐시)"""
    global _option_map
    if _option_map is not None:
        return _option_map

    try:
        resp = await client.get(f"{_ENCAR_API_BASE}/vehicles/car/options/standard")
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        _option_map = {}
        return _option_map

    result: dict[str, str] = {}
    for item in data.get("options", []):
        code = str(item.get("optionCd", ""))
        name = item.get("optionName", "")
        if code and name:
            result[code] = name
        for sub in item.get("subOptions") or []:
            sc = str(sub.get("optionCd", ""))
            sn = sub.get("optionName", "")
            if sc and sn:
                result[sc] = sn

    _option_map = result
    return result


# ── HTTP 요청 + 재시도 ──

def _get_with_retry(client: httpx.Client, url: str,
                    max_retries: int = _LIST_MAX_RETRIES,
                    retry_delay: float = _LIST_RETRY_DELAY) -> httpx.Response:
    """GET 요청 + 재시도 (407/429/5xx 시 백오프)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.get(url)
            if resp.status_code in (407, 429):
                logger.warning("엔카 API %d (attempt %d/%d)", resp.status_code, attempt + 1, max_retries)
                time.sleep(retry_delay * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            last_exc = e
            logger.warning("엔카 API 오류 (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(retry_delay * (attempt + 1))
    raise last_exc or httpx.HTTPError("max retries exceeded")


async def _aget_with_retry(client: httpx.AsyncClient, url: str,
                           max_retries: int = _DETAIL_MAX_RETRIES,
                           retry_delay: float = _DETAIL_RETRY_DELAY) -> httpx.Response:
    """비동기 GET 요청 + 재시도."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.get(url)
            if resp.status_code in (407, 429):
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            last_exc = e
            await asyncio.sleep(retry_delay * (attempt + 1))
    raise last_exc or httpx.HTTPError("max retries exceeded")


# ── List API 검색 ──

def search_encar_retail(
    maker: str,
    model_keyword: str,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    limit: int = 200,
    sort_by: str = "가격",
) -> list[dict]:
    """엔카 List API 페이지네이션 검색 (재시도)."""
    keyword_lower = model_keyword.lower().strip()
    normalized_fuel = _normalize_fuel(fuel)
    fuel_lower = normalized_fuel.lower() if normalized_fuel else None
    trim_lower = trim.lower().strip() if trim else None

    # DSL 빌드: Manufacturer 필수 + FuelType/Year.range/Mileage.range 선택
    dsl_parts = ["Hidden.N", "CarType.Y", f"Manufacturer.{maker}"]
    if fuel_lower:
        dsl_parts.append(f"FuelType.{normalized_fuel}")
    if year_min and year_max:
        dsl_parts.append(f"Year.range({year_min * 100}..{(year_max + 1) * 100})")
    elif year_min:
        dsl_parts.append(f"Year.range({year_min * 100}..)")
    elif year_max:
        dsl_parts.append(f"Year.range(..{(year_max + 1) * 100})")
    if mileage_max:
        dsl_parts.append(f"Mileage.range(..{mileage_max})")
    dsl = "(And." + "._.".join(dsl_parts) + ".)"

    results: list[dict] = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    with httpx.Client(timeout=10.0, headers=headers) as client:
        for page in range(_MAX_PAGES):
            if page > 0:
                time.sleep(_PAGE_DELAY)

            offset = page * _PAGE_SIZE
            sr_value = quote(f"|ModifiedDate|{offset}|{_PAGE_SIZE}", safe="")
            url = f"{_ENCAR_LIST_URL}?count=true&q={quote(dsl, safe='')}&sr={sr_value}"

            try:
                resp = _get_with_retry(client, url)
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("엔카 List API 최종 실패 (page=%d): %s", page, e)
                break

            items = data.get("SearchResults", [])
            if not items:
                break

            for item in items:
                item_model = (item.get("Model") or "").lower()
                if keyword_lower not in item_model:
                    continue

                form_year = int(item.get("FormYear") or item.get("Year") or 0)
                if year_min and form_year < year_min:
                    continue
                if year_max and form_year > year_max:
                    continue

                if fuel_lower:
                    item_fuel = (item.get("FuelType") or "").lower()
                    if fuel_lower not in item_fuel:
                        continue

                if trim_lower:
                    item_badge = (item.get("Badge") or "").lower()
                    if trim_lower not in item_badge:
                        continue

                item_mileage = item.get("Mileage") or 0
                if mileage_max and item_mileage > mileage_max:
                    continue

                results.append(_map_encar_item(item))

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

    if sort_by == "가격":
        results.sort(key=lambda x: x.get("retail_price") or 0)
    else:
        results.sort(key=lambda x: x.get("year") or 0, reverse=True)

    return results[:limit]


# ── Detail + 사고이력 + 성능검사 보강 ──

async def enrich_with_details(results: list[dict], max_concurrent: int = 10) -> None:
    """Detail·사고이력·성능검사 API 병렬 호출로 상세 정보 보강."""
    if not results:
        return

    enrich_count = min(len(results), _ENRICH_LIMIT)
    semaphore = asyncio.Semaphore(max_concurrent)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        opt_map = await _load_option_map(client)

        async def fetch_one(idx: int, car_id: str):
            async with semaphore:
                try:
                    await _enrich_single(idx, car_id, client, opt_map, results)
                except Exception:
                    pass

        tasks = [
            fetch_one(i, r["auction_id"])
            for i, r in enumerate(results[:enrich_count])
            if r.get("auction_id")
        ]
        await asyncio.gather(*tasks)


async def _enrich_single(
    idx: int,
    car_id: str,
    client: httpx.AsyncClient,
    opt_map: dict[str, str],
    results: list[dict],
) -> None:
    """단일 차량 보강: Detail → 사고이력 + 성능검사 병렬 호출 (재시도 포함)."""
    # 1) Detail API
    resp = await _aget_with_retry(client, f"{_ENCAR_API_BASE}/vehicle/{car_id}")
    detail = resp.json()

    spec = detail.get("spec") or {}
    category = detail.get("category") or {}
    manage = detail.get("manage") or {}
    condition = detail.get("condition") or {}
    adv = detail.get("advertisement") or {}
    accident_cond = condition.get("accident") or {}
    seizing = condition.get("seizing") or {}
    inspection_cond = condition.get("inspection") or {}

    r = results[idx]

    # 기본 필드
    r["color"] = spec.get("colorName") or ""
    origin_price = category.get("originPrice")
    if origin_price is not None:
        r["factory_price"] = int(origin_price)
    displacement = spec.get("displacement")
    if displacement:
        r["displacement"] = int(displacement)
    regist_dt = manage.get("registDateTime") or ""
    if regist_dt:
        r["listing_date"] = regist_dt[:10]
    r["has_accident_record"] = bool(accident_cond.get("recordView"))
    r["seizing_count"] = int(seizing.get("seizingCount") or 0)
    r["pledge_count"] = int(seizing.get("pledgeCount") or 0)
    r["has_diagnosis"] = bool(adv.get("diagnosisCar"))
    r["has_inspection"] = bool(inspection_cond.get("formats"))
    r["view_count"] = int(manage.get("viewCount") or 0)

    # 옵션 디코딩
    opt_data = detail.get("options") or {}
    std_codes = opt_data.get("standard") or []
    choice_codes = opt_data.get("choice") or []
    all_codes = [str(c) for c in std_codes + choice_codes]
    if all_codes and opt_map:
        names = [opt_map[c] for c in all_codes if c in opt_map]
        if names:
            r["options"] = ", ".join(names)

    # 2) real vehicleId 확보 (사고이력·성능검사 API에 필요)
    real_id = detail.get("vehicleId")
    if not real_id:
        return

    # 3) 사고이력 + 성능검사 병렬 호출
    sub_tasks = [_fetch_accident_summary(real_id, client)]
    if r["has_inspection"]:
        sub_tasks.append(_fetch_inspection_data(real_id, client))

    sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)

    if not isinstance(sub_results[0], Exception) and sub_results[0]:
        r["accident_summary"] = sub_results[0]

    if len(sub_results) > 1 and not isinstance(sub_results[1], Exception) and sub_results[1]:
        insp = sub_results[1]
        r["frame_exchange"] = insp["frame_exchange"]
        r["frame_bodywork"] = insp["frame_bodywork"]
        r["frame_corrosion"] = insp["frame_corrosion"]
        r["exterior_exchange"] = insp["exterior_exchange"]
        r["exterior_bodywork"] = insp["exterior_bodywork"]
        r["exterior_corrosion"] = insp["exterior_corrosion"]


async def _fetch_accident_summary(real_id: int | str, client: httpx.AsyncClient) -> str:
    """보험이력 요약: '내차 1건(148만) / 상대 1건(38만)' 또는 '무사고'"""
    resp = await _aget_with_retry(client, f"{_ENCAR_API_BASE}/record/vehicle/{real_id}/summary")
    d = resp.json()

    my_cnt = int(d.get("myAccidentCnt") or 0)
    other_cnt = int(d.get("otherAccidentCnt") or 0)

    if my_cnt == 0 and other_cnt == 0:
        return "무사고"

    parts = []
    my_cost = int(d.get("myAccidentCost") or 0)
    other_cost = int(d.get("otherAccidentCost") or 0)

    if my_cnt > 0:
        cost_str = f"({my_cost // 10000}만)" if my_cost >= 10000 else ""
        parts.append(f"내차피해 {my_cnt}건{cost_str}")
    if other_cnt > 0:
        cost_str = f"({other_cost // 10000}만)" if other_cost >= 10000 else ""
        parts.append(f"상대차피해 {other_cnt}건{cost_str}")

    return " / ".join(parts)


async def _fetch_inspection_data(real_id: int | str, client: httpx.AsyncClient) -> dict:
    """성능검사 파싱: 프레임/외판별 교환·판금·부식 카운트 반환."""
    resp = await _aget_with_retry(client, f"{_ENCAR_API_BASE}/inspection/vehicle/{real_id}")
    d = resp.json()

    outers = d.get("outers") or []
    frame = {"exchange": 0, "bodywork": 0, "corrosion": 0}
    exterior = {"exchange": 0, "bodywork": 0, "corrosion": 0}

    for o in outers:
        status_types = o.get("statusTypes") or []
        attributes = o.get("attributes") or []
        target = frame if "RANK_ONE" in attributes else exterior

        for st in status_types:
            code = st.get("code") or ""
            if code == "X":      # 교환
                target["exchange"] += 1
            elif code == "W":    # 판금/용접
                target["bodywork"] += 1
            elif code == "C":    # 부식
                target["corrosion"] += 1

    return {
        "frame_exchange": frame["exchange"],
        "frame_bodywork": frame["bodywork"],
        "frame_corrosion": frame["corrosion"],
        "exterior_exchange": exterior["exchange"],
        "exterior_bodywork": exterior["bodywork"],
        "exterior_corrosion": exterior["corrosion"],
    }


# ── List API 매핑 ──

def _map_encar_item(item: dict) -> dict:
    """엔카 List API 항목 → search-retail 응답 형식 매핑."""
    car_id = str(item.get("Id", ""))
    manufacturer = item.get("Manufacturer") or ""
    model = item.get("Model") or ""
    badge = item.get("Badge") or ""
    badge_detail = item.get("BadgeDetail") or ""

    parts = [manufacturer, model, badge]
    if badge_detail:
        parts.append(badge_detail)
    vehicle_name = " ".join(p for p in parts if p)

    return {
        "auction_id": car_id,
        "vehicle_name": vehicle_name,
        "year": int(item.get("FormYear") or item.get("Year") or 0),
        "mileage": int(item.get("Mileage") or 0),
        "retail_price": int(item.get("Price") or 0),
        "fuel": item.get("FuelType") or "",
        "trim": badge,
        "color": "",
        "options": "",
        "region": item.get("OfficeCityState") or "",
        "source_url": f"{_ENCAR_DETAIL_URL}/{car_id}" if car_id else "",
        "factory_price": None,
        "displacement": None,
        "listing_date": "",
        "has_accident_record": False,
        "seizing_count": 0,
        "pledge_count": 0,
        "has_diagnosis": False,
        "has_inspection": False,
        "view_count": 0,
        "accident_summary": "",
        "inspection_summary": "",
        "frame_exchange": 0,
        "frame_bodywork": 0,
        "frame_corrosion": 0,
        "exterior_exchange": 0,
        "exterior_bodywork": 0,
        "exterior_corrosion": 0,
    }
