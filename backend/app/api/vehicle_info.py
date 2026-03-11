"""
차량 정보 자동완성 & 시세 통계 API — taxonomy + 낙찰 DB 기반
"""
from fastapi import APIRouter, Query

from app.services.taxonomy_search import (
    search_vehicles, get_makers, get_models, get_generations, get_variants, get_trims,
)
from app.services.firestore_db import search_auction_db, get_price_stats
from app.services.encar_api import enrich_with_details

router = APIRouter()

_ENCAR_COMPANY_ID = "KYMaGfcnzwGsvbDm6Z91"


async def _enrich_encar_data(mapped: list[dict]) -> None:
    """엔카 API로 보강: 기본가(originPrice→base_price), 옵션, 엔카진단"""
    original_ids = {}
    for r in mapped:
        aid = r["auction_id"]
        if aid.startswith("encar_"):
            stripped = aid[len("encar_"):]
            original_ids[stripped] = aid
            r["auction_id"] = stripped

    await enrich_with_details(mapped)

    for r in mapped:
        aid = r["auction_id"]
        if aid in original_ids:
            r["auction_id"] = original_ids[aid]
        # 엔카진단
        if r.get("has_diagnosis"):
            r["has_encar_diagnosis"] = True


@router.get("/vehicle-info")
async def search_vehicle_info(
    q: str = Query(..., min_length=1, description="검색어"),
    limit: int = Query(20, ge=1, le=100),
):
    """차명 자동완성 검색 (taxonomy 기반)"""
    return {"query": q, "results": search_vehicles(q, limit=limit)}


@router.get("/makers")
async def list_makers():
    """전체 제작사 목록"""
    return {"makers": get_makers()}


@router.get("/models/{maker}")
async def list_models(maker: str):
    """제작사별 모델 목록"""
    return {"maker": maker, "models": get_models(maker)}


@router.get("/generations/{maker}/{model}")
async def list_generations(maker: str, model: str):
    """모델별 세대 목록"""
    return {"maker": maker, "model": model, "generations": get_generations(maker, model)}


@router.get("/variants/{maker}/{model}/{generation}")
async def list_variants(maker: str, model: str, generation: str):
    """세대별 변형(연료/배기량) 목록"""
    return {"variants": get_variants(maker, model, generation)}


@router.get("/trims/{maker}/{model}/{generation}")
async def list_trims(maker: str, model: str, generation: str, variant_key: str | None = None):
    """세대별 트림 목록 (variant_key로 필터 가능)"""
    return {"trims": get_trims(maker, model, generation, variant_key=variant_key)}


@router.get("/price-stats/{maker}/{model}")
async def price_stats(
    maker: str,
    model: str,
    generation: str | None = None,
    year: int | None = None,
    months: int = 3,
):
    """모델별 시세 통계 (최근 N개월)"""
    return get_price_stats(maker, model, generation, year, months)


@router.get("/search-retail")
async def search_retail(
    model: str = Query(...),
    maker: str | None = None,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    limit: int = Query(500, ge=1, le=9999),
    sort_by: str = "날짜",
):
    """소매가 차량 검색 — Firestore 엔카(KYMaGfcnzwGsvbDm6Z91) 데이터"""
    results = search_auction_db(
        model=model, maker=maker, generation=generation,
        year_min=year_min, year_max=year_max,
        fuel=fuel, trim=trim, mileage_max=mileage_max,
        limit=limit, sort_by=sort_by,
        company_id="KYMaGfcnzwGsvbDm6Z91",
    )
    # 소매가용 영문키 매핑
    mapped = []
    for r in results:
        mapped.append({
            "auction_id": r.get("auction_id", ""),
            "vehicle_name": r.get("차명", ""),
            "year": r.get("연식"),
            "mileage": r.get("주행거리"),
            "retail_price": r.get("낙찰가"),  # 낙찰가를 소매가로 매핑
            "color": r.get("색상", ""),
            "options": r.get("옵션", ""),
            "fuel": r.get("연료", ""),
            "trim": r.get("trim", ""),
            "base_price": r.get("base_price"),
            "factory_price": r.get("factory_price"),
            "listing_date": r.get("개최일", ""),
            "source_url": "",
            "region": "",
            "displacement": r.get("배기량"),
            "has_diagnosis": False,
            "has_accident_record": False,
            "seizing_count": 0,
            "pledge_count": 0,
            "view_count": 0,
            "accident_summary": "",
            "inspection_summary": "",
            "frame_exchange": r.get("frame_exchange", 0),
            "frame_bodywork": r.get("frame_bodywork", 0),
            "frame_corrosion": r.get("frame_corrosion", 0),
            "exterior_exchange": r.get("exterior_exchange", 0),
            "exterior_bodywork": r.get("exterior_bodywork", 0),
            "exterior_corrosion": r.get("exterior_corrosion", 0),
            "company_id": r.get("company_id", ""),
        })
    return {"count": len(mapped), "results": mapped}


@router.get("/search-auction")
async def search_auction_endpoint(
    model: str = Query(...),
    maker: str | None = None,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    usage: str | None = None,
    company_id: str = "cRFWlHv4PZczXpd8bEw2",
    limit: int = 500,
    sort_by: str = "날짜",
):
    """낙찰가 차량 검색 — vehicleModel로 searchTokens 검색 (영문키 반환)"""
    results = search_auction_db(
        model=model, maker=maker, generation=generation,
        year_min=year_min, year_max=year_max,
        fuel=fuel, trim=trim, mileage_max=mileage_max,
        usage=usage, limit=limit, sort_by=sort_by,
        company_id=company_id,
    )
    # 한글키 → 영문키 변환 (차명 + 트림 결합)
    mapped = []
    for r in results:
        name = r.get("차명", "")
        trim_val = r.get("trim", "")
        if trim_val and trim_val not in name:
            name = f"{name} {trim_val}".strip()
        mapped.append({
            "auction_id": r.get("auction_id", ""),
            "vehicle_name": name,
            "year": r.get("연식"),
            "mileage": r.get("주행거리"),
            "auction_price": r.get("낙찰가"),
            "auction_date": r.get("개최일", ""),
            "color": r.get("색상", ""),
            "options": r.get("옵션", ""),
            "fuel": r.get("연료", ""),
            "trim": r.get("trim", ""),
            "inspection_grade": r.get("평가점", ""),
            "base_price": r.get("base_price"),
            "factory_price": r.get("factory_price"),
            "frame_exchange": r.get("frame_exchange", 0),
            "frame_bodywork": r.get("frame_bodywork", 0),
            "frame_corrosion": r.get("frame_corrosion", 0),
            "exterior_exchange": r.get("exterior_exchange", 0),
            "exterior_bodywork": r.get("exterior_bodywork", 0),
            "exterior_corrosion": r.get("exterior_corrosion", 0),
            "company_id": r.get("company_id", ""),
            "has_encar_diagnosis": False,
        })

    # 엔카 데이터: 기본가·옵션·엔카진단 보강
    if company_id == _ENCAR_COMPANY_ID and mapped:
        await _enrich_encar_data(mapped)

    return {"count": len(mapped), "results": mapped}


@router.get("/search")
async def search_auction(
    model: str = Query(...),
    maker: str | None = None,
    generation: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    fuel: str | None = None,
    trim: str | None = None,
    mileage_max: int | None = None,
    usage: str | None = None,
    limit: int = 30,
    sort_by: str = "날짜",
):
    """낙찰 DB 검색 (LLM 도구로도 사용)"""
    results = search_auction_db(
        model=model, maker=maker, generation=generation,
        year_min=year_min, year_max=year_max,
        fuel=fuel, trim=trim, mileage_max=mileage_max,
        usage=usage, limit=limit, sort_by=sort_by,
    )
    return {"count": len(results), "results": results}
