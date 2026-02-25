"""
차량 정보 자동완성 & 시세 통계 API — taxonomy + 낙찰 DB 기반
"""
from fastapi import APIRouter, Query

from app.services.taxonomy_search import (
    search_vehicles, get_makers, get_models, get_generations, get_variants, get_trims,
)
from app.services.auction_db import search_auction_db, get_price_stats

router = APIRouter()


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


@router.get("/search")
async def search_auction(
    maker: str = Query(...),
    model: str = Query(...),
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
        maker=maker, model=model, generation=generation,
        year_min=year_min, year_max=year_max,
        fuel=fuel, trim=trim, mileage_max=mileage_max,
        usage=usage, limit=limit, sort_by=sort_by,
    )
    return {"count": len(results), "results": results}
