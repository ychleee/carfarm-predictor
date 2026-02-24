"""
CarFarm v2 — 차량 택소노미 검색 서비스

vehicle_taxonomy.json 기반 차명 자동완성 및 계층 검색.

구조: 제작사(47) → 모델(343) → 세대(762) → 변형(1,382) → 트림[]
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache


TAXONOMY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "car_price_prediction" / "output" / "vehicle_taxonomy.json"
)


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict:
    """택소노미 JSON 로드"""
    with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def search_vehicles(query: str, limit: int = 20) -> list[dict]:
    """
    차명 자동완성 검색.

    query가 포함된 제작사/모델/세대/트림을 검색하여 반환.
    """
    taxonomy = _load_taxonomy()
    results = []
    q = query.strip().lower()

    for maker, maker_data in taxonomy.items():
        maker_lower = maker.lower()
        for model_name, model_data in maker_data.get('models', {}).items():
            model_lower = model_name.lower()
            for gen_key, gen_data in model_data.get('generations', {}).items():
                for variant_key, variant_data in gen_data.get('variants', {}).items():
                    trims = variant_data.get('trims', [])

                    # 검색 대상: 제작사, 모델, 세대, 변형, 트림
                    searchable = f"{maker_lower} {model_lower} {gen_key.lower()} {variant_key.lower()} {' '.join(t.lower() for t in trims)}"

                    if q in searchable:
                        results.append({
                            "maker": maker,
                            "model": model_name,
                            "generation": gen_key,
                            "variant": variant_key,
                            "trims": trims,
                            "segment": model_data.get("segment", ""),
                        })

                        if len(results) >= limit:
                            return results

    return results


def get_makers() -> list[str]:
    """전체 제작사 목록"""
    return list(_load_taxonomy().keys())


def get_models(maker: str) -> list[dict]:
    """특정 제작사의 모델 목록"""
    taxonomy = _load_taxonomy()
    maker_data = taxonomy.get(maker, {})
    return [
        {"model": name, "segment": data.get("segment", "")}
        for name, data in maker_data.get("models", {}).items()
    ]


def get_generations(maker: str, model: str) -> list[dict]:
    """특정 모델의 세대 목록"""
    taxonomy = _load_taxonomy()
    model_data = taxonomy.get(maker, {}).get("models", {}).get(model, {})
    return [
        {"generation": gen, "variants": list(data.get("variants", {}).keys())}
        for gen, data in model_data.get("generations", {}).items()
    ]


def get_trims(maker: str, model: str, generation: str) -> list[str]:
    """특정 세대의 트림 목록"""
    taxonomy = _load_taxonomy()
    gen_data = (
        taxonomy.get(maker, {})
        .get("models", {}).get(model, {})
        .get("generations", {}).get(generation, {})
    )
    trims = set()
    for variant_data in gen_data.get("variants", {}).values():
        trims.update(variant_data.get("trims", []))
    return sorted(trims)
