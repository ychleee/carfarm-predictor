"""
CarFarm v2 — 차량 택소노미 검색 서비스

vehicle_taxonomy.json 기반 차명 자동완성 및 계층 검색.

구조: 제작사(47) → 모델(343) → 세대(762) → 변형(1,382) → 트림[]
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from collections import Counter


def _resolve_taxonomy_path() -> Path:
    """vehicle_taxonomy.json 경로 결정 (로컬/Docker 모두 지원)"""
    import os
    # 1) 환경변수 우선
    env = os.environ.get("CARFARM_DATA_ROOT")
    if env:
        p = Path(env) / "vehicle_taxonomy.json"
        if p.exists():
            return p
    # 2) backend/data/ (Docker COPY 포함)
    backend_data = Path(__file__).parent.parent.parent / "data" / "vehicle_taxonomy.json"
    if backend_data.exists():
        return backend_data
    # 3) Docker: /app/data/
    docker = Path("/app/data/vehicle_taxonomy.json")
    if docker.exists():
        return docker
    # 4) 기존 경로 (로컬 개발 호환)
    legacy = Path(__file__).parent.parent.parent.parent / "car_price_prediction" / "output" / "vehicle_taxonomy.json"
    if legacy.exists():
        return legacy
    return backend_data  # fallback

TAXONOMY_PATH = _resolve_taxonomy_path()


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict:
    """택소노미 JSON 로드"""
    with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# =========================================================================
# 세대 표시명 계산
# =========================================================================

def _compute_display_name(gen_key: str, gen_data: dict) -> str:
    """
    트림명 공통 접두사에서 세대 표시명을 추출.

    예시:
      - "17년~현재" + 트림들이 모두 "뉴 라이즈 ..." → "뉴 라이즈 (17년~현재)"
      - "14년~현재" + 트림들이 모두 "LF ..." → "LF (14년~현재)"
      - "19년~현재" + 트림 접두사 없음 → "19년~현재"
    """
    all_trims = []
    for vd in gen_data.get('variants', {}).values():
        all_trims.extend(vd.get('trims', []))

    if not all_trims:
        return gen_key

    prefix = _common_trim_prefix(all_trims)
    if prefix:
        return f"{prefix} ({gen_key})"
    return gen_key


def _common_trim_prefix(trims: list[str]) -> str:
    """
    트림명 리스트에서 공통 접두사(첫 N 단어)를 추출.

    전체 트림의 70% 이상이 공유하는 가장 긴 접두사를 반환.
    """
    if not trims:
        return ""

    # 각 트림의 첫 단어 추출
    first_words = []
    for t in trims:
        words = t.strip().split()
        if words:
            first_words.append(words[0])

    if not first_words:
        return ""

    # 가장 흔한 첫 단어가 70% 이상이면 공통 접두사
    counter = Counter(first_words)
    most_common_word, count = counter.most_common(1)[0]
    threshold = len(trims) * 0.7

    if count < threshold:
        return ""

    # 첫 단어가 공통이면, 두 번째 단어도 확인
    matching_trims = [t for t in trims if t.strip().startswith(most_common_word)]
    second_words = []
    for t in matching_trims:
        words = t.strip().split()
        if len(words) >= 2:
            second_words.append(words[1])

    if second_words:
        counter2 = Counter(second_words)
        most_common_2nd, count2 = counter2.most_common(1)[0]
        if count2 >= len(matching_trims) * 0.7:
            return f"{most_common_word} {most_common_2nd}"

    return most_common_word


# =========================================================================
# 모델명 해석 — isaac 앱 입력 → 택소노미 기준 모델명
# =========================================================================

import re as _re

_GEN_PREFIX_RE = _re.compile(
    r"^(더\s*올\s*뉴|디\s*올\s*뉴|올\s*뉴|더\s*뉴|뉴)\s*", flags=_re.IGNORECASE
)

# 흔한 오타 보정 (정규화 시 적용)
_TYPO_MAP = {
    "클레스": "클래스",
}


def _normalize_for_match(s: str) -> str:
    """매칭용 정규화: 소문자 + 하이픈/공백 제거 + 오타 보정"""
    result = s.lower().replace("-", "").replace(" ", "")
    for typo, correct in _TYPO_MAP.items():
        result = result.replace(typo, correct)
    return result


@lru_cache(maxsize=256)
def resolve_base_model(raw_model: str, maker: str | None = None) -> str:
    """
    isaac 앱의 차량 모델명을 택소노미 기준 모델명으로 해석.

    예:
      - "그랜드스타렉스" → "스타렉스"  (포함 매칭)
      - "카니발 R 리무진" → "카니발"   (포함 매칭)
      - "더 뉴 아반떼 MD" → "아반떼"   (세대접두사 제거 + 포함 매칭)
      - "E클래스" / "E클레스" → "E-클래스" (하이픈·오타 정규화)
    """
    if not raw_model or not raw_model.strip():
        return raw_model or ""

    cleaned = _GEN_PREFIX_RE.sub("", raw_model.strip()).strip()

    taxonomy = _load_taxonomy()

    # maker가 있으면 해당 제작사의 모델만, 없으면 전체
    if maker:
        maker_data = taxonomy.get(maker, {})
        model_names = list(maker_data.get("models", {}).keys())
    else:
        model_names = []
        for m_data in taxonomy.values():
            model_names.extend(m_data.get("models", {}).keys())

    if not model_names:
        return cleaned

    cleaned_norm = _normalize_for_match(cleaned)

    # 1) 정확 매칭 (정규화 후 비교)
    for name in model_names:
        if _normalize_for_match(name) == cleaned_norm:
            return name

    # 2) 포함 매칭 — taxonomy 모델이 cleaned에 포함되는지 (정규화 후, 길이 내림차순)
    sorted_names = sorted(model_names, key=len, reverse=True)
    for name in sorted_names:
        if _normalize_for_match(name) in cleaned_norm:
            return name

    return cleaned


# =========================================================================
# 공개 API 함수
# =========================================================================

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
    """특정 모델의 세대 목록 (display_name 포함)"""
    taxonomy = _load_taxonomy()
    model_data = taxonomy.get(maker, {}).get("models", {}).get(model, {})
    results = []
    for gen, data in model_data.get("generations", {}).items():
        display_name = _compute_display_name(gen, data)
        results.append({
            "generation": gen,
            "display_name": display_name,
            "variants": list(data.get("variants", {}).keys()),
        })
    return results


def get_variants(maker: str, model: str, generation: str) -> list[dict]:
    """특정 세대의 변형(연료/배기량) 목록"""
    taxonomy = _load_taxonomy()
    gen_data = (
        taxonomy.get(maker, {})
        .get("models", {}).get(model, {})
        .get("generations", {}).get(generation, {})
    )
    variants = []
    for vk, vd in gen_data.get("variants", {}).items():
        parts = vk.split("|")
        fuel = parts[0] if len(parts) > 0 and parts[0] != "N/A" else ""
        drive = parts[1] if len(parts) > 1 and parts[1] != "N/A" else ""
        displacement = parts[2] if len(parts) > 2 and parts[2] != "N/A" else ""

        # 사용자에게 보여줄 라벨 생성
        label_parts = []
        if fuel:
            label_parts.append(fuel)
        if displacement:
            label_parts.append(f"{displacement}L")
        if drive:
            label_parts.append(drive)
        label = " ".join(label_parts) if label_parts else vk

        trim_count = len(vd.get("trims", []))

        variants.append({
            "variant_key": vk,
            "fuel": fuel,
            "displacement": displacement,
            "drive": drive,
            "label": label,
            "trim_count": trim_count,
        })
    return variants


def get_trims(maker: str, model: str, generation: str,
              variant_key: str | None = None) -> list[str]:
    """특정 세대의 트림 목록 (variant_key로 필터 가능)"""
    taxonomy = _load_taxonomy()
    gen_data = (
        taxonomy.get(maker, {})
        .get("models", {}).get(model, {})
        .get("generations", {}).get(generation, {})
    )

    if variant_key:
        # 특정 variant만
        variant_data = gen_data.get("variants", {}).get(variant_key, {})
        return sorted(variant_data.get("trims", []))

    # 전체 variant의 트림 합산
    trims = set()
    for variant_data in gen_data.get("variants", {}).values():
        trims.update(variant_data.get("trims", []))
    return sorted(trims)
