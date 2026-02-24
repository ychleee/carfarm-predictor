"""
CarFarm v2 — 낙찰 DB 검색 서비스

LLM 리즈닝 모델이 호출하는 Tool로 사용됨.
정제된 CSV 데이터를 pandas로 로드하여 검색.

제공 기능:
  1. search_auction_db() — 조건 기반 낙찰 차량 검색
  2. get_vehicle_detail() — 특정 차량 상세 정보
  3. get_price_stats() — 모델별 시세 통계

향후: CSV → PostgreSQL/SQLite 전환 예정
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache


DATA_PATH = Path(__file__).parent.parent.parent.parent / "car_price_prediction" / "output"

# 한글 혼용 표기 정규화 (ㅓ↔ㅕ, ㅐ↔ㅔ 등)
_NORMALIZE_MAP = {
    '그랜저': '그랜져', '그랜져': '그랜져',
    '쏘나타': '쏘나타', '소나타': '쏘나타',
    '싼타페': '싼타페', '산타페': '싼타페',
    '투싼': '투싼', '투산': '투싼',
    '쏘렌토': '쏘렌토', '소렌토': '쏘렌토',
    '아반떼': '아반떼', '아반테': '아반떼',
}


def _normalize_query(q: str) -> str:
    """검색어 정규화 — 한글 혼용 표기 처리"""
    for alt, canonical in _NORMALIZE_MAP.items():
        if alt in q:
            q = q.replace(alt, canonical)
    return q


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    """정제된 낙찰 데이터 로드 (싱글톤)"""
    clean_file = DATA_PATH / "domestic_clean_data.csv"
    if not clean_file.exists():
        # fallback
        clean_file = DATA_PATH / "cleaned_auction_data.csv"
    if not clean_file.exists():
        raise FileNotFoundError(f"정제된 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    df = pd.read_csv(clean_file, low_memory=False)

    # 날짜 파싱
    if '개최일' in df.columns:
        df['개최일'] = pd.to_datetime(df['개최일'], errors='coerce')

    # 인덱스 생성 (auction_id)
    df['auction_id'] = df.index.astype(str).str.zfill(6)

    return df


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
    낙찰 DB에서 조건에 맞는 차량 검색.

    Args:
        maker: 제작사 (필수)
        model: 모델명 (필수)
        generation: 세대 코드 (예: GN7)
        year_min/max: 연식 범위
        fuel: 연료 (가솔린/디젤/하이브리드 등)
        drive: 구동방식 (2WD/4WD)
        trim: 트림명
        mileage_max: 최대 주행거리 (km)
        usage: 차량경력 (personal/rental)
        domestic_only: 내수만 검색 (기본 True)
        limit: 최대 결과 수
        sort_by: 정렬 기준 (날짜/가격)

    Returns:
        검색 결과 리스트 (dict)
    """
    df = _load_data()
    mask = pd.Series(True, index=df.index)

    # 검색어 정규화
    maker = _normalize_query(maker)
    model = _normalize_query(model)

    # 필수 필터: 제작사 + 모델
    if 'maker' in df.columns:
        mask &= df['maker'].str.contains(maker, na=False, case=False)
    elif '제작사' in df.columns:
        mask &= df['제작사'].str.contains(maker, na=False, case=False)

    if 'model_name' in df.columns:
        mask &= df['model_name'].str.contains(model, na=False, case=False)
    elif '모델' in df.columns:
        mask &= df['모델'].str.contains(model, na=False, case=False)

    # 선택 필터
    if generation and 'generation' in df.columns:
        mask &= df['generation'].str.contains(generation, na=False, case=False)

    if year_min and '연식' in df.columns:
        mask &= df['연식'] >= year_min
    if year_max and '연식' in df.columns:
        mask &= df['연식'] <= year_max

    if fuel and 'fuel' in df.columns:
        mask &= df['fuel'].str.contains(fuel, na=False, case=False)

    if drive and 'drive' in df.columns:
        mask &= df['drive'].str.contains(drive, na=False, case=False)

    if trim and 'trim' in df.columns:
        mask &= df['trim'].str.contains(trim, na=False, case=False)

    if mileage_max and '주행거리' in df.columns:
        mask &= df['주행거리'] <= mileage_max

    if usage and 'usage_type' in df.columns:
        mask &= df['usage_type'] == usage

    if domestic_only and 'is_export' in df.columns:
        mask &= df['is_export'] == 0

    # 낙찰가 있는 것만
    if '낙찰가' in df.columns:
        mask &= df['낙찰가'] > 0

    result = df[mask].copy()

    # 정렬
    if sort_by == "날짜" and '개최일' in result.columns:
        result = result.sort_values('개최일', ascending=False)
    elif sort_by == "가격" and '낙찰가' in result.columns:
        result = result.sort_values('낙찰가', ascending=False)

    result = result.head(limit)

    # 반환용 컬럼
    return_cols = ['auction_id', '차명', '연식', '주행거리', '낙찰가', '색상',
                   '옵션', '개최일', 'usage_type', 'trim', 'segment',
                   'grade_score', 'exchange_count', 'bodywork_count',
                   'color_group', 'accident_severity']
    available = [c for c in return_cols if c in result.columns]

    records = result[available].to_dict('records')
    # datetime → string
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif hasattr(v, 'isoformat'):
                r[k] = v.isoformat()[:10]
    return records


def get_vehicle_detail(auction_id: str) -> dict | None:
    """특정 낙찰 차량의 전체 상세 정보"""
    df = _load_data()
    match = df[df['auction_id'] == auction_id]
    if match.empty:
        return None

    record = match.iloc[0].to_dict()
    # NaN/datetime 처리
    for k, v in record.items():
        if pd.isna(v):
            record[k] = None
        elif hasattr(v, 'isoformat'):
            record[k] = v.isoformat()[:10]
    return record


def get_price_stats(
    maker: str,
    model: str,
    generation: str | None = None,
    year: int | None = None,
    months: int = 3,
) -> dict:
    """
    모델별 시세 통계 (최근 N개월).

    Returns:
        {count, mean, median, min, max, std, period}
    """
    df = _load_data()
    mask = pd.Series(True, index=df.index)

    maker = _normalize_query(maker)
    model = _normalize_query(model)

    if 'maker' in df.columns:
        mask &= df['maker'].str.contains(maker, na=False, case=False)
    if 'model_name' in df.columns:
        mask &= df['model_name'].str.contains(model, na=False, case=False)
    if generation and 'generation' in df.columns:
        mask &= df['generation'].str.contains(generation, na=False, case=False)
    if year and '연식' in df.columns:
        mask &= df['연식'] == year
    if 'is_export' in df.columns:
        mask &= df['is_export'] == 0
    if '낙찰가' in df.columns:
        mask &= df['낙찰가'] > 0

    # 최근 N개월 필터
    if '개최일' in df.columns:
        max_date = df['개최일'].max()
        if pd.notna(max_date):
            min_date = max_date - pd.DateOffset(months=months)
            mask &= df['개최일'] >= min_date

    result = df.loc[mask, '낙찰가'] if '낙찰가' in df.columns else pd.Series()

    if result.empty:
        return {"count": 0, "message": "해당 조건의 데이터가 없습니다"}

    return {
        "maker": maker,
        "model": model,
        "generation": generation,
        "year": year,
        "period_months": months,
        "count": int(len(result)),
        "mean": round(float(result.mean()), 1),
        "median": round(float(result.median()), 1),
        "min": round(float(result.min()), 1),
        "max": round(float(result.max()), 1),
        "std": round(float(result.std()), 1),
    }
