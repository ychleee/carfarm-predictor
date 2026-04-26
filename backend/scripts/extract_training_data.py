"""
Firestore vehicles 컬렉션에서 ML 학습용 데이터 추출.

AI 가격예측과 동일한 품질 필터 적용:
- companyId별 분류: 엔카(소매) / 헤이딜러(낙찰)
- 출고가/기본가 적절성 검증
- 가격 범위 검증
- 사고차(출고가=낙찰가) 제외

Usage:
    cd backend
    python -m scripts.extract_training_data
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud.firestore_v1.base_query import FieldFilter
from app.services.firestore_client import get_firestore_db

ENCAR_COMPANY_ID = "KYMaGfcnzwGsvbDm6Z91"
HEYDEALER_COMPANY_ID = "cRFWlHv4PZczXpd8bEw2"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _safe_number(val, default=0) -> float:
    if val is None:
        return default
    try:
        if isinstance(val, str):
            result = float(val.replace(",", ""))
        else:
            result = float(val)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _normalize_price(price: float) -> float:
    """가격 → 만원 단위"""
    if price > 100000:
        return round(price / 10000, 1)
    return round(price, 1)


def _parse_date(doc_dict: dict) -> str:
    """판매일 추출"""
    for field in ("saleDate", "updatedAt", "createdAt"):
        val = doc_dict.get(field)
        if val is None:
            continue
        try:
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            if hasattr(val, "seconds"):
                return datetime.fromtimestamp(val.seconds, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def _extract_vehicle(doc_dict: dict, doc_id: str, data_type: str) -> dict | None:
    """
    Firestore 문서 → 학습용 dict.

    AI 가격예측과 동일한 품질 필터:
    1. 가격 유효 범위
    2. 출고가/기본가 적절성
    3. 출고가 ≈ 낙찰가(소매가) → 이상치 제외
    """
    # ── 가격 ──
    raw_price = _safe_number(doc_dict.get("actualBidPrice", 0))
    if raw_price <= 0:
        return None
    price = _normalize_price(raw_price)
    if price < 100 or price > 20000:  # 100만~2억
        return None

    # ── 기본 정보 ──
    maker = (doc_dict.get("vehicleMaker") or "").strip()
    model = (doc_dict.get("vehicleModel") or "").strip()
    if not maker or not model:
        return None

    year = int(_safe_number(doc_dict.get("vehicleYear", 0)))
    if year < 2005 or year > 2026:
        return None

    mileage = int(_safe_number(doc_dict.get("mileage", 0)))
    if mileage < 0 or mileage > 500000:
        return None

    # ── 출고가/기본가 검증 (핵심 필터) ──
    factory_price = _normalize_price(_safe_number(doc_dict.get("vehicleFactoryPrice", 0)))
    base_price = _normalize_price(_safe_number(doc_dict.get("vehicleBasePrice", 0)))
    ref_price = factory_price if factory_price > 0 else base_price

    # 출고가가 없거나 비정상인 데이터 제외
    if ref_price < 500:  # 출고가 500만 미만 → 이상치
        return None

    # 출고가 ≈ 실거래가 → 이상치 (AI예측에서 _remove_same_as_factory와 동일)
    if ref_price > 0 and price > 0:
        ratio = price / ref_price
        if ratio > 0.98:  # 거래가가 출고가의 98% 이상 → 비정상
            return None

    # 거래가가 출고가의 5% 미만 → 이상치
    if ref_price > 0 and price > 0:
        if price / ref_price < 0.05:
            return None

    # ── 기타 필드 ──
    fuel = (doc_dict.get("fuelType") or "").strip()
    trim = (doc_dict.get("vehicleTrim") or "").strip()
    color = (doc_dict.get("vehicleColor") or "").strip()
    generation = (doc_dict.get("generation") or "").strip()
    category = (doc_dict.get("vehicleCategory") or "").strip()

    # 차량 용도
    purpose = (doc_dict.get("vehiclePurpose") or "").strip()
    usage_map = {"자가용": "personal", "렌터카": "rental", "영업용": "commercial", "관용": "government"}
    usage_type = usage_map.get(purpose, "personal")

    # 수출 여부
    sale_dest = (doc_dict.get("saleDestination") or "")
    is_export = 1 if "수출" in sale_dest else 0

    # 사고 이력
    exchange_count = int(_safe_number(doc_dict.get("exchangeCount", 0)))
    bodywork_count = int(_safe_number(doc_dict.get("bodyworkCount", 0)))

    # 배기량
    displacement = (doc_dict.get("engineDisplacement") or "").strip()

    sale_date = _parse_date(doc_dict)

    return {
        "doc_id": doc_id,
        "data_type": data_type,
        "maker": maker,
        "model": model,
        "year": year,
        "mileage": mileage,
        "fuel": fuel,
        "trim": trim,
        "color": color,
        "generation": generation,
        "category": category,
        "displacement": displacement,
        "factory_price": factory_price,
        "base_price": base_price,
        "usage_type": usage_type,
        "is_export": is_export,
        "exchange_count": exchange_count,
        "bodywork_count": bodywork_count,
        "sale_date": sale_date,
        "price": price,
    }


def _query_by_company(db, company_id: str, data_type: str) -> list[dict]:
    """companyId 인덱스 기반 쿼리 — 배치 페이징으로 타임아웃 방지"""
    col = db.collection("vehicles")
    results = []
    skipped = 0
    last_doc = None
    batch_size = 5000
    batch_num = 0

    while True:
        batch_num += 1
        query = col.where(filter=FieldFilter("companyId", "==", company_id))
        query = query.order_by("__name__").limit(batch_size)

        if last_doc:
            query = query.start_after(last_doc)

        docs = list(query.stream())
        if not docs:
            break

        for doc in docs:
            row = _extract_vehicle(doc.to_dict(), doc.id, data_type)
            if row:
                results.append(row)
            else:
                skipped += 1

        last_doc = docs[-1]
        print(f"  [{data_type}] 배치 {batch_num}: {len(docs)}건 → 유효 {len(results)}건 (스킵 {skipped}건)")

        if len(docs) < batch_size:
            break

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    db = get_firestore_db()

    # 1) 낙찰 데이터 (헤이딜러)
    print("=" * 60)
    print("낙찰 데이터 추출 (헤이딜러)...")
    print("=" * 60)
    auction_data = _query_by_company(db, HEYDEALER_COMPANY_ID, "auction")

    # 2) 소매 데이터 (엔카)
    print("\n" + "=" * 60)
    print("소매 데이터 추출 (엔카)...")
    print("=" * 60)
    retail_data = _query_by_company(db, ENCAR_COMPANY_ID, "retail")

    # 3) 저장
    print(f"\n{'=' * 60}")
    print(f"결과 요약")
    print(f"{'=' * 60}")
    print(f"  낙찰 데이터: {len(auction_data)}건")
    print(f"  소매 데이터: {len(retail_data)}건")

    # 제조사별 분포
    for dtype, data in [("낙찰", auction_data), ("소매", retail_data)]:
        from collections import Counter
        maker_cnt = Counter(r["maker"] for r in data)
        print(f"\n  [{dtype}] 제조사 분포 (상위 10):")
        for m, c in maker_cnt.most_common(10):
            print(f"    {m}: {c}건")

    auction_path = os.path.join(OUTPUT_DIR, "auction_training.json")
    retail_path = os.path.join(OUTPUT_DIR, "retail_training.json")

    with open(auction_path, "w", encoding="utf-8") as f:
        json.dump(auction_data, f, ensure_ascii=False)
    print(f"\n낙찰 데이터 저장: {auction_path}")

    with open(retail_path, "w", encoding="utf-8") as f:
        json.dump(retail_data, f, ensure_ascii=False)
    print(f"소매 데이터 저장: {retail_path}")


if __name__ == "__main__":
    main()
