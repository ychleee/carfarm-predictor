"""
CarFarm i3 모델 — LightGBM 기반 ML 가격 예측.

학습된 모델로 출고가 대비 비율을 예측하고, 출고가를 곱하여 가격 산출.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from datetime import datetime

import pandas as pd

from app.services.llm_price_predictor import PricePrediction

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

CURRENT_YEAR = datetime.now().year

# 싱글톤 모델 캐시
_auction_domestic_model = None
_auction_export_model = None
_retail_model = None


def _load_model(data_type: str):
    """pickle 모델 로드 (싱글톤)"""
    model_path = os.path.join(MODEL_DIR, f"{data_type}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML 모델 파일 없음: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("[ML] %s 모델 로드 완료 — %s", data_type, bundle.get("trained_at", ""))
    return bundle


def _get_auction_domestic_model():
    global _auction_domestic_model
    if _auction_domestic_model is None:
        _auction_domestic_model = _load_model("auction_domestic")
    return _auction_domestic_model


def _get_auction_export_model():
    global _auction_export_model
    if _auction_export_model is None:
        _auction_export_model = _load_model("auction_export")
    return _auction_export_model


def _get_retail_model():
    global _retail_model
    if _retail_model is None:
        _retail_model = _load_model("retail")
    return _retail_model


# 제조사 정규화 맵
_MAKER_MAP = {
    "현대": "hyundai", "기아": "kia", "제네시스": "genesis",
    "쉐보레(GM대우)": "chevrolet", "쉐보레": "chevrolet",
    "KG모빌리티(쌍용)": "ssangyong", "쌍용": "ssangyong",
    "kg모빌리티": "ssangyong",
    "르노코리아(삼성)": "renault", "르노삼성": "renault", "르노코리아": "renault",
    "BMW": "bmw", "벤츠": "benz", "메르세데스-��츠": "benz",
    "아우디": "audi", "폭스바겐": "volkswagen", "볼보": "volvo",
    "테슬라": "tesla", "토요타": "toyota", "혼다": "honda",
    "렉서스": "lexus", "미니": "mini", "포르쉐": "porsche",
    "링컨": "lincoln", "캐딜락": "cadillac", "지프": "jeep",
    "랜드로버": "landrover", "재규어": "jaguar",
    "마세라티": "maserati", "인피니티": "infiniti",
    "푸조": "peugeot", "포드": "ford",
}

# 연료 정규화 맵
_FUEL_MAP = {
    "가솔린": "gasoline", "디젤": "diesel", "경유": "diesel",
    "LPG": "lpg", "하이브리드": "hybrid", "전기": "electric",
    "가솔린+전기": "hybrid", "디젤+전기": "hybrid",
    "LPG+전기": "hybrid", "수소": "hydrogen",
}


def _build_features(target: dict) -> pd.DataFrame:
    """대상 차량 → 모델 입력 DataFrame"""
    year = target.get("year", 0) or 0
    mileage = target.get("mileage", 0) or 0
    factory_price = target.get("factory_price", 0) or 0
    base_price = target.get("base_price", 0) or 0

    # 만원 단위로 정규화
    if factory_price > 100000:
        factory_price = round(factory_price / 10000, 1)
    if base_price > 100000:
        base_price = round(base_price / 10000, 1)

    ref_price = factory_price if factory_price > 0 else base_price
    vehicle_age = CURRENT_YEAR - year if year > 0 else 0

    maker = target.get("maker", "")
    maker_norm = _MAKER_MAP.get(maker, "other")

    fuel = target.get("fuel", "")
    fuel_norm = _FUEL_MAP.get(fuel, "other")

    model_name = target.get("model", "")
    # 모델명 정규화: 택소노미 base model로 통일
    model_name = re.sub(r"\s*[\(（][^\)）]*[\)）]\s*$", "", model_name).strip()
    try:
        from app.services.taxonomy_search import resolve_base_model
        resolved = resolve_base_model(model_name, maker)
        if resolved:
            model_name = resolved
    except Exception:
        pass

    trim = target.get("trim", "") or ""
    _commercial_kw = ("영업", "택시", "렌터카", "렌트")
    is_commercial = 1 if any(kw in trim for kw in _commercial_kw) else 0
    is_rental = 1 if "렌터카" in trim or "렌트" in trim else 0

    row = {
        "vehicle_age": vehicle_age,
        "mileage": mileage,
        "ref_price": ref_price,
        "mileage_per_year": mileage / max(vehicle_age, 1),
        "has_damage": 0,
        "exchange_count": target.get("exchange_count", 0) or 0,
        "bodywork_count": target.get("bodywork_count", 0) or 0,
        "is_export": 0,
        "is_commercial": is_commercial,
        "is_rental": is_rental,
        "maker_norm": maker_norm,
        "model": model_name,
        "fuel_norm": fuel_norm,
    }

    df = pd.DataFrame([row])
    for col in ["maker_norm", "model", "fuel_norm"]:
        df[col] = df[col].astype("category")

    return df, ref_price


def _generate_ml_brackets(
    target: dict, ref_price: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """ML 모델로 10,000km 구간별 가격추이 생성."""
    mileage = target.get("mileage", 0) or 0
    year = target.get("year", 0) or 0
    vehicle_age = CURRENT_YEAR - year if year > 0 else 0

    max_km = max(mileage + 50000, 200000)
    max_km = ((max_km // 10000) + 1) * 10000
    bracket_starts = list(range(0, max_km, 10000))

    # 기본 feature 1행 생성 후 mileage만 바꿔가며 복제
    base_df, _ = _build_features(target)
    base_row = base_df.iloc[0]

    rows = []
    for start in bracket_starts:
        mid = start + 5000
        row = base_row.copy()
        row["mileage"] = mid
        row["mileage_per_year"] = mid / max(vehicle_age, 1)
        rows.append(row)

    all_df = pd.DataFrame(rows)
    for col in ["maker_norm", "model", "fuel_norm"]:
        all_df[col] = all_df[col].astype("category")

    # 배치 예측
    domestic_ratios = _get_auction_domestic_model()["model"].predict(all_df)
    export_ratios = _get_auction_export_model()["model"].predict(all_df)
    retail_ratios = _get_retail_model()["model"].predict(all_df)

    auction_brackets: list[dict] = []
    export_brackets: list[dict] = []
    retail_brackets: list[dict] = []

    for i, start in enumerate(bracket_starts):
        end = start + 10000

        d_r = max(float(domestic_ratios[i]), 0)
        d_p = round(d_r * ref_price / 10) * 10
        auction_brackets.append({"s": start, "e": end, "n": 0, "r": round(d_r * 100, 1), "mn": d_p, "mx": d_p})

        e_r = max(float(export_ratios[i]), 0)
        e_p = round(e_r * ref_price / 10) * 10
        export_brackets.append({"s": start, "e": end, "n": 0, "r": round(e_r * 100, 1), "mn": e_p, "mx": e_p})

        r_r = max(float(retail_ratios[i]), 0)
        r_p = round(r_r * ref_price / 10) * 10
        retail_brackets.append({"s": start, "e": end, "n": 0, "r": round(r_r * 100, 1), "mn": r_p, "mx": r_p})

    return auction_brackets, export_brackets, retail_brackets


def predict_price_ml(target: dict) -> PricePrediction:
    """
    ML 모델 기반 가격 예측 (i3).

    출고가 대비 비율을 예측하고, 출고가를 곱해서 최종 가격 산출.
    """
    factory_price = target.get("factory_price", 0) or 0
    base_price = target.get("base_price", 0) or 0
    if factory_price > 100000:
        factory_price = round(factory_price / 10000, 1)
    if base_price > 100000:
        base_price = round(base_price / 10000, 1)
    ref_price = factory_price if factory_price > 0 else base_price

    if ref_price <= 0:
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning="[i3-ML] 출고가/기본가 정보가 없어 ML 예측이 불가합니다.",
            vehicles_analyzed=0,
        )

    try:
        features_df, _ = _build_features(target)
        print(f"[ML input] model={target.get('model')}, maker={target.get('maker')}, "
              f"year={target.get('year')}, km={target.get('mileage')}, fuel={target.get('fuel')}, "
              f"fp={factory_price}, bp={base_price}, ref={ref_price}")

        # 낙찰(내수) 예측
        domestic_bundle = _get_auction_domestic_model()
        domestic_ratio = domestic_bundle["model"].predict(features_df)[0]
        estimated_auction = round(domestic_ratio * ref_price / 10) * 10

        # 낙찰(수출) 예측
        export_bundle = _get_auction_export_model()
        export_ratio = export_bundle["model"].predict(features_df)[0]
        estimated_auction_export = round(export_ratio * ref_price / 10) * 10

        # 소매 예측
        retail_bundle = _get_retail_model()
        retail_ratio = retail_bundle["model"].predict(features_df)[0]
        estimated_retail = round(retail_ratio * ref_price / 10) * 10

        print(f"[ML result] domestic={domestic_ratio:.4f}({estimated_auction}), "
              f"export={export_ratio:.4f}({estimated_auction_export}), "
              f"retail={retail_ratio:.4f}({estimated_retail})")

        # 정합성: 낙찰(내수) < 소매
        if estimated_auction >= estimated_retail * 0.97:
            estimated_auction = round(estimated_retail * 0.90 / 10) * 10

        # 바닥값
        estimated_auction = max(estimated_auction, 0)
        estimated_auction_export = max(estimated_auction_export, 0)
        estimated_retail = max(estimated_retail, 0)

        year = target.get("year", 0) or 0
        mileage = target.get("mileage", 0) or 0

        domestic_metrics = domestic_bundle.get("metrics", {})
        export_metrics = export_bundle.get("metrics", {})
        retail_metrics = retail_bundle.get("metrics", {})

        # 신뢰도 판단
        confidence = "보통"
        if year >= CURRENT_YEAR - 5 and mileage < 150000:
            confidence = "높음"
        elif year < CURRENT_YEAR - 10 or mileage > 250000:
            confidence = "낮음"

        reasoning = (
            f"[i3-ML] LightGBM 모델 기반 예측\n"
            f"출고가 {ref_price:,.0f}만원 대비 "
            f"내수 {domestic_ratio*100:.1f}%, 수출 {export_ratio*100:.1f}%, "
            f"소매 {retail_ratio*100:.1f}%\n"
            f"학습: 내수 {domestic_metrics.get('train_size', 0):,}건, "
            f"수출 {export_metrics.get('train_size', 0):,}건, "
            f"소매 {retail_metrics.get('train_size', 0):,}건"
        )

        # 주행거리별 가격추이 생성
        auction_brackets, export_brackets, retail_brackets = _generate_ml_brackets(target, ref_price)

        # 구간별 비율추이 텍스트 생성
        target_bucket = (mileage // 10000) * 10000

        def _brackets_text(brackets: list[dict], label: str) -> str:
            lines = [f"\n\n[{label} 구간별 비율추이]"]
            for b in brackets:
                s = b["s"] // 10000
                mark = "★" if b["s"] == target_bucket else " "
                lines.append(f"{mark} {s}~{s+1}만km: {b['r']}% ({b['mn']:,}만)")
            return "\n".join(lines)

        auction_reasoning = (
            f"[ML 내수] 출고가 {ref_price:,.0f}만 × {domestic_ratio*100:.1f}% = "
            f"{estimated_auction:,.0f}만원 (MAPE {domestic_metrics.get('mape', 0):.1f}%)"
        )
        export_reasoning = (
            f"[ML 수출] 출고가 {ref_price:,.0f}만 × {export_ratio*100:.1f}% = "
            f"{estimated_auction_export:,.0f}만원 (MAPE {export_metrics.get('mape', 0):.1f}%)"
        )
        retail_reasoning = (
            f"[ML 소매] 출고가 {ref_price:,.0f}만 × {retail_ratio*100:.1f}% = "
            f"{estimated_retail:,.0f}만원 (MAPE {retail_metrics.get('mape', 0):.1f}%)"
        )

        return PricePrediction(
            estimated_auction=estimated_auction,
            estimated_auction_export=estimated_auction_export,
            estimated_retail=estimated_retail,
            confidence=confidence,
            reasoning=reasoning,
            auction_reasoning=auction_reasoning,
            export_reasoning=export_reasoning,
            retail_reasoning=retail_reasoning,
            vehicles_analyzed=0,
            auction_stats={},
            retail_stats={},
            auction_brackets=auction_brackets,
            export_brackets=export_brackets,
            retail_brackets=retail_brackets,
        )

    except Exception as e:
        logger.exception("[ML] 예측 오류")
        return PricePrediction(
            estimated_auction=0,
            estimated_retail=0,
            confidence="낮음",
            reasoning=f"[i3-ML] 예측 오류: {e}",
            vehicles_analyzed=0,
        )
