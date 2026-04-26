"""
LightGBM 기반 차량 가격 예측 모델 학습.

낙찰 모델 + 소매 모델을 각각 학습하여 저장.

Features:
- maker (categorical)
- model (categorical)
- year → vehicle_age (numeric)
- mileage (numeric)
- fuel (categorical)
- factory_price (numeric)
- base_price (numeric)
- price_ratio = price / ref_price (target as ratio)
- usage_type (categorical)
- is_export (binary)
- exchange_count, bodywork_count (numeric)

Usage:
    cd backend
    python -m scripts.train_ml_model
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

CURRENT_YEAR = datetime.now().year


def load_data(data_type: str) -> pd.DataFrame:
    """JSON → DataFrame"""
    path = os.path.join(DATA_DIR, f"{data_type}_training.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"[{data_type}] 원본: {len(df)}건")
    return df


def _normalize_model_name(name: str, maker: str = "") -> str:
    """모델명 정규화: 택소노미 기준 base model명으로 통일.

    '디 올 뉴 그랜저(GN7)' → '그랜져'
    '그랜저 IG' → '그랜져'
    """
    import re
    # 1) 세대 코드 괄호 제거
    name = re.sub(r"\s*[\(（][^\)）]*[\)）]\s*$", "", name).strip()
    # 2) 택소노미 base model 해석
    try:
        from app.services.taxonomy_search import resolve_base_model
        resolved = resolve_base_model(name, maker if maker else None)
        if resolved:
            return resolved
    except Exception:
        pass
    return name


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """특성 엔지니어링"""
    df = df.copy()

    # 모델명 정규화 (택소노미 base model로 통일)
    df["model"] = df.apply(lambda r: _normalize_model_name(r["model"], r["maker"]), axis=1)

    # 차량 나이
    df["vehicle_age"] = CURRENT_YEAR - df["year"]

    # 출고가 기준 (factory > base)
    df["ref_price"] = df["factory_price"].where(df["factory_price"] > 0, df["base_price"])

    # 가격 비율 (출고가 대비)
    df["price_ratio"] = df["price"] / df["ref_price"]
    # 비율 이상치 제거
    df = df[(df["price_ratio"] > 0.03) & (df["price_ratio"] < 0.98)]

    # 주행거리/나이 비율
    df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].clip(lower=1)

    # 사고 여부
    df["has_damage"] = ((df["exchange_count"] > 0) | (df["bodywork_count"] > 0)).astype(int)

    # 연료 정규화
    fuel_map = {
        "가솔린": "gasoline", "디젤": "diesel", "경유": "diesel",
        "LPG": "lpg", "하이브리드": "hybrid", "전기": "electric",
        "가솔린+전기": "hybrid", "디젤+전기": "hybrid",
        "LPG+전기": "hybrid", "수소": "hydrogen",
    }
    df["fuel_norm"] = df["fuel"].map(fuel_map).fillna("other")

    # 제조사 정규화
    maker_map = {
        "현대": "hyundai", "기아": "kia", "제네시스": "genesis",
        "쉐보레(GM대우)": "chevrolet", "쉐보레": "chevrolet",
        "KG모빌리티(쌍용)": "ssangyong", "쌍용": "ssangyong",
        "르노코리아(삼성)": "renault", "르노삼성": "renault",
        "BMW": "bmw", "벤츠": "benz", "아우디": "audi",
        "폭스바겐": "volkswagen", "볼보": "volvo",
        "테슬라": "tesla", "토요타": "toyota", "혼다": "honda",
        "렉서스": "lexus", "미니": "mini", "포르쉐": "porsche",
        "링컨": "lincoln", "캐딜락": "cadillac", "지프": "jeep",
        "랜드로버": "landrover", "재규어": "jaguar",
        "마세라티": "maserati", "인피니티": "infiniti",
        "푸조": "peugeot", "포드": "ford",
    }
    df["maker_norm"] = df["maker"].map(maker_map).fillna("other")

    # 용도 정규화 (영업용/렌터카 플래그)
    df["is_commercial"] = (df["usage_type"] == "commercial").astype(int)
    df["is_rental"] = (df["usage_type"] == "rental").astype(int)

    return df


FEATURE_COLS = [
    "vehicle_age",
    "mileage",
    "ref_price",
    "mileage_per_year",
    "has_damage",
    "exchange_count",
    "bodywork_count",
    "is_export",
    "is_commercial",
    "is_rental",
]

CAT_FEATURE_COLS = [
    "maker_norm",
    "model",
    "fuel_norm",
]


def train_model(df: pd.DataFrame, data_type: str) -> dict:
    """LightGBM 모델 학습"""
    df = engineer_features(df)
    print(f"[{data_type}] 특성 엔지니어링 후: {len(df)}건")

    # 타겟: 출고가 대비 가격 비율 (비율 예측 → 출고가 곱해서 가격 산출)
    target = "price_ratio"

    # Categorical encoding
    for col in CAT_FEATURE_COLS:
        df[col] = df[col].astype("category")

    features = FEATURE_COLS + CAT_FEATURE_COLS

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42,
    )

    # 실제 가격으로 변환하여 평가하기 위해 ref_price 보관
    ref_prices_test = df.loc[X_test.index, "ref_price"].values
    actual_prices_test = df.loc[X_test.index, "price"].values

    print(f"  학습: {len(X_train)}건, 테스트: {len(X_test)}건")

    # LightGBM 파라미터
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CAT_FEATURE_COLS)
    valid_data = lgb.Dataset(X_test, label=y_test, categorical_feature=CAT_FEATURE_COLS, reference=train_data)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    # 평가
    y_pred = model.predict(X_test)
    pred_prices = y_pred * ref_prices_test

    mae_ratio = mean_absolute_error(y_test, y_pred)
    mae_price = mean_absolute_error(actual_prices_test, pred_prices)
    mape = mean_absolute_percentage_error(actual_prices_test, pred_prices) * 100
    r2 = r2_score(actual_prices_test, pred_prices)

    print(f"\n[{data_type}] 평가 결과:")
    print(f"  MAE (비율): {mae_ratio:.4f}")
    print(f"  MAE (가격): {mae_price:.1f}만원")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²: {r2:.4f}")

    # 가격 구간별 정확도
    bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10000), (10000, 30000)]
    print(f"\n  가격 구간별 MAPE:")
    for lo, hi in bins:
        mask = (actual_prices_test >= lo) & (actual_prices_test < hi)
        if mask.sum() > 10:
            seg_mape = mean_absolute_percentage_error(actual_prices_test[mask], pred_prices[mask]) * 100
            print(f"    {lo:>5}~{hi:>5}만: {seg_mape:5.1f}% ({mask.sum()}건)")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importance (top 10):")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat}: {imp:.0f}")

    # 모델 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{data_type}_model.pkl")

    model_bundle = {
        "model": model,
        "features": features,
        "cat_features": CAT_FEATURE_COLS,
        "target": target,
        "metrics": {
            "mae_ratio": mae_ratio,
            "mae_price": mae_price,
            "mape": mape,
            "r2": r2,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n  모델 저장: {model_path}")

    return model_bundle


def main():
    auction_df = load_data("auction")

    # 낙찰 내수
    domestic_df = auction_df[auction_df["is_export"] == 0].copy()
    print("=" * 60)
    print(f"낙찰(내수) 모델 학습 — {len(domestic_df)}건")
    print("=" * 60)
    domestic_bundle = train_model(domestic_df, "auction_domestic")

    # 낙찰 수출
    export_df = auction_df[auction_df["is_export"] == 1].copy()
    print("\n" + "=" * 60)
    print(f"낙찰(수출) 모델 학습 — {len(export_df)}건")
    print("=" * 60)
    export_bundle = train_model(export_df, "auction_export")

    # 소매
    print("\n" + "=" * 60)
    print("소매 모델 학습")
    print("=" * 60)
    retail_df = load_data("retail")
    retail_bundle = train_model(retail_df, "retail")

    print("\n" + "=" * 60)
    print("학습 완료 요약")
    print("=" * 60)
    for name, bundle in [("낙찰(내수)", domestic_bundle), ("낙찰(수출)", export_bundle), ("소매", retail_bundle)]:
        m = bundle["metrics"]
        print(f"  [{name}] MAE={m['mae_price']:.1f}만, MAPE={m['mape']:.1f}%, R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()
