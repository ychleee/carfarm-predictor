"""
피드백 기반 다중 파라미터 학습 엔진

세그먼트별 5개 파라미터를 순차적 잔차 분해로 학습:
  1) scale_factor  — 스케일 보정 (×0.70 ~ ×1.30)
  2) price_bias    — 고정 바이어스 (-200 ~ +200 만원)
  3) mileage_slope — 주행거리별 보정 (-0.05 ~ +0.05 /만km)
  4) exclusion_pct — 이상치 제거 비율 (0 ~ 0.40)
  5) direction     — 제거 방향 ("up" | "down" | "none")

- 세그먼트 키: {maker}::{model}::{trim첫토큰}::{2년구간}::{3만km구간}
- 시간감쇠: half_life=60일
- 신뢰도: min(effective_n / 5, 1.0)
- Warm-up: effective_n에 따라 활성 파라미터 점진 확대
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from app.services.firestore_client import get_firestore_db

logger = logging.getLogger(__name__)

HALF_LIFE_DAYS = 60
MAX_EXCLUSION_PCT = 0.40
MIN_RATIO = 0.70
MAX_RATIO = 1.30
MAX_SCALE = 1.30
MIN_SCALE = 0.70
MAX_BIAS = 200.0       # 만원
MAX_SLOPE = 0.05       # /만km


@dataclass
class LearnedParams:
    # 사전 필터 (Gaussian 평활 전 적용)
    exclusion_pct: float = 0.0     # 0 ~ 0.40
    direction: str = "none"        # "up" | "down" | "none"

    # 사후 보정 (Gaussian 평활 후 적용)
    scale_factor: float = 1.0      # 0.70 ~ 1.30
    price_bias: float = 0.0        # -200 ~ +200 만원
    mileage_slope: float = 0.0     # -0.05 ~ +0.05 (10,000km당 비율 변화)
    ref_mileage: float = 0.0       # mileage_slope 기준점

    # 메타
    confidence: float = 0.0
    feedback_count: int = 0
    effective_n: float = 0.0
    segment_key: str = ""
    auto_cal_count: int = 0
    manual_feedback_count: int = 0


# ── 하위 호환: 기존 코드에서 FeedbackFilter 참조 시 ──
FeedbackFilter = LearnedParams


def build_segment_key(maker: str, model: str, trim: str, year: int, mileage: int) -> str:
    """세그먼트 키 생성: maker::model::trim첫토큰::2년구간::3만km구간"""
    trim_token = trim.split()[0] if trim and trim.strip() else "_"
    year_bucket_start = (year // 2) * 2
    year_bucket = f"{year_bucket_start}-{year_bucket_start + 1}"
    mileage_bucket = (mileage // 30000) * 30000
    return f"{maker}::{model}::{trim_token}::{year_bucket}::{mileage_bucket}"


# =========================================================================
# 헬퍼: 가중 중앙값 / 가중 선형회귀
# =========================================================================

def _weighted_median(values: list[float], weights: list[float]) -> float:
    """가중 중앙값 (이상치에 robust)."""
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= total * 0.5:
            return v
    return pairs[-1][0]


def _weighted_regression_slope(
    xs: list[float], ys: list[float], weights: list[float],
) -> float:
    """가중 선형회귀 기울기. 데이터 부족 시 0 반환."""
    n = len(xs)
    if n < 2:
        return 0.0
    w_sum = sum(weights)
    if w_sum <= 0:
        return 0.0
    wx = sum(w * x for w, x in zip(weights, xs))
    wy = sum(w * y for w, y in zip(weights, ys))
    wxx = sum(w * x * x for w, x in zip(weights, xs))
    wxy = sum(w * x * y for w, x, y in zip(weights, xs, ys))
    denom = w_sum * wxx - wx ** 2
    if abs(denom) < 1e-10:
        return 0.0
    return (w_sum * wxy - wx * wy) / denom


# =========================================================================
# 학습 알고리즘: 순차적 잔차 분해
# =========================================================================

def _compute_learned_params(feedbacks: list[dict], price_type: str) -> LearnedParams:
    """피드백 리스트로부터 5개 파라미터 산출 (시간감쇠 가중, 순차적 잔차 분해)."""
    now = datetime.now(timezone.utc)

    # 피드백 파싱 + 시간 가중치
    entries: list[dict] = []
    for fb in feedbacks:
        if fb.get("price_type") != price_type:
            continue
        predicted = fb.get("predicted_price", 0)
        target = fb.get("target_price", 0)
        if predicted <= 0 or target <= 0:
            continue
        ratio = max(MIN_RATIO, min(MAX_RATIO, target / predicted))
        mileage = fb.get("mileage", 0) or 0

        created_at = fb.get("created_at")
        if created_at is None:
            age_days = 0.0
        elif isinstance(created_at, datetime):
            age_days = max(
                (now - created_at.replace(
                    tzinfo=timezone.utc if created_at.tzinfo is None else created_at.tzinfo
                )).total_seconds() / 86400,
                0,
            )
        else:
            age_days = 0.0

        decay = math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)
        entries.append({
            "ratio": ratio,
            "predicted": predicted,
            "target": target,
            "mileage": mileage,
            "decay": decay,
        })

    if not entries:
        return LearnedParams(segment_key="")

    count = len(entries)
    weights = [e["decay"] for e in entries]
    effective_n = sum(weights)
    confidence = min(effective_n / 5.0, 1.0)

    # ── Phase 1: scale_factor (n≥1) ──
    ratios = [e["ratio"] for e in entries]
    raw_scale = _weighted_median(ratios, weights)
    raw_scale = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

    # Warm-up 감쇠: effective_n < 2 이면 강하게 감쇠
    if effective_n < 2:
        scale_damping = min(effective_n / 2.0, 1.0) * confidence
    else:
        scale_damping = confidence
    effective_scale = 1.0 + (raw_scale - 1.0) * scale_damping

    # ── Phase 2: price_bias (n≥3) ──
    raw_bias = 0.0
    if effective_n >= 3:
        residuals_bias = [
            e["target"] - e["predicted"] * effective_scale
            for e in entries
        ]
        raw_bias = _weighted_median(residuals_bias, weights)
        raw_bias = max(-MAX_BIAS, min(MAX_BIAS, raw_bias))
    effective_bias = raw_bias * confidence

    # ── Phase 3: mileage_slope (n≥5, spread≥20000km) ──
    raw_slope = 0.0
    ref_mileage = 0.0
    mileages = [e["mileage"] for e in entries]
    mileage_spread = max(mileages) - min(mileages) if mileages else 0

    if effective_n >= 5 and mileage_spread >= 20000:
        ref_mileage = _weighted_median(
            [float(m) for m in mileages], weights,
        )
        residuals_slope = [
            e["target"] - (e["predicted"] * effective_scale + effective_bias)
            for e in entries
        ]
        xs_km = [(e["mileage"] - ref_mileage) / 10000 for e in entries]
        raw_slope = _weighted_regression_slope(xs_km, residuals_slope, weights)
        # slope를 비율로 변환 (가격 절대값 → 비율)
        avg_predicted = sum(e["predicted"] * w for e, w in zip(entries, weights)) / effective_n
        if avg_predicted > 0:
            raw_slope = raw_slope / avg_predicted
        raw_slope = max(-MAX_SLOPE, min(MAX_SLOPE, raw_slope))

    mileage_confidence = min(confidence, mileage_spread / 30000) if mileage_spread > 0 else 0
    effective_slope = raw_slope * mileage_confidence

    # ── Phase 4: exclusion_pct / direction (최종 잔차의 체계적 오차) ──
    final_residuals = []
    for e in entries:
        delta_km = (e["mileage"] - ref_mileage) / 10000 if ref_mileage else 0
        predicted_corrected = (
            e["predicted"] * (effective_scale + effective_slope * delta_km)
            + effective_bias
        )
        if predicted_corrected > 0:
            final_residuals.append(e["target"] / predicted_corrected)

    if final_residuals and effective_n >= 3:
        final_ratio = _weighted_median(final_residuals, weights[:len(final_residuals)])
        excl = min(MAX_EXCLUSION_PCT, abs(final_ratio - 1.0) * confidence)
        if final_ratio > 1.01:
            direction = "up"
        elif final_ratio < 0.99:
            direction = "down"
        else:
            direction = "none"
    else:
        excl = 0.0
        direction = "none"

    return LearnedParams(
        exclusion_pct=round(excl, 4),
        direction=direction,
        scale_factor=round(effective_scale, 4),
        price_bias=round(effective_bias, 1),
        mileage_slope=round(effective_slope, 6),
        ref_mileage=round(ref_mileage, 0),
        confidence=round(confidence, 4),
        feedback_count=count,
        effective_n=round(effective_n, 2),
        segment_key="",
    )


# =========================================================================
# Firestore 조회 / 저장
# =========================================================================

def _params_from_doc(data: dict, segment_key: str, fallback: bool = False) -> LearnedParams:
    """Firestore 문서 → LearnedParams (하위 호환: 새 필드 없으면 기본값)."""
    params = LearnedParams(
        direction=data.get("direction", "none"),
        exclusion_pct=data.get("exclusion_pct", 0.0),
        scale_factor=data.get("scale_factor", 1.0),
        price_bias=data.get("price_bias", 0.0),
        mileage_slope=data.get("mileage_slope", 0.0),
        ref_mileage=data.get("ref_mileage", 0.0),
        confidence=data.get("confidence", 0.0),
        feedback_count=data.get("feedback_count", 0),
        effective_n=data.get("effective_n", 0.0),
        segment_key=segment_key,
    )
    # 폴백(maker::model)이면 세그먼트 특화 파라미터 제외
    if fallback:
        params.mileage_slope = 0.0
        params.ref_mileage = 0.0
        params.exclusion_pct = 0.0
        params.direction = "none"
    return params


def get_learned_params(
    maker: str, model: str, trim: str, year: int, mileage: int,
    price_type: str,
) -> LearnedParams:
    """세그먼트별 학습 파라미터 조회 (Firestore feedbackFilters)."""
    db = get_firestore_db()
    segment_key = build_segment_key(maker, model, trim, year, mileage)

    # 1차: 정확 매칭 → 5개 파라미터 전부
    doc = db.collection("feedbackFilters").document(f"{segment_key}::{price_type}").get()
    if doc.exists:
        return _params_from_doc(doc.to_dict(), segment_key, fallback=False)

    # 2차: maker::model 폴백 → scale_factor + price_bias만
    fallback_key = f"{maker}::{model}"
    doc = db.collection("feedbackFilters").document(f"{fallback_key}::{price_type}").get()
    if doc.exists:
        return _params_from_doc(doc.to_dict(), fallback_key, fallback=True)

    # 매칭 없음 → 전부 중립값
    return LearnedParams(segment_key=segment_key)


# 하위 호환 alias
get_feedback_filter = get_learned_params


def store_feedback_and_recalculate(
    *,
    vehicle_id: str | None = None,
    maker: str,
    model: str,
    trim: str,
    year: int,
    mileage: int,
    price_type: str,
    predicted_price: float,
    target_price: float,
    user_note: str | None = None,
) -> LearnedParams:
    """피드백 저장 + 해당 세그먼트 학습 파라미터 재계산."""
    db = get_firestore_db()
    segment_key = build_segment_key(maker, model, trim, year, mileage)

    if predicted_price <= 0:
        logger.warning("predicted_price <= 0, 피드백 무시")
        return LearnedParams(segment_key=segment_key)

    ratio = target_price / predicted_price

    # 피드백 저장
    feedback_doc = {
        "vehicle_id": vehicle_id,
        "maker": maker,
        "model": model,
        "trim": trim,
        "year": year,
        "mileage": mileage,
        "price_type": price_type,
        "predicted_price": predicted_price,
        "target_price": target_price,
        "ratio": ratio,
        "segment_key": segment_key,
        "user_note": user_note,
        "created_at": SERVER_TIMESTAMP,
    }
    db.collection("calibrationFeedback").add(feedback_doc)
    logger.info("피드백 저장: segment=%s, type=%s, ratio=%.4f", segment_key, price_type, ratio)

    # 해당 세그먼트 피드백 전체 조회 → 재계산
    query = (
        db.collection("calibrationFeedback")
        .where("segment_key", "==", segment_key)
        .where("price_type", "==", price_type)
        .order_by("created_at")
    )
    docs = query.stream()
    feedbacks = [d.to_dict() for d in docs]

    result = _compute_learned_params(feedbacks, price_type)
    result.segment_key = segment_key

    # feedbackFilters 갱신
    filter_doc = {
        "direction": result.direction,
        "exclusion_pct": result.exclusion_pct,
        "scale_factor": result.scale_factor,
        "price_bias": result.price_bias,
        "mileage_slope": result.mileage_slope,
        "ref_mileage": result.ref_mileage,
        "confidence": result.confidence,
        "feedback_count": result.feedback_count,
        "effective_n": result.effective_n,
        "segment_key": segment_key,
        "price_type": price_type,
        "updated_at": SERVER_TIMESTAMP,
    }
    db.collection("feedbackFilters").document(f"{segment_key}::{price_type}").set(filter_doc)

    # maker::model 폴백 키도 재계산
    _recalculate_fallback(db, maker, model, price_type)

    logger.info(
        "학습 파라미터 갱신: segment=%s::%s → scale=%.3f bias=%.1f slope=%.4f excl=%.2f%% dir=%s (n=%d, eff_n=%.1f, conf=%.2f)",
        segment_key, price_type,
        result.scale_factor, result.price_bias, result.mileage_slope,
        result.exclusion_pct * 100, result.direction,
        result.feedback_count, result.effective_n, result.confidence,
    )
    return result


def _recalculate_fallback(db, maker: str, model: str, price_type: str):
    """maker::model 폴백 학습 파라미터 재계산."""
    fallback_key = f"{maker}::{model}"
    query = (
        db.collection("calibrationFeedback")
        .where("maker", "==", maker)
        .where("model", "==", model)
        .where("price_type", "==", price_type)
        .order_by("created_at")
    )
    docs = query.stream()
    feedbacks = [d.to_dict() for d in docs]

    if not feedbacks:
        return

    result = _compute_learned_params(feedbacks, price_type)

    filter_doc = {
        "direction": result.direction,
        "exclusion_pct": result.exclusion_pct,
        "scale_factor": result.scale_factor,
        "price_bias": result.price_bias,
        "mileage_slope": result.mileage_slope,
        "ref_mileage": result.ref_mileage,
        "confidence": result.confidence,
        "feedback_count": result.feedback_count,
        "effective_n": result.effective_n,
        "segment_key": fallback_key,
        "price_type": price_type,
        "updated_at": SERVER_TIMESTAMP,
    }
    db.collection("feedbackFilters").document(f"{fallback_key}::{price_type}").set(filter_doc)


# =========================================================================
# Leave-One-Out 자동보정 + 수동 피드백 블렌딩
# =========================================================================

@dataclass
class CalibrationEntry:
    """자동/수동 캘리브레이션 항목."""
    predicted_price: float
    target_price: float
    mileage: int
    weight: float       # 사전 계산된 가중치
    source: str         # "auto" | "manual"


def _leave_one_out_residuals(
    price_data: list[tuple[int, float]],
    target_mileage: int,
    conservative: bool = False,
) -> list[CalibrationEntry]:
    """
    Leave-One-Out 자동보정 잔차 계산.

    price_data: estimator가 정규화한 [(mileage, price), ...]
    len < 15이면 빈 리스트 반환 (희소 데이터 과적합 방지).
    """
    n = len(price_data)
    if n < 15:
        logger.info("LOO 자동보정 스킵: 데이터 %d건 < 최소 15건", n)
        return []

    from app.services.retail_estimator import (
        _gaussian_weights, _weighted_local_regression, _adaptive_bandwidth,
    )

    bandwidth = _adaptive_bandwidth(price_data, target_mileage)
    mileages = [d[0] for d in price_data]
    prices = [d[1] for d in price_data]
    auto_weight = 5.0 / n

    entries: list[CalibrationEntry] = []
    for i in range(n):
        # i번째 차량을 제외한 나머지로 i의 가격 예측
        loo_mileages = mileages[:i] + mileages[i + 1:]
        loo_prices = prices[:i] + prices[i + 1:]
        ws = _gaussian_weights(loo_mileages, mileages[i], bandwidth)

        predicted_i = _weighted_local_regression(
            [float(m) for m in loo_mileages],
            loo_prices,
            ws,
            float(mileages[i]),
        )
        if predicted_i <= 0:
            continue

        entries.append(CalibrationEntry(
            predicted_price=predicted_i,
            target_price=prices[i],
            mileage=mileages[i],
            weight=auto_weight,
            source="auto",
        ))

    return entries


def _compute_learned_params_from_entries(
    entries: list[CalibrationEntry],
    segment_key: str = "",
) -> LearnedParams:
    """CalibrationEntry 리스트로 5개 파라미터 산출 (entry.weight 사용)."""
    if not entries:
        return LearnedParams(segment_key=segment_key)

    auto_count = sum(1 for e in entries if e.source == "auto")
    manual_count = sum(1 for e in entries if e.source == "manual")

    parsed: list[dict] = []
    for e in entries:
        if e.predicted_price <= 0 or e.target_price <= 0:
            continue
        ratio = max(MIN_RATIO, min(MAX_RATIO, e.target_price / e.predicted_price))
        parsed.append({
            "ratio": ratio,
            "predicted": e.predicted_price,
            "target": e.target_price,
            "mileage": e.mileage,
            "decay": e.weight,   # weight를 decay 슬롯에 넣어 기존 알고리즘 재사용
        })

    if not parsed:
        return LearnedParams(segment_key=segment_key)

    count = len(parsed)
    weights = [p["decay"] for p in parsed]
    effective_n = sum(weights)
    confidence = min(effective_n / 5.0, 1.0)

    # ── Phase 1: scale_factor (n≥1) ──
    ratios = [p["ratio"] for p in parsed]
    raw_scale = _weighted_median(ratios, weights)
    raw_scale = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

    if effective_n < 2:
        scale_damping = min(effective_n / 2.0, 1.0) * confidence
    else:
        scale_damping = confidence
    effective_scale = 1.0 + (raw_scale - 1.0) * scale_damping

    # ── Phase 2: price_bias (n≥3) ──
    raw_bias = 0.0
    if effective_n >= 3:
        residuals_bias = [
            p["target"] - p["predicted"] * effective_scale
            for p in parsed
        ]
        raw_bias = _weighted_median(residuals_bias, weights)
        raw_bias = max(-MAX_BIAS, min(MAX_BIAS, raw_bias))
    effective_bias = raw_bias * confidence

    # ── Phase 3: mileage_slope (n≥5, spread≥20000km) ──
    raw_slope = 0.0
    ref_mileage = 0.0
    mileages = [p["mileage"] for p in parsed]
    mileage_spread = max(mileages) - min(mileages) if mileages else 0

    if effective_n >= 5 and mileage_spread >= 20000:
        ref_mileage = _weighted_median(
            [float(m) for m in mileages], weights,
        )
        residuals_slope = [
            p["target"] - (p["predicted"] * effective_scale + effective_bias)
            for p in parsed
        ]
        xs_km = [(p["mileage"] - ref_mileage) / 10000 for p in parsed]
        raw_slope = _weighted_regression_slope(xs_km, residuals_slope, weights)
        avg_predicted = sum(p["predicted"] * w for p, w in zip(parsed, weights)) / effective_n
        if avg_predicted > 0:
            raw_slope = raw_slope / avg_predicted
        raw_slope = max(-MAX_SLOPE, min(MAX_SLOPE, raw_slope))

    mileage_confidence = min(confidence, mileage_spread / 30000) if mileage_spread > 0 else 0
    effective_slope = raw_slope * mileage_confidence

    # ── Phase 4: exclusion_pct / direction ──
    final_residuals = []
    for p in parsed:
        delta_km = (p["mileage"] - ref_mileage) / 10000 if ref_mileage else 0
        predicted_corrected = (
            p["predicted"] * (effective_scale + effective_slope * delta_km)
            + effective_bias
        )
        if predicted_corrected > 0:
            final_residuals.append(p["target"] / predicted_corrected)

    if final_residuals and effective_n >= 3:
        final_ratio = _weighted_median(final_residuals, weights[:len(final_residuals)])
        excl = min(MAX_EXCLUSION_PCT, abs(final_ratio - 1.0) * confidence)
        if final_ratio > 1.01:
            direction = "up"
        elif final_ratio < 0.99:
            direction = "down"
        else:
            direction = "none"
    else:
        excl = 0.0
        direction = "none"

    return LearnedParams(
        exclusion_pct=round(excl, 4),
        direction=direction,
        scale_factor=round(effective_scale, 4),
        price_bias=round(effective_bias, 1),
        mileage_slope=round(effective_slope, 6),
        ref_mileage=round(ref_mileage, 0),
        confidence=round(confidence, 4),
        feedback_count=count,
        effective_n=round(effective_n, 2),
        segment_key=segment_key,
        auto_cal_count=auto_count,
        manual_feedback_count=manual_count,
    )


def compute_blended_params(
    price_data: list[tuple[int, float]],
    target_mileage: int,
    maker: str,
    model: str,
    trim: str,
    year: int,
    price_type: str,
    conservative: bool = False,
) -> LearnedParams:
    """
    DB 데이터 자동보정(LOO) + 수동 피드백을 블렌딩하여 학습 파라미터 산출.

    1. _leave_one_out_residuals → auto entries (weight = 5.0/N)
    2. Firestore calibrationFeedback → manual entries (weight = time_decay)
    3. _compute_learned_params_from_entries(auto + manual)
    """
    now = datetime.now(timezone.utc)
    segment_key = build_segment_key(maker, model, trim, year, target_mileage)

    # 1) 자동보정 (LOO)
    auto_entries = _leave_one_out_residuals(price_data, target_mileage, conservative)
    logger.info(
        "자동보정(LOO): %s::%s — %d건 (price_data=%d건)",
        segment_key, price_type, len(auto_entries), len(price_data),
    )

    # 2) 수동 피드백 조회
    manual_entries: list[CalibrationEntry] = []
    try:
        db = get_firestore_db()
        query = (
            db.collection("calibrationFeedback")
            .where("segment_key", "==", segment_key)
            .where("price_type", "==", price_type)
            .order_by("created_at")
        )
        docs = query.stream()
        for d in docs:
            fb = d.to_dict()
            predicted = fb.get("predicted_price", 0)
            target = fb.get("target_price", 0)
            mileage_fb = fb.get("mileage", 0) or 0
            if predicted <= 0 or target <= 0:
                continue

            created_at = fb.get("created_at")
            if created_at is None:
                age_days = 0.0
            elif isinstance(created_at, datetime):
                age_days = max(
                    (now - created_at.replace(
                        tzinfo=timezone.utc if created_at.tzinfo is None else created_at.tzinfo
                    )).total_seconds() / 86400,
                    0,
                )
            else:
                age_days = 0.0

            decay = math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)
            manual_entries.append(CalibrationEntry(
                predicted_price=predicted,
                target_price=target,
                mileage=mileage_fb,
                weight=decay,
                source="manual",
            ))
    except Exception as e:
        logger.warning("수동 피드백 조회 실패: %s", e)

    if manual_entries:
        logger.info("수동 피드백: %s::%s — %d건", segment_key, price_type, len(manual_entries))

    # 3) 블렌딩
    all_entries = auto_entries + manual_entries
    result = _compute_learned_params_from_entries(all_entries, segment_key)

    logger.info(
        "블렌딩 결과: %s::%s → scale=%.3f bias=%.1f slope=%.4f excl=%.2f%% dir=%s (auto=%d, manual=%d, eff_n=%.1f)",
        segment_key, price_type,
        result.scale_factor, result.price_bias, result.mileage_slope,
        result.exclusion_pct * 100, result.direction,
        result.auto_cal_count, result.manual_feedback_count, result.effective_n,
    )
    return result
