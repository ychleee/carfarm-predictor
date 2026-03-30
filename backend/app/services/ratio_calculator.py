"""
비율 기반 가격 계산 엔진

계산 흐름:
  1. 기준비율 = 기준낙찰가 / 기준출고가(또는 기본가)
  2. 주행거리 보정: 기준비율 -%p (보정율 × 만km 차이)
  3. 연식 보정: ±%p (보정율 × 연식 차이)
  4. 검차 보정: 대상=AA 기준, 기준차 사고이력 만큼 +%p 가산
  5. 조정비율 × 대상출고가(또는 기본가) = 예상 낙찰가

출고가/기본가 없으면 절대금액 방식으로 폴백.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.services.retail_estimator import estimate_retail_by_market, estimate_auction_by_market

logger = logging.getLogger(__name__)

# 골격(프레임) 부위 — firestore_db._FRAME_PARTS 와 동일
_FRAME_PARTS = {
    "FRONT_PANEL", "FRONT_CROSS_MEMBER", "FLOOR_PANEL",
    "SIDE_MEMBER", "REAR_CROSS_MEMBER", "TRUNK_FLOOR_PANEL", "REAR_PANEL",
    "LEFT_A_PILLAR", "RIGHT_A_PILLAR",
    "LEFT_B_PILLAR", "RIGHT_B_PILLAR",
    "LEFT_C_PILLAR", "RIGHT_C_PILLAR",
    "LEFT_FRONT_INSIDE_PANEL", "RIGHT_FRONT_INSIDE_PANEL",
    "LEFT_REAR_INSIDE_PANEL", "RIGHT_REAR_INSIDE_PANEL",
    "LEFT_FRONT_WHEELS_HOUSE", "RIGHT_FRONT_WHEELS_HOUSE",
    "LEFT_REAR_WHEELS_HOUSE", "RIGHT_REAR_WHEELS_HOUSE",
}

# 검차 보정 기본 비율
_EXCHANGE_RATE_PER_PART = 2.0   # 외판 교환 1부위당 %p
_BODYWORK_RATE_PER_PART = 1.0   # 외판 판금 1부위당 %p
_STRUCTURAL_GRADE = {            # 골격 손상 부위수 → %p
    1: 15.0, 2: 17.0, 3: 18.0, 4: 19.0,
}
_STRUCTURAL_DEFAULT = 20.0       # 5부위 이상


def _inspection_age_weight(year: int, mileage: int = 0) -> tuple[float, str]:
    """연식/주행거리에 따른 검차 보정 가중치 (신차=1.0, 오래될수록 감소).

    Returns:
        (weight, 설명 문자열)
    """
    age = datetime.now().year - year if year > 0 else 0

    if age <= 3:
        w = 1.0
    elif age <= 6:
        w = 0.5
    elif age <= 9:
        w = 0.3
    else:
        w = 0.15

    # 고주행 추가 감소 (15만km 이상)
    mileage_discount = ""
    if mileage > 150000:
        w *= 0.7
        mileage_discount = f", 고주행({mileage:,}km) ×0.7"

    desc = f"연식 가중치: {age}년차 → ×{w:.1f}{mileage_discount}"
    return w, desc


def _calc_inspection_adj(
    reference: dict,
    target_year: int = 0,
    target_mileage: int = 0,
) -> tuple[float, list[str]]:
    """기준차량 사고이력 → 대상(AA) 기준 가산 %p 와 상세 내역.

    target_year/target_mileage 제공 시 연식/주행거리 기반 가중치 적용.
    """
    part_damages = reference.get("part_damages") or []

    # part_damages가 있으면 거기서 파생, 없으면 직접 필드 사용
    if part_damages:
        exchange_count = reference.get("exchange_count", 0) or 0
        bodywork_count = reference.get("bodywork_count", 0) or 0
        # 골격 교환 부위 (EXCHANGE on frame parts)
        structural_parts = {
            d["part"] for d in part_damages
            if d.get("part", "") in _FRAME_PARTS
            and d.get("damage_type", "").upper() == "EXCHANGE"
        }
        structural_count = len(structural_parts)
    else:
        # part_damages 없을 때 개별 필드에서 직접 읽기
        frame_exchange = reference.get("frame_exchange", 0) or 0
        frame_bodywork = reference.get("frame_bodywork", 0) or 0
        exterior_exchange = reference.get("exterior_exchange", 0) or 0
        exterior_bodywork = reference.get("exterior_bodywork", 0) or 0
        exchange_count = frame_exchange + exterior_exchange
        bodywork_count = frame_bodywork + exterior_bodywork
        structural_count = frame_exchange  # 프레임 교환 = 골격사고

    # 비골격 교환 = 전체 교환 - 골격 교환
    non_structural_exchange = max(exchange_count - structural_count, 0)

    raw_pct = 0.0
    details: list[str] = []

    # 1) 골격사고
    if structural_count > 0:
        grade_pct = _STRUCTURAL_GRADE.get(structural_count, _STRUCTURAL_DEFAULT)
        raw_pct += grade_pct
        details.append(
            f"골격손상 {structural_count}부위 → +{grade_pct:.0f}%p"
        )

    # 2) 외판 교환
    if non_structural_exchange > 0:
        ex_pct = non_structural_exchange * _EXCHANGE_RATE_PER_PART
        raw_pct += ex_pct
        details.append(
            f"외판교환 {non_structural_exchange}부위 × "
            f"{_EXCHANGE_RATE_PER_PART:.0f}%p = +{ex_pct:.0f}%p"
        )

    # 3) 판금
    if bodywork_count > 0:
        bw_pct = bodywork_count * _BODYWORK_RATE_PER_PART
        raw_pct += bw_pct
        details.append(
            f"판금 {bodywork_count}부위 × "
            f"{_BODYWORK_RATE_PER_PART:.0f}%p = +{bw_pct:.0f}%p"
        )

    # 4) 연식/주행거리 기반 가중치 적용
    if target_year > 0 and raw_pct > 0:
        weight, weight_desc = _inspection_age_weight(target_year, target_mileage)
        if weight < 1.0:
            adj_pct = round(raw_pct * weight, 1)
            details.append(f"{weight_desc}")
            details.append(f"보정전 {raw_pct:.1f}%p × {weight:.1f} = {adj_pct:.1f}%p")
            return adj_pct, details

    return raw_pct, details


def _to_man_won(value: float) -> float:
    """원 단위 → 만원 변환 (100000 초과면 원 단위로 간주)"""
    if value > 100000:
        return round(value / 10000, 1)
    return value


def _pick_price(vehicle: dict) -> tuple[float, str]:
    """출고가 우선, 없으면 기본가 사용. (가격, 라벨) 반환."""
    factory = vehicle.get("factory_price", 0) or 0
    if factory > 0:
        return factory, "출고가"
    base = vehicle.get("base_price", 0) or 0
    if base > 0:
        return base, "기본가"
    return 0, ""


def calculate_with_criteria(
    target: dict,
    reference: dict,
    criteria: dict,
) -> dict:
    """
    비율(%) 기반 가격 계산.

    예시:
      기준: 출고가 2960만, 낙찰가 775만 → 기준비율 26.2%
      대상: 출고가 2635만, 주행거리 +9만km, 보정율 1%/만km
      → 26.2% - (1% × 9) = 17.2%
      → 2635만 × 17.2% = 453만원
    """
    ref_auction = _to_man_won(reference.get("auction_price", 0) or 0)
    ref_price, ref_label = _pick_price(reference)
    tgt_price, tgt_label = _pick_price(target)

    # 대상 출고가 없으면 기준차량 출고가를 폴백으로 사용
    if tgt_price == 0 and ref_price > 0:
        tgt_price = ref_price
        tgt_label = f"기준{ref_label}"

    # 보정 기준 추출
    mileage_rate = criteria.get("mileage_rate_per_10k", 1.5)
    ceiling_km = criteria.get("mileage_ceiling_km", 200000)
    year_rate = criteria.get("year_rate_per_year", 2.5)

    adjustments = []

    # ── 비율 방식: 기준차량 출고가/기본가가 있을 때 ──
    if ref_price > 0 and tgt_price > 0:
        base_ratio = ref_auction / ref_price  # 기준비율 (예: 0.262)
        base_ratio_pct = base_ratio * 100     # 26.2%

        # ── 1. 주행거리 보정 ──
        target_mileage = target.get("mileage", 0) or 0
        ref_mileage = reference.get("mileage", 0) or 0

        effective_target = min(target_mileage, ceiling_km)
        effective_ref = min(ref_mileage, ceiling_km)
        effective_diff_10k = (effective_target - effective_ref) / 10000

        mileage_adj_pct = -effective_diff_10k * mileage_rate  # %p 단위

        ceiling_note = ""
        if target_mileage > ceiling_km or ref_mileage > ceiling_km:
            ceiling_note = f"\n{ceiling_km // 10000}만km 천장 적용"

        ratio_after_mileage = base_ratio_pct + mileage_adj_pct

        mileage_amount = round(mileage_adj_pct / 100 * tgt_price, 1)  # 만원

        adjustments.append({
            "rule_name": "주행거리 보정",
            "rule_id": "mileage",
            "description": (
                f"{'감가' if effective_diff_10k > 0 else '가산'} "
                f"({mileage_rate:.1f}%p/만km × {abs(effective_diff_10k):.1f}만km = {mileage_adj_pct:+.1f}%p)"
            ),
            "amount": mileage_amount,  # 만원
            "details": (
                f"기준비율: {base_ratio_pct:.1f}% (낙찰 {ref_auction:,.0f}만 / {ref_label} {ref_price:,.0f}만)\n"
                f"주행거리: 대상 {target_mileage:,}km − 기준 {ref_mileage:,}km = {effective_diff_10k:+.1f}만km{ceiling_note}\n"
                f"보정: {mileage_rate:.1f}%p/만km × {effective_diff_10k:.1f}만km = {mileage_adj_pct:+.1f}%p\n"
                f"→ {base_ratio_pct:.1f}% {mileage_adj_pct:+.1f}%p = {ratio_after_mileage:.1f}%"
            ),
            "data_source": f"LLM 분석 기준: {mileage_rate:.1f}%p/만km",
        })

        # ── 2. 연식 보정 ──
        target_year = target.get("year", 0) or 0
        ref_year = reference.get("year", 0) or 0
        year_diff = target_year - ref_year  # 양수 = 대상이 더 새로움

        year_adj_pct = year_diff * year_rate  # %p 단위

        ratio_after_year = ratio_after_mileage + year_adj_pct

        year_amount = round(year_adj_pct / 100 * tgt_price, 1)  # 만원

        adjustments.append({
            "rule_name": "연식 보정",
            "rule_id": "year",
            "description": (
                f"연식 {abs(year_diff)}년 {'가산' if year_diff > 0 else '감가'} "
                f"({year_rate:.1f}%p/년 × {abs(year_diff)}년 = {year_adj_pct:+.1f}%p)"
                if year_diff != 0 else "연식 동일"
            ),
            "amount": year_amount,  # 만원
            "details": (
                f"연식: 대상 {target_year}년 − 기준 {ref_year}년 = {year_diff:+d}년\n"
                f"보정: {year_rate:.1f}%p/년 × {abs(year_diff)}년 = {year_adj_pct:+.1f}%p\n"
                f"→ {ratio_after_mileage:.1f}% {year_adj_pct:+.1f}%p = {ratio_after_year:.1f}%"
            ),
            "data_source": f"LLM 분석 기준: 연당 {year_rate:.1f}%p",
        })

        # ── 최종 산출 ──
        adjusted_ratio_pct = ratio_after_year
        adjusted_ratio = adjusted_ratio_pct / 100
        total_adj_pct = mileage_adj_pct + year_adj_pct

        estimated_auction = round(adjusted_ratio * tgt_price, 1)

        # ── 낙찰가: 시장 데이터 단독 ──
        target_mileage_val = target.get("mileage", 0) or 0
        target_year_val = target.get("year", 0) or 0
        target_fuel = target.get("fuel", "") or ""
        auction_market = estimate_auction_by_market(
            maker=target.get("maker", ""),
            model=target.get("model", ""),
            trim=target.get("trim", ""),
            year=target_year_val,
            mileage=target_mileage_val,
            factory_price=target.get("factory_price", 0) or 0,
            base_price=target.get("base_price", 0) or 0,
            fuel=target_fuel,
        )
        if auction_market.success:
            estimated_auction = auction_market.estimated_auction

        # total_adjustment: 금액 기준 (기준낙찰가 대비 차이)
        total_adjustment = round(estimated_auction - ref_auction, 1)

        # ── 소매가 추정: 시장 데이터 기반 ──
        retail_result = estimate_retail_by_market(
            maker=target.get("maker", ""),
            model=target.get("model", ""),
            trim=target.get("trim", ""),
            year=target_year_val,
            mileage=target_mileage_val,
            factory_price=target.get("factory_price", 0) or 0,
            base_price=target.get("base_price", 0) or 0,
            fuel=target_fuel,
        )

        estimated_retail = retail_result.estimated_retail if retail_result.success else 0
        adjustments.append({
            "rule_name": "소매가 추정 (비율 추이)",
            "rule_id": "retail_estimation",
            "description": (
                f"같은 트림·연식 소매 {retail_result.vehicles_found}건 분석 → "
                f"비율 {retail_result.estimated_ratio:.1f}% × {tgt_label} {tgt_price:,.0f}만 = "
                f"추정 소매가 {estimated_retail:,.0f}만원"
                if retail_result.success
                else f"소매가 추정 불가: {retail_result.details}"
            ),
            "amount": 0,
            "details": retail_result.details,
            "data_source": (
                f"시장 데이터 비율 추이 ({retail_result.vehicles_found}건)"
                if retail_result.success else "데이터 부족"
            ),
        })

        # 신뢰도
        auction_confidence = "높음" if abs(total_adj_pct) < 5 else "보통"
        if retail_result.success:
            confidence = retail_result.confidence
            if auction_confidence == "보통":
                confidence = "보통"
        else:
            confidence = auction_confidence

        summary = (
            f"기준비율 {base_ratio_pct:.1f}%"
            f" → 보정 {total_adj_pct:+.1f}%p"
            f" → 조정비율 {adjusted_ratio_pct:.1f}%"
            f" × 대상{tgt_label} {tgt_price:,.0f}만"
            f" = 예상 낙찰가 {estimated_auction:,.0f}만원"
            + (f" | 소매가 {estimated_retail:,.0f}만원 (비율 {retail_result.estimated_ratio:.1f}%)"
               if retail_result.success else " | 소매가 추정 불가")
        )

    else:
        # ── 폴백: 출고가/기본가 없을 때 절대금액 방식 ──
        logger.info("출고가/기본가 부재 — 절대금액 방식으로 폴백")

        target_mileage = target.get("mileage", 0) or 0
        ref_mileage = reference.get("mileage", 0) or 0
        effective_target = min(target_mileage, ceiling_km)
        effective_ref = min(ref_mileage, ceiling_km)
        effective_diff_10k = (effective_target - effective_ref) / 10000

        mileage_amount = round(-effective_diff_10k * mileage_rate / 100 * ref_auction, 1)
        adjustments.append({
            "rule_name": "주행거리 보정",
            "rule_id": "mileage",
            "description": f"{'감가' if effective_diff_10k > 0 else '가산'} ({mileage_rate:.1f}%/만km)",
            "amount": mileage_amount,
            "details": (
                f"주행거리 차이: {effective_diff_10k:+.1f}만km\n"
                f"계산: {mileage_rate:.1f}% × {abs(effective_diff_10k):.1f}만km × {ref_auction:,.0f}만원 = {abs(mileage_amount):,.1f}만원\n"
                f"(출고가/기본가 부재 — 낙찰가 기준 절대금액 방식)"
            ),
            "data_source": f"LLM 분석 기준: {mileage_rate:.1f}%/만km (절대금액 방식)",
        })

        target_year = target.get("year", 0) or 0
        ref_year = reference.get("year", 0) or 0
        year_diff = target_year - ref_year

        year_amount = round(year_diff * year_rate / 100 * ref_auction, 1)
        adjustments.append({
            "rule_name": "연식 보정",
            "rule_id": "year",
            "description": (
                f"연식 {abs(year_diff)}년 {'가산' if year_diff > 0 else '감가'} (연당 {year_rate:.1f}%)"
                if year_diff != 0 else "연식 동일"
            ),
            "amount": year_amount,
            "details": (
                f"대상: {target_year}년식, 기준: {ref_year}년식 (차이: {year_diff:+d}년)\n"
                f"계산: {year_rate:.1f}% × {abs(year_diff)}년 × {ref_auction:,.0f}만원 = {abs(year_amount):,.1f}만원\n"
                f"(출고가/기본가 부재 — 낙찰가 기준 절대금액 방식)"
            ),
            "data_source": f"LLM 분석 기준: 연당 {year_rate:.1f}% (절대금액 방식)",
        })

        total_adjustment = sum(a["amount"] for a in adjustments)
        estimated_auction = round(ref_auction + total_adjustment, 1)

        # ── 낙찰가: 시장 데이터 단독 ──
        target_mileage_val = target.get("mileage", 0) or 0
        target_year_val = target.get("year", 0) or 0
        target_fuel = target.get("fuel", "") or ""
        auction_market = estimate_auction_by_market(
            maker=target.get("maker", ""),
            model=target.get("model", ""),
            trim=target.get("trim", ""),
            year=target_year_val,
            mileage=target_mileage_val,
            factory_price=target.get("factory_price", 0) or 0,
            base_price=target.get("base_price", 0) or 0,
            fuel=target_fuel,
        )
        if auction_market.success:
            estimated_auction = auction_market.estimated_auction
        total_adjustment = round(estimated_auction - ref_auction, 1)

        # ── 소매가 추정: 시장 데이터 기반 ──
        retail_result = estimate_retail_by_market(
            maker=target.get("maker", ""),
            model=target.get("model", ""),
            trim=target.get("trim", ""),
            year=target_year_val,
            mileage=target_mileage_val,
            factory_price=target.get("factory_price", 0) or 0,
            base_price=target.get("base_price", 0) or 0,
            fuel=target_fuel,
        )

        estimated_retail = retail_result.estimated_retail if retail_result.success else 0
        adjustments.append({
            "rule_name": "소매가 추정 (비율 추이)",
            "rule_id": "retail_estimation",
            "description": (
                f"같은 트림·연식 소매 {retail_result.vehicles_found}건 분석 → "
                f"비율 {retail_result.estimated_ratio:.1f}% = "
                f"추정 소매가 {estimated_retail:,.0f}만원"
                if retail_result.success
                else f"소매가 추정 불가: {retail_result.details}"
            ),
            "amount": 0,
            "details": retail_result.details,
            "data_source": (
                f"시장 데이터 비율 추이 ({retail_result.vehicles_found}건)"
                if retail_result.success else "데이터 부족"
            ),
        })

        confidence = retail_result.confidence if retail_result.success else "보통"
        summary = (
            f"기준가 {ref_auction:,.0f}만원"
            f" → 보정 {total_adjustment:+,.0f}만원"
            f" → 예상 낙찰가 {estimated_auction:,.0f}만원"
            + (f" | 소매가 {estimated_retail:,.0f}만원 (비율 {retail_result.estimated_ratio:.1f}%)"
               if retail_result.success else " | 소매가 추정 불가")
            + f" [출고가/기본가 부재 — 절대금액 방식]"
        )

    return {
        "reference_price": ref_auction,
        "adjustments": adjustments,
        "total_adjustment": round(total_adjustment, 1),
        "estimated_auction": estimated_auction,
        "estimated_retail": estimated_retail,
        "confidence": confidence,
        "summary": summary,
    }
