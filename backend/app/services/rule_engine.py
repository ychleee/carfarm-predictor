"""
CarFarm v2 — 룰 엔진 (Rule Engine)

기준차량 낙찰가를 기반으로 대상차량의 예상 가격을 산출한다.
각 보정 단계가 투명하게 기록되어 사용자에게 "왜 이 가격인지" 설명할 수 있다.

적용 룰 목록 (pricing_rules.yaml 기반, 프라이싱 매니저 업계 룰):
  1. 주행거리 감가 — 연식별 차등 (2%/1.5%/1%/10만원), 20만km 천장
  2. 교환 보정 — 부위별 가중치 적용, 골격 부위 제외
  3. 판금·도색 보정 — 부위별 가중치 적용, 골격 부위 제외
  4. 골격사고 보정 — C-F등급 15~20% 감가
  5. 색상 보정 — 선호도별 20만원, 차급별 선호 순서
  6. 옵션 보정 — 옵션 리스트 차이 비교
  7. 연식 보정 — 연당 2%
  8. 최종 가격 산출 — 소매가 × 경매할인율, 최소 마진 적용

사용법:
    engine = RuleEngine()
    result = engine.calculate(target, reference)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field, replace


# =========================================================================
# 색상 정규화
# =========================================================================

COLOR_NORMALIZE: dict[str, str] = {
    # 한국어 → 정규화 그룹
    "흰색": "white", "백색": "white", "화이트": "white", "아이보리": "white",
    "크림": "white", "진주": "white", "펄화이트": "white", "미색": "cream",
    "검정": "black", "검정색": "black", "블랙": "black",
    "은색": "silver", "실버": "silver", "은회색": "silver",
    "회색": "gray", "메탈": "gray", "그레이": "gray", "건메탈": "gray",
    "다크그레이": "gray", "티탄": "gray", "쥐색": "gray",
    "빨강": "other", "파랑": "other", "파란색": "other", "노랑": "other",
    "갈색": "other", "베이지": "other", "브라운": "other",
    # 영문 passthrough
    "white": "white", "black": "black", "silver": "silver",
    "gray": "gray", "cream": "cream", "other": "other",
}


def normalize_color(color: str) -> str:
    """색상 문자열을 정규화 그룹으로 변환 (white/black/silver/gray/cream/other)"""
    if not color:
        return "other"
    return COLOR_NORMALIZE.get(color.strip(), "other")


# =========================================================================
# 데이터 모델
# =========================================================================

@dataclass
class Vehicle:
    """차량 정보"""
    maker: str = ""
    model: str = ""
    generation: str = ""
    year: int = 0                  # 연식 (예: 2023)
    mileage: int = 0               # 주행거리 (km)
    fuel: str = ""
    drive: str = ""
    trim: str = ""
    segment: str = ""              # 경차/소형/중형/대형/SUV/MPV/수입_프리미엄 등
    color: str = ""                # 색상 원본
    color_group: str = ""          # 정규화 색상 (white/black/silver/gray/other)
    usage: str = "personal"        # personal/rental/commercial
    options: list[str] = field(default_factory=list)
    exchange_count: int = 0        # X교환 부위 수
    bodywork_count: int = 0        # W판금 부위 수
    part_damages: list[dict] = field(default_factory=list)  # [{part, damage_type}]
    auction_price: float = 0       # 낙찰가 (만원) — 기준차량만 해당
    new_car_price: float = 0       # 신차가격 (만원) — 있으면 사용
    base_price: float = 0          # 기본가 (만원) — 출고가 역산용
    factory_price: float = 0       # 출고가 (만원) — 옵션가 역산용
    option_unit_price: float = 0   # 같은 차종 출고가-기본가 역산 옵션 단가 (만원/개)


@dataclass
class AdjustmentStep:
    """보정 단계 하나"""
    rule_name: str                 # 룰 이름 (한글)
    rule_id: str                   # 룰 ID (영문)
    description: str               # 적용 설명
    amount: float                  # 보정 금액 (만원, +/-/0)
    details: str                   # 상세 계산 과정
    data_source: str = ""          # 데이터 근거


@dataclass
class PriceResult:
    """가격 산출 결과"""
    reference_price: float         # 기준차량 낙찰가 (만원)
    adjustments: list[AdjustmentStep] = field(default_factory=list)
    total_adjustment: float = 0    # 총 보정액
    estimated_retail: float = 0    # 추정 소매가
    estimated_auction: float = 0   # 예상 낙찰가
    confidence: str = "보통"       # 높음/보통/낮음
    summary: str = ""              # 한 줄 요약


# =========================================================================
# 룰 엔진
# =========================================================================

class RuleEngine:
    """
    기준차량 기반 가격 보정 룰 엔진.

    사용법:
        engine = RuleEngine()
        result = engine.calculate(target_vehicle, reference_vehicle)
        for step in result.adjustments:
            print(f"{step.rule_name}: {step.amount:+.0f}만원 — {step.description}")
    """

    def __init__(self, rules_path: str | Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent.parent.parent / "rules" / "pricing_rules.yaml"
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        """YAML 룰 파일 로드"""
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------
    # 메인 계산
    # -----------------------------------------------------------------

    def calculate(self, target: Vehicle, reference: Vehicle) -> PriceResult:
        """
        기준차량 대비 대상차량의 예상 가격 산출.

        Args:
            target: 대상차량 (가격을 알고 싶은 차량)
            reference: 기준차량 (낙찰가가 알려진 비교 차량)

        Returns:
            PriceResult: 보정 단계별 결과 + 최종 가격
        """
        result = PriceResult(reference_price=reference.auction_price)
        base_price = reference.auction_price

        # ── AA등급 보정: 대상차량을 항상 무사고(AA)로 간주 ──
        # 사고 이력을 제거하여 AA등급 기준 가격을 예측
        target = replace(
            target,
            exchange_count=0,
            bodywork_count=0,
            part_damages=[],
        )

        # ── 룰 1: 주행거리 감가 ──
        step = self._adjust_mileage(target, reference, base_price)
        result.adjustments.append(step)

        # ── 룰 5: 색상 보정 ──
        step = self._adjust_color(target, reference)
        result.adjustments.append(step)

        # ── 룰 6: 선호옵션 보정 ──
        step = self._adjust_options(target, reference)
        result.adjustments.append(step)

        # ── 룰 7: 연식 차이 보정 ──
        step = self._adjust_year_diff(target, reference, base_price)
        result.adjustments.append(step)

        # ── 룰 8: 트림 차이 경고 ──
        trim_warning = self._warn_trim_diff(target, reference)
        if trim_warning:
            result.adjustments.append(trim_warning)

        # 총 보정액
        result.total_adjustment = sum(s.amount for s in result.adjustments)

        # 최종 가격 산출
        result = self._calculate_final_price(target, result, has_trim_warning=trim_warning is not None)

        return result

    # -----------------------------------------------------------------
    # 룰 1: 주행거리 감가
    # -----------------------------------------------------------------

    def _adjust_mileage(self, target: Vehicle, reference: Vehicle,
                        base_price: float) -> AdjustmentStep:
        """
        주행거리 차이에 따른 감가 보정.

        업계 룰:
        - 2~3년: 소매가의 2%/만km
        - 4~6년: 1.5%/만km
        - 7~9년: 1%/만km
        - 10년+ & 2000만원+: 1%/만km
        - 10년+ & 2000만원-: 10만원/만km (정액)
        - 20만km 초과: 증감 미적용 (천장)
        """
        rules = self.rules['mileage_depreciation']
        diff_km = target.mileage - reference.mileage
        diff_10k = diff_km / 10000

        if abs(diff_10k) < 0.1:
            return AdjustmentStep(
                rule_name="주행거리 보정",
                rule_id="mileage",
                description="주행거리 차이 없음",
                amount=0,
                details=f"대상 {target.mileage:,}km ≈ 기준 {reference.mileage:,}km",
            )

        # 20만km 천장: 양쪽 다 20만km 넘으면 보정 없음
        ceiling = rules.get('ceiling', {})
        ceiling_km = ceiling.get('ceiling_km', 200000)
        if ceiling.get('enabled') and target.mileage > ceiling_km and reference.mileage > ceiling_km:
            return AdjustmentStep(
                rule_name="주행거리 보정",
                rule_id="mileage",
                description="20만km 초과 — 증감 미적용 (천장 효과)",
                amount=0,
                details=(
                    f"대상 {target.mileage:,}km, 기준 {reference.mileage:,}km\n"
                    f"둘 다 20만km 초과 → 천장 효과 적용, 보정 없음"
                ),
                data_source="업계 룰: 20만km 초과 증감 미적용",
            )

        # 20만km 이하 부분만 감가 적용
        effective_target = min(target.mileage, ceiling_km) if ceiling.get('enabled') else target.mileage
        effective_ref = min(reference.mileage, ceiling_km) if ceiling.get('enabled') else reference.mileage
        effective_diff = effective_target - effective_ref
        effective_diff_10k = effective_diff / 10000

        # 연식별 감가율 조회
        age = 2026 - target.year
        rate, age_band = self._get_mileage_rate(age, base_price, rules)

        amount = -abs(effective_diff_10k) * (rate / 100) * base_price
        if effective_diff < 0:  # 대상이 더 적게 주행 → 가산
            amount = -amount

        ceiling_note = ""
        if ceiling.get('enabled') and (target.mileage > ceiling_km or reference.mileage > ceiling_km):
            ceiling_note = f"\n20만km 천장 적용: 유효 차이 {effective_diff:+,}km"

        return AdjustmentStep(
            rule_name="주행거리 보정",
            rule_id="mileage",
            description=f"{'감가' if effective_diff > 0 else '가산'} ({age_band}, {rate:.1f}%/만km)",
            amount=round(amount, 1),
            details=(
                f"차이: {diff_km:+,}km ({diff_10k:+.1f}만km){ceiling_note}\n"
                f"감가율: {rate:.1f}%/만km ({age_band})\n"
                f"계산: {abs(effective_diff_10k):.1f}만km × {rate:.1f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: {age_band} {rate:.1f}%/만km",
        )

    def _get_mileage_rate(self, age: int, price: float,
                          rules: dict) -> tuple[float, str]:
        """연식별 감가율 조회"""
        if age <= 3:
            band_key, band_name = 'age_2_3', '2~3년'
        elif age <= 6:
            band_key, band_name = 'age_4_6', '4~6년'
        elif age <= 9:
            band_key, band_name = 'age_7_9', '7~9년'
        else:
            band_key, band_name = 'age_10_plus', '10년+'

        default = rules['default'][band_key]

        # 10년+ 저가차는 flat 방식
        if band_key == 'age_10_plus':
            threshold = default.get('price_threshold', 2000)
            if price < threshold:
                flat = default.get('flat_per_10k', 10)
                if price > 0:
                    return (flat / price) * 100, f"{band_name}, 정액 {flat}만원/만km"

        return default['rate_per_10k'], band_name

    # -----------------------------------------------------------------
    # 룰 2: 교환 보정
    # -----------------------------------------------------------------

    def _adjust_exchange(self, target: Vehicle, reference: Vehicle,
                         base_price: float) -> AdjustmentStep:
        """
        교환(X) 부위 수 차이에 따른 보정.

        part_damages 있으면 → 부위별 가중치 적용 (골격 부위 제외)
        없으면 → 기존 count × 일괄 비율 유지 (하위 호환)
        """
        if target.part_damages or reference.part_damages:
            return self._adjust_exchange_by_part(target, reference, base_price)

        # 기존 로직 (하위 호환)
        rules = self.rules['accident_adjustment']
        rate = rules['exchange']['rate_per_count']  # 0.03

        diff = target.exchange_count - reference.exchange_count

        if diff == 0:
            return AdjustmentStep(
                rule_name="교환(X) 보정",
                rule_id="exchange",
                description="교환 부위 수 동일",
                amount=0,
                details=f"대상: {target.exchange_count}개, 기준: {reference.exchange_count}개",
            )

        amount = -diff * rate * base_price

        return AdjustmentStep(
            rule_name="교환(X) 보정",
            rule_id="exchange",
            description=f"교환 {abs(diff)}회 {'감가' if diff > 0 else '가산'} (1회당 {rate*100:.0f}%)",
            amount=round(amount, 1),
            details=(
                f"대상: {target.exchange_count}개, 기준: {reference.exchange_count}개\n"
                f"차이: {diff:+d}개\n"
                f"계산: {abs(diff)}회 × {rate*100:.0f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: 교환 1회당 {rate*100:.0f}%",
        )

    def _adjust_exchange_by_part(self, target: Vehicle, reference: Vehicle,
                                 base_price: float) -> AdjustmentStep:
        """부위별 가중 교환 보정 (골격 부위 제외 — 골격사고 룰에서 별도 처리)"""
        rules = self.rules['accident_adjustment']
        rate = rules['exchange']['rate_per_count']  # 0.03
        structural = self._get_structural_parts()

        t_exchanges = [d for d in target.part_damages
                       if d['damage_type'] == 'EXCHANGE' and d['part'] not in structural]
        r_exchanges = [d for d in reference.part_damages
                       if d['damage_type'] == 'EXCHANGE' and d['part'] not in structural]

        t_weighted = sum(self._get_part_weight(d['part']) for d in t_exchanges)
        r_weighted = sum(self._get_part_weight(d['part']) for d in r_exchanges)
        diff = t_weighted - r_weighted

        if abs(diff) < 0.01:
            return AdjustmentStep(
                rule_name="교환(X) 보정",
                rule_id="exchange",
                description="교환 보정 차이 없음 (부위별 가중)",
                amount=0,
                details=(
                    f"대상 교환: {[d['part'] for d in t_exchanges] or '없음'} (가중합: {t_weighted:.1f})\n"
                    f"기준 교환: {[d['part'] for d in r_exchanges] or '없음'} (가중합: {r_weighted:.1f})"
                ),
            )

        amount = -diff * rate * base_price

        t_detail = ", ".join(f"{d['part']}(×{self._get_part_weight(d['part'])})" for d in t_exchanges) or "없음"
        r_detail = ", ".join(f"{d['part']}(×{self._get_part_weight(d['part'])})" for d in r_exchanges) or "없음"

        return AdjustmentStep(
            rule_name="교환(X) 보정",
            rule_id="exchange",
            description=f"교환 {'감가' if diff > 0 else '가산'} (부위별 가중, 골격 제외)",
            amount=round(amount, 1),
            details=(
                f"대상 교환: {t_detail} (가중합: {t_weighted:.1f})\n"
                f"기준 교환: {r_detail} (가중합: {r_weighted:.1f})\n"
                f"가중 차이: {diff:+.1f}\n"
                f"계산: {abs(diff):.1f} × {rate*100:.0f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: 교환 1회당 {rate*100:.0f}% × 부위별 가중치",
        )

    # -----------------------------------------------------------------
    # 룰 3: 판금·도색 보정
    # -----------------------------------------------------------------

    def _adjust_bodywork(self, target: Vehicle, reference: Vehicle,
                         base_price: float) -> AdjustmentStep:
        """
        판금·도색(W) 부위 수 차이에 따른 보정.

        part_damages 있으면 → 부위별 가중치 적용 (골격 부위 제외)
        없으면 → 기존 count × 일괄 비율 유지 (하위 호환)
        """
        if target.part_damages or reference.part_damages:
            return self._adjust_bodywork_by_part(target, reference, base_price)

        # 기존 로직 (하위 호환)
        diff = target.bodywork_count - reference.bodywork_count

        if diff == 0:
            return AdjustmentStep(
                rule_name="판금(W) 보정",
                rule_id="bodywork",
                description="판금 부위 수 동일",
                amount=0,
                details=f"대상: {target.bodywork_count}개, 기준: {reference.bodywork_count}개",
            )

        # 연식/가격대별 감가율 결정
        age = 2026 - target.year
        rate, tier_label = self._get_bodywork_rate(age, base_price)

        if rate == 0:
            return AdjustmentStep(
                rule_name="판금(W) 보정",
                rule_id="bodywork",
                description=f"판금 보정 미적용 ({tier_label})",
                amount=0,
                details=(
                    f"대상: {target.bodywork_count}개, 기준: {reference.bodywork_count}개\n"
                    f"조건: {tier_label} → 미적용"
                ),
                data_source="업계 룰: 500만원 미만 미적용",
            )

        amount = -diff * rate * base_price

        return AdjustmentStep(
            rule_name="판금(W) 보정",
            rule_id="bodywork",
            description=f"판금 {abs(diff)}회 {'감가' if diff > 0 else '가산'} ({tier_label}, {rate*100:.1f}%)",
            amount=round(amount, 1),
            details=(
                f"대상: {target.bodywork_count}개, 기준: {reference.bodywork_count}개\n"
                f"차이: {diff:+d}개\n"
                f"구간: {tier_label} → {rate*100:.1f}%/회\n"
                f"계산: {abs(diff)}회 × {rate*100:.1f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: {tier_label} {rate*100:.1f}%/회",
        )

    def _adjust_bodywork_by_part(self, target: Vehicle, reference: Vehicle,
                                 base_price: float) -> AdjustmentStep:
        """부위별 가중 판금 보정 (골격 부위 제외)"""
        BODYWORK_TYPES = {'PAINT_PANEL_BEATING', 'PANEL_WELDING'}
        structural = self._get_structural_parts()

        age = 2026 - target.year
        rate, tier_label = self._get_bodywork_rate(age, base_price)

        t_bodywork = [d for d in target.part_damages
                      if d['damage_type'] in BODYWORK_TYPES and d['part'] not in structural]
        r_bodywork = [d for d in reference.part_damages
                      if d['damage_type'] in BODYWORK_TYPES and d['part'] not in structural]

        t_weighted = sum(self._get_part_weight(d['part']) for d in t_bodywork)
        r_weighted = sum(self._get_part_weight(d['part']) for d in r_bodywork)
        diff = t_weighted - r_weighted

        if abs(diff) < 0.01 or rate == 0:
            desc = "판금 보정 차이 없음 (부위별 가중)" if abs(diff) < 0.01 else f"판금 보정 미적용 ({tier_label})"
            return AdjustmentStep(
                rule_name="판금(W) 보정",
                rule_id="bodywork",
                description=desc,
                amount=0,
                details=(
                    f"대상 판금: {[d['part'] for d in t_bodywork] or '없음'} (가중합: {t_weighted:.1f})\n"
                    f"기준 판금: {[d['part'] for d in r_bodywork] or '없음'} (가중합: {r_weighted:.1f})\n"
                    f"구간: {tier_label} → {rate*100:.1f}%/회"
                ),
            )

        amount = -diff * rate * base_price

        t_detail = ", ".join(f"{d['part']}(×{self._get_part_weight(d['part'])})" for d in t_bodywork) or "없음"
        r_detail = ", ".join(f"{d['part']}(×{self._get_part_weight(d['part'])})" for d in r_bodywork) or "없음"

        return AdjustmentStep(
            rule_name="판금(W) 보정",
            rule_id="bodywork",
            description=f"판금 {'감가' if diff > 0 else '가산'} ({tier_label}, 부위별 가중)",
            amount=round(amount, 1),
            details=(
                f"대상 판금: {t_detail} (가중합: {t_weighted:.1f})\n"
                f"기준 판금: {r_detail} (가중합: {r_weighted:.1f})\n"
                f"가중 차이: {diff:+.1f}\n"
                f"구간: {tier_label} → {rate*100:.1f}%/회\n"
                f"계산: {abs(diff):.1f} × {rate*100:.1f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: {tier_label} {rate*100:.1f}%/회 × 부위별 가중치",
        )

    def _get_bodywork_rate(self, age: int, price: float) -> tuple[float, str]:
        """판금 감가율 조회 (연식/가격대별)"""
        if price < 500:
            return 0.0, "500만원 미만"
        if age < 3 or price >= 2500:
            return 0.02, "3년 미만 또는 2500만원+"
        if (3 <= age < 7) or (1500 <= price < 2500):
            return 0.015, "3~7년 또는 1500~2500만원"
        return 0.01, "기타"

    # -----------------------------------------------------------------
    # 룰 4: 골격사고 보정
    # -----------------------------------------------------------------

    def _adjust_structural(self, target: Vehicle, reference: Vehicle,
                           base_price: float) -> AdjustmentStep:
        """골격사고(C-F등급) 감가 보정"""
        rules = self.rules.get('structural_damage', {})
        structural_parts = self._get_structural_parts()
        severe_types = set(rules.get('severe_damage_types', ['EXCHANGE', 'PANEL_WELDING', 'BENT']))

        # part_damages가 없으면 골격사고 판단 불가 → 스킵
        if not target.part_damages and not reference.part_damages:
            return AdjustmentStep(
                rule_name="골격사고 보정",
                rule_id="structural",
                description="부위별 데이터 없음 (판단 불가)",
                amount=0,
                details="part_damages 데이터가 없어 골격사고 판단을 생략합니다.",
            )

        # 대상/기준 골격 손상 부위 수
        t_structural = {d['part'] for d in target.part_damages
                        if d['part'] in structural_parts and d['damage_type'] in severe_types}
        r_structural = {d['part'] for d in reference.part_damages
                        if d['part'] in structural_parts and d['damage_type'] in severe_types}

        t_count = len(t_structural)
        r_count = len(r_structural)

        if t_count == 0 and r_count == 0:
            return AdjustmentStep(
                rule_name="골격사고 보정",
                rule_id="structural",
                description="골격사고 없음",
                amount=0,
                details="대상/기준 모두 골격 부위 손상 없음",
            )

        # 등급 판정
        t_grade, t_rate, t_label = self._classify_structural_grade(t_count, rules)
        r_grade, r_rate, r_label = self._classify_structural_grade(r_count, rules)

        # 차이 적용
        amount = -(t_rate - r_rate) * base_price

        if abs(amount) < 0.1:
            desc = f"골격사고 등급 동일 ({t_label})"
        else:
            desc = f"골격사고 {'감가' if amount < 0 else '가산'} ({t_label} vs {r_label})"

        return AdjustmentStep(
            rule_name="골격사고 보정",
            rule_id="structural",
            description=desc,
            amount=round(amount, 1),
            details=(
                f"대상 골격손상: {', '.join(sorted(t_structural)) or '없음'} → {t_label}\n"
                f"기준 골격손상: {', '.join(sorted(r_structural)) or '없음'} → {r_label}\n"
                f"감가율 차이: {(t_rate - r_rate)*100:+.0f}%\n"
                f"계산: {abs(t_rate - r_rate)*100:.0f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source="업계 룰: 골격사고 C-F등급 15-20%",
        )

    def _classify_structural_grade(self, count: int, rules: dict) -> tuple[str, float, str]:
        """골격 손상 부위 수 → 등급/감가율"""
        grades = rules.get('grades', {})
        if count == 0:
            return "무사고", 0.0, "무사고"
        if count >= grades.get('F', {}).get('min_parts', 5):
            g = grades['F']
            return "F", g['rate'], g['label']
        if count >= grades.get('E', {}).get('min_parts', 3):
            g = grades['E']
            return "E", g['rate'], g['label']
        if count >= grades.get('D', {}).get('min_parts', 2):
            g = grades['D']
            return "D", g['rate'], g['label']
        g = grades.get('C', {'rate': 0.15, 'label': 'C등급'})
        return "C", g['rate'], g['label']

    # -----------------------------------------------------------------
    # 부위별 가중치 헬퍼
    # -----------------------------------------------------------------

    def _get_structural_parts(self) -> set[str]:
        """골격+반구조 부위 목록"""
        part_rules = self.rules.get('part_weight', {})
        return set(
            part_rules.get('structural_parts', []) +
            part_rules.get('semi_structural_parts', [])
        )

    def _get_part_weight(self, part: str) -> float:
        """부위명 → 가중치 반환"""
        part_rules = self.rules.get('part_weight', {})
        major = part_rules.get('major_outer', {})
        minor = part_rules.get('minor_parts', {})
        if part in major.get('parts', []):
            return major.get('weight', 1.5)
        if part in minor.get('parts', []):
            return minor.get('weight', 0.3)
        return part_rules.get('default_weight', 1.0)

    # -----------------------------------------------------------------
    # 룰 5: 색상 보정
    # -----------------------------------------------------------------

    def _adjust_color(self, target: Vehicle, reference: Vehicle) -> AdjustmentStep:
        """
        색상 차이에 따른 가격 보정.

        업계 룰: 선호도별 20만원 차등
        - 대형: 흰색=검정 > 메탈 > 실버 > 원색
        - 일반: 흰색 > 검정 > 메탈 > 실버 > 원색
        - 경차: 흰색 > 미색 > 메탈=실버 > 검정 > 원색
        """
        rules = self.rules['color_adjustment']

        # 세그먼트 → 색상 테이블 매핑
        segment = target.segment or ""
        if '대형' in segment or '프리미엄' in segment:
            table_key = 'large'
        elif 'SUV' in segment.upper():
            table_key = 'suv'
        elif '경차' in segment:
            table_key = 'compact'
        else:
            table_key = 'medium'

        table = rules.get(table_key, rules.get('medium', {}))
        target_adj = table.get(target.color_group, table.get('other', 0))
        ref_adj = table.get(reference.color_group, table.get('other', 0))
        raw_diff = target_adj - ref_adj

        # 연식 가중치
        age = 2026 - target.year
        weight = 1.0
        for w in rules.get('age_weight', []):
            if age <= w['max_age']:
                weight = w['weight']
                break

        amount = raw_diff * weight

        if abs(amount) < 1:
            desc = "색상 차이 없음"
        else:
            desc = f"{target.color_group} vs {reference.color_group} ({table_key}급, 가중치 {weight})"

        return AdjustmentStep(
            rule_name="색상 보정",
            rule_id="color",
            description=desc,
            amount=round(amount, 1),
            details=(
                f"차급: {table_key} (세그먼트: {segment})\n"
                f"대상 [{target.color_group}]: {target_adj:+.0f}만원\n"
                f"기준 [{reference.color_group}]: {ref_adj:+.0f}만원\n"
                f"차이: {raw_diff:+.0f}만원 × 연식가중치 {weight} = {amount:+.1f}만원"
            ),
            data_source=f"업계 룰: {table_key}급, 선호도별 20만원/단계",
        )

    # -----------------------------------------------------------------
    # 룰 6: 선호옵션 보정 (선스네후)
    # -----------------------------------------------------------------

    def _adjust_options(self, target: Vehicle, reference: Vehicle) -> AdjustmentStep:
        """
        옵션 차이 보정 (v2.4 개선).

        항상 옵션 리스트의 set 차이로 비교 (같은 옵션 무시, 차이만 적용).
        출고가-기본가 데이터는 개별 옵션 단가 추정에만 활용.
        """
        rules = self.rules['preferred_options']
        SUNROOF_VALUE = rules.get('sunroof', {}).get('default', 50)
        GENERAL_OPTION_DEFAULT = rules.get('general_option_default', 20)

        # ── 연식 가중치 ──
        age = 2026 - target.year
        weight = 1.0
        color_rules = self.rules.get('color_adjustment', {})
        for w in color_rules.get('age_weight', []):
            if age <= w['max_age']:
                weight = w['weight']
                break

        # ── 기본옵션 키워드 로드 ──
        basic_keywords = rules.get('basic_option_keywords', [])

        # ── 항상 set 비교 ──
        t_set = set(self._normalize_option_names(target.options))
        r_set = set(self._normalize_option_names(reference.options))
        only_target_raw = t_set - r_set
        only_ref_raw = r_set - t_set

        # 기본옵션 필터링 (키워드 매칭되면 제외)
        only_target = {o for o in only_target_raw
                       if not self._is_basic_option(o, basic_keywords)}
        only_ref = {o for o in only_ref_raw
                    if not self._is_basic_option(o, basic_keywords)}
        filtered_count = (len(only_target_raw) - len(only_target) +
                          len(only_ref_raw) - len(only_ref))

        # 옵션 단가 추정 (출고가-기본가 → 단가 추정에만 사용)
        per_option_avg = GENERAL_OPTION_DEFAULT
        price_source = "기본값"
        # 1) 개별 차량의 출고가-기본가 역산
        for veh in (target, reference):
            opt_total = self._calc_option_total(veh)
            n_opts = len(veh.options)
            if opt_total > 0 and n_opts > 0:
                per_option_avg = round(opt_total / n_opts, 1)
                price_source = "출고가-기본가 역산"
                break
        # 2) 같은 차종 통계 (Firestore에서 사전 계산된 값)
        if price_source == "기본값":
            for veh in (target, reference):
                if veh.option_unit_price > 0:
                    per_option_avg = veh.option_unit_price
                    price_source = "동일 차종 통계"
                    break

        # 선루프(50만원) / 일반옵션(per_option_avg) 분류
        target_val = 0
        ref_val = 0
        detail_lines = []

        for opt in sorted(only_target):
            if self._is_sunroof(opt):
                val = SUNROOF_VALUE
                detail_lines.append(f"  + {opt} (선루프): +{val}만원")
            else:
                val = per_option_avg
                detail_lines.append(f"  + {opt}: +{val}만원")
            target_val += val

        for opt in sorted(only_ref):
            if self._is_sunroof(opt):
                val = SUNROOF_VALUE
                detail_lines.append(f"  - {opt} (선루프): -{val}만원")
            else:
                val = per_option_avg
                detail_lines.append(f"  - {opt}: -{val}만원")
            ref_val += val

        raw_amount = target_val - ref_val
        amount = raw_amount * weight

        if abs(amount) < 1:
            desc = "옵션 차이 없음"
        else:
            n_diff = len(only_target) + len(only_ref)
            weight_note = f", 연식가중치 {weight}" if weight < 1.0 else ""
            desc = f"옵션 {n_diff}개 차이 {'가산' if amount > 0 else '감가'}{weight_note}"

        # 필터링된 기본옵션 목록 (디버깅용)
        filtered_target = sorted(only_target_raw - only_target)
        filtered_ref = sorted(only_ref_raw - only_ref)
        filter_info = ""
        if filtered_count > 0:
            filter_info = f"\n---\n기본옵션 제외 {filtered_count}개"
            if filtered_target:
                filter_info += f"\n  대상 제외: {', '.join(filtered_target)}"
            if filtered_ref:
                filter_info += f"\n  기준 제외: {', '.join(filtered_ref)}"

        return AdjustmentStep(
            rule_name="옵션 보정",
            rule_id="options",
            description=desc,
            amount=round(amount, 1),
            details=(
                f"대상 옵션({len(t_set)}개): {', '.join(sorted(t_set)) or '없음'}\n"
                f"기준 옵션({len(r_set)}개): {', '.join(sorted(r_set)) or '없음'}\n"
                f"---\n"
                + ("\n".join(detail_lines) if detail_lines else "  차이 없음")
                + filter_info
                + (f"\n---\n연식가중치: {weight} (차령 {age}년)" if weight < 1.0 else "")
                + (f"\n옵션 단가: {per_option_avg}만원/개 ({price_source})" if price_source != "기본값" else "")
            ),
            data_source="옵션 리스트 비교: 기본옵션 제외, 선루프 50만원, 일반옵션 추정가 × 연식가중치",
        )

    @staticmethod
    def _calc_option_total(vehicle: Vehicle) -> float:
        """출고가 - 기본가 = 옵션가 (양수일 때만, 음수는 할인이므로 0)"""
        if vehicle.factory_price > 0 and vehicle.base_price > 0:
            diff = vehicle.factory_price - vehicle.base_price
            return diff if diff > 0 else 0
        return 0

    @staticmethod
    def _normalize_option_names(options: list[str]) -> list[str]:
        """옵션명 정규화 (괄호 정리, 공백 정리)"""
        result = []
        for opt in options:
            name = opt.strip()
            if not name:
                continue
            # "전동시트 (운전석" + "조수석)" 같은 분리된 옵션 스킵
            if name.endswith(')') and '(' not in name:
                continue
            result.append(name)
        return result

    @staticmethod
    def _is_sunroof(option_name: str) -> bool:
        """선루프 여부 판별 (부분 매칭: 선루프, 썬루프, sunroof)"""
        name = option_name.lower()
        return '선루프' in name or '썬루프' in name or 'sunroof' in name

    @staticmethod
    def _is_basic_option(option_name: str, keywords: list[str]) -> bool:
        """기본옵션 여부 판별 (키워드 부분 매칭)"""
        name = option_name.lower()
        for kw in keywords:
            if kw.lower() in name:
                return True
        return False

    @staticmethod
    def _option_diff_text(target_opts: list[str], ref_opts: list[str]) -> str:
        """두 옵션 리스트의 차이를 텍스트로"""
        t_set = set(target_opts)
        r_set = set(ref_opts)
        only_t = sorted(t_set - r_set)
        only_r = sorted(r_set - t_set)
        if not only_t and not only_r:
            return "\n옵션 구성 동일"
        lines = ["\n--- 옵션 차이 ---"]
        if only_t:
            lines.append(f"대상만: {', '.join(only_t)}")
        if only_r:
            lines.append(f"기준만: {', '.join(only_r)}")
        return "\n" + "\n".join(lines)

    # -----------------------------------------------------------------
    # 룰 7: 연식 차이 보정
    # -----------------------------------------------------------------

    def _adjust_year_diff(self, target: Vehicle, reference: Vehicle,
                          base_price: float) -> AdjustmentStep:
        """
        연식 차이에 따른 보정.

        업계 룰: 연당 2% (낙찰가 기준)
        대상이 기준보다 오래되면 감가, 새로우면 가산.
        """
        year_rules = self.rules.get('year_adjustment', {})
        rate = year_rules.get('rate_per_year', 0.02)

        diff = target.year - reference.year  # 양수 = 대상이 더 새로움

        if diff == 0:
            return AdjustmentStep(
                rule_name="연식 보정",
                rule_id="year_diff",
                description="연식 동일",
                amount=0,
                details=f"대상: {target.year}년식, 기준: {reference.year}년식",
            )

        amount = diff * rate * base_price

        return AdjustmentStep(
            rule_name="연식 보정",
            rule_id="year_diff",
            description=f"연식 {abs(diff)}년 {'가산' if diff > 0 else '감가'} (연당 {rate*100:.0f}%)",
            amount=round(amount, 1),
            details=(
                f"대상: {int(target.year)}년식, 기준: {int(reference.year)}년식\n"
                f"차이: {diff:+.0f}년\n"
                f"계산: {abs(diff)}년 × {rate*100:.0f}% × {base_price:.0f}만원 = {abs(amount):.1f}만원"
            ),
            data_source=f"업계 룰: 연식 차이 연당 {rate*100:.0f}%",
        )

    # -----------------------------------------------------------------
    # 룰 8: 트림 차이 경고
    # -----------------------------------------------------------------

    def _warn_trim_diff(self, target: Vehicle, reference: Vehicle) -> AdjustmentStep | None:
        """
        트림이 다를 때 경고를 반환.

        가격 보정은 하지 않지만, 신뢰도를 낮추는 데 사용.
        - 둘 다 트림 정보 없음 → 경고 안 함
        - 둘 다 있고 동일 → 경고 안 함
        - 한쪽만 없거나, 둘 다 있는데 다름 → 경고
        """
        t_trim = target.trim.strip() if target.trim else ""
        r_trim = reference.trim.strip() if reference.trim else ""

        # 둘 다 트림 정보가 없으면 비교 불가 → 경고 안 함
        if not t_trim and not r_trim:
            return None

        # 동일하면 경고 안 함
        if t_trim == r_trim:
            return None

        # 한쪽만 트림 정보 없는 경우
        if not r_trim:
            return AdjustmentStep(
                rule_name="⚠ 트림 차이 경고",
                rule_id="trim_warning",
                description=f"기준차량 트림 정보 없음 (대상: {t_trim})",
                amount=0,
                details=(
                    f"대상 트림: {t_trim}\n"
                    f"기준 트림: (정보 없음)\n"
                    f"기준차량의 트림 정보가 없어 트림 일치 여부를 확인할 수 없습니다.\n"
                    f"이 산출 결과의 신뢰도는 '낮음'으로 조정됩니다."
                ),
                data_source="기준차량 트림 정보 부재 → 경고 표시",
            )

        if not t_trim:
            return AdjustmentStep(
                rule_name="⚠ 트림 차이 경고",
                rule_id="trim_warning",
                description=f"대상차량 트림 미입력 (기준: {r_trim})",
                amount=0,
                details=(
                    f"대상 트림: (미입력)\n"
                    f"기준 트림: {r_trim}\n"
                    f"대상차량의 트림이 입력되지 않아 트림 일치 여부를 확인할 수 없습니다.\n"
                    f"이 산출 결과의 신뢰도는 '낮음'으로 조정됩니다."
                ),
                data_source="대상차량 트림 미입력 → 경고 표시",
            )

        # 둘 다 있는데 다른 경우
        return AdjustmentStep(
            rule_name="⚠ 트림 차이 경고",
            rule_id="trim_warning",
            description=f"트림이 다릅니다: {t_trim} vs {r_trim}",
            amount=0,
            details=(
                f"대상 트림: {t_trim}\n"
                f"기준 트림: {r_trim}\n"
                f"트림이 다르면 가격 차이가 클 수 있습니다.\n"
                f"이 산출 결과의 신뢰도는 '낮음'으로 조정됩니다."
            ),
            data_source="트림 차이에 의한 가격 변동은 정형화 불가 → 경고만 표시",
        )

    # -----------------------------------------------------------------
    # 최종 가격 산출
    # -----------------------------------------------------------------

    def _calculate_final_price(self, target: Vehicle,
                               result: PriceResult,
                               has_trim_warning: bool = False) -> PriceResult:
        """
        최종 가격 산출.

        공식:
          보정된 가격 = 기준가 + 총 보정액
          예상 낙찰가 = 보정된 가격 (이미 낙찰가 기준이므로)
          추정 소매가 = 시장 데이터 기반 추정 (폴백: 경매 할인율 역산)
        """
        from app.services.retail_estimator import estimate_retail_by_market, estimate_auction_by_market

        calc_rules = self.rules['price_calculation']

        adjusted_price = result.reference_price + result.total_adjustment
        result.estimated_auction = round(adjusted_price, 1)

        # ── 낙찰가: 시장 데이터 단독 ──
        auction_market = estimate_auction_by_market(
            maker=target.maker,
            model=target.model,
            trim=target.trim,
            year=target.year,
            mileage=target.mileage,
            factory_price=target.factory_price,
            base_price=target.base_price,
            fuel=target.fuel,
        )
        if auction_market.success:
            result.estimated_auction = auction_market.estimated_auction
        result.adjustments.append(AdjustmentStep(
            rule_name="낙찰가 추정 (시장 데이터)",
            rule_id="auction_market",
            description=(
                f"같은 트림·연식 낙찰 {auction_market.vehicles_found}건 분석 → "
                f"추정 낙찰가 {result.estimated_auction:,.0f}만원"
                if auction_market.success
                else f"시장 데이터 부족 — 룰엔진 값 유지 ({result.estimated_auction:,.0f}만원)"
            ),
            amount=0,
            details=auction_market.details if auction_market.success else "시장 데이터 부족",
            data_source=(
                f"시장 데이터 추이 ({auction_market.vehicles_found}건)"
                if auction_market.success else "룰엔진 폴백"
            ),
        ))

        # ── 소매가 추정: 시장 데이터 기반 ──
        retail_result = estimate_retail_by_market(
            maker=target.maker,
            model=target.model,
            trim=target.trim,
            year=target.year,
            mileage=target.mileage,
            factory_price=target.factory_price,
            base_price=target.base_price,
            fuel=target.fuel,
        )

        result.estimated_retail = retail_result.estimated_retail if retail_result.success else 0
        result.adjustments.append(AdjustmentStep(
            rule_name="소매가 추정 (비율 추이)",
            rule_id="retail_estimation",
            description=(
                f"같은 트림·연식 소매 {retail_result.vehicles_found}건 분석 → "
                f"비율 {retail_result.estimated_ratio:.1f}% × "
                f"{target.factory_price or target.base_price:,.0f}만 = "
                f"추정 소매가 {result.estimated_retail:,.0f}만원"
                if retail_result.success
                else f"소매가 추정 불가: {retail_result.details}"
            ),
            amount=0,
            details=retail_result.details,
            data_source=(
                f"시장 데이터 비율 추이 ({retail_result.vehicles_found}건)"
                if retail_result.success else "데이터 부족"
            ),
        ))

        # 신뢰도 판단
        if has_trim_warning:
            result.confidence = "낮음"
        elif retail_result.success:
            non_zero = sum(1 for s in result.adjustments
                          if abs(s.amount) > 0 and s.rule_id != "retail_estimation")
            if retail_result.confidence == "높음" and non_zero <= 1:
                result.confidence = "높음"
            elif retail_result.confidence == "낮음" or non_zero >= 4:
                result.confidence = "낮음"
            else:
                result.confidence = "보통"
        else:
            non_zero = sum(1 for s in result.adjustments
                          if abs(s.amount) > 0 and s.rule_id != "retail_estimation")
            if non_zero <= 1:
                result.confidence = "높음"
            elif non_zero <= 3:
                result.confidence = "보통"
            else:
                result.confidence = "낮음"

        # 요약
        result.summary = (
            f"기준가 {result.reference_price:.0f}만원"
            f" → 보정 {result.total_adjustment:+.0f}만원"
            f" → 예상 낙찰가 {result.estimated_auction:.0f}만원"
            + (f" | 소매가 {result.estimated_retail:.0f}만원 (비율 {retail_result.estimated_ratio:.1f}%)"
               if retail_result.success else " | 소매가 추정 불가")
        )

        return result

    # -----------------------------------------------------------------
    # 유틸리티
    # -----------------------------------------------------------------

    def get_rules_summary(self) -> list[dict]:
        """현재 적용 중인 룰 목록 반환 (문서화용)"""
        return [
            {
                "id": "mileage",
                "name": "주행거리 감가",
                "description": "연식별 감가율: 2~3년 2%, 4~6년 1.5%, 7~9년 1%, 10년+ 1%/10만원",
                "ceiling": "20만km 초과 증감 미적용",
            },
            {
                "id": "exchange",
                "name": "교환(X) 보정",
                "description": "1회당 신차대비율 3% (부위별 가중치 적용, 골격 부위 제외)",
            },
            {
                "id": "bodywork",
                "name": "판금(W) 보정",
                "description": "연식/가격대별 1~2% (부위별 가중치 적용, 골격 부위 제외)",
            },
            {
                "id": "structural",
                "name": "골격사고 보정",
                "description": "C등급(1부위) 15%, D등급(2부위) 17%, E등급(3부위) 18%, F등급(5부위+) 20%",
            },
            {
                "id": "color",
                "name": "색상 보정",
                "description": "선호도별 20만원 차등",
                "segments": {
                    "대형": "흰색=검정 > 메탈 > 실버 > 원색",
                    "일반": "흰색 > 검정 > 메탈 > 실버 > 원색",
                    "경차": "흰색 > 미색 > 메탈=실버 > 검정 > 원색",
                },
            },
            {
                "id": "options",
                "name": "옵션 보정",
                "description": "옵션 리스트 차이 비교 (선루프 50만원, 일반옵션 추정가) × 연식가중치",
            },
            {
                "id": "year_diff",
                "name": "연식 보정",
                "description": "연식 차이별 보정: 연당 2%",
            },
            {
                "id": "trim_warning",
                "name": "트림 차이 경고",
                "description": "기준차량과 트림이 다를 때 경고 표시 (가격 보정 없음, 신뢰도 하향)",
            },
        ]


# =========================================================================
# 간편 사용 함수
# =========================================================================

_engine: RuleEngine | None = None


def get_engine() -> RuleEngine:
    """싱글톤 룰 엔진 인스턴스"""
    global _engine
    if _engine is None:
        _engine = RuleEngine()
    return _engine


def calculate_price(target: dict, reference: dict) -> dict:
    """
    dict 기반 간편 호출.

    Args:
        target: 대상차량 정보 dict
        reference: 기준차량 정보 dict (auction_price 필수)

    Returns:
        dict: 가격 산출 결과
    """
    engine = get_engine()
    t = Vehicle(**{k: v for k, v in target.items() if k in Vehicle.__dataclass_fields__})
    r = Vehicle(**{k: v for k, v in reference.items() if k in Vehicle.__dataclass_fields__})

    # 색상 정규화 — target은 "흰색" 같은 한국어, reference는 이미 color_group일 수 있음
    if t.color and not t.color_group:
        t.color_group = normalize_color(t.color)
    if r.color and not r.color_group:
        r.color_group = normalize_color(r.color)

    result = engine.calculate(t, r)

    return {
        "reference_price": result.reference_price,
        "adjustments": [
            {
                "rule_name": s.rule_name,
                "rule_id": s.rule_id,
                "description": s.description,
                "amount": s.amount,
                "details": s.details,
                "data_source": s.data_source,
            }
            for s in result.adjustments
        ],
        "total_adjustment": result.total_adjustment,
        "estimated_auction": result.estimated_auction,
        "estimated_retail": result.estimated_retail,
        "confidence": result.confidence,
        "summary": result.summary,
    }
