"""
CarFarm v2 — 룰 엔진 (Rule Engine)

기준차량 낙찰가를 기반으로 대상차량의 예상 가격을 산출한다.
각 보정 단계가 투명하게 기록되어 사용자에게 "왜 이 가격인지" 설명할 수 있다.

적용 룰 목록 (pricing_rules.yaml 기반, 프라이싱 매니저 업계 룰):
  1. 주행거리 감가 — 연식별 차등 (2%/1.5%/1%/10만원), 20만km 천장
  2. 교환 보정 — 1회당 신차대비율 3%
  3. 판금·도색 보정 — 연식/가격대별 1~2%
  4. 색상 보정 — 선호도별 20만원, 차급별 선호 순서
  5. 렌터카 감가 — 단일 계수
  6. 선호옵션(선스네후) — 1개당 50만원
  7. 최종 가격 산출 — 소매가 × 경매할인율, 최소 마진 적용

사용법:
    engine = RuleEngine()
    result = engine.calculate(target, reference)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field


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
    auction_price: float = 0       # 낙찰가 (만원) — 기준차량만 해당
    new_car_price: float = 0       # 신차가격 (만원) — 있으면 사용


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

        # ── 룰 1: 주행거리 감가 ──
        step = self._adjust_mileage(target, reference, base_price)
        result.adjustments.append(step)

        # ── 룰 2: 교환 보정 ──
        step = self._adjust_exchange(target, reference, base_price)
        result.adjustments.append(step)

        # ── 룰 3: 판금 보정 ──
        step = self._adjust_bodywork(target, reference, base_price)
        result.adjustments.append(step)

        # ── 룰 4: 색상 보정 ──
        step = self._adjust_color(target, reference)
        result.adjustments.append(step)

        # ── 룰 5: 렌터카 감가 ──
        step = self._adjust_rental(target, reference, base_price)
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

        업계 룰: 1회당 신차대비율 3% (선형)
        대상이 기준보다 교환 많으면 감가, 적으면 가산.
        """
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

    # -----------------------------------------------------------------
    # 룰 3: 판금·도색 보정
    # -----------------------------------------------------------------

    def _adjust_bodywork(self, target: Vehicle, reference: Vehicle,
                         base_price: float) -> AdjustmentStep:
        """
        판금·도색(W) 부위 수 차이에 따른 보정.

        업계 룰 (연식/가격대별 차등):
        - 3년 미만 또는 2500만원+: 1회당 2%
        - 3~7년 또는 1500~2500만원: 1회당 1.5%
        - 500만원 미만: 미적용
        - 기타: 1회당 1%
        """
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
    # 룰 4: 색상 보정
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
    # 룰 5: 렌터카 감가
    # -----------------------------------------------------------------

    def _adjust_rental(self, target: Vehicle, reference: Vehicle,
                       base_price: float) -> AdjustmentStep:
        """
        렌터카 경력에 따른 감가 보정.

        업계 룰: 단일 계수 적용 (약 5%)
        """
        rules = self.rules['rental_discount']
        target_rental = target.usage == 'rental'
        ref_rental = reference.usage == 'rental'

        # 둘 다 렌터카이거나 둘 다 아니면 차이 없음
        if target_rental == ref_rental:
            return AdjustmentStep(
                rule_name="렌터카 보정",
                rule_id="rental",
                description="경력 동일 (보정 없음)",
                amount=0,
                details=f"대상: {target.usage}, 기준: {reference.usage}",
            )

        rate = rules.get('default_rate', -0.05)

        # 대상이 렌터카면 감가, 기준이 렌터카면 가산
        if target_rental:
            amount = rate * base_price
            direction = "감가"
        else:
            amount = -rate * base_price
            direction = "가산"

        return AdjustmentStep(
            rule_name="렌터카 보정",
            rule_id="rental",
            description=f"렌터카 {direction} ({rate*100:+.1f}%)",
            amount=round(amount, 1),
            details=(
                f"대상: {target.usage}, 기준: {reference.usage}\n"
                f"계수: {rate*100:+.1f}%\n"
                f"계산: {rate*100:+.1f}% × {base_price:.0f}만원 = {amount:+.1f}만원"
            ),
            data_source=f"업계 룰: 렌터카 {rate*100:+.1f}%",
        )

    # -----------------------------------------------------------------
    # 룰 6: 선호옵션 보정 (선스네후)
    # -----------------------------------------------------------------

    def _adjust_options(self, target: Vehicle, reference: Vehicle) -> AdjustmentStep:
        """
        선호옵션(선루프/스마트키/네비/후방카메라) 차이 보정.

        업계 룰: 1개당 약 50만원
        """
        rules = self.rules['preferred_options']

        option_map = {
            '선루프': 'sunroof',
            '스마트키': 'smart_key',
            '네비게이션': 'navigation',
            '후방카메라': 'rear_camera',
        }

        target_total = 0
        ref_total = 0
        detail_lines = []

        for kor_name, eng_key in option_map.items():
            option_rule = rules.get(eng_key, {})
            value = option_rule.get('default', 50)

            t_has = kor_name in target.options
            r_has = kor_name in reference.options

            if t_has:
                target_total += value
            if r_has:
                ref_total += value

            if t_has != r_has:
                diff = value if t_has else -value
                detail_lines.append(
                    f"  {kor_name}: {'있음' if t_has else '없음'} vs "
                    f"{'있음' if r_has else '없음'} → {diff:+.0f}만원"
                )

        raw_amount = target_total - ref_total

        # 연식 가중치 — 오래된 차량일수록 옵션 가치 감소
        age = 2026 - target.year
        weight = 1.0
        color_rules = self.rules.get('color_adjustment', {})
        for w in color_rules.get('age_weight', []):
            if age <= w['max_age']:
                weight = w['weight']
                break

        amount = raw_amount * weight

        if abs(amount) < 1:
            desc = "옵션 차이 없음"
        else:
            weight_note = f", 연식가중치 {weight}" if weight < 1.0 else ""
            desc = f"선호옵션 {'가산' if amount > 0 else '감가'} (1개당 50만원{weight_note})"

        return AdjustmentStep(
            rule_name="선호옵션 보정",
            rule_id="options",
            description=desc,
            amount=round(amount, 1),
            details=(
                f"대상: {target_total:.0f}만원\n"
                f"기준: {ref_total:.0f}만원\n"
                + ("\n".join(detail_lines) if detail_lines else "  차이 없음")
                + (f"\n연식가중치: {weight} (차령 {age}년)" if weight < 1.0 else "")
            ),
            data_source="업계 룰: 선호옵션 1개당 50만원 × 연식가중치",
        )

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
                f"대상: {target.year}년식, 기준: {reference.year}년식\n"
                f"차이: {diff:+d}년\n"
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
          추정 소매가 = 예상 낙찰가 / 경매 할인율
          최소 마진 150만원 적용
        """
        calc_rules = self.rules['price_calculation']

        adjusted_price = result.reference_price + result.total_adjustment

        # 경매 할인율 역산 → 소매가 추정
        is_import = target.segment and '수입' in target.segment
        discount = calc_rules['auction_discount'].get(
            'import' if is_import else 'domestic', 0.90
        )
        estimated_retail = adjusted_price / discount

        # 최소 마진
        min_margin = calc_rules.get('min_margin', 150)
        if estimated_retail - adjusted_price < min_margin:
            estimated_retail = adjusted_price + min_margin

        result.estimated_auction = round(adjusted_price, 1)
        result.estimated_retail = round(estimated_retail, 1)

        # 신뢰도 판단
        if has_trim_warning:
            result.confidence = "낮음"
        else:
            non_zero = sum(1 for s in result.adjustments if abs(s.amount) > 0)
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
            f" (소매가 {result.estimated_retail:.0f}만원)"
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
                "description": "1회당 신차대비율 3%",
            },
            {
                "id": "bodywork",
                "name": "판금(W) 보정",
                "description": "연식/가격대별 1~2% (3년미만/2500만+: 2%, 3~7년/1500~2500: 1.5%, 기타: 1%, 500만-: 미적용)",
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
                "id": "rental",
                "name": "렌터카 감가",
                "description": f"단일 계수 {self.rules['rental_discount']['default_rate']*100:.0f}%",
            },
            {
                "id": "options",
                "name": "선호옵션 (선스네후)",
                "description": "선루프/스마트키/네비/후방카메라, 1개당 50만원 × 연식가중치",
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
