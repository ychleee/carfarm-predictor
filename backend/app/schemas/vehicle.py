"""
차량 입력 스키마 — Firestore(camelCase) & CarFarm(snake_case) 양방향 호환

Pydantic v2 alias + populate_by_name으로 두 형식 모두 수용.
Firestore 전용 필드(검수, 워크플로 등)는 extra="ignore"로 자동 무시.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PartDamageSchema(BaseModel):
    """부위별 손상 정보"""
    part: str          # VehiclePart enum name (예: "HOOD", "FRONT_PANEL")
    damage_type: str   # DamageType enum name (예: "EXCHANGE", "PAINT_PANEL_BEATING")


class TargetVehicleSchema(BaseModel):
    """대상차량 입력 스키마 — API 경계에서 정규화 수행"""

    model_config = ConfigDict(
        populate_by_name=True,  # alias와 필드명 모두 허용
        extra="ignore",         # Firestore 전용 필드 자동 무시
    )

    # --- 식별 필드 ---
    vehicle_id: str = Field(default="", alias="vehicleId")

    # --- 핵심 필드 (Isaac VehicleModel 필드명 통일) ---
    maker: str = Field(alias="vehicleMaker")
    model: str = Field(alias="vehicleModel")
    year: int = Field(alias="vehicleYear")
    mileage: int = Field(default=0)
    fuel: str | None = Field(default=None, alias="fuelType")
    displacement: str | None = Field(default=None, alias="engineDisplacement")
    drive: str | None = Field(default=None, alias="driveType")
    trim: str | None = Field(default=None, alias="vehicleTrim")
    color: str | None = Field(default=None, alias="vehicleColor")
    usage: str | None = Field(default=None, alias="vehicleCategory")
    domestic: bool = True
    options: list[str] = Field(default_factory=list, alias="vehicleOptions")
    exchange_count: int = Field(default=0, alias="exchangeCount")
    bodywork_count: int = Field(default=0, alias="bodyworkCount")
    generation: str | None = None
    base_price: float = Field(default=0, alias="vehicleBasePrice")
    factory_price: float = Field(default=0, alias="vehicleFactoryPrice")
    exclude_auction_ids: list[str] = Field(
        default_factory=list, alias="excludeAuctionIds"
    )
    part_damages: list[PartDamageSchema] = Field(
        default_factory=list, alias="partDamages"
    )

    # --- 타입 변환 validators ---

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        """문자열 → 정수: "2014" → 2014"""
        return int(v) if isinstance(v, str) and v.strip() else v

    @field_validator("mileage", mode="before")
    @classmethod
    def coerce_mileage(cls, v):
        """문자열 → 정수: "194362" → 194362"""
        return int(v) if isinstance(v, str) and v.strip() else v

    @field_validator("displacement", mode="before")
    @classmethod
    def normalize_displacement(cls, v):
        """cc → 리터 변환: "1968" → "2.0", "2.0" → "2.0" (그대로)"""
        if not v:
            return v
        try:
            num = float(str(v).replace(",", ""))
            if num > 100:  # cc 단위로 판단
                return f"{num / 1000:.1f}"
            return str(v)
        except ValueError:
            return str(v)

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, v):
        """Firestore [{name, price}] → ["선루프", "네비게이션"]"""
        if not v:
            return []
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [item.get("name", "") for item in v if item.get("name")]
        return v  # 이미 string list

    @field_validator("usage", mode="before")
    @classmethod
    def normalize_usage(cls, v):
        """한국어 → 영문 매핑: "자가용" → "personal" """
        if not v:
            return v
        mapping = {
            "자가용": "personal",
            "렌터카": "rental",
            "영업용": "commercial",
            "관용": "commercial",
        }
        return mapping.get(v, v)

    @field_validator("exchange_count", "bodywork_count", mode="before")
    @classmethod
    def coerce_int_field(cls, v):
        """문자열/None → 정수"""
        if v is None:
            return 0
        return int(v) if isinstance(v, str) and v.strip() else (v or 0)

    @field_validator("base_price", "factory_price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        """원 단위 → 만원 변환: "50600000" → 5060.0, 이미 만원이면 그대로"""
        if not v:
            return 0
        try:
            num = float(str(v).replace(",", ""))
            return round(num / 10000, 1) if num > 100000 else num
        except (ValueError, TypeError):
            return 0

    @field_validator("part_damages", mode="before")
    @classmethod
    def normalize_part_damages(cls, v):
        """Firestore [{part, damageType, ...}] → [{part, damage_type}] 정규화"""
        if not v:
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                part = item.get("part", "")
                dt = item.get("damage_type") or item.get("damageType", "")
                if part and dt:
                    result.append({"part": part, "damage_type": dt})
        return result
