// === 백엔드 Pydantic 모델 미러 ===

/** 대상차량 (POST /api/recommend 입력) — Isaac VehicleModel 필드명 통일 */
export interface TargetVehicle {
  vehicleMaker: string;
  vehicleModel: string;
  generation?: string | null;
  vehicleYear: number;
  mileage: number;
  fuelType?: string | null;
  engineDisplacement?: string | null;
  driveType?: string | null;
  vehicleTrim?: string | null;
  vehicleColor?: string | null;
  vehicleCategory?: string | null;
  domestic?: boolean;
  vehicleOptions?: string[];
  vehicleFactoryPrice?: string | null;
  exchangeCount?: number;
  bodyworkCount?: number;
}

/** 낙찰가 차량 */
export interface AuctionVehicle {
  auction_id: string;
  company_id?: string;
  vehicle_name: string | null;
  year: number | null;
  mileage: number | null;
  auction_price: number | null;   // 낙찰가 (만원)
  auction_date: string | null;
  color: string | null;
  trim: string | null;
  options: string | null;
  factory_price: number | null;   // 출고가 (만원)
  base_price: number | null;      // 기본가 (만원)
  inspection_grade: string | null; // 검차등급
  is_export?: boolean;             // 수출여부
  fuel: string | null;             // 연료
  // 프레임 검차
  frame_exchange: number;
  frame_bodywork: number;
  frame_corrosion: number;
  // 외부패널 검차
  exterior_exchange: number;
  exterior_bodywork: number;
  exterior_corrosion: number;
  // LLM 추천 이유
  reason?: string | null;
  // 엔카진단 여부
  has_encar_diagnosis?: boolean;
  // 차량 상태 (엔카: "완료"=판매완료)
  status?: string | null;
}

/** search-auction 응답 */
export interface SearchAuctionResponse {
  count: number;
  results: AuctionVehicle[];
}

/** 회사 탭 정보 */
export interface CompanyTab {
  id: string;
  label: string;
  color: string;        // tailwind color prefix (e.g. "green", "orange", "purple")
  bgClass: string;       // 배지 bg
  textClass: string;     // 배지 text
}

export const COMPANY_TABS: CompanyTab[] = [
  {
    id: "KYMaGfcnzwGsvbDm6Z91",
    label: "엔카",
    color: "green",
    bgClass: "bg-green-100",
    textClass: "text-green-700",
  },
  {
    id: "cRFWlHv4PZczXpd8bEw2",
    label: "헤이딜러",
    color: "orange",
    bgClass: "bg-orange-100",
    textClass: "text-orange-700",
  },
  {
    id: "vF8hj91n0tgzqUfWsuvJ",
    label: "셀카",
    color: "purple",
    bgClass: "bg-purple-100",
    textClass: "text-purple-700",
  },
];

/** 필터 인터페이스 */
export interface AuctionFilters {
  trim: string;
  fuel: string;
  color: string;
  yearMin: number | null;
  yearMax: number | null;
  mileageMin: number | null;
  mileageMax: number | null;
  soldOnly: boolean;
}

/** 필터 옵션 (데이터에서 추출) */
export interface FilterOptions {
  trims: string[];
  fuels: string[];
  colors: string[];
  yearRange: [number, number];
  mileageRange: [number, number];
}

/** 보정 단계 */
export interface AdjustmentStep {
  rule_name: string;
  rule_id: string;
  description: string;
  amount: number;
  details: string;
  data_source: string;
}

/** 가격 산출 응답 (POST /api/calculate 응답) */
export interface CalculateResponse {
  reference_price: number;
  adjustments: AdjustmentStep[];
  total_adjustment: number;
  estimated_retail: number;
  estimated_auction: number;
  confidence: string;
  summary: string;
}

/** 기준차량 검차 상태 */
export interface ReferenceInspection {
  frame_exchange: number;
  frame_bodywork: number;
  frame_corrosion: number;
  exterior_exchange: number;
  exterior_bodywork: number;
  exterior_corrosion: number;
}

/** 가격 산출 요청 */
export interface CalculateRequest {
  target_vehicle: TargetVehicle;
  reference_auction_id: string;
  reference_auction_price: number;
  reference_inspection?: ReferenceInspection;
}

/** 보정 기준 (LLM 분석 결과 or 사용자 수정) */
export interface PricingCriteria {
  mileage_rate_per_10k: number;   // %/만km
  mileage_ceiling_km: number;     // 천장 km
  year_rate_per_year: number;     // %/년
}

/** analyze-criteria 응답 */
export interface AnalyzeCriteriaResponse {
  criteria: PricingCriteria;
  analysis_summary: string;
  vehicles_analyzed: number;
  confidence: string;
}

/** criteria 포함 가격 산출 요청 */
export interface CalculateWithCriteriaRequest extends CalculateRequest {
  criteria?: PricingCriteria;
}

/** 피드백 요청 — 전체 맥락 저장 (학습 데이터용) */
export interface FeedbackRequest {
  // 대상차량
  target_vehicle: TargetVehicle;

  // 추천 맥락
  selected_reference_id: string | null;
  recommended_references: string[];              // ID 목록
  recommendations_detail: AuctionVehicle[];      // 낙찰가 기준차 상세

  // 가격 산출 맥락 (산출했을 경우)
  calculation_result: CalculateResponse | null;

  // 피드백
  estimated_price: number | null;
  actual_price: number | null;
  feedback_type: string;  // wrong_recommendation / price_high / price_low / price_ok
  comment: string | null;
}

// === AI 가격 예측 ===

export interface PriceFactor {
  factor: string;
  impact: number;
  description: string;
}

export interface PriceStats {
  count: number;
  mean: number;
  median: number;
  min: number;
  max: number;
}

export interface PricePredictionResponse {
  estimated_auction: number;
  estimated_retail: number;
  confidence: string;
  reasoning: string;
  factors: PriceFactor[];
  comparable_summary: string;
  key_comparables: string[];
  vehicles_analyzed: number;
  auction_stats: PriceStats;
  retail_stats: PriceStats;
}

// === 멀티 모델 예측 (모델 개발용) ===

export interface ModelResult {
  model_id: string;
  model_name: string;
  elapsed_ms: number;
  error?: string | null;
  estimated_auction: number;
  estimated_auction_export: number;
  estimated_retail: number;
  confidence: string;
  reasoning: string;
  auction_reasoning: string;
  retail_reasoning: string;
  export_reasoning: string;
  factors: PriceFactor[];
  auction_factors: PriceFactor[];
  retail_factors: PriceFactor[];
  comparable_summary: string;
  key_comparables: string[];
  vehicles_analyzed: number;
  auction_stats: PriceStats;
  retail_stats: PriceStats;
}

export interface MultiModelResponse {
  results: ModelResult[];
}

// === Taxonomy 타입 ===

export interface ModelInfo {
  model: string;
  segment: string;
}

export interface GenerationInfo {
  generation: string;
  display_name: string;
  variants: string[];
}

export interface VariantInfo {
  variant_key: string;
  fuel: string;
  displacement: string;
  drive: string;
  label: string;
  trim_count: number;
}
