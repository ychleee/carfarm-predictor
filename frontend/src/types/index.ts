// === 백엔드 Pydantic 모델 미러 ===

/** 대상차량 (POST /api/recommend 입력) */
export interface TargetVehicle {
  maker: string;
  model: string;
  generation?: string | null;
  year: number;
  mileage: number;
  fuel?: string | null;
  displacement?: string | null;
  drive?: string | null;
  trim?: string | null;
  color?: string | null;
  usage?: string | null;
  domestic?: boolean;
  options?: string[];
  exchange_count?: number;
  bodywork_count?: number;
}

/** 엔카 소매가 차량 */
export interface RetailVehicle {
  auction_id: string;
  vehicle_name: string | null;
  year: number | null;
  mileage: number | null;
  retail_price: number | null;   // 소매가 (만원)
  color: string | null;
  trim: string | null;
  source_url: string | null;
  factory_price: number | null;  // 출고가 (만원)
  options: string | null;        // 옵션
  fuel: string | null;           // 연료
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
}

/** 낙찰가 차량 */
export interface AuctionVehicle {
  auction_id: string;
  vehicle_name: string | null;
  year: number | null;
  mileage: number | null;
  auction_price: number | null;   // 낙찰가 (만원)
  auction_date: string | null;
  color: string | null;
  trim: string | null;
  options: string | null;
  factory_price: number | null;   // 출고가 (만원)
  inspection_grade: string | null; // 검차등급
  is_export: boolean;              // 수출여부
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
}

/** 추천 응답 (POST /api/recommend 응답) — 소매가/낙찰가 분리 */
export interface RecommendResponse {
  target: TargetVehicle;
  retail_vehicles: RetailVehicle[];    // 엔카 소매가 (최대 15)
  auction_vehicles: AuctionVehicle[];  // 낙찰가 (최대 15)
  reasoning?: string | null;           // LLM 전체 선별 근거
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

/** 가격 산출 요청 */
export interface CalculateRequest {
  target_vehicle: TargetVehicle;
  reference_auction_id: string;
  reference_auction_price: number;
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
