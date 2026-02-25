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

/** 추천된 기준차량 */
export interface ReferenceVehicle {
  auction_id: string;
  vehicle_name: string | null;
  year: number | null;
  mileage: number | null;
  auction_price: number | null;
  auction_date: string | null;
  color: string | null;
  options: string | null;
  usage_type: string | null;
  is_export: boolean;          // 내수(false) / 수출(true)
  trim: string | null;         // 트림명
  similarity_reason: string;
}

/** 추천 응답 (POST /api/recommend 응답) */
export interface RecommendResponse {
  target: TargetVehicle;
  recommendations: ReferenceVehicle[];
  reasoning: string;
  tool_calls_count: number;
  tokens_used: { input: number; output: number };
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

/** 피드백 요청 — 전체 맥락 저장 (few-shot 학습 데이터용) */
export interface FeedbackRequest {
  // 대상차량
  target_vehicle: TargetVehicle;

  // 추천 맥락
  selected_reference_id: string | null;
  recommended_references: string[];              // ID 목록 (하위호환)
  recommendations_detail: ReferenceVehicle[];    // 기준차 전체 상세
  llm_reasoning: string | null;                  // LLM 추론 전문
  tokens_used: { input: number; output: number } | null;
  tool_calls_count: number | null;

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
