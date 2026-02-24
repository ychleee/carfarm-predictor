// === 백엔드 Pydantic 모델 미러 ===

/** 대상차량 (POST /api/recommend 입력) */
export interface TargetVehicle {
  maker: string;
  model: string;
  generation?: string | null;
  year: number;
  mileage: number;
  fuel?: string | null;
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

/** 피드백 요청 */
export interface FeedbackRequest {
  target_vehicle: TargetVehicle;
  selected_reference_id: string | null;
  recommended_references: string[];
  estimated_price: number | null;
  actual_price: number | null;
  feedback_type: string;
  comment: string | null;
}

// === Taxonomy 타입 ===

export interface ModelInfo {
  model: string;
  segment: string;
}

export interface GenerationInfo {
  generation: string;
  variants: string[];
}
