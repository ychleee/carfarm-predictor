import type {
  CalculateRequest,
  CalculateResponse,
  CalculateWithCriteriaRequest,
  FeedbackRequest,
  SearchAuctionResponse,
  PricePredictionResponse,
  AnalyzeCriteriaResponse,
  MultiModelResponse,
  ModelInfo,
  GenerationInfo,
  VariantInfo,
  TargetVehicle,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";
console.log("[CarFarm] API BASE_URL:", BASE_URL);

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (options?.body) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(`${BASE_URL}${path}`, {
    headers,
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `API 오류 (${res.status})`);
  }
  return res.json();
}

// === 경매 검색 ===

export async function searchAuction(params: {
  maker: string;
  model: string;
  company_id: string;
  generation?: string;
  limit?: number;
}): Promise<SearchAuctionResponse> {
  const query = new URLSearchParams();
  query.set("company_id", params.company_id);
  query.set("maker", params.maker);
  query.set("model", params.model);
  if (params.generation) query.set("generation", params.generation);
  if (params.limit) query.set("limit", String(params.limit));
  return request<SearchAuctionResponse>(`/search-auction?${query.toString()}`);
}

// === 가격 산출 ===

export async function calculatePrice(
  req: CalculateRequest
): Promise<CalculateResponse> {
  return request<CalculateResponse>("/calculate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// === 보정 기준 분석 ===

export async function analyzeCriteria(req: {
  target_vehicle: TargetVehicle;
  reference_auction_id: string;
  reference_auction_price: number;
}): Promise<AnalyzeCriteriaResponse> {
  return request<AnalyzeCriteriaResponse>("/analyze-criteria", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// === 보정 기준 포함 가격 산출 ===

export async function calculateWithCriteria(
  req: CalculateWithCriteriaRequest
): Promise<CalculateResponse> {
  return request<CalculateResponse>("/calculate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// === AI 가격 예측 ===

export async function predictPrice(
  target: TargetVehicle
): Promise<PricePredictionResponse> {
  return request<PricePredictionResponse>("/predict-price", {
    method: "POST",
    body: JSON.stringify(target),
  });
}

// === 멀티 모델 AI 가격 예측 (모델 개발용) ===

export async function predictPriceMulti(
  target: TargetVehicle
): Promise<MultiModelResponse> {
  return request<MultiModelResponse>("/predict-price-multi", {
    method: "POST",
    body: JSON.stringify(target),
  });
}

// === 피드백 ===

export async function submitFeedback(
  req: FeedbackRequest
): Promise<{ status: string }> {
  return request("/feedback", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// === Taxonomy ===

export async function getMakers(): Promise<string[]> {
  const data = await request<{ makers: string[] }>("/makers");
  return data.makers;
}

export async function getModels(maker: string): Promise<ModelInfo[]> {
  const data = await request<{ models: ModelInfo[] }>(
    `/models/${encodeURIComponent(maker)}`
  );
  return data.models;
}

export async function getGenerations(
  maker: string,
  model: string
): Promise<GenerationInfo[]> {
  const data = await request<{ generations: GenerationInfo[] }>(
    `/generations/${encodeURIComponent(maker)}/${encodeURIComponent(model)}`
  );
  return data.generations;
}

export async function getVariants(
  maker: string,
  model: string,
  generation: string
): Promise<VariantInfo[]> {
  const data = await request<{ variants: VariantInfo[] }>(
    `/variants/${encodeURIComponent(maker)}/${encodeURIComponent(model)}/${encodeURIComponent(generation)}`
  );
  return data.variants;
}

export async function getTrims(
  maker: string,
  model: string,
  generation: string,
  variantKey?: string
): Promise<string[]> {
  let url = `/trims/${encodeURIComponent(maker)}/${encodeURIComponent(model)}/${encodeURIComponent(generation)}`;
  if (variantKey) {
    url += `?variant_key=${encodeURIComponent(variantKey)}`;
  }
  const data = await request<{ trims: string[] }>(url);
  return data.trims;
}
