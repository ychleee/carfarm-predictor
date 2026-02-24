import type {
  TargetVehicle,
  RecommendResponse,
  CalculateRequest,
  CalculateResponse,
  FeedbackRequest,
  ModelInfo,
  GenerationInfo,
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

// === 추천 ===

export async function recommendReferences(
  target: TargetVehicle,
  excludeIds?: string[]
): Promise<RecommendResponse> {
  const body = excludeIds?.length
    ? { ...target, exclude_auction_ids: excludeIds }
    : target;
  return request<RecommendResponse>("/recommend", {
    method: "POST",
    body: JSON.stringify(body),
  });
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

export async function getTrims(
  maker: string,
  model: string,
  generation: string
): Promise<string[]> {
  const data = await request<{ trims: string[] }>(
    `/trims/${encodeURIComponent(maker)}/${encodeURIComponent(model)}/${encodeURIComponent(generation)}`
  );
  return data.trims;
}
