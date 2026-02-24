import { useState } from "react";
import type {
  TargetVehicle,
  RecommendResponse,
  ReferenceVehicle,
  CalculateResponse,
} from "../types";
import { calculatePrice } from "../api/client";
import ReasoningPanel from "./ReasoningPanel";
import ReferenceCard from "./ReferenceCard";

interface Props {
  target: TargetVehicle;
  data: RecommendResponse;
  onSelect: (ref: ReferenceVehicle, calc: CalculateResponse) => void;
  onBack: () => void;
}

export default function RecommendationResult({
  target,
  data,
  onSelect,
  onBack,
}: Props) {
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelect = async (ref: ReferenceVehicle) => {
    if (!ref.auction_price) return;
    setLoadingId(ref.auction_id);
    setError(null);
    try {
      const calc = await calculatePrice({
        target_vehicle: target,
        reference_auction_id: ref.auction_id,
        reference_auction_price: ref.auction_price,
      });
      onSelect(ref, calc);
    } catch (err) {
      setError(err instanceof Error ? err.message : "가격 산출 실패");
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <div>
      {/* 대상차량 요약 */}
      <div className="bg-gray-100 rounded-lg px-4 py-2.5 mb-4 text-sm text-gray-700 flex flex-wrap gap-x-4 gap-y-1">
        <span className="font-medium">대상:</span>
        <span>{target.maker} {target.model}</span>
        {target.trim && <span>| {target.trim}</span>}
        <span>| {target.year}년</span>
        <span>| {target.mileage.toLocaleString()}km</span>
        {target.color && <span>| {target.color}</span>}
      </div>

      {/* LLM 추론 과정 */}
      <ReasoningPanel
        reasoning={data.reasoning}
        toolCallsCount={data.tool_calls_count}
        tokensUsed={data.tokens_used}
      />

      {/* 기준차량 카드 */}
      <h3 className="text-base font-semibold text-gray-900 mb-3">
        추천 기준차량 ({data.recommendations.length}건)
      </h3>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {data.recommendations.map((ref, i) => (
          <ReferenceCard
            key={ref.auction_id}
            reference={ref}
            index={i}
            loading={loadingId === ref.auction_id}
            onSelect={() => handleSelect(ref)}
          />
        ))}
      </div>

      {/* 뒤로가기 */}
      <button
        onClick={onBack}
        className="text-sm text-gray-500 hover:text-gray-700 underline"
      >
        다른 차량으로 다시 검색
      </button>
    </div>
  );
}
