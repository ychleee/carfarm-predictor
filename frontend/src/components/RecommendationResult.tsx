import { useState } from "react";
import type {
  TargetVehicle,
  RecommendResponse,
  RetailVehicle,
  AuctionVehicle,
  CalculateResponse,
} from "../types";
import { calculatePrice, recommendReferences, submitFeedback } from "../api/client";
import RetailCard from "./RetailCard";
import AuctionCard from "./AuctionCard";
import type { CalcState } from "./AuctionCard";
import PriceDetailModal from "./PriceDetailModal";
import DeleteModal from "./DeleteModal";

interface Props {
  target: TargetVehicle;
  data: RecommendResponse;
  onBack: () => void;
}

export default function RecommendationResult({
  target,
  data,
  onBack,
}: Props) {
  const [retailCards, setRetailCards] = useState<RetailVehicle[]>(data.retail_vehicles);
  const [auctionCards, setAuctionCards] = useState<AuctionVehicle[]>(data.auction_vehicles);
  const [calcStates, setCalcStates] = useState<Record<string, CalcState>>({});
  const [deleteTarget, setDeleteTarget] = useState<AuctionVehicle | null>(null);
  const [detailTarget, setDetailTarget] = useState<{
    ref: AuctionVehicle;
    calc: CalculateResponse;
  } | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState(false);

  const handleCalc = async (ref: AuctionVehicle) => {
    if (!ref.auction_price) return;
    setCalcStates((prev) => ({
      ...prev,
      [ref.auction_id]: { status: "loading" },
    }));
    try {
      const calc = await calculatePrice({
        target_vehicle: target,
        reference_auction_id: ref.auction_id,
        reference_auction_price: ref.auction_price,
      });
      setCalcStates((prev) => ({
        ...prev,
        [ref.auction_id]: { status: "done", data: calc },
      }));
    } catch (err) {
      setCalcStates((prev) => ({
        ...prev,
        [ref.auction_id]: {
          status: "error",
          error: err instanceof Error ? err.message : "산출 실패",
        },
      }));
    }
  };

  const handleCardClick = (ref: AuctionVehicle) => {
    const state = calcStates[ref.auction_id];
    if (state?.status === "done" && state.data) {
      setDetailTarget({ ref, calc: state.data });
    }
  };

  const handleDeleteConfirm = () => {
    if (!deleteTarget) return;
    setAuctionCards((prev) => prev.filter((c) => c.auction_id !== deleteTarget.auction_id));
    setDeleteTarget(null);
  };

  const handleLoadMore = async () => {
    setLoadingMore(true);
    setError(null);
    try {
      const currentIds = [
        ...retailCards.map((c) => c.auction_id),
        ...auctionCards.map((c) => c.auction_id),
      ];
      const moreData = await recommendReferences(target, currentIds);
      if (moreData.retail_vehicles.length === 0 && moreData.auction_vehicles.length === 0) {
        setError("추가 추천할 차량이 없습니다.");
      } else {
        setRetailCards((prev) => [...prev, ...moreData.retail_vehicles]);
        setAuctionCards((prev) => [...prev, ...moreData.auction_vehicles]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "추가 추천 실패");
    } finally {
      setLoadingMore(false);
    }
  };

  return (
    <div>
      {/* 대상차량 요약 */}
      <div className="bg-gray-100 rounded-lg px-4 py-2.5 mb-4 text-sm text-gray-700 flex flex-wrap gap-x-4 gap-y-1">
        <span className="font-medium">대상:</span>
        <span>
          {target.maker} {target.model}
        </span>
        {target.trim && <span>| {target.trim}</span>}
        <span>| {target.year}년</span>
        <span>| {target.mileage.toLocaleString()}km</span>
        {target.color && <span>| {target.color}</span>}
      </div>

      {/* LLM 선별 근거 */}
      {data.reasoning && (
        <div className="mb-4 border border-blue-200 rounded-lg overflow-hidden">
          <button
            onClick={() => setReasoningOpen((o) => !o)}
            className="w-full flex items-center justify-between px-4 py-2.5 bg-blue-50 hover:bg-blue-100 transition-colors text-sm text-blue-800 font-medium"
          >
            <span>AI 선별 근거</span>
            <span className="text-blue-500 text-xs">{reasoningOpen ? "접기" : "펼치기"}</span>
          </button>
          {reasoningOpen && (
            <div className="px-4 py-3 text-sm text-gray-700 whitespace-pre-wrap bg-white">
              {data.reasoning}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* 엔카 소매가 섹션 */}
      {retailCards.length > 0 && (
        <>
          <h3 className="text-base font-semibold text-gray-900 mb-3">
            <span className="text-green-700">엔카 소매가</span> ({retailCards.length}건)
          </h3>
          <div className="flex flex-col gap-3 mb-6">
            {retailCards.map((v, i) => (
              <RetailCard
                key={v.auction_id}
                vehicle={v}
                index={i}
                onDelete={() => setRetailCards((prev) => prev.filter((c) => c.auction_id !== v.auction_id))}
              />
            ))}
          </div>
        </>
      )}

      {/* 낙찰가 섹션 */}
      <h3 className="text-base font-semibold text-gray-900 mb-3">
        <span className="text-blue-700">낙찰가</span> ({auctionCards.length}건)
      </h3>
      <div className="flex flex-col gap-3 mb-4">
        {auctionCards.map((ref, i) => (
          <AuctionCard
            key={ref.auction_id}
            vehicle={ref}
            index={i}
            calcState={calcStates[ref.auction_id] ?? { status: "idle" }}
            onCalc={() => handleCalc(ref)}
            onClick={() => handleCardClick(ref)}
            onDelete={() => setDeleteTarget(ref)}
          />
        ))}
      </div>

      {/* 추가 추천 + 뒤로 */}
      <div className="flex gap-3 items-center">
        <button
          onClick={handleLoadMore}
          disabled={loadingMore}
          className="flex-1 bg-blue-50 hover:bg-blue-100 disabled:bg-blue-50 border border-blue-200 text-blue-700 font-medium py-3 px-4 rounded-lg transition-colors text-sm"
        >
          {loadingMore ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              검색 중...
            </span>
          ) : (
            "+ 추가 추천 받기"
          )}
        </button>
        <button
          onClick={onBack}
          className="text-sm text-gray-500 hover:text-gray-700 underline px-4"
        >
          다른 차량으로 다시 검색
        </button>
      </div>

      {/* 삭제 모달 */}
      {deleteTarget && (
        <DeleteModal
          vehicle={deleteTarget}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {/* 가격 상세 모달 (피드백 포함) */}
      {detailTarget && (
        <PriceDetailModal
          target={target}
          reference={detailTarget.ref}
          data={detailTarget.calc}
          onClose={() => setDetailTarget(null)}
          onPriceFeedback={(type, comment) => {
            submitFeedback({
              target_vehicle: target,
              selected_reference_id: detailTarget.ref.auction_id,
              recommended_references: auctionCards.map((c) => c.auction_id),
              recommendations_detail: auctionCards,
              calculation_result: detailTarget.calc,
              estimated_price: detailTarget.calc.estimated_retail,
              actual_price: null,
              feedback_type: type,
              comment: comment || null,
            }).catch(() => {});
          }}
        />
      )}
    </div>
  );
}
