import type {
  TargetVehicle,
  AuctionVehicle,
  CalculateResponse,
} from "../types";
import AdjustmentTable from "./AdjustmentTable";

interface Props {
  target: TargetVehicle;
  reference: AuctionVehicle;
  data: CalculateResponse;
  onBack: () => void;
  onReset: () => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-800",
  보통: "bg-yellow-100 text-yellow-800",
  낮음: "bg-red-100 text-red-800",
};

export default function PriceCalculation({
  target,
  reference,
  data,
  onBack,
  onReset,
}: Props) {
  const confClass = CONFIDENCE_COLORS[data.confidence] ?? "bg-gray-100 text-gray-800";

  return (
    <div>
      {/* 대상차량 & 기준차량 요약 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">대상차량</h4>
          <p className="text-sm font-medium text-gray-900">
            {target.maker} {target.model} {target.trim ?? ""}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {target.year}년 | {target.mileage.toLocaleString()}km
            {target.color ? ` | ${target.color}` : ""}
          </p>
        </div>
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-blue-400 uppercase mb-2">기준차량</h4>
          <p className="text-sm font-medium text-blue-900 line-clamp-1">
            {reference.vehicle_name ?? `ID: ${reference.auction_id}`}
          </p>
          <p className="text-xs text-blue-600 mt-1">
            {reference.year}년 | {reference.mileage?.toLocaleString()}km |
            낙찰 {reference.auction_price?.toLocaleString()}만원
          </p>
        </div>
      </div>

      {/* 보정 내역 테이블 — 리즈닝의 핵심 */}
      <h3 className="text-base font-semibold text-gray-900 mb-3">
        보정 내역 (룰 엔진)
      </h3>
      <div className="mb-6">
        <AdjustmentTable
          adjustments={data.adjustments}
          referencePrice={data.reference_price}
          totalAdjustment={data.total_adjustment}
        />
      </div>

      {/* 최종 결과 */}
      <div className="bg-white border-2 border-blue-200 rounded-xl p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-gray-900">산출 결과</h3>
          <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${confClass}`}>
            신뢰도: {data.confidence}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-xs text-gray-400 mb-1">추정 소매가</p>
            <p className="text-2xl font-bold text-gray-800">
              {data.estimated_retail.toLocaleString()}
              <span className="text-base font-normal text-gray-500">만원</span>
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">예상 낙찰가</p>
            <p className="text-2xl font-bold text-blue-700">
              {data.estimated_auction.toLocaleString()}
              <span className="text-base font-normal text-blue-400">만원</span>
            </p>
          </div>
        </div>

        <p className="mt-4 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
          {data.summary}
        </p>
      </div>

      {/* 액션 버튼 */}
      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="flex-1 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-3 px-4 rounded-lg transition-colors text-sm"
        >
          다른 기준차량 선택
        </button>
        <button
          onClick={onReset}
          className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-4 rounded-lg transition-colors text-sm"
        >
          새 차량 산출하기
        </button>
      </div>
    </div>
  );
}
