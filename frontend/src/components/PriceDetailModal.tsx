import type {
  TargetVehicle,
  ReferenceVehicle,
  CalculateResponse,
} from "../types";
import AdjustmentTable from "./AdjustmentTable";

interface Props {
  target: TargetVehicle;
  reference: ReferenceVehicle;
  data: CalculateResponse;
  onClose: () => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-800",
  보통: "bg-yellow-100 text-yellow-800",
  낮음: "bg-red-100 text-red-800",
};

export default function PriceDetailModal({
  target,
  reference,
  data,
  onClose,
}: Props) {
  const confClass =
    CONFIDENCE_COLORS[data.confidence] ?? "bg-gray-100 text-gray-800";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              가격 산출 상세
            </h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-xl leading-none"
            >
              &times;
            </button>
          </div>

          {/* 대상차량 & 기준차량 요약 */}
          <div className="grid grid-cols-2 gap-3 mb-5">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-gray-400 uppercase mb-1">
                대상차량
              </p>
              <p className="text-sm font-medium text-gray-900">
                {target.maker} {target.model} {target.trim ?? ""}
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {target.year}년 | {target.mileage.toLocaleString()}km
              </p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-blue-400 uppercase mb-1">
                기준차량
              </p>
              <p className="text-sm font-medium text-blue-900">
                {reference.vehicle_name ?? `ID: ${reference.auction_id}`}
              </p>
              <p className="text-xs text-blue-600 mt-0.5">
                {reference.year}년 | {reference.mileage?.toLocaleString()}km |
                낙찰 {reference.auction_price?.toLocaleString()}만원
              </p>
            </div>
          </div>

          {/* 보정 내역 */}
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            보정 내역 (룰 엔진)
          </h4>
          <div className="mb-5">
            <AdjustmentTable
              adjustments={data.adjustments}
              referencePrice={data.reference_price}
              totalAdjustment={data.total_adjustment}
            />
          </div>

          {/* 최종 결과 */}
          <div className="bg-white border-2 border-blue-200 rounded-xl p-5 mb-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-base font-bold text-gray-900">산출 결과</h4>
              <span
                className={`text-xs font-medium px-2.5 py-1 rounded-full ${confClass}`}
              >
                신뢰도: {data.confidence}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-400 mb-1">추정 소매가</p>
                <p className="text-xl font-bold text-gray-800">
                  {data.estimated_retail.toLocaleString()}
                  <span className="text-sm font-normal text-gray-500">
                    만원
                  </span>
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">예상 낙찰가</p>
                <p className="text-xl font-bold text-blue-700">
                  {data.estimated_auction.toLocaleString()}
                  <span className="text-sm font-normal text-blue-400">
                    만원
                  </span>
                </p>
              </div>
            </div>

            <p className="mt-3 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
              {data.summary}
            </p>
          </div>

          {/* 닫기 */}
          <button
            onClick={onClose}
            className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2.5 rounded-lg transition-colors text-sm"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
