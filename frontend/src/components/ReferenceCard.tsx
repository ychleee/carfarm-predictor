import type { ReferenceVehicle, CalculateResponse } from "../types";

export interface CalcState {
  status: "idle" | "loading" | "done" | "error";
  data?: CalculateResponse;
  error?: string;
}

interface Props {
  reference: ReferenceVehicle;
  index: number;
  calcState: CalcState;
  onCalc: () => void;
  onClick: () => void;
  onDelete: () => void;
}

export default function ReferenceCard({
  reference,
  index,
  calcState,
  onCalc,
  onClick,
  onDelete,
}: Props) {
  const r = reference;

  const optionList = r.options?.split(",").map((o) => o.trim()).filter(Boolean) ?? [];
  const displayOptions = optionList.slice(0, 5);
  const moreCount = optionList.length - displayOptions.length;

  const hasPriceResult = calcState.status === "done" && calcState.data;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-md transition-all">
      {/* 헤더: 번호 + 차명 + 삭제 */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-start gap-2 min-w-0">
          <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-0.5 rounded shrink-0">
            #{index + 1}
          </span>
          <h4 className="text-sm font-semibold text-gray-900">
            {r.vehicle_name ?? `차량 ${r.auction_id}`}
          </h4>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="text-gray-300 hover:text-red-500 text-lg leading-none shrink-0 transition-colors"
          title="삭제"
        >
          &times;
        </button>
      </div>

      {/* 스펙 한 줄 */}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500 mb-2">
        <span>{r.year}년</span>
        <span>{r.mileage?.toLocaleString()}km</span>
        {r.color && <span>{r.color}</span>}
        {r.trim && <span className="font-medium text-gray-700">{r.trim}</span>}
        <span className={`font-medium px-1.5 py-0.5 rounded ${r.is_export ? "bg-orange-100 text-orange-700" : "bg-green-100 text-green-700"}`}>
          {r.is_export ? "수출" : "내수"}
        </span>
        <span className="font-medium text-blue-600">
          낙찰 {r.auction_price?.toLocaleString()}만원
        </span>
        {r.auction_date && <span>{r.auction_date}</span>}
      </div>

      {/* 옵션 태그 */}
      {displayOptions.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {displayOptions.map((opt) => (
            <span
              key={opt}
              className="bg-gray-100 text-gray-600 text-xs px-1.5 py-0.5 rounded"
            >
              {opt}
            </span>
          ))}
          {moreCount > 0 && (
            <span className="text-xs text-gray-400">+{moreCount}개</span>
          )}
        </div>
      )}

      {/* 선택 이유 */}
      <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 mb-3">
        <p className="text-xs font-medium text-blue-700 mb-1">선택 이유</p>
        <p className="text-sm text-blue-900 leading-relaxed">
          {r.similarity_reason}
        </p>
      </div>

      {/* 가격 산출 영역 */}
      <div className="flex items-center gap-3">
        {calcState.status === "idle" && (
          <button
            onClick={(e) => { e.stopPropagation(); onCalc(); }}
            className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-4 rounded-lg transition-colors"
          >
            가격 산출
          </button>
        )}

        {calcState.status === "loading" && (
          <span className="flex items-center gap-2 text-sm text-blue-600">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            산출 중...
          </span>
        )}

        {calcState.status === "error" && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-red-600">{calcState.error}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onCalc(); }}
              className="text-sm text-blue-600 hover:underline"
            >
              재시도
            </button>
          </div>
        )}

        {hasPriceResult && (
          <button
            onClick={onClick}
            className="flex items-center gap-3 bg-green-50 border border-green-200 hover:bg-green-100 rounded-lg px-4 py-2 transition-colors"
          >
            <div className="text-left">
              <span className="text-xs text-gray-500">추정 소매가</span>
              <p className="text-sm font-bold text-gray-800">
                {calcState.data!.estimated_retail.toLocaleString()}만원
              </p>
            </div>
            <div className="text-left">
              <span className="text-xs text-gray-500">예상 낙찰가</span>
              <p className="text-sm font-bold text-blue-700">
                {calcState.data!.estimated_auction.toLocaleString()}만원
              </p>
            </div>
            <span className="text-xs text-gray-400 ml-1">상세 &rsaquo;</span>
          </button>
        )}
      </div>
    </div>
  );
}
