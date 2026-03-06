import type { AuctionVehicle, CompanyTab, CalculateResponse } from "../types";
import DamageInfo from "./DamageInfo";

export interface CalcState {
  status: "idle" | "loading" | "done" | "error";
  data?: CalculateResponse;
  error?: string;
}

interface Props {
  vehicle: AuctionVehicle;
  index: number;
  calcState: CalcState;
  companyTab?: CompanyTab;
  onCalc: () => void;
  onClick: () => void;
  onDelete: () => void;
}

export default function AuctionCard({
  vehicle,
  index,
  calcState,
  companyTab,
  onCalc,
  onClick,
  onDelete,
}: Props) {
  const v = vehicle;

  const optionList = v.options?.split(",").map((o) => o.trim()).filter(Boolean) ?? [];

  const hasPriceResult = calcState.status === "done" && calcState.data;

  // 회사 태그 색상
  const badgeBg = companyTab?.bgClass ?? "bg-blue-100";
  const badgeText = companyTab?.textClass ?? "text-blue-700";

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-md transition-all">
      {/* 헤더: 번호 + 회사 태그 + 차명 + 삭제 */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-start gap-2 min-w-0">
          <span className={`${badgeBg} ${badgeText} text-xs font-bold px-2 py-0.5 rounded shrink-0`}>
            #{index + 1}
          </span>
          {companyTab && (
            <span className={`${badgeBg} ${badgeText} text-[10px] font-semibold px-1.5 py-0.5 rounded shrink-0`}>
              {companyTab.label}
            </span>
          )}
          <h4 className="text-sm font-semibold text-gray-900">
            {v.vehicle_name ?? `차량 ${v.auction_id}`}
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
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500 mb-1">
        <span>{v.year}년</span>
        <span>{v.mileage?.toLocaleString()}km</span>
        {v.color && <span>{v.color}</span>}
        {v.trim && <span className="font-medium text-gray-700">{v.trim}</span>}
        {v.fuel && <span>{v.fuel}</span>}
        {v.inspection_grade && (
          <span className="font-medium text-gray-600">{v.inspection_grade}</span>
        )}
        {v.is_export && (
          <span className="bg-blue-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">수출</span>
        )}
      </div>

      {/* 검차 상태 (프레임/외판) */}
      <DamageInfo
        frame={{ exchange: v.frame_exchange, bodywork: v.frame_bodywork, corrosion: v.frame_corrosion }}
        exterior={{ exchange: v.exterior_exchange, bodywork: v.exterior_bodywork, corrosion: v.exterior_corrosion }}
      />

      {/* 출고가 · 판매일 */}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-400 mb-2">
        {v.factory_price != null && v.factory_price > 0 && (
          <span>출고가 {v.factory_price.toLocaleString()}만원</span>
        )}
        {v.auction_date && <span>판매일 {v.auction_date}</span>}
      </div>

      {/* 낙찰가 */}
      <div className="flex items-center gap-3 mb-2">
        <span className="font-medium text-red-600 text-sm">
          낙찰가 {v.auction_price?.toLocaleString()}만원
        </span>
      </div>

      {/* 옵션 태그 */}
      {optionList.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-2">
          {optionList.map((opt, i) => (
            <span
              key={i}
              className="text-[10px] text-gray-600 bg-gray-100 border border-gray-200 px-1.5 py-0.5 rounded"
            >
              {opt}
            </span>
          ))}
        </div>
      )}

      {/* LLM 추천 이유 */}
      {v.reason && (
        <p className="text-[11px] text-gray-400 mb-2 leading-relaxed">
          {v.reason}
        </p>
      )}

      {/* 가격 산출 영역 */}
      <div className="flex items-center gap-3 mt-3">
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
