import type { RetailVehicle } from "../types";
import DamageInfo from "./DamageInfo";

interface Props {
  vehicle: RetailVehicle;
  index: number;
  onDelete: () => void;
}

export default function RetailCard({ vehicle, index, onDelete }: Props) {
  const v = vehicle;

  // 신차대비 %
  const ratioPercent =
    v.factory_price && v.factory_price > 0 && v.retail_price && v.retail_price > 0
      ? Math.round((v.retail_price / v.factory_price) * 100)
      : null;

  return (
    <div
      className="bg-white border border-gray-200 rounded-xl p-5 hover:border-green-300 hover:shadow-md transition-all cursor-pointer"
      onClick={() => v.source_url && window.open(v.source_url, "_blank")}
    >
      {/* 헤더: 번호 + 차명 + 엔카진단/삭제 */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-start gap-2 min-w-0">
          <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-0.5 rounded shrink-0">
            #{index + 1}
          </span>
          <h4 className="text-sm font-semibold text-gray-900">
            {v.vehicle_name ?? `차량 ${v.auction_id}`}
          </h4>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {v.has_diagnosis && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-500 text-white">
              엔카진단
            </span>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            className="text-gray-300 hover:text-red-500 text-lg leading-none transition-colors"
            title="삭제"
          >
            &times;
          </button>
        </div>
      </div>

      {/* 스펙 한 줄 */}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500 mb-2">
        <span>{v.year}년</span>
        <span>{v.mileage?.toLocaleString()}km</span>
        {v.color && <span>{v.color}</span>}
        {v.trim && <span className="font-medium text-gray-700">{v.trim}</span>}
        {v.fuel && <span>{v.fuel}</span>}
        {v.displacement != null && v.displacement > 0 && (
          <span>{v.displacement.toLocaleString()}cc</span>
        )}
        {v.region && <span>{v.region}</span>}
      </div>

      {/* 사고이력 요약 */}
      {v.accident_summary && (
        <div className="flex items-center gap-1.5 text-xs mb-2">
          <span className={`font-medium ${v.accident_summary === "무사고" ? "text-emerald-600" : "text-red-600"}`}>
            {v.accident_summary === "무사고" ? "무사고" : "보험이력"}
          </span>
          {v.accident_summary !== "무사고" && (
            <span className="text-gray-500">{v.accident_summary}</span>
          )}
        </div>
      )}

      {/* 상태 배지: 압류/저당 · 등록일 · 조회수 */}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {(v.seizing_count > 0 || v.pledge_count > 0) && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-50 text-orange-600 border border-orange-200">
            압류{v.seizing_count} 저당{v.pledge_count}
          </span>
        )}
        {v.listing_date && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-50 text-gray-500 border border-gray-200">
            등록 {v.listing_date}
          </span>
        )}
        {v.view_count > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-50 text-gray-500 border border-gray-200">
            조회 {v.view_count.toLocaleString()}
          </span>
        )}
      </div>

      {/* 검차 상태 (프레임/외판) */}
      <DamageInfo
        frame={{ exchange: v.frame_exchange, bodywork: v.frame_bodywork, corrosion: v.frame_corrosion }}
        exterior={{ exchange: v.exterior_exchange, bodywork: v.exterior_bodywork, corrosion: v.exterior_corrosion }}
      />

      {/* 소매가 + 출고가 + 신차대비% */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="font-medium text-green-700 text-sm">
          소매가 {v.retail_price?.toLocaleString()}만원
        </span>
        {v.factory_price != null && v.factory_price > 0 && (
          <span className="text-xs text-gray-400">
            출고가 {v.factory_price.toLocaleString()}만원
          </span>
        )}
        {ratioPercent != null && (
          <span className="text-xs text-gray-500 font-medium">
            신차대비 {ratioPercent}%
          </span>
        )}
      </div>

      {/* 옵션 */}
      {v.options && (
        <div className="flex flex-wrap gap-1 mt-2">
          {v.options.split(",").map((opt, i) => (
            <span
              key={i}
              className="text-[10px] text-gray-600 bg-gray-100 border border-gray-200 px-1.5 py-0.5 rounded"
            >
              {opt.trim()}
            </span>
          ))}
        </div>
      )}

      {/* LLM 추천 이유 */}
      {v.reason && (
        <p className="text-[11px] text-gray-400 mt-2 leading-relaxed">
          {v.reason}
        </p>
      )}
    </div>
  );
}
