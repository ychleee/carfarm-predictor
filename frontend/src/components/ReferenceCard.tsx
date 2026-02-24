import type { ReferenceVehicle } from "../types";

interface Props {
  reference: ReferenceVehicle;
  index: number;
  loading: boolean;
  onSelect: () => void;
}

export default function ReferenceCard({
  reference,
  index,
  loading,
  onSelect,
}: Props) {
  const r = reference;

  // 옵션 요약 (처음 4개만)
  const optionList = r.options?.split(",").map((o) => o.trim()) ?? [];
  const displayOptions = optionList.slice(0, 4);
  const moreCount = optionList.length - displayOptions.length;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:border-blue-300 hover:shadow-md transition-all">
      {/* 헤더 */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-0.5 rounded">
            #{index + 1}
          </span>
          <h4 className="text-sm font-semibold text-gray-900 line-clamp-1">
            {r.vehicle_name ?? `차량 ${r.auction_id}`}
          </h4>
        </div>
        <span className="text-lg font-bold text-blue-600 whitespace-nowrap">
          {r.auction_price?.toLocaleString()}만원
        </span>
      </div>

      {/* 스펙 정보 */}
      <div className="grid grid-cols-3 gap-2 mb-3 text-xs text-gray-500">
        <div>
          <span className="text-gray-400">연식</span>
          <p className="text-gray-800 font-medium">{r.year}년</p>
        </div>
        <div>
          <span className="text-gray-400">주행거리</span>
          <p className="text-gray-800 font-medium">
            {r.mileage?.toLocaleString()}km
          </p>
        </div>
        <div>
          <span className="text-gray-400">색상</span>
          <p className="text-gray-800 font-medium">{r.color ?? "-"}</p>
        </div>
      </div>

      {/* 옵션 */}
      {displayOptions.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
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

      {/* 선택 이유 — 핵심! */}
      <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 mb-3">
        <p className="text-xs font-medium text-blue-700 mb-1">선택 이유</p>
        <p className="text-sm text-blue-900 leading-relaxed">
          {r.similarity_reason}
        </p>
      </div>

      {/* 선택 버튼 */}
      <button
        onClick={onSelect}
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-medium py-2 rounded-lg transition-colors"
      >
        {loading ? "산출 중..." : "이 차량으로 가격 산출"}
      </button>
    </div>
  );
}
