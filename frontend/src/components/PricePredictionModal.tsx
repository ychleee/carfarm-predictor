import type { TargetVehicle, PricePredictionResponse } from "../types";

interface Props {
  target: TargetVehicle;
  data: PricePredictionResponse;
  onClose: () => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-800",
  보통: "bg-yellow-100 text-yellow-800",
  낮음: "bg-red-100 text-red-800",
};

export default function PricePredictionModal({ target, data, onClose }: Props) {
  const confClass = CONFIDENCE_COLORS[data.confidence] ?? "bg-gray-100 text-gray-800";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 rounded-t-2xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-violet-500 to-purple-600 text-white text-xs font-bold px-3 py-1 rounded-full">
              AI 예측
            </div>
            <h2 className="text-lg font-bold text-gray-900">가격 예측 결과</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        <div className="px-6 py-5 space-y-5">
          {/* 대상차량 요약 */}
          <div className="bg-gray-50 rounded-lg px-4 py-3 text-sm text-gray-600">
            <span className="font-medium text-gray-800">대상: </span>
            {target.vehicleMaker} {target.vehicleModel}
            {target.generation && ` | ${target.generation}`}
            {target.vehicleTrim && ` | ${target.vehicleTrim}`}
            {` | ${target.vehicleYear}년`}
            {` | ${target.mileage.toLocaleString()}km`}
            {target.fuelType && ` | ${target.fuelType}`}
          </div>

          {/* 가격 카드 */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-4 text-center">
              <p className="text-xs text-blue-600 mb-1">예상 낙찰가</p>
              <p className="text-2xl font-bold text-blue-800">
                {data.estimated_auction > 0
                  ? `${data.estimated_auction.toLocaleString()}만원`
                  : "-"}
              </p>
            </div>
            <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border border-emerald-200 rounded-xl p-4 text-center">
              <p className="text-xs text-emerald-600 mb-1">추정 소매가</p>
              <p className="text-2xl font-bold text-emerald-800">
                {data.estimated_retail > 0
                  ? `${data.estimated_retail.toLocaleString()}만원`
                  : "-"}
              </p>
            </div>
          </div>

          {/* 신뢰도 + 분석 요약 */}
          <div className="flex items-center gap-3">
            <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${confClass}`}>
              신뢰도: {data.confidence}
            </span>
            <span className="text-xs text-gray-500">
              {data.vehicles_analyzed}건 유사차량 분석
            </span>
          </div>

          {/* 요인 분석 */}
          {data.factors.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">가격 요인 분석</h3>
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="text-left px-3 py-2 text-xs font-medium text-gray-500">요인</th>
                      <th className="text-right px-3 py-2 text-xs font-medium text-gray-500">영향</th>
                      <th className="text-left px-3 py-2 text-xs font-medium text-gray-500">설명</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {data.factors.map((f, i) => (
                      <tr key={i}>
                        <td className="px-3 py-2 font-medium text-gray-700">{f.factor}</td>
                        <td className={`px-3 py-2 text-right font-semibold ${
                          f.impact > 0 ? "text-green-600" : f.impact < 0 ? "text-red-600" : "text-gray-500"
                        }`}>
                          {f.impact > 0 ? "+" : ""}{f.impact.toLocaleString()}만
                        </td>
                        <td className="px-3 py-2 text-gray-500 text-xs">{f.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* LLM 추론 근거 */}
          {data.reasoning && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">분석 근거</h3>
              <div className="bg-violet-50 border border-violet-200 rounded-lg px-4 py-3 text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                {data.reasoning}
              </div>
            </div>
          )}

          {/* 시세 통계 */}
          {(data.auction_stats.count > 0 || data.retail_stats.count > 0) && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">시세 통계</h3>
              <div className="grid grid-cols-2 gap-3 text-xs">
                {data.auction_stats.count > 0 && (
                  <div className="bg-blue-50 rounded-lg px-3 py-2">
                    <p className="font-medium text-blue-700 mb-1">낙찰가 ({data.auction_stats.count}건)</p>
                    <p>평균: {data.auction_stats.mean.toLocaleString()}만</p>
                    <p>중앙값: {data.auction_stats.median.toLocaleString()}만</p>
                    <p className="text-gray-500">
                      {data.auction_stats.min.toLocaleString()} ~ {data.auction_stats.max.toLocaleString()}만
                    </p>
                  </div>
                )}
                {data.retail_stats.count > 0 && (
                  <div className="bg-emerald-50 rounded-lg px-3 py-2">
                    <p className="font-medium text-emerald-700 mb-1">소매가 ({data.retail_stats.count}건)</p>
                    <p>평균: {data.retail_stats.mean.toLocaleString()}만</p>
                    <p>중앙값: {data.retail_stats.median.toLocaleString()}만</p>
                    <p className="text-gray-500">
                      {data.retail_stats.min.toLocaleString()} ~ {data.retail_stats.max.toLocaleString()}만
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 유사차량 요약 */}
          {data.comparable_summary && (
            <p className="text-xs text-gray-400">{data.comparable_summary}</p>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-white border-t border-gray-200 px-6 py-3 rounded-b-2xl flex justify-end">
          <button
            onClick={onClose}
            className="text-sm text-gray-600 hover:text-gray-800 px-4 py-2"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
