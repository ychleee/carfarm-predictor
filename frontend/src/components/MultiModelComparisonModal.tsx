import { useState } from "react";
import type { TargetVehicle, MultiModelResponse, ModelResult } from "../types";

interface Props {
  target: TargetVehicle;
  data: MultiModelResponse;
  onClose: () => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-800",
  보통: "bg-yellow-100 text-yellow-800",
  낮음: "bg-red-100 text-red-800",
};

const MODEL_COLORS: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
  i1: { bg: "bg-violet-50", border: "border-violet-200", text: "text-violet-700", gradient: "from-violet-500 to-purple-600" },
  i2: { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-700", gradient: "from-amber-500 to-orange-500" },
  i3: { bg: "bg-teal-50", border: "border-teal-200", text: "text-teal-700", gradient: "from-teal-500 to-cyan-500" },
};

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function PriceComparisonRow({
  label,
  values,
  unit = "만원",
}: {
  label: string;
  values: { modelId: string; value: number }[];
  unit?: string;
}) {
  const validValues = values.filter((v) => v.value > 0);
  const max = Math.max(...validValues.map((v) => v.value), 0);
  const min = Math.min(...validValues.map((v) => v.value), Infinity);
  const diff = validValues.length >= 2 ? max - min : 0;

  return (
    <div className="flex items-center gap-3 py-2">
      <span className="text-xs text-gray-500 w-20 shrink-0">{label}</span>
      <div className="flex-1 flex items-center gap-3">
        {values.map((v) => {
          const colors = MODEL_COLORS[v.modelId] ?? MODEL_COLORS.i1;
          return (
            <div key={v.modelId} className="flex-1 text-center">
              <span className={`text-lg font-bold ${colors.text}`}>
                {v.value > 0 ? `${v.value.toLocaleString()}` : "-"}
              </span>
              <span className="text-xs text-gray-400 ml-1">{unit}</span>
            </div>
          );
        })}
      </div>
      {diff > 0 && (
        <span className="text-xs text-gray-400 w-16 text-right shrink-0">
          차이 {diff.toLocaleString()}만
        </span>
      )}
    </div>
  );
}

function ModelDetailPanel({ result }: { result: ModelResult }) {
  const [expanded, setExpanded] = useState(false);
  const colors = MODEL_COLORS[result.model_id] ?? MODEL_COLORS.i1;
  const confClass = CONFIDENCE_COLORS[result.confidence] ?? "bg-gray-100 text-gray-800";

  if (result.error) {
    return (
      <div className={`${colors.bg} ${colors.border} border rounded-xl p-4`}>
        <div className="flex items-center gap-2 mb-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-gradient-to-r ${colors.gradient} text-white`}>
            {result.model_id.toUpperCase()}
          </span>
          <span className="text-xs text-gray-500">{result.model_name}</span>
        </div>
        <div className="text-sm text-red-600 bg-red-50 rounded-lg p-3">
          오류: {result.error}
        </div>
      </div>
    );
  }

  return (
    <div className={`${colors.bg} ${colors.border} border rounded-xl p-4`}>
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-gradient-to-r ${colors.gradient} text-white`}>
            {result.model_id.toUpperCase()}
          </span>
          <span className="text-xs text-gray-500">{result.model_name}</span>
          <span className="text-[10px] text-gray-400">{formatElapsed(result.elapsed_ms)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${confClass}`}>
            {result.confidence}
          </span>
          <span className="text-[10px] text-gray-400">{result.vehicles_analyzed}건</span>
        </div>
      </div>

      {/* 가격 */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="text-center bg-white/60 rounded-lg p-2">
          <p className="text-[10px] text-blue-600">내수 낙찰</p>
          <p className="text-sm font-bold text-blue-800">
            {result.estimated_auction > 0 ? `${result.estimated_auction.toLocaleString()}만` : "-"}
          </p>
        </div>
        <div className="text-center bg-white/60 rounded-lg p-2">
          <p className="text-[10px] text-emerald-600">소매가</p>
          <p className="text-sm font-bold text-emerald-800">
            {result.estimated_retail > 0 ? `${result.estimated_retail.toLocaleString()}만` : "-"}
          </p>
        </div>
        <div className="text-center bg-white/60 rounded-lg p-2">
          <p className="text-[10px] text-orange-600">수출 낙찰</p>
          <p className="text-sm font-bold text-orange-800">
            {result.estimated_auction_export > 0 ? `${result.estimated_auction_export.toLocaleString()}만` : "-"}
          </p>
        </div>
      </div>

      {/* 요인 분석 */}
      {(result.auction_factors.length > 0 || result.retail_factors.length > 0) && (
        <div className="mb-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
          >
            <span>{expanded ? "▼" : "▶"}</span>
            요인 분석 / 근거
          </button>
          {expanded && (
            <div className="mt-2 space-y-3">
              {result.auction_factors.length > 0 && (
                <div>
                  <p className="text-[10px] font-medium text-gray-500 mb-1">낙찰가 요인</p>
                  <div className="space-y-1">
                    {result.auction_factors.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs bg-white/50 rounded px-2 py-1">
                        <span className="font-medium text-gray-700 w-16">{f.factor}</span>
                        <span className={`font-semibold w-14 text-right ${f.impact > 0 ? "text-green-600" : f.impact < 0 ? "text-red-600" : "text-gray-500"}`}>
                          {f.impact > 0 ? "+" : ""}{f.impact}만
                        </span>
                        <span className="text-gray-400 flex-1 truncate">{f.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {result.retail_factors.length > 0 && (
                <div>
                  <p className="text-[10px] font-medium text-gray-500 mb-1">소매가 요인</p>
                  <div className="space-y-1">
                    {result.retail_factors.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs bg-white/50 rounded px-2 py-1">
                        <span className="font-medium text-gray-700 w-16">{f.factor}</span>
                        <span className={`font-semibold w-14 text-right ${f.impact > 0 ? "text-green-600" : f.impact < 0 ? "text-red-600" : "text-gray-500"}`}>
                          {f.impact > 0 ? "+" : ""}{f.impact}만
                        </span>
                        <span className="text-gray-400 flex-1 truncate">{f.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {result.auction_reasoning && (
                <div>
                  <p className="text-[10px] font-medium text-gray-500 mb-1">낙찰가 분석 근거</p>
                  <div className="text-xs text-gray-600 bg-white/50 rounded-lg px-3 py-2 whitespace-pre-line leading-relaxed">
                    {result.auction_reasoning}
                  </div>
                </div>
              )}
              {result.retail_reasoning && (
                <div>
                  <p className="text-[10px] font-medium text-gray-500 mb-1">소매가 분석 근거</p>
                  <div className="text-xs text-gray-600 bg-white/50 rounded-lg px-3 py-2 whitespace-pre-line leading-relaxed">
                    {result.retail_reasoning}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function MultiModelComparisonModal({ target, data, onClose }: Props) {
  const results = data.results;
  const validResults = results.filter((r) => !r.error);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      <div className="relative bg-white rounded-2xl shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 rounded-t-2xl flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-rose-500 to-orange-500 text-white text-xs font-bold px-3 py-1 rounded-full">
              모델 비교
            </div>
            <h2 className="text-lg font-bold text-gray-900">AI 가격 예측 모델 비교</h2>
            <span className="text-xs text-gray-400">{results.length}개 모델</span>
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

          {/* 가격 비교 요약 */}
          {validResults.length >= 2 && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-800 mb-3">가격 비교</h3>

              {/* 모델 라벨 헤더 */}
              <div className="flex items-center gap-3 mb-2 pb-2 border-b border-gray-100">
                <span className="w-20 shrink-0" />
                <div className="flex-1 flex items-center gap-3">
                  {results.map((r) => {
                    const colors = MODEL_COLORS[r.model_id] ?? MODEL_COLORS.i1;
                    return (
                      <div key={r.model_id} className="flex-1 text-center">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full bg-gradient-to-r ${colors.gradient} text-white`}>
                          {r.model_id.toUpperCase()}
                        </span>
                        <span className="text-[10px] text-gray-400 ml-1">{r.model_name}</span>
                      </div>
                    );
                  })}
                </div>
                <span className="w-16 shrink-0" />
              </div>

              <PriceComparisonRow
                label="내수 낙찰가"
                values={results.map((r) => ({ modelId: r.model_id, value: r.estimated_auction }))}
              />
              <PriceComparisonRow
                label="소매가"
                values={results.map((r) => ({ modelId: r.model_id, value: r.estimated_retail }))}
              />
              <PriceComparisonRow
                label="수출 낙찰가"
                values={results.map((r) => ({ modelId: r.model_id, value: r.estimated_auction_export }))}
              />
            </div>
          )}

          {/* 개별 모델 상세 */}
          <div className="space-y-4">
            {results.map((r) => (
              <ModelDetailPanel key={r.model_id} result={r} />
            ))}
          </div>

          {/* 시세 통계 (첫 번째 유효 모델에서 가져옴) */}
          {validResults.length > 0 && (validResults[0].auction_stats.count > 0 || validResults[0].retail_stats.count > 0) && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">시세 통계 (공통 데이터)</h3>
              <div className="grid grid-cols-2 gap-3 text-xs">
                {validResults[0].auction_stats.count > 0 && (
                  <div className="bg-blue-50 rounded-lg px-3 py-2">
                    <p className="font-medium text-blue-700 mb-1">낙찰가 ({validResults[0].auction_stats.count}건)</p>
                    <p>평균: {validResults[0].auction_stats.mean.toLocaleString()}만</p>
                    <p>중앙값: {validResults[0].auction_stats.median.toLocaleString()}만</p>
                    <p className="text-gray-500">
                      {validResults[0].auction_stats.min.toLocaleString()} ~ {validResults[0].auction_stats.max.toLocaleString()}만
                    </p>
                  </div>
                )}
                {validResults[0].retail_stats.count > 0 && (
                  <div className="bg-emerald-50 rounded-lg px-3 py-2">
                    <p className="font-medium text-emerald-700 mb-1">소매가 ({validResults[0].retail_stats.count}건)</p>
                    <p>평균: {validResults[0].retail_stats.mean.toLocaleString()}만</p>
                    <p>중앙값: {validResults[0].retail_stats.median.toLocaleString()}만</p>
                    <p className="text-gray-500">
                      {validResults[0].retail_stats.min.toLocaleString()} ~ {validResults[0].retail_stats.max.toLocaleString()}만
                    </p>
                  </div>
                )}
              </div>
            </div>
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
