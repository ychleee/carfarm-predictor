import { useState, useEffect } from "react";
import type {
  PricingCriteria,
  AnalyzeCriteriaResponse,
  TargetVehicle,
} from "../types";

interface Props {
  target: TargetVehicle;
  data: AnalyzeCriteriaResponse | null;
  loading: boolean;
  error: string | null;
  onCriteriaChange: (criteria: PricingCriteria) => void;
  onReanalyze: () => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-700",
  보통: "bg-yellow-100 text-yellow-700",
  낮음: "bg-red-100 text-red-700",
};

export default function CriteriaPanel({
  target,
  data,
  loading,
  error,
  onCriteriaChange,
  onReanalyze,
}: Props) {
  // Local criteria state for editing
  const [editedCriteria, setEditedCriteria] = useState<PricingCriteria | null>(
    null
  );

  // When data changes externally, reset edited state
  useEffect(() => {
    if (data?.criteria) {
      setEditedCriteria(null);
    }
  }, [data]);

  // Use edited values or original data
  const criteria = editedCriteria ?? data?.criteria;

  const handleFieldChange = (field: keyof PricingCriteria, value: number) => {
    const updated = { ...criteria!, [field]: value };
    setEditedCriteria(updated);
    onCriteriaChange(updated);
  };

  // Loading state
  if (loading) {
    return (
      <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-5 mb-4">
        <div className="flex items-center justify-center gap-3 py-4">
          <svg
            className="animate-spin h-5 w-5 text-indigo-600"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-sm font-medium text-indigo-700">
            보정 기준 분석 중...
          </span>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-5 mb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-red-700">{error}</span>
          <button
            onClick={onReanalyze}
            className="text-sm font-medium text-red-600 hover:text-red-800 border border-red-300 rounded-lg px-3 py-1.5 hover:bg-red-100 transition-colors"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  // No data yet
  if (!data || !criteria) {
    return null;
  }

  const confClass =
    CONFIDENCE_COLORS[data.confidence] ?? "bg-gray-100 text-gray-700";

  // Compute age band label based on target year
  const currentYear = new Date().getFullYear();
  const age = currentYear - target.vehicleYear;
  const ageBandLabel =
    age <= 3
      ? "1~3년 기준"
      : age <= 6
        ? "4~6년 기준"
        : age <= 10
          ? "7~10년 기준"
          : "10년+ 기준";

  return (
    <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-5 mb-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-indigo-900">
          보정 기준 (LLM 분석 결과)
        </h3>
        <button
          onClick={onReanalyze}
          className="text-xs font-medium text-indigo-600 hover:text-indigo-800 border border-indigo-300 rounded-lg px-3 py-1.5 hover:bg-indigo-100 transition-colors"
        >
          다시 분석
        </button>
      </div>

      {/* Editable fields - 2 column grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* 주행거리 보정율 */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            주행거리 보정율
          </label>
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              step="0.1"
              value={criteria.mileage_rate_per_10k}
              onChange={(e) =>
                handleFieldChange(
                  "mileage_rate_per_10k",
                  parseFloat(e.target.value) || 0
                )
              }
              className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
            <span className="text-xs text-gray-500 whitespace-nowrap">
              %p/만km
            </span>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">{ageBandLabel}</p>
        </div>

        {/* 연식 보정율 */}
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            연식 보정율
          </label>
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              step="0.1"
              value={criteria.year_rate_per_year}
              onChange={(e) =>
                handleFieldChange(
                  "year_rate_per_year",
                  parseFloat(e.target.value) || 0
                )
              }
              className="w-full border border-indigo-300 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
            <span className="text-xs text-gray-500 whitespace-nowrap">
              %p/년
            </span>
          </div>
        </div>
      </div>

      {/* Footer: vehicle count, confidence, summary */}
      <div className="flex items-start gap-3 pt-3 border-t border-indigo-200">
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs text-gray-500">
            분석 차량:{" "}
            <span className="font-semibold text-gray-700">
              {data.vehicles_analyzed}대
            </span>
          </span>
          <span className="text-gray-300">|</span>
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${confClass}`}
          >
            신뢰도: {data.confidence}
          </span>
        </div>
        <p className="text-xs text-gray-500 leading-relaxed">
          {data.analysis_summary}
        </p>
      </div>
    </div>
  );
}
