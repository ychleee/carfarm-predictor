import type { AdjustmentStep } from "../types";

interface Props {
  adjustments: AdjustmentStep[];
  referencePrice: number;
  totalAdjustment: number;
}

export default function AdjustmentTable({
  adjustments,
  referencePrice,
  totalAdjustment,
}: Props) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            <th className="text-left px-4 py-3 font-medium text-gray-600">보정 항목</th>
            <th className="text-left px-4 py-3 font-medium text-gray-600">설명</th>
            <th className="text-right px-4 py-3 font-medium text-gray-600">보정액</th>
            <th className="text-left px-4 py-3 font-medium text-gray-600">상세</th>
          </tr>
        </thead>
        <tbody>
          {/* 기준가 */}
          <tr className="border-b border-gray-100">
            <td className="px-4 py-2.5 font-medium text-gray-900">기준 낙찰가</td>
            <td className="px-4 py-2.5 text-gray-500">선택한 기준차량</td>
            <td className="px-4 py-2.5 text-right font-mono font-medium text-gray-900">
              {referencePrice.toLocaleString()}만원
            </td>
            <td className="px-4 py-2.5 text-gray-400">-</td>
          </tr>

          {/* 각 보정 단계 */}
          {adjustments.map((adj) => {
            const isPositive = adj.amount > 0;
            const isZero = adj.amount === 0;
            const isWarning = adj.rule_id === "trim_warning";
            return (
              <tr key={adj.rule_id} className={`border-b border-gray-100 ${isWarning ? "bg-amber-50" : ""}`}>
                <td className={`px-4 py-2.5 font-medium ${isWarning ? "text-amber-700" : "text-gray-800"}`}>
                  {adj.rule_name}
                </td>
                <td className="px-4 py-2.5 text-gray-500">{adj.description}</td>
                <td
                  className={`px-4 py-2.5 text-right font-mono font-medium ${
                    isZero
                      ? "text-gray-400"
                      : isPositive
                        ? "text-blue-600"
                        : "text-red-600"
                  }`}
                >
                  {isZero
                    ? "0"
                    : `${isPositive ? "+" : ""}${adj.amount.toLocaleString()}`}
                  만원
                </td>
                <td className="px-4 py-2.5 text-gray-400 text-xs">
                  {adj.details}
                </td>
              </tr>
            );
          })}

          {/* 합계 */}
          <tr className="bg-gray-50 font-semibold">
            <td className="px-4 py-3 text-gray-900">총 보정액</td>
            <td className="px-4 py-3"></td>
            <td
              className={`px-4 py-3 text-right font-mono ${
                totalAdjustment >= 0 ? "text-blue-700" : "text-red-700"
              }`}
            >
              {totalAdjustment >= 0 ? "+" : ""}
              {totalAdjustment.toLocaleString()}만원
            </td>
            <td className="px-4 py-3"></td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
