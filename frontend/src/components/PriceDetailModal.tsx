import { useState } from "react";
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
  onClose: () => void;
  onPriceFeedback?: (type: string, comment: string) => void;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  높음: "bg-green-100 text-green-800",
  보통: "bg-yellow-100 text-yellow-800",
  낮음: "bg-red-100 text-red-800",
};

const PRICE_FEEDBACK_OPTIONS = [
  { value: "price_ok", label: "적절", color: "bg-green-600 hover:bg-green-700 text-white" },
  { value: "price_high", label: "높음", color: "bg-orange-500 hover:bg-orange-600 text-white" },
  { value: "price_low", label: "낮음", color: "bg-blue-500 hover:bg-blue-600 text-white" },
  { value: "wrong_recommendation", label: "추천 부적절", color: "bg-red-600 hover:bg-red-700 text-white" },
];

export default function PriceDetailModal({
  target,
  reference,
  data,
  onClose,
  onPriceFeedback,
}: Props) {
  const confClass =
    CONFIDENCE_COLORS[data.confidence] ?? "bg-gray-100 text-gray-800";

  const [feedbackSent, setFeedbackSent] = useState(false);
  const [comment, setComment] = useState("");

  const handleFeedback = (type: string) => {
    if (!onPriceFeedback) return;
    onPriceFeedback(type, comment);
    setFeedbackSent(true);
  };

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

          {/* 차량 비교 테이블 */}
          <div className="mb-5">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">차량 비교</h4>
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="text-left px-3 py-2 text-gray-500 font-medium w-[70px]"></th>
                    <th className="text-left px-3 py-2 text-gray-700 font-semibold">대상차량</th>
                    <th className="text-left px-3 py-2 text-blue-700 font-semibold">기준차량</th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const targetName = `${target.vehicleMaker} ${target.vehicleModel}`;
                    const refName = reference.vehicle_name ?? `ID: ${reference.auction_id}`;
                    const targetOptions = target.vehicleOptions ?? [];
                    const refOptions = reference.options?.split(",").map(o => o.trim()).filter(Boolean) ?? [];
                    const commonOptions = targetOptions.filter(o => refOptions.includes(o));
                    const targetOnly = targetOptions.filter(o => !refOptions.includes(o));
                    const refOnly = refOptions.filter(o => !targetOptions.includes(o));

                    // 만원 변환 헬퍼 (100000 초과면 원 단위 → 만원)
                    const toManWon = (v: number) => v > 100000 ? Math.round(v / 10000) : v;

                    const rows: { label: string; targetVal: string; refVal: string }[] = [
                      { label: "차명", targetVal: targetName, refVal: refName },
                      { label: "트림", targetVal: target.vehicleTrim ?? "-", refVal: reference.trim ?? "-" },
                      { label: "연식", targetVal: `${target.vehicleYear}년`, refVal: reference.year != null ? `${reference.year}년` : "-" },
                      { label: "주행", targetVal: `${target.mileage.toLocaleString()}km`, refVal: reference.mileage != null ? `${reference.mileage.toLocaleString()}km` : "-" },
                      { label: "연료", targetVal: target.fuelType ?? "-", refVal: reference.fuel ?? "-" },
                      { label: "색상", targetVal: target.vehicleColor ?? "-", refVal: reference.color ?? "-" },
                      { label: "출고가", targetVal: target.vehicleFactoryPrice ? `${toManWon(Number(target.vehicleFactoryPrice)).toLocaleString()}만원` : "-", refVal: (() => { const p = (reference.factory_price != null && reference.factory_price > 0) ? reference.factory_price : reference.base_price; return p != null && p > 0 ? `${toManWon(p).toLocaleString()}만원` : "-"; })() },
                      {
                        label: "검차",
                        targetVal: "AA (무사고)",
                        refVal: (() => {
                          const parts: string[] = [];
                          if (reference.inspection_grade) parts.push(reference.inspection_grade);
                          const dmg: string[] = [];
                          if (reference.frame_exchange > 0) dmg.push(`프레임교환${reference.frame_exchange}`);
                          if (reference.frame_bodywork > 0) dmg.push(`프레임판금${reference.frame_bodywork}`);
                          if (reference.exterior_exchange > 0) dmg.push(`외판교환${reference.exterior_exchange}`);
                          if (reference.exterior_bodywork > 0) dmg.push(`외판판금${reference.exterior_bodywork}`);
                          if (dmg.length > 0) parts.push(dmg.join(" "));
                          return parts.length > 0 ? parts.join(" ") : "-";
                        })(),
                      },
                    ];

                    return (
                      <>
                        {rows.map((row, i) => {
                          const isDiff = row.targetVal !== row.refVal;
                          return (
                            <tr key={row.label} className={`${i % 2 === 0 ? "bg-white" : "bg-gray-50/50"} ${isDiff ? "bg-amber-50/60" : ""}`}>
                              <td className="px-3 py-1.5 text-gray-400 font-medium">{row.label}</td>
                              <td className="px-3 py-1.5 text-gray-800">{row.targetVal}</td>
                              <td className="px-3 py-1.5 text-blue-800">{row.refVal}</td>
                            </tr>
                          );
                        })}
                        {(targetOptions.length > 0 || refOptions.length > 0) && (
                          <tr className="border-t border-gray-200">
                            <td colSpan={3} className="px-3 py-2">
                              <p className="text-gray-500 font-medium mb-1">옵션 비교</p>
                              <div className="flex flex-wrap gap-1">
                                {commonOptions.map(o => (
                                  <span key={`c-${o}`} className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded text-[10px]">{o}</span>
                                ))}
                                {targetOnly.map(o => (
                                  <span key={`t-${o}`} className="bg-green-100 text-green-700 px-1.5 py-0.5 rounded text-[10px]">[대상] {o}</span>
                                ))}
                                {refOnly.map(o => (
                                  <span key={`r-${o}`} className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-[10px]">[기준] {o}</span>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })()}
                </tbody>
              </table>
            </div>
          </div>

          {/* 보정 내역 */}
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            보정 내역 (LLM 분석)
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

            <div>
              <p className="text-xs text-gray-400 mb-1">예상 낙찰가</p>
              <p className="text-2xl font-bold text-blue-700">
                {data.estimated_auction.toLocaleString()}
                <span className="text-sm font-normal text-blue-400">
                  만원
                </span>
              </p>
            </div>

            <p className="mt-3 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
              {data.summary}
            </p>
          </div>

          {/* 가격 평가 피드백 */}
          {onPriceFeedback && (
            <div className="mb-4 border border-gray-200 rounded-xl p-4">
              {feedbackSent ? (
                <div className="text-center text-sm text-green-700 bg-green-50 rounded-lg py-3">
                  피드백이 저장되었습니다
                </div>
              ) : (
                <>
                  <p className="text-sm font-semibold text-gray-700 mb-2">
                    이 산출 가격이 적절한가요?
                  </p>
                  <textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="(선택) 보정이 잘못된 부분, 적정 가격 의견 등"
                    rows={2}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none mb-3"
                  />
                  <div className="flex gap-2">
                    {PRICE_FEEDBACK_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => handleFeedback(opt.value)}
                        className={`flex-1 font-medium py-2 px-3 rounded-lg transition-colors text-sm ${opt.color}`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

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
