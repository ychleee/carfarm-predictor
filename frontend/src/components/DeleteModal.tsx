import { useState } from "react";
import type { TargetVehicle, ReferenceVehicle } from "../types";

interface Props {
  target: TargetVehicle;
  reference: ReferenceVehicle;
  onConfirm: (comment: string) => void;
  onCancel: () => void;
}

export default function DeleteModal({
  target,
  reference,
  onConfirm,
  onCancel,
}: Props) {
  const [comment, setComment] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onCancel} />
      <div className="relative bg-white rounded-xl shadow-2xl max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            이 기준차량을 삭제하시겠어요?
          </h3>

          {/* 대상차량 */}
          <div className="mb-3">
            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">대상차량</p>
            <p className="text-sm text-gray-800">
              {target.maker} {target.model} {target.trim ?? ""}
            </p>
            <p className="text-xs text-gray-500">
              {target.year}년 | {target.mileage.toLocaleString()}km
              {target.color ? ` | ${target.color}` : ""}
            </p>
          </div>

          {/* 기준차량 */}
          <div className="mb-3">
            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">기준차량</p>
            <p className="text-sm text-gray-800">
              {reference.vehicle_name ?? `ID: ${reference.auction_id}`}
            </p>
            <p className="text-xs text-gray-500">
              {reference.year}년 | {reference.mileage?.toLocaleString()}km |
              낙찰 {reference.auction_price?.toLocaleString()}만원
              {reference.color ? ` | ${reference.color}` : ""}
            </p>
          </div>

          {/* LLM 선택 이유 */}
          <div className="mb-4">
            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">LLM 선택 이유</p>
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 text-sm text-blue-800">
              {reference.similarity_reason}
            </div>
          </div>

          {/* 삭제 이유 입력 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              LLM 추론의 어떤 부분이 잘못됐나요?
            </label>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="예) 트림이 다릅니다. 이 차량은 스탠다드인데 대상차량은 프리미엄입니다."
              rows={3}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 resize-none"
            />
            <p className="text-xs text-gray-400 mt-1">
              {comment.length < 10
                ? `${10 - comment.length}자 더 입력해주세요`
                : "입력 완료"}
            </p>
          </div>

          {/* 버튼 */}
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              취소
            </button>
            <button
              onClick={() => onConfirm(comment)}
              disabled={comment.length < 10}
              className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-red-300 text-white font-medium py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              삭제 및 피드백 저장
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
