import type { AuctionVehicle } from "../types";

interface Props {
  vehicle: AuctionVehicle;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteModal({
  vehicle,
  onConfirm,
  onCancel,
}: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onCancel} />
      <div className="relative bg-white rounded-xl shadow-2xl max-w-sm w-full mx-4">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            이 기준차량을 삭제할까요?
          </h3>

          <div className="mb-4 bg-gray-50 rounded-lg p-3">
            <p className="text-sm font-medium text-gray-800">
              {vehicle.vehicle_name ?? `ID: ${vehicle.auction_id}`}
            </p>
            <p className="text-xs text-gray-500 mt-0.5">
              {vehicle.year}년 | {vehicle.mileage?.toLocaleString()}km |
              낙찰 {vehicle.auction_price?.toLocaleString()}만원
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              취소
            </button>
            <button
              onClick={onConfirm}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white font-medium py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              삭제
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
