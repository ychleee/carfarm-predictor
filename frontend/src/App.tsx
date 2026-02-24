import { useState } from "react";
import type {
  TargetVehicle,
  RecommendResponse,
  ReferenceVehicle,
  CalculateResponse,
} from "./types";
import StepIndicator from "./components/StepIndicator";
import VehicleForm from "./components/VehicleForm";
import RecommendationResult from "./components/RecommendationResult";
import PriceCalculation from "./components/PriceCalculation";

type Step = "input" | "recommend" | "calculate";

export default function App() {
  const [step, setStep] = useState<Step>("input");

  // 각 단계 데이터
  const [target, setTarget] = useState<TargetVehicle | null>(null);
  const [recommendData, setRecommendData] =
    useState<RecommendResponse | null>(null);
  const [selectedRef, setSelectedRef] = useState<ReferenceVehicle | null>(null);
  const [calcData, setCalcData] = useState<CalculateResponse | null>(null);

  // Step 1 → Step 2
  const handleRecommend = (t: TargetVehicle, data: RecommendResponse) => {
    setTarget(t);
    setRecommendData(data);
    setStep("recommend");
  };

  // Step 2 → Step 3
  const handleSelectRef = (ref: ReferenceVehicle, calc: CalculateResponse) => {
    setSelectedRef(ref);
    setCalcData(calc);
    setStep("calculate");
  };

  // 처음으로
  const handleReset = () => {
    setStep("input");
    setTarget(null);
    setRecommendData(null);
    setSelectedRef(null);
    setCalcData(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          CarFarm v2
        </h1>
        <p className="text-gray-500 mb-6 text-sm">
          기준차량 기반 경매 가격 산출 시스템
        </p>

        <StepIndicator current={step} />

        {step === "input" && (
          <VehicleForm onSubmit={handleRecommend} />
        )}

        {step === "recommend" && recommendData && target && (
          <RecommendationResult
            target={target}
            data={recommendData}
            onSelect={handleSelectRef}
            onBack={handleReset}
          />
        )}

        {step === "calculate" && calcData && selectedRef && target && (
          <PriceCalculation
            target={target}
            reference={selectedRef}
            data={calcData}
            onReset={handleReset}
          />
        )}
      </div>
    </div>
  );
}
