import { useState } from "react";
import type { TargetVehicle, RecommendResponse } from "./types";
import StepIndicator from "./components/StepIndicator";
import VehicleForm from "./components/VehicleForm";
import RecommendationResult from "./components/RecommendationResult";

type Step = "input" | "recommend";

export default function App() {
  const [step, setStep] = useState<Step>("input");
  const [target, setTarget] = useState<TargetVehicle | null>(null);
  const [recommendData, setRecommendData] =
    useState<RecommendResponse | null>(null);

  const handleRecommend = (t: TargetVehicle, data: RecommendResponse) => {
    setTarget(t);
    setRecommendData(data);
    setStep("recommend");
  };

  const handleReset = () => {
    setStep("input");
    setTarget(null);
    setRecommendData(null);
  };

  const handleStepClick = (clickedStep: Step) => {
    if (clickedStep === "input" && step !== "input") {
      handleReset();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">CarFarm v2</h1>
        <p className="text-gray-500 mb-6 text-sm">
          기준차량 기반 경매 가격 산출 시스템
        </p>

        <StepIndicator current={step} onStepClick={handleStepClick} />

        {step === "input" && <VehicleForm onSubmit={handleRecommend} />}

        {step === "recommend" && recommendData && target && (
          <RecommendationResult
            target={target}
            data={recommendData}
            onBack={handleReset}
          />
        )}
      </div>
    </div>
  );
}
