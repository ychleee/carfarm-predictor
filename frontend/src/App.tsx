import { useState } from "react";
import type { TargetVehicle } from "./types";
import StepIndicator from "./components/StepIndicator";
import VehicleForm from "./components/VehicleForm";
import SearchResult from "./components/SearchResult";

type Step = "input" | "search";

export default function App() {
  const [step, setStep] = useState<Step>("input");
  const [target, setTarget] = useState<TargetVehicle | null>(null);

  const handleSearch = (t: TargetVehicle) => {
    setTarget(t);
    setStep("search");
  };

  const handleReset = () => {
    setStep("input");
    setTarget(null);
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

        {step === "input" && <VehicleForm onSubmit={handleSearch} />}

        {step === "search" && target && (
          <SearchResult target={target} onBack={handleReset} />
        )}
      </div>
    </div>
  );
}
