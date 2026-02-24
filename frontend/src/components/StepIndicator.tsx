type Step = "input" | "recommend";

const STEPS = [
  { key: "input" as Step, label: "1. 대상차량 입력" },
  { key: "recommend" as Step, label: "2. 기준차량 추천 & 가격 산출" },
];

interface Props {
  current: Step;
  onStepClick?: (step: Step) => void;
}

export default function StepIndicator({ current, onStepClick }: Props) {
  const idx = STEPS.findIndex((s) => s.key === current);

  return (
    <div className="flex items-center gap-2 mb-8">
      {STEPS.map((s, i) => {
        const done = i < idx;
        const active = i === idx;
        const clickable = done && onStepClick;
        return (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={`h-0.5 w-8 ${done ? "bg-blue-500" : "bg-gray-300"}`}
              />
            )}
            <button
              type="button"
              disabled={!clickable}
              onClick={() => clickable && onStepClick(s.key)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                active
                  ? "bg-blue-600 text-white"
                  : done
                    ? "bg-blue-100 text-blue-700 hover:bg-blue-200 cursor-pointer"
                    : "bg-gray-100 text-gray-400"
              } ${!clickable ? "cursor-default" : ""}`}
            >
              {s.label}
            </button>
          </div>
        );
      })}
    </div>
  );
}
