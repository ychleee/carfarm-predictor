type Step = "input" | "recommend" | "calculate";

const STEPS = [
  { key: "input" as Step, label: "1. 대상차량 입력" },
  { key: "recommend" as Step, label: "2. 기준차량 추천" },
  { key: "calculate" as Step, label: "3. 가격 산출" },
];

export default function StepIndicator({ current }: { current: Step }) {
  const idx = STEPS.findIndex((s) => s.key === current);

  return (
    <div className="flex items-center gap-2 mb-8">
      {STEPS.map((s, i) => {
        const done = i < idx;
        const active = i === idx;
        return (
          <div key={s.key} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={`h-0.5 w-8 ${done ? "bg-blue-500" : "bg-gray-300"}`}
              />
            )}
            <div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                active
                  ? "bg-blue-600 text-white"
                  : done
                    ? "bg-blue-100 text-blue-700"
                    : "bg-gray-100 text-gray-400"
              }`}
            >
              {s.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}
