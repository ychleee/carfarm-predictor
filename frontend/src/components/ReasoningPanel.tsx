interface Props {
  reasoning: string;
  toolCallsCount: number;
  tokensUsed: { input: number; output: number };
}

export default function ReasoningPanel({
  reasoning,
  toolCallsCount,
  tokensUsed,
}: Props) {
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 mb-6">
      <h3 className="text-sm font-semibold text-amber-800 mb-2 flex items-center gap-1.5">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        LLM 추론 과정
      </h3>
      <p className="text-sm text-amber-900 leading-relaxed whitespace-pre-wrap">
        {reasoning}
      </p>
      <div className="mt-3 flex gap-4 text-xs text-amber-600">
        <span>DB 도구 호출: {toolCallsCount}회</span>
        <span>
          토큰: {tokensUsed.input.toLocaleString()} in / {tokensUsed.output.toLocaleString()} out
        </span>
      </div>
    </div>
  );
}
