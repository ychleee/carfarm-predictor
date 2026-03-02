interface DamageGroup {
  exchange: number;
  bodywork: number;
  corrosion: number;
}

interface Props {
  frame: DamageGroup;
  exterior: DamageGroup;
}

export default function DamageInfo({ frame, exterior }: Props) {
  const hasFrame = frame.exchange > 0 || frame.bodywork > 0 || frame.corrosion > 0;
  const hasExterior = exterior.exchange > 0 || exterior.bodywork > 0 || exterior.corrosion > 0;

  if (!hasFrame && !hasExterior) return null;

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] mt-1.5 mb-1">
      {hasFrame && (
        <span className="text-red-600 font-medium">
          프레임
          {frame.exchange > 0 && ` 교환${frame.exchange}`}
          {frame.bodywork > 0 && ` 판금${frame.bodywork}`}
          {frame.corrosion > 0 && ` 부식${frame.corrosion}`}
        </span>
      )}
      {hasExterior && (
        <span className="text-orange-600 font-medium">
          외판
          {exterior.exchange > 0 && ` 교환${exterior.exchange}`}
          {exterior.bodywork > 0 && ` 판금${exterior.bodywork}`}
          {exterior.corrosion > 0 && ` 부식${exterior.corrosion}`}
        </span>
      )}
    </div>
  );
}
