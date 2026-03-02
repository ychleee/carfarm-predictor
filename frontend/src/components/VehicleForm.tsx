import { useState, useEffect } from "react";
import type { TargetVehicle, RecommendResponse } from "../types";
import {
  getMakers,
  getModels,
  getGenerations,
  getVariants,
  getTrims,
  recommendReferences,
} from "../api/client";
import type { ModelInfo, GenerationInfo, VariantInfo } from "../types";

const COLOR_OPTIONS = ["흰색", "검정", "은색", "메탈", "기타"];
const USAGE_OPTIONS = [
  { value: "personal", label: "자가용" },
  { value: "rental", label: "렌터카" },
];
const PREFERRED_OPTIONS = ["선루프", "스마트키", "네비", "후방카메라"];

interface Props {
  onSubmit: (target: TargetVehicle, data: RecommendResponse) => void;
}

export default function VehicleForm({ onSubmit }: Props) {
  // Taxonomy 캐스케이딩
  const [makers, setMakers] = useState<string[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [generations, setGenerations] = useState<GenerationInfo[]>([]);
  const [variants, setVariants] = useState<VariantInfo[]>([]);
  const [trims, setTrims] = useState<string[]>([]);

  // 폼 데이터
  const [maker, setMaker] = useState("");
  const [model, setModel] = useState("");
  const [generation, setGeneration] = useState("");
  const [variantKey, setVariantKey] = useState("");
  const [trim, setTrim] = useState("");
  const [year, setYear] = useState<number>(2022);
  const [mileage, setMileage] = useState<number>(50000);
  const [fuel, setFuel] = useState("");
  const [displacement, setDisplacement] = useState("");
  const [color, setColor] = useState("");
  const [usage, setUsage] = useState("personal");
  const [options, setOptions] = useState<string[]>([]);
  const [exchangeCount, setExchangeCount] = useState(0);
  const [bodyworkCount, setBodyworkCount] = useState(0);

  // UI 상태
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 제작사 목록 로드
  useEffect(() => {
    getMakers()
      .then(setMakers)
      .catch((err) => console.error("[CarFarm] 제작사 로드 실패:", err));
  }, []);

  // 모델 목록 로드
  useEffect(() => {
    if (!maker) {
      setModels([]);
      return;
    }
    setModel("");
    setGeneration("");
    setVariantKey("");
    setTrim("");
    getModels(maker).then(setModels).catch(() => {});
  }, [maker]);

  // 세대 목록 로드
  useEffect(() => {
    if (!maker || !model) {
      setGenerations([]);
      return;
    }
    setGeneration("");
    setVariantKey("");
    setTrim("");
    getGenerations(maker, model).then(setGenerations).catch(() => {});
  }, [maker, model]);

  // 변형 목록 로드 + 자동 선택
  useEffect(() => {
    if (!maker || !model || !generation) {
      setVariants([]);
      setVariantKey("");
      setTrim("");
      return;
    }
    setVariantKey("");
    setTrim("");
    getVariants(maker, model, generation).then((v) => {
      setVariants(v);
      // variant가 1개면 자동 선택
      if (v.length === 1) {
        setVariantKey(v[0].variant_key);
        // 연료/배기량 자동 세팅
        if (v[0].fuel) setFuel(v[0].fuel);
        if (v[0].displacement) setDisplacement(v[0].displacement);
      }
    }).catch(() => {});
  }, [maker, model, generation]);

  // 트림 목록 로드
  useEffect(() => {
    if (!maker || !model || !generation) {
      setTrims([]);
      return;
    }
    setTrim("");
    // variant가 선택되었으면 해당 variant의 트림만, 아니면 전체 트림
    const vk = variantKey || undefined;
    getTrims(maker, model, generation, vk).then(setTrims).catch(() => {});
  }, [maker, model, generation, variantKey]);

  // variant 선택 시 연료/배기량 자동 세팅
  const handleVariantChange = (vk: string) => {
    setVariantKey(vk);
    const v = variants.find((v) => v.variant_key === vk);
    if (v) {
      if (v.fuel) setFuel(v.fuel);
      if (v.displacement) setDisplacement(v.displacement);
    }
  };

  const toggleOption = (opt: string) => {
    setOptions((prev) =>
      prev.includes(opt) ? prev.filter((o) => o !== opt) : [...prev, opt]
    );
  };

  const handleSubmit = async () => {
    if (!maker || !model || !year || !mileage) {
      setError("제작사, 모델, 연식, 주행거리는 필수입니다.");
      return;
    }

    const target: TargetVehicle = {
      maker,
      model,
      generation: generation || undefined,
      year,
      mileage,
      fuel: fuel || undefined,
      displacement: displacement || undefined,
      trim: trim || undefined,
      color: color || undefined,
      usage: usage || undefined,
      domestic: true,
      options,
      exchange_count: exchangeCount,
      bodywork_count: bodyworkCount,
    };

    setError(null);
    setLoading(true);
    try {
      const data = await recommendReferences(target);
      if ((!data.retail_vehicles || data.retail_vehicles.length === 0) &&
          (!data.auction_vehicles || data.auction_vehicles.length === 0)) {
        setError("추천 결과가 없습니다. 검색 조건을 확인해주세요.");
        return;
      }
      onSubmit(target, data);
    } catch (err) {
      console.error("[CarFarm] 추천 에러:", err);
      setError(err instanceof Error ? err.message : "추천 요청 실패");
    } finally {
      setLoading(false);
    }
  };

  const selectClass =
    "w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white";
  const labelClass = "block text-sm font-medium text-gray-700 mb-1";

  // variant가 1개뿐이면 드롭다운 숨김
  const showVariants = variants.length > 1;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        대상차량 정보 입력
      </h2>

      {/* 차량 선택 (캐스케이딩) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className={labelClass}>제작사 *</label>
          <select
            className={selectClass}
            value={maker}
            onChange={(e) => setMaker(e.target.value)}
          >
            <option value="">선택</option>
            {makers.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>모델 *</label>
          <select
            className={selectClass}
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={!maker}
          >
            <option value="">선택</option>
            {models.map((m) => (
              <option key={m.model} value={m.model}>{m.model}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>세대</label>
          <select
            className={selectClass}
            value={generation}
            onChange={(e) => setGeneration(e.target.value)}
            disabled={!model}
          >
            <option value="">전체</option>
            {generations.map((g) => (
              <option key={g.generation} value={g.generation}>{g.display_name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>트림</label>
          <select
            className={selectClass}
            value={trim}
            onChange={(e) => setTrim(e.target.value)}
            disabled={!generation}
          >
            <option value="">전체</option>
            {trims.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
      </div>

      {/* 변형 선택 (variant가 2개 이상일 때만 표시) */}
      {showVariants && (
        <div className="mb-4">
          <label className={labelClass}>연료/배기량 (세부 변형)</label>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => { setVariantKey(""); setFuel(""); setDisplacement(""); }}
              className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                !variantKey
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
              }`}
            >
              전체
            </button>
            {variants.map((v) => (
              <button
                key={v.variant_key}
                type="button"
                onClick={() => handleVariantChange(v.variant_key)}
                className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                  variantKey === v.variant_key
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"
                }`}
              >
                {v.label} <span className="text-xs opacity-70">({v.trim_count})</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 기본 정보 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className={labelClass}>연식 *</label>
          <input
            type="number"
            className={selectClass}
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            min={2000}
            max={2026}
          />
        </div>

        <div>
          <label className={labelClass}>주행거리(km) *</label>
          <input
            type="number"
            className={selectClass}
            value={mileage}
            onChange={(e) => setMileage(Number(e.target.value))}
            min={0}
            step={1000}
          />
        </div>

        <div>
          <label className={labelClass}>연료</label>
          <select
            className={selectClass}
            value={fuel}
            onChange={(e) => setFuel(e.target.value)}
          >
            <option value="">전체</option>
            <option value="가솔린">가솔린</option>
            <option value="디젤">디젤</option>
            <option value="하이브리드">하이브리드</option>
            <option value="LPG">LPG</option>
          </select>
        </div>

        <div>
          <label className={labelClass}>색상</label>
          <select
            className={selectClass}
            value={color}
            onChange={(e) => setColor(e.target.value)}
          >
            <option value="">선택</option>
            {COLOR_OPTIONS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {/* 차량경력 + 사고이력 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className={labelClass}>차량경력</label>
          <select
            className={selectClass}
            value={usage}
            onChange={(e) => setUsage(e.target.value)}
          >
            {USAGE_OPTIONS.map((u) => (
              <option key={u.value} value={u.value}>{u.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>교환(X) 부위수</label>
          <input
            type="number"
            className={selectClass}
            value={exchangeCount}
            onChange={(e) => setExchangeCount(Number(e.target.value))}
            min={0}
            max={20}
          />
        </div>

        <div>
          <label className={labelClass}>판금(W) 부위수</label>
          <input
            type="number"
            className={selectClass}
            value={bodyworkCount}
            onChange={(e) => setBodyworkCount(Number(e.target.value))}
            min={0}
            max={20}
          />
        </div>
      </div>

      {/* 선호옵션 체크박스 */}
      <div className="mb-6">
        <label className={labelClass}>선호옵션 (선스네후)</label>
        <div className="flex gap-4 mt-1">
          {PREFERRED_OPTIONS.map((opt) => (
            <label key={opt} className="flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.includes(opt)}
                onChange={() => toggleOption(opt)}
                className="rounded border-gray-300"
              />
              {opt}
            </label>
          ))}
        </div>
      </div>

      {/* 에러 메시지 */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* 제출 버튼 */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium py-3 px-4 rounded-lg transition-colors text-sm"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            LLM 추론 중... (20~40초 소요)
          </span>
        ) : (
          "기준차량 추천 받기"
        )}
      </button>
    </div>
  );
}
