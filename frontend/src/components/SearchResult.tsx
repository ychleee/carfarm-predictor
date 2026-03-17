import { useState, useEffect, useMemo, useRef } from "react";
import type {
  TargetVehicle,
  AuctionVehicle,
  AuctionFilters,
  FilterOptions,
  CalculateResponse,
  PricePredictionResponse,
  AnalyzeCriteriaResponse,
  PricingCriteria,
} from "../types";
import { COMPANY_TABS } from "../types";
import { searchAuction, calculateWithCriteria, analyzeCriteria, predictPrice, submitFeedback } from "../api/client";
import AuctionCard from "./AuctionCard";
import type { CalcState } from "./AuctionCard";
import CriteriaPanel from "./CriteriaPanel";
import PriceDetailModal from "./PriceDetailModal";
import PricePredictionModal from "./PricePredictionModal";
import DeleteModal from "./DeleteModal";

// === TabData ===

interface TabData {
  vehicles: AuctionVehicle[];
  loading: boolean;
  error: string | null;
  loaded: boolean;
  filters: AuctionFilters;
  filterOptions: FilterOptions;
  deletedIds: Set<string>;
}

const EMPTY_FILTERS: AuctionFilters = {
  trim: "",
  fuel: "",
  color: "",
  yearMin: null,
  yearMax: null,
  mileageMin: null,
  mileageMax: null,
  soldOnly: false,
};

const EMPTY_FILTER_OPTIONS: FilterOptions = {
  trims: [],
  fuels: [],
  colors: [],
  yearRange: [2000, 2026],
  mileageRange: [0, 500000],
};

function extractFilterOptions(vehicles: AuctionVehicle[]): FilterOptions {
  const trims = new Set<string>();
  const fuels = new Set<string>();
  const colors = new Set<string>();
  let minYear = 9999, maxYear = 0;
  let minMileage = Infinity, maxMileage = 0;

  for (const v of vehicles) {
    if (v.trim) trims.add(v.trim);
    if (v.fuel) fuels.add(v.fuel);
    if (v.color) colors.add(v.color);
    if (v.year != null) {
      if (v.year < minYear) minYear = v.year;
      if (v.year > maxYear) maxYear = v.year;
    }
    if (v.mileage != null) {
      if (v.mileage < minMileage) minMileage = v.mileage;
      if (v.mileage > maxMileage) maxMileage = v.mileage;
    }
  }

  return {
    trims: [...trims].sort(),
    fuels: [...fuels].sort(),
    colors: [...colors].sort(),
    yearRange: [minYear === 9999 ? 2000 : minYear, maxYear === 0 ? 2026 : maxYear],
    mileageRange: [minMileage === Infinity ? 0 : minMileage, maxMileage === 0 ? 500000 : maxMileage],
  };
}

function buildSmartDefaults(target: TargetVehicle): AuctionFilters {
  return {
    trim: "",
    fuel: target.fuelType ?? "",
    color: "",
    yearMin: target.vehicleYear - 2,
    yearMax: target.vehicleYear + 2,
    mileageMin: null,
    mileageMax: null,
    soldOnly: false,
  };
}

function applyFilters(
  vehicles: AuctionVehicle[],
  filters: AuctionFilters,
  deletedIds: Set<string>
): AuctionVehicle[] {
  return vehicles.filter((v) => {
    if (deletedIds.has(v.auction_id)) return false;
    if (filters.trim && v.trim !== filters.trim) return false;
    if (filters.fuel && v.fuel !== filters.fuel) return false;
    if (filters.color && v.color !== filters.color) return false;
    if (filters.yearMin != null && v.year != null && v.year < filters.yearMin) return false;
    if (filters.yearMax != null && v.year != null && v.year > filters.yearMax) return false;
    if (filters.mileageMin != null && v.mileage != null && v.mileage < filters.mileageMin) return false;
    if (filters.mileageMax != null && v.mileage != null && v.mileage > filters.mileageMax) return false;
    if (filters.soldOnly && v.status !== "완료") return false;
    return true;
  });
}

// === Props ===

interface Props {
  target: TargetVehicle;
  onBack: () => void;
}

export default function SearchResult({ target, onBack }: Props) {
  // 탭별 독립 데이터
  const [tabDataMap, setTabDataMap] = useState<Record<string, TabData>>(() => {
    const map: Record<string, TabData> = {};
    for (const tab of COMPANY_TABS) {
      map[tab.id] = {
        vehicles: [],
        loading: false,
        error: null,
        loaded: false,
        filters: buildSmartDefaults(target),
        filterOptions: EMPTY_FILTER_OPTIONS,
        deletedIds: new Set(),
      };
    }
    return map;
  });

  // 공유 state
  const [activeTabId, setActiveTabId] = useState(COMPANY_TABS[0].id);
  const [calcStates, setCalcStates] = useState<Record<string, CalcState>>({});
  const [deleteTarget, setDeleteTarget] = useState<AuctionVehicle | null>(null);
  const [detailTarget, setDetailTarget] = useState<{
    ref: AuctionVehicle;
    calc: CalculateResponse;
  } | null>(null);

  // AI 가격 예측
  const [predictionState, setPredictionState] = useState<{
    status: "idle" | "loading" | "done" | "error";
    data?: PricePredictionResponse;
    error?: string;
  }>({ status: "idle" });
  const [showPredictionModal, setShowPredictionModal] = useState(false);

  // 보정 기준 분석
  const [criteriaState, setCriteriaState] = useState<{
    data: AnalyzeCriteriaResponse | null;
    loading: boolean;
    error: string | null;
  }>({ data: null, loading: false, error: null });

  const [currentCriteria, setCurrentCriteria] = useState<PricingCriteria | null>(null);

  const handlePredictPrice = async () => {
    setPredictionState({ status: "loading" });
    try {
      const result = await predictPrice(target);
      setPredictionState({ status: "done", data: result });
      setShowPredictionModal(true);
    } catch (err) {
      setPredictionState({
        status: "error",
        error: err instanceof Error ? err.message : "예측 실패",
      });
    }
  };

  // 보정 기준 분석
  const handleAnalyzeCriteria = async (refId?: string, refPrice?: number) => {
    setCriteriaState({ data: null, loading: true, error: null });
    try {
      const result = await analyzeCriteria({
        target_vehicle: target,
        reference_auction_id: refId || "auto",
        reference_auction_price: refPrice || 0,
      });
      setCriteriaState({ data: result, loading: false, error: null });
      setCurrentCriteria(result.criteria);
    } catch (err) {
      // LLM 실패 시 업계 기본값 사용
      const age = new Date().getFullYear() - target.vehicleYear;
      const defaults: PricingCriteria = {
        mileage_rate_per_10k: age <= 3 ? 2.0 : age <= 6 ? 1.5 : 1.0,
        mileage_ceiling_km: 200000,
        year_rate_per_year: age <= 6 ? 2.5 : age <= 9 ? 2.0 : 1.5,
      };
      setCurrentCriteria(defaults);
      setCriteriaState({
        data: {
          criteria: defaults,
          analysis_summary: "LLM 분석 실패 — 업계 기본값 사용",
          vehicles_analyzed: 0,
          confidence: "낮음",
        },
        loading: false,
        error: null,
      });
    }
  };

  // 범위 필터 팝오버
  const [openPopover, setOpenPopover] = useState<string | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  // 탭 데이터 로드
  const loadTab = async (tabId: string) => {
    setTabDataMap((prev) => ({
      ...prev,
      [tabId]: { ...prev[tabId], loading: true, error: null },
    }));

    try {
      const res = await searchAuction({
        maker: target.vehicleMaker,
        model: target.vehicleModel,
        company_id: tabId,
        limit: 500,
      });

      const filterOptions = extractFilterOptions(res.results);

      setTabDataMap((prev) => ({
        ...prev,
        [tabId]: {
          ...prev[tabId],
          vehicles: res.results,
          loading: false,
          loaded: true,
          filterOptions,
        },
      }));

      // 첫 번째 탭 로드 시 보정 기준 분석 자동 실행
      if (!criteriaState.data && !criteriaState.loading && res.results.length > 0) {
        const firstWithPrice = res.results.find((v) => v.auction_price);
        if (firstWithPrice) {
          handleAnalyzeCriteria(firstWithPrice.auction_id, firstWithPrice.auction_price!);
        }
      }
    } catch (err) {
      setTabDataMap((prev) => ({
        ...prev,
        [tabId]: {
          ...prev[tabId],
          loading: false,
          error: err instanceof Error ? err.message : "검색 실패",
        },
      }));
    }
  };

  // 마운트 시 첫 번째 탭 로드
  useEffect(() => {
    loadTab(COMPANY_TABS[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 탭 전환 시 lazy load
  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId);
    const tabData = tabDataMap[tabId];
    if (!tabData.loaded && !tabData.loading) {
      loadTab(tabId);
    }
  };

  // 팝오버 바깥 클릭 닫기
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpenPopover(null);
      }
    };
    if (openPopover) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [openPopover]);

  // 현재 탭
  const activeTab = COMPANY_TABS.find((t) => t.id === activeTabId)!;
  const currentTabData = tabDataMap[activeTabId];

  // 필터 적용된 목록
  const filteredVehicles = useMemo(
    () => applyFilters(currentTabData.vehicles, currentTabData.filters, currentTabData.deletedIds),
    [currentTabData.vehicles, currentTabData.filters, currentTabData.deletedIds]
  );

  // 필터 업데이트 헬퍼
  const updateFilter = (patch: Partial<AuctionFilters>) => {
    setTabDataMap((prev) => ({
      ...prev,
      [activeTabId]: {
        ...prev[activeTabId],
        filters: { ...prev[activeTabId].filters, ...patch },
      },
    }));
  };

  const resetFilters = () => {
    setTabDataMap((prev) => ({
      ...prev,
      [activeTabId]: {
        ...prev[activeTabId],
        filters: { ...EMPTY_FILTERS },
      },
    }));
  };

  const hasActiveFilter = () => {
    const f = currentTabData.filters;
    return f.trim !== "" || f.fuel !== "" || f.color !== "" ||
      f.yearMin != null || f.yearMax != null ||
      f.mileageMin != null || f.mileageMax != null ||
      f.soldOnly;
  };

  // 삭제
  const handleDelete = (vehicle: AuctionVehicle) => {
    setDeleteTarget(vehicle);
  };

  const handleDeleteConfirm = () => {
    if (!deleteTarget) return;
    setTabDataMap((prev) => ({
      ...prev,
      [activeTabId]: {
        ...prev[activeTabId],
        deletedIds: new Set(prev[activeTabId].deletedIds).add(deleteTarget.auction_id),
      },
    }));
    setDeleteTarget(null);
  };

  // 가격 산출
  const handleCalc = async (ref: AuctionVehicle) => {
    if (!ref.auction_price) return;
    setCalcStates((prev) => ({
      ...prev,
      [ref.auction_id]: { status: "loading" },
    }));
    try {
      const calc = await calculateWithCriteria({
        target_vehicle: target,
        reference_auction_id: ref.auction_id,
        reference_auction_price: ref.auction_price,
        criteria: currentCriteria ?? {
          mileage_rate_per_10k: 1.5,
          mileage_ceiling_km: 200000,
          year_rate_per_year: 2.5,
        },
        reference_inspection: {
          frame_exchange: ref.frame_exchange ?? 0,
          frame_bodywork: ref.frame_bodywork ?? 0,
          frame_corrosion: ref.frame_corrosion ?? 0,
          exterior_exchange: ref.exterior_exchange ?? 0,
          exterior_bodywork: ref.exterior_bodywork ?? 0,
          exterior_corrosion: ref.exterior_corrosion ?? 0,
        },
      });
      setCalcStates((prev) => ({
        ...prev,
        [ref.auction_id]: { status: "done", data: calc },
      }));
    } catch (err) {
      setCalcStates((prev) => ({
        ...prev,
        [ref.auction_id]: {
          status: "error",
          error: err instanceof Error ? err.message : "산출 실패",
        },
      }));
    }
  };

  const handleCardClick = (ref: AuctionVehicle) => {
    const state = calcStates[ref.auction_id];
    if (state?.status === "done" && state.data) {
      setDetailTarget({ ref, calc: state.data });
    }
  };

  // UI 헬퍼
  const selectChipClass = (active: boolean) =>
    active
      ? "bg-blue-600 text-white border-blue-600"
      : "bg-white text-gray-700 border-gray-300 hover:border-blue-400";

  const chipSelectClass =
    "border rounded-lg px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500";

  return (
    <div>
      {/* 대상차량 요약 + AI 예측 버튼 */}
      <div className="bg-gray-100 rounded-lg px-4 py-2.5 mb-4 text-sm text-gray-700 flex items-center justify-between gap-2">
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <span className="font-medium">대상:</span>
          <span>
            {target.vehicleMaker} {target.vehicleModel}
          </span>
          {target.generation && <span>| {target.generation}</span>}
          {target.vehicleTrim && <span>| {target.vehicleTrim}</span>}
          <span>| {target.vehicleYear}년</span>
          <span>| {target.mileage.toLocaleString()}km</span>
          {target.fuelType && <span>| {target.fuelType}</span>}
          {target.vehicleColor && <span>| {target.vehicleColor}</span>}
        </div>

        <div className="shrink-0">
          {predictionState.status === "idle" && (
            <button
              onClick={handlePredictPrice}
              className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-all shadow-sm"
            >
              AI 가격 예측
            </button>
          )}
          {predictionState.status === "loading" && (
            <span className="flex items-center gap-2 text-xs text-purple-600 font-medium">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              AI 분석 중...
            </span>
          )}
          {predictionState.status === "done" && predictionState.data && (
            <button
              onClick={() => setShowPredictionModal(true)}
              className="bg-gradient-to-r from-emerald-500 to-green-600 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-sm flex items-center gap-2"
            >
              <span>예상 낙찰가 {predictionState.data.estimated_auction.toLocaleString()}만</span>
            </button>
          )}
          {predictionState.status === "error" && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-red-500">{predictionState.error}</span>
              <button
                onClick={handlePredictPrice}
                className="text-xs text-purple-600 hover:underline"
              >
                재시도
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 탭 바 */}
      <div className="flex border-b border-gray-200 mb-4">
        {COMPANY_TABS.map((tab) => {
          const td = tabDataMap[tab.id];
          const isActive = tab.id === activeTabId;
          const count = td.loaded
            ? applyFilters(td.vehicles, td.filters, td.deletedIds).length
            : null;

          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? `border-current ${tab.textClass}`
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              {tab.label}
              {td.loading && (
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {count != null && (
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                  isActive ? `${tab.bgClass} ${tab.textClass}` : "bg-gray-100 text-gray-500"
                }`}>
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* 필터 칩 행 */}
      {currentTabData.loaded && (
        <div className="flex flex-wrap items-center gap-2 mb-4 overflow-x-auto" ref={popoverRef}>
          {/* 트림 */}
          {currentTabData.filterOptions.trims.length > 0 && (
            <select
              value={currentTabData.filters.trim}
              onChange={(e) => updateFilter({ trim: e.target.value })}
              className={`${chipSelectClass} ${selectChipClass(!!currentTabData.filters.trim)}`}
            >
              <option value="">트림 전체</option>
              {currentTabData.filterOptions.trims.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          )}

          {/* 연료 */}
          {currentTabData.filterOptions.fuels.length > 0 && (
            <select
              value={currentTabData.filters.fuel}
              onChange={(e) => updateFilter({ fuel: e.target.value })}
              className={`${chipSelectClass} ${selectChipClass(!!currentTabData.filters.fuel)}`}
            >
              <option value="">연료 전체</option>
              {currentTabData.filterOptions.fuels.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          )}

          {/* 색상 */}
          {currentTabData.filterOptions.colors.length > 0 && (
            <select
              value={currentTabData.filters.color}
              onChange={(e) => updateFilter({ color: e.target.value })}
              className={`${chipSelectClass} ${selectChipClass(!!currentTabData.filters.color)}`}
            >
              <option value="">색상 전체</option>
              {currentTabData.filterOptions.colors.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          )}

          {/* 연식 범위 */}
          <div className="relative">
            <button
              onClick={() => setOpenPopover(openPopover === "year" ? null : "year")}
              className={`${chipSelectClass} cursor-pointer ${selectChipClass(
                currentTabData.filters.yearMin != null || currentTabData.filters.yearMax != null
              )}`}
            >
              연식{currentTabData.filters.yearMin != null || currentTabData.filters.yearMax != null
                ? ` ${currentTabData.filters.yearMin ?? "~"}-${currentTabData.filters.yearMax ?? "~"}`
                : " 전체"}
            </button>
            {openPopover === "year" && (
              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-3 z-10 w-56">
                <div className="flex items-center gap-2 mb-2">
                  <input
                    type="number"
                    placeholder="최소"
                    value={currentTabData.filters.yearMin ?? ""}
                    onChange={(e) => updateFilter({ yearMin: e.target.value ? Number(e.target.value) : null })}
                    className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
                    min={2000}
                    max={2026}
                  />
                  <span className="text-gray-400 text-xs">~</span>
                  <input
                    type="number"
                    placeholder="최대"
                    value={currentTabData.filters.yearMax ?? ""}
                    onChange={(e) => updateFilter({ yearMax: e.target.value ? Number(e.target.value) : null })}
                    className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
                    min={2000}
                    max={2026}
                  />
                </div>
                <button
                  onClick={() => { updateFilter({ yearMin: null, yearMax: null }); setOpenPopover(null); }}
                  className="text-xs text-blue-600 hover:underline"
                >
                  초기화
                </button>
              </div>
            )}
          </div>

          {/* 주행거리 범위 */}
          <div className="relative">
            <button
              onClick={() => setOpenPopover(openPopover === "mileage" ? null : "mileage")}
              className={`${chipSelectClass} cursor-pointer ${selectChipClass(
                currentTabData.filters.mileageMin != null || currentTabData.filters.mileageMax != null
              )}`}
            >
              주행거리{currentTabData.filters.mileageMin != null || currentTabData.filters.mileageMax != null
                ? ` ${currentTabData.filters.mileageMin != null ? (currentTabData.filters.mileageMin / 10000).toFixed(0) + "만" : "~"}-${currentTabData.filters.mileageMax != null ? (currentTabData.filters.mileageMax / 10000).toFixed(0) + "만" : "~"}km`
                : " 전체"}
            </button>
            {openPopover === "mileage" && (
              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-3 z-10 w-64">
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex-1">
                    <label className="text-[10px] text-gray-400">최소 (km)</label>
                    <input
                      type="number"
                      placeholder="0"
                      value={currentTabData.filters.mileageMin ?? ""}
                      onChange={(e) => updateFilter({ mileageMin: e.target.value ? Number(e.target.value) : null })}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
                      min={0}
                      step={10000}
                    />
                  </div>
                  <span className="text-gray-400 text-xs mt-3">~</span>
                  <div className="flex-1">
                    <label className="text-[10px] text-gray-400">최대 (km)</label>
                    <input
                      type="number"
                      placeholder="999999"
                      value={currentTabData.filters.mileageMax ?? ""}
                      onChange={(e) => updateFilter({ mileageMax: e.target.value ? Number(e.target.value) : null })}
                      className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
                      min={0}
                      step={10000}
                    />
                  </div>
                </div>
                <button
                  onClick={() => { updateFilter({ mileageMin: null, mileageMax: null }); setOpenPopover(null); }}
                  className="text-xs text-blue-600 hover:underline"
                >
                  초기화
                </button>
              </div>
            )}
          </div>

          {/* 판매완료만 (엔카 탭) */}
          {activeTabId === "KYMaGfcnzwGsvbDm6Z91" && (
            <button
              onClick={() => updateFilter({ soldOnly: !currentTabData.filters.soldOnly })}
              className={`${chipSelectClass} cursor-pointer ${selectChipClass(currentTabData.filters.soldOnly)}`}
            >
              판매완료만
            </button>
          )}

          {/* 필터 초기화 */}
          {hasActiveFilter() && (
            <button
              onClick={resetFilters}
              className="text-xs text-red-500 hover:text-red-700 border border-red-300 rounded-lg px-2 py-1 hover:bg-red-50 transition-colors"
            >
              필터 초기화
            </button>
          )}
        </div>
      )}

      {/* 보정 기준 패널 */}
      <CriteriaPanel
        target={target}
        data={criteriaState.data}
        loading={criteriaState.loading}
        error={criteriaState.error}
        onCriteriaChange={setCurrentCriteria}
        onReanalyze={handleAnalyzeCriteria}
      />

      {/* 에러 */}
      {currentTabData.error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm flex items-center justify-between">
          <span>{currentTabData.error}</span>
          <button
            onClick={() => loadTab(activeTabId)}
            className="text-sm text-blue-600 hover:underline ml-3"
          >
            재시도
          </button>
        </div>
      )}

      {/* 로딩 */}
      {currentTabData.loading && (
        <div className="flex items-center justify-center py-12 text-gray-500">
          <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          {activeTab.label} 데이터 검색 중...
        </div>
      )}

      {/* 결과 리스트 */}
      {currentTabData.loaded && !currentTabData.loading && (
        <>
          {filteredVehicles.length === 0 ? (
            <div className="text-center py-12 text-gray-400 text-sm">
              {currentTabData.vehicles.length === 0
                ? `${activeTab.label}에 해당 차량 데이터가 없습니다.`
                : "필터 조건에 맞는 차량이 없습니다. 필터를 조정해보세요."}
            </div>
          ) : (
            <div className="flex flex-col gap-3 mb-4">
              {filteredVehicles.map((v, i) => (
                <AuctionCard
                  key={v.auction_id}
                  vehicle={v}
                  index={i}
                  companyTab={activeTab}
                  calcState={calcStates[v.auction_id] ?? { status: "idle" }}
                  onCalc={() => handleCalc(v)}
                  onClick={() => handleCardClick(v)}
                  onDelete={() => handleDelete(v)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* 뒤로 */}
      <div className="flex justify-end mt-4">
        <button
          onClick={onBack}
          className="text-sm text-gray-500 hover:text-gray-700 underline px-4"
        >
          다른 차량으로 다시 검색
        </button>
      </div>

      {/* 삭제 모달 */}
      {deleteTarget && (
        <DeleteModal
          vehicle={deleteTarget}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {/* AI 가격 예측 모달 */}
      {showPredictionModal && predictionState.data && (
        <PricePredictionModal
          target={target}
          data={predictionState.data}
          onClose={() => setShowPredictionModal(false)}
        />
      )}

      {/* 가격 상세 모달 (피드백 포함) */}
      {detailTarget && (
        <PriceDetailModal
          target={target}
          reference={detailTarget.ref}
          data={detailTarget.calc}
          onClose={() => setDetailTarget(null)}
          onPriceFeedback={(type, comment) => {
            // 현재 탭의 전체 차량 목록으로 피드백 전송
            const allVehicles = Object.values(tabDataMap).flatMap((td) => td.vehicles);
            submitFeedback({
              target_vehicle: target,
              selected_reference_id: detailTarget.ref.auction_id,
              recommended_references: allVehicles.map((c) => c.auction_id),
              recommendations_detail: allVehicles,
              calculation_result: detailTarget.calc,
              estimated_price: detailTarget.calc.estimated_retail,
              actual_price: null,
              feedback_type: type,
              comment: comment || null,
            }).catch(() => {});
          }}
        />
      )}
    </div>
  );
}
