"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import { usePanelState } from "@/components/PanelStateProvider";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type ScatterColorMode } from "@/components/ScatterChart";
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import { formatPValue } from "@/lib/numberFormat";
import {
  ApiNetworkError,
  ApiResponseError,
  ApiTimeoutError,
  getAnalysisReport,
  getScreeningHeatmap,
  getScreeningPareto,
  getScreeningScatter,
  getScreeningScatterCategorical,
} from "@/lib/api";
import type {
  CategoricalScatterResponse,
  ConfidenceTier,
  ParetoRankingItem,
  ParetoRankingResponse,
  ScatterPoint,
  ScreeningScatterResponse,
} from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
const TIER_LABEL: Record<ConfidenceTier, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };
const RUN_STAGES = ["인자 스크리닝 중 (5개 타깃)", "Pareto 집계 중", "산점도 준비 중", "히트맵 집계 중"];

type ColorMode = ScatterColorMode;
type RunState = "idle" | "running" | "error" | "done";

function hasReliableEvidence(tier: ConfidenceTier): boolean {
  return tier === "strong" || tier === "moderate";
}

type AnalysisFailureKind = "network" | "timeout" | "server" | "model_not_ready" | "unknown";

const ANALYSIS_FAILURE_MESSAGE: Record<AnalysisFailureKind, string> = {
  network: "서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  timeout: "분석 시간이 초과되었습니다. 다시 시도해 주세요.",
  server: "분석 중 오류가 발생했습니다. 다시 시도해 주세요.",
  model_not_ready: "모델 학습이 완료되지 않았습니다. 모델 학습 탭에서 먼저 학습을 실행해 주세요.",
  unknown: "분석을 완료하지 못했습니다. 다시 시도해 주세요.",
};

/** Turns a raw fetch/HTTP failure into a screen-safe message -- never
 * shows "Failed to fetch"/"500 Internal Server Error" verbatim (spec
 * §5-2). The raw detail is kept separately for the collapsible
 * "자세히 보기" section and the console, not shown by default. */
function classifyAnalysisFailure(error: unknown): { kind: AnalysisFailureKind; detail: string } {
  if (error instanceof ApiTimeoutError) return { kind: "timeout", detail: error.message };
  if (error instanceof ApiNetworkError) return { kind: "network", detail: error.message };
  if (error instanceof ApiResponseError) {
    if (error.status >= 500) return { kind: "server", detail: `HTTP ${error.status}: ${error.message}` };
    if (/모델|학습/.test(error.message)) return { kind: "model_not_ready", detail: `HTTP ${error.status}: ${error.message}` };
    return { kind: "unknown", detail: `HTTP ${error.status}: ${error.message}` };
  }
  if (error instanceof Error) return { kind: "unknown", detail: error.message };
  return { kind: "unknown", detail: String(error) };
}

function ConfidenceBadge({ tier }: { tier: ConfidenceTier }) {
  return <span className={`confidenceBadge tier-${tier}`}>{TIER_LABEL[tier]}</span>;
}

function buildCategoricalSpec(data: CategoricalScatterResponse) {
  const x = data.groups.flatMap((group) => group.values.map(() => group.category));
  const y = data.groups.flatMap((group) => group.values);
  return {
    data: [
      {
        type: "box",
        x,
        y,
        boxpoints: "outliers",
        marker: { color: "#1D4ED8" },
        line: { color: "#1D4ED8" },
      },
    ],
    layout: {
      xaxis: { title: { text: factorAxisLabel(data.axis.x_label) }, tickangle: 0 },
      yaxis: { title: { text: targetAxisLabel(data.axis.y_label) } },
      margin: { t: 20, b: 90 },
    },
  };
}

export default function RootCausePage() {
  return (
    <Suspense fallback={null}>
      <RootCauseContent />
    </Suspense>
  );
}

function RootCauseContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setAnalysisDataset } = usePanelState();
  const [datasetId, setDatasetId] = useState("train");
  const [activeTarget, setActiveTarget] = useState(searchParams.get("target") || "Y1");
  const [selectedWafer, setSelectedWafer] = useState<ScatterPoint | null>(null);
  const [compareFeature, setCompareFeature] = useState<string | null>(null);

  const [runState, setRunState] = useState<RunState>("idle");
  // Bumped once per "원인 분석 실행"/"다시 실행" -- folded into each factor
  // card's React `key` so every chart's per-card Color By state (spec
  // §5-3) is forced back to "기본" on a fresh run, even when the same
  // feature reappears at the same list position.
  const [runGeneration, setRunGeneration] = useState(0);
  const [runStageIndex, setRunStageIndex] = useState(0);
  const [runError, setRunError] = useState("");
  const [runErrorDetail, setRunErrorDetail] = useState("");
  const [runElapsedSeconds, setRunElapsedSeconds] = useState(0);
  const [reportSaving, setReportSaving] = useState(false);
  const [reportToast, setReportToast] = useState("");

  const [paretoByTarget, setParetoByTarget] = useState<Record<string, ParetoRankingResponse>>({});
  const [scatterByKey, setScatterByKey] = useState<Record<string, ScreeningScatterResponse>>({});
  const [categoricalByKey, setCategoricalByKey] = useState<Record<string, CategoricalScatterResponse>>({});

  const [pendingScrollFeature, setPendingScrollFeature] = useState<string | null>(null);
  const [quickLook, setQuickLook] = useState<{ target: string; feature: string; isConfig: boolean } | null>(null);
  const [quickLookData, setQuickLookData] = useState<ScreeningScatterResponse | CategoricalScatterResponse | null>(null);
  const [quickLookError, setQuickLookError] = useState("");
  const [quickLookColorMode, setQuickLookColorMode] = useState<ColorMode>("default");
  const initialDeepLinkHandled = useRef(false);

  // A dataset change invalidates every cached result -- back to "not yet run".
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setParetoByTarget({});
      setScatterByKey({});
      setCategoricalByKey({});
      setQuickLook(null);
      setQuickLookData(null);
      setRunState("idle");
      setRunError("");
      setRunErrorDetail("");
      setAnalysisDataset(null);
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  async function runAnalysis() {
    setRunState("running");
    setRunError("");
    setRunErrorDetail("");
    setRunStageIndex(0);
    setRunElapsedSeconds(0);
    setRunGeneration((generation) => generation + 1);
    const startedAt = Date.now();
    const elapsedTimer = window.setInterval(() => {
      setRunElapsedSeconds(Math.round((Date.now() - startedAt) / 1000));
    }, 1000);
    try {
      setRunStageIndex(1);
      const paretoResults = await Promise.all(
        TARGETS.map((t) => getScreeningPareto(datasetId, t).then((response) => [t, response] as const)),
      );
      const paretoMap: Record<string, ParetoRankingResponse> = Object.fromEntries(paretoResults);

      setRunStageIndex(2);
      const fetched = await Promise.all(
        TARGETS.flatMap((t) =>
          (paretoMap[t]?.items ?? []).map(async (item) => {
            const key = `${t}::${item.feature}`;
            if (item.kind === "Config") {
              return { key, type: "categorical" as const, data: await getScreeningScatterCategorical(datasetId, t, item.feature) };
            }
            return { key, type: "numeric" as const, data: await getScreeningScatter(datasetId, t, item.feature) };
          }),
        ),
      );
      const scatterMap: Record<string, ScreeningScatterResponse> = {};
      const categoricalMap: Record<string, CategoricalScatterResponse> = {};
      for (const result of fetched) {
        if (result.type === "categorical") categoricalMap[result.key] = result.data;
        else scatterMap[result.key] = result.data;
      }

      setRunStageIndex(3);
      // Warms the server-side cache with the same computation the
      // heatmap will read -- not a second independent calculation, just
      // a second (cheap, cached) round trip.
      await getScreeningHeatmap(datasetId, "spearman").catch(() => {});

      setParetoByTarget(paretoMap);
      setScatterByKey(scatterMap);
      setCategoricalByKey(categoricalMap);
      setRunState("done");
      setAnalysisDataset(datasetId);
    } catch (failure) {
      // Never leave a stale result on screen after a failure -- it could
      // be mistaken for the new run's output (spec §5-2).
      setParetoByTarget({});
      setScatterByKey({});
      setCategoricalByKey({});
      const { kind, detail } = classifyAnalysisFailure(failure);
      console.error("원인 분석 실행 실패:", failure);
      setRunError(ANALYSIS_FAILURE_MESSAGE[kind]);
      setRunErrorDetail(detail);
      setRunState("error");
      setAnalysisDataset(null);
    } finally {
      window.clearInterval(elapsedTimer);
    }
  }

  async function saveJsonReport() {
    setReportSaving(true);
    try {
      const report = await getAnalysisReport(datasetId);
      const now = new Date();
      const stamp = [
        now.getFullYear(),
        String(now.getMonth() + 1).padStart(2, "0"),
        String(now.getDate()).padStart(2, "0"),
      ].join("") + "_" + [String(now.getHours()).padStart(2, "0"), String(now.getMinutes()).padStart(2, "0")].join("");
      const filename = `suni_analysis_${datasetId}_${stamp}.json`;
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setReportToast("보고서를 저장했습니다");
      window.setTimeout(() => setReportToast(""), 2500);
    } catch (failure) {
      setReportToast(failure instanceof Error ? failure.message : "보고서 저장에 실패했습니다.");
      window.setTimeout(() => setReportToast(""), 3500);
    } finally {
      setReportSaving(false);
    }
  }

  const activeDisplayFactors: ParetoRankingItem[] = useMemo(
    () => paretoByTarget[activeTarget]?.items ?? [],
    [paretoByTarget, activeTarget],
  );

  function updateUrl(target: string, feature?: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("target", target);
    if (feature) params.set("feature", feature);
    else params.delete("feature");
    router.replace(`/root-cause?${params.toString()}`, { scroll: false });
  }

  function selectTarget(target: string) {
    setActiveTarget(target);
    updateUrl(target);
  }

  function openFactor(target: string, feature: string) {
    const isConfig = /_Config$/.test(feature);
    setActiveTarget(target);
    updateUrl(target, feature);
    setQuickLook(null);
    setQuickLookData(null);
    setQuickLookError("");
    // Every newly opened quick-look factor starts at 기본, same as the
    // main list's per-card Color By (spec §5-3).
    setQuickLookColorMode("default");
    const isDisplayed = (paretoByTarget[target]?.items ?? []).some((f) => f.feature === feature);
    if (isDisplayed) {
      setPendingScrollFeature(feature);
    } else {
      setPendingScrollFeature(null);
      setQuickLook({ target, feature, isConfig });
    }
  }

  function handleHeatmapSelect(selection: HeatmapCellSelection) {
    openFactor(selection.target, selection.feature);
  }

  function handleParetoBarClick(item: ParetoRankingItem) {
    openFactor(activeTarget, item.feature);
  }

  // Deep-link support: `?target=&feature=` resolves once the execution's
  // results are available.
  useEffect(() => {
    if (initialDeepLinkHandled.current || runState !== "done") return;
    initialDeepLinkHandled.current = true;
    const featureFromUrl = searchParams.get("feature");
    if (!featureFromUrl) return;
    const timer = window.setTimeout(() => openFactor(activeTarget, featureFromUrl), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runState]);

  useEffect(() => {
    if (!pendingScrollFeature) return;
    const timer = window.setTimeout(() => {
      const displayed = paretoByTarget[activeTarget]?.items ?? [];
      if (!displayed.some((f) => f.feature === pendingScrollFeature)) {
        setPendingScrollFeature(null);
        return;
      }
      const element = document.getElementById(`factor-${pendingScrollFeature}`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
        setPendingScrollFeature(null);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [pendingScrollFeature, activeTarget, paretoByTarget]);

  useEffect(() => {
    if (!quickLook) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const result = quickLook.isConfig
            ? await getScreeningScatterCategorical(datasetId, quickLook.target, quickLook.feature)
            : await getScreeningScatter(datasetId, quickLook.target, quickLook.feature);
          if (!cancelled) setQuickLookData(result);
        } catch (failure) {
          if (!cancelled) setQuickLookError(failure instanceof Error ? failure.message : "산점도를 불러오지 못했습니다.");
        }
      })();
    }, 0);
    document.getElementById("heatmapQuickLook")?.scrollIntoView({ behavior: "smooth", block: "start" });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quickLook]);

  const quickLookNumeric = quickLook && !quickLook.isConfig ? (quickLookData as ScreeningScatterResponse | null) : null;
  const quickLookCategorical = quickLook && quickLook.isConfig ? (quickLookData as CategoricalScatterResponse | null) : null;

  return (
    <DashboardShell activeItem="원인 분석">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">ROOT CAUSE</span>
        <h1>원인 분석</h1>
        <p>타깃(Y1~Y5)별로 전체 인자 풀 기준 Pareto 상위 5개 인자와 산점도를 확인합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(220px,1fr)" }}>
          <DatasetSelector label="분석 데이터셋" value={datasetId} onChange={setDatasetId} />
        </div>
      </section>

      <section className="uploadCard">
        <div className="paretoRunBar">
          <div>
            <h2 style={{ margin: 0, fontSize: "var(--text-section, 17px)" }}>원인 분석 실행</h2>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--text-secondary)" }}>
              {runState === "done"
                ? "완료된 결과입니다. 데이터셋을 바꾸면 다시 실행해야 합니다."
                : "타깃 5개 각각의 전체 인자 풀(R+D+Eq.) 기준 Pareto 상위 5개를 계산합니다."}
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <button type="button" className="button" disabled={runState === "running"} onClick={() => void runAnalysis()}>
              {runState === "running" ? "원인 분석 중..." : runState === "done" ? "다시 실행" : runState === "error" ? "다시 시도" : "원인 분석 실행"}
            </button>
            <button
              type="button"
              className="reportButton"
              disabled={runState !== "done" || reportSaving}
              title={runState !== "done" ? "원인 분석을 먼저 실행하세요" : "보고서는 전체 인자 기준으로 생성됩니다"}
              onClick={() => void saveJsonReport()}
            >
              {reportSaving ? "저장 중..." : "JSON 보고서 저장"}
            </button>
          </div>
        </div>
        {runState === "done" && (
          <p className="reportButtonCaption">보고서는 전체 인자 기준으로 생성됩니다.</p>
        )}
        {runState === "running" && (
          <div className="paretoRunProgress" style={{ marginTop: 12 }}>
            <div className="paretoRunProgressTrack">
              <span style={{ width: `${((runStageIndex + 1) / RUN_STAGES.length) * 100}%` }} />
            </div>
            <span className="paretoRunStage">{RUN_STAGES[runStageIndex]} · 경과 {runElapsedSeconds}초</span>
          </div>
        )}
        {runState === "error" && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">{runError}</p>
              {runErrorDetail && (
                <details className="analysisErrorDetail">
                  <summary>자세히 보기</summary>
                  <code>{runErrorDetail}</code>
                </details>
              )}
            </div>
            <button type="button" className="button" onClick={() => void runAnalysis()}>다시 시도</button>
          </div>
        )}
      </section>

      <HeatmapParetoSection
        datasetId={datasetId}
        enabled={runState === "done"}
        paretoByTarget={paretoByTarget}
        activeTarget={activeTarget}
        onActiveTargetChange={selectTarget}
        onBarClick={handleParetoBarClick}
        onHeatmapCellSelect={handleHeatmapSelect}
      />

      {runState === "done" && (
        <>
          {quickLook && (
            <article id="heatmapQuickLook" className="resultCard factorChartCard">
              <div className="factorChartMeta">
                <div className="factorChartTitleBlock">
                  <span className="sectionLabel">선택한 인자</span>
                  <div className="factorChartTitleRow">
                    <h2>{quickLook.feature} vs {quickLook.target}</h2>
                    {!quickLook.isConfig && <ColorBySelect value={quickLookColorMode} onChange={setQuickLookColorMode} />}
                  </div>
                </div>
                <button
                  className="button"
                  type="button"
                  onClick={() => {
                    setQuickLook(null);
                    setQuickLookData(null);
                  }}
                >
                  닫기
                </button>
              </div>
              {quickLookError && <p className="errorMessage">{quickLookError}</p>}
              {!quickLookError && quickLookNumeric && !hasReliableEvidence(quickLookNumeric.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  이 인자와 {quickLook.target}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(quickLookNumeric.p_value)}, 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}).
                  아래 관리한계선은 인자 자체의 분포에서 산출된 것으로 별개이지만, 원인으로 단정할 근거는 부족합니다.
                </p>
              )}
              {!quickLookError && quickLookCategorical && !hasReliableEvidence(quickLookCategorical.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  통계적 신뢰도가 낮습니다 (p = {formatPValue(quickLookCategorical.p_value)}, 등급 {TIER_LABEL[quickLookCategorical.confidence_tier]}).
                </p>
              )}
              {quickLookNumeric ? (
                <ScatterChart data={quickLookNumeric} colorMode={quickLookColorMode} onSelectWafer={setSelectedWafer} height={420} />
              ) : quickLookCategorical ? (
                <PlotlyChart spec={buildCategoricalSpec(quickLookCategorical)} height={420} />
              ) : !quickLookError ? (
                <p className="emptyMessage">불러오는 중…</p>
              ) : null}
            </article>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
            {activeDisplayFactors.map((item, index) => {
              const isConfig = item.kind === "Config";
              const key = `${activeTarget}::${item.feature}`;
              if (!isConfig) {
                return (
                  <NumericFactorCard
                    key={`${runGeneration}-${activeTarget}-${item.feature}`}
                    item={item}
                    index={index}
                    activeTarget={activeTarget}
                    numericData={scatterByKey[key]}
                    onSelectWafer={setSelectedWafer}
                    onCompare={setCompareFeature}
                  />
                );
              }
              const categoricalData = categoricalByKey[key];
              return (
                <article className="resultCard factorChartCard" id={`factor-${item.feature}`} key={item.feature}>
                  <div className="factorChartMeta">
                    <div className="factorChartTitleBlock">
                      <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
                      <div className="factorChartTitleRow">
                        <h2>{item.feature} vs {activeTarget}</h2>
                        <ConfidenceBadge tier={item.confidence_tier} />
                      </div>
                    </div>
                    {categoricalData && (
                      <small className="factorChartStats">
                        <span>n={categoricalData.n}</span>
                        <span>기여율={item.contribution_pct.toFixed(1)}%</span>
                        <span className="metaCumulative">누적={item.cumulative_pct.toFixed(1)}%</span>
                        <span>p-value {formatPValue(item.p_value)}</span>
                      </small>
                    )}
                  </div>
                  {!hasReliableEvidence(item.confidence_tier) && (
                    <p className="heatmapSignificanceBanner">
                      이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(item.p_value)}). 아래 관리한계선은 인자 자체의 분포에서 산출된 것이라 별개이지만, 원인으로 단정할 근거는 부족합니다.
                    </p>
                  )}
                  {categoricalData ? (
                    <PlotlyChart spec={buildCategoricalSpec(categoricalData)} height={420} />
                  ) : (
                    <p className="emptyMessage">불러오는 중…</p>
                  )}
                </article>
              );
            })}
          </div>
        </>
      )}

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>이 분석은 해당 인자가 계측된 wafer만 대상으로 합니다. R은 전체의 15%, D는 5%입니다. 미계측 wafer로의 일반화는 보장되지 않습니다.</li>
          <li>ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행/후행 관계나 교락 인자는 반영되지 않았습니다.</li>
          <li>&ldquo;약함&rdquo;·&ldquo;참고&rdquo; 등급 인자는 통계적 신뢰도가 낮아 원인으로 단정할 근거가 부족합니다. 사전 알람은 p&lt;0.05(강함·보통 등급) 인자에서만 생성됩니다.</li>
        </ul>
      </section>

      {selectedWafer && (
        <WaferDetailPopover point={selectedWafer} target={activeTarget} onClose={() => setSelectedWafer(null)} />
      )}
      {reportToast && <div className="jsonReportToast" role="status">{reportToast}</div>}
      {compareFeature && (
        <CompareAcrossTargetsModal
          feature={compareFeature}
          originTarget={activeTarget}
          datasetId={datasetId}
          onClose={() => setCompareFeature(null)}
          onSelectTarget={(target) => selectTarget(target)}
        />
      )}
    </DashboardShell>
  );
}

/** One dropdown per scatter card (spec §5-3) -- no server round-trip on
 * change, `lot_id`/`config` already ride along in the point data
 * ScatterChart already has. */
function ColorBySelect({ value, onChange }: { value: ColorMode; onChange: (mode: ColorMode) => void }) {
  return (
    <label className="colorBySelectField">
      <span>색상</span>
      <select
        className="colorBySelect"
        value={value}
        onChange={(event) => onChange(event.target.value as ColorMode)}
      >
        <option value="default">기본</option>
        <option value="config_model">Eq. 모델별</option>
        <option value="lot">LOT별</option>
        <option value="alarm">알람 여부</option>
      </select>
    </label>
  );
}

/** Owns its own Color By state locally (spec §5-3: "전역 store에 넣지
 * 마라") -- the parent forces a reset to 기본 by changing this
 * component's `key` (remount) on every new analysis run or target
 * switch, rather than lifting the state up. */
function NumericFactorCard({
  item,
  index,
  activeTarget,
  numericData,
  onSelectWafer,
  onCompare,
}: {
  item: ParetoRankingItem;
  index: number;
  activeTarget: string;
  numericData: ScreeningScatterResponse | undefined;
  onSelectWafer: (point: ScatterPoint) => void;
  onCompare: (feature: string) => void;
}) {
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`}>
      <div className="factorChartMeta">
        <div className="factorChartTitleBlock">
          <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
            <button
              type="button"
              className="compareTriggerButton"
              title="이 인자가 다른 불량 유형에도 영향을 주는지 확인"
              onClick={() => onCompare(item.feature)}
            >
              ⊞ Y1~Y5 비교
            </button>
            <ColorBySelect value={colorMode} onChange={setColorMode} />
          </div>
        </div>
        {numericData && (
          <small className="factorChartStats">
            <span>n={numericData.n}</span>
            <span>기여율={item.contribution_pct.toFixed(1)}%</span>
            <span className="metaCumulative">누적={item.cumulative_pct.toFixed(1)}%</span>
            <span>p-value {formatPValue(item.p_value)}</span>
          </small>
        )}
      </div>
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(item.p_value)}). 아래 관리한계선은 인자 자체의 분포에서 산출된 것이라 별개이지만, 원인으로 단정할 근거는 부족합니다.
        </p>
      )}
      {numericData ? (
        <ScatterChart data={numericData} colorMode={colorMode} onSelectWafer={onSelectWafer} height={480} />
      ) : (
        <p className="emptyMessage">불러오는 중…</p>
      )}
    </article>
  );
}

function WaferDetailPopover({
  point,
  target,
  onClose,
}: {
  point: ScatterPoint;
  target: string;
  onClose: () => void;
}) {
  return (
    <div className="waferDetailPopover" style={{ right: 24, bottom: 24 }} role="dialog" aria-label="wafer 상세">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">WAFER</span>
          <h2>{point.lot_wafer_id ?? "미상"}</h2>
        </div>
        <button className="button" type="button" onClick={onClose}>닫기</button>
      </div>
      <dl>
        <dt>LOT</dt><dd>{point.lot_id ?? "-"}</dd>
        <dt>{target}</dt><dd>{point.y.toFixed(2)}</dd>
        <dt>인자값</dt><dd>{point.x.toFixed(2)}</dd>
        <dt>정상범위 내</dt><dd>{point.in_range ? "예" : "아니오 (알람)"}</dd>
        <dt>Eq.</dt><dd>{point.config ?? "미계측"}</dd>
      </dl>
    </div>
  );
}
