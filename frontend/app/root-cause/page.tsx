"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type ScatterColorMode } from "@/components/ScatterChart";
import {
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
const KIND_LABEL: Record<string, string> = { R: "계측값", D: "결함수", Config: "장비 설정" };
const TIER_LABEL: Record<ConfidenceTier, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };
const RUN_STAGES = ["인자 스크리닝 중 (5개 타깃)", "Pareto 집계 중", "산점도 준비 중", "히트맵 집계 중"];

type ColorMode = ScatterColorMode;
type RunState = "idle" | "running" | "error" | "done";

function formatP(p: number): string {
  return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
}

function hasReliableEvidence(tier: ConfidenceTier): boolean {
  return tier === "strong" || tier === "moderate";
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
      xaxis: { title: { text: data.axis.x_label }, tickangle: 0 },
      yaxis: { title: { text: data.axis.y_label } },
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
  const [datasetId, setDatasetId] = useState("train");
  const [activeTarget, setActiveTarget] = useState(searchParams.get("target") || "Y1");
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  const [selectedWafer, setSelectedWafer] = useState<ScatterPoint | null>(null);
  const [compareFeature, setCompareFeature] = useState<string | null>(null);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runStageIndex, setRunStageIndex] = useState(0);
  const [runError, setRunError] = useState("");
  const [reportSaving, setReportSaving] = useState(false);
  const [reportToast, setReportToast] = useState("");

  const [paretoByTarget, setParetoByTarget] = useState<Record<string, ParetoRankingResponse>>({});
  const [scatterByKey, setScatterByKey] = useState<Record<string, ScreeningScatterResponse>>({});
  const [categoricalByKey, setCategoricalByKey] = useState<Record<string, CategoricalScatterResponse>>({});

  const [pendingScrollFeature, setPendingScrollFeature] = useState<string | null>(null);
  const [quickLook, setQuickLook] = useState<{ target: string; feature: string; isConfig: boolean } | null>(null);
  const [quickLookData, setQuickLookData] = useState<ScreeningScatterResponse | CategoricalScatterResponse | null>(null);
  const [quickLookError, setQuickLookError] = useState("");
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
    }, 0);
    return () => window.clearTimeout(timer);
  }, [datasetId]);

  async function runAnalysis() {
    setRunState("running");
    setRunError("");
    setRunStageIndex(0);
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
    } catch (failure) {
      setParetoByTarget({});
      setScatterByKey({});
      setCategoricalByKey({});
      setRunError(failure instanceof Error ? failure.message : "원인 분석 실행에 실패했습니다.");
      setRunState("error");
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
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(220px,1fr) minmax(180px,1fr)" }}>
          <DatasetSelector label="분석 데이터셋" value={datasetId} onChange={setDatasetId} />
          <div className="fieldGroup">
            <span>Color By</span>
            <select value={colorMode} onChange={(event) => setColorMode(event.target.value as ColorMode)}>
              <option value="default">기본</option>
              <option value="config_model">Config 모델별</option>
              <option value="lot">LOT별</option>
              <option value="alarm">알람 여부</option>
            </select>
          </div>
        </div>
      </section>

      <section className="uploadCard">
        <div className="paretoRunBar">
          <div>
            <h2 style={{ margin: 0, fontSize: "var(--text-section, 17px)" }}>원인 분석 실행</h2>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--text-secondary)" }}>
              {runState === "done"
                ? "완료된 결과입니다. 데이터셋을 바꾸면 다시 실행해야 합니다."
                : "타깃 5개 각각의 전체 인자 풀(R+D+Config) 기준 Pareto 상위 5개를 계산합니다."}
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
            <span className="paretoRunStage">{RUN_STAGES[runStageIndex]}</span>
          </div>
        )}
        {runState === "error" && <p className="errorMessage">{runError}</p>}
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
                <div>
                  <span className="sectionLabel">선택한 인자</span>
                  <h2>{quickLook.feature} · {quickLook.target}</h2>
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
                  이 인자와 {quickLook.target}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatP(quickLookNumeric.p_value)}, 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}).
                  아래 관리한계선은 인자 자체의 분포에서 산출된 것으로 별개이지만, 원인으로 단정할 근거는 부족합니다.
                </p>
              )}
              {!quickLookError && quickLookCategorical && !hasReliableEvidence(quickLookCategorical.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  통계적 신뢰도가 낮습니다 (p = {formatP(quickLookCategorical.p_value)}, 등급 {TIER_LABEL[quickLookCategorical.confidence_tier]}).
                </p>
              )}
              {quickLookNumeric ? (
                <ScatterChart data={quickLookNumeric} colorMode={colorMode} onSelectWafer={setSelectedWafer} height={420} />
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
              const numericData = !isConfig ? scatterByKey[key] : undefined;
              const categoricalData = isConfig ? categoricalByKey[key] : undefined;
              const n = numericData?.n ?? categoricalData?.n;
              return (
                <article className="resultCard factorChartCard" id={`factor-${item.feature}`} key={item.feature}>
                  <div className="factorChartMeta">
                    <div>
                      <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
                      <h2>{item.feature} ({KIND_LABEL[item.kind]}) <ConfidenceBadge tier={item.confidence_tier} /></h2>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {n != null && (
                        <small>
                          n={n} · 기여율={item.contribution_pct.toFixed(1)}% · 누적={item.cumulative_pct.toFixed(1)}% · p={formatP(item.p_value)}
                        </small>
                      )}
                      {!isConfig && (
                        <button
                          type="button"
                          className="compareTriggerButton"
                          title="이 인자가 다른 불량 유형에도 영향을 주는지 확인"
                          onClick={() => setCompareFeature(item.feature)}
                        >
                          ⊞ Y1~Y5 비교
                        </button>
                      )}
                    </div>
                  </div>
                  {!hasReliableEvidence(item.confidence_tier) && (
                    <p className="heatmapSignificanceBanner">
                      이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatP(item.p_value)}). 아래 관리한계선은 인자 자체의 분포에서 산출된 것이라 별개이지만, 원인으로 단정할 근거는 부족합니다.
                    </p>
                  )}
                  {isConfig ? (
                    categoricalData ? (
                      <PlotlyChart spec={buildCategoricalSpec(categoricalData)} height={420} />
                    ) : (
                      <p className="emptyMessage">불러오는 중…</p>
                    )
                  ) : numericData ? (
                    <ScatterChart data={numericData} colorMode={colorMode} onSelectWafer={setSelectedWafer} height={480} />
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
        <dt>Config</dt><dd>{point.config ?? "미계측"}</dd>
      </dl>
    </div>
  );
}
