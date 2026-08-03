"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import CorrelationHeatmap, { type HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import PlotlyChart from "@/components/PlotlyChart";
import { getScreening, getScreeningScatter } from "@/lib/api";
import type { ScatterPoint, ScreeningResponse, ScreeningScatterResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
const KIND_LABEL: Record<string, string> = { R: "계측값", D: "결함수", Config: "장비 설정" };
const SHAPE_LABEL: Record<string, string> = {
  monotonic_increasing: "단조 증가",
  monotonic_decreasing: "단조 감소",
  u_shape: "U자형",
  unclear: "불명확",
};
type ColorMode = "default" | "config_model" | "lot" | "alarm";

function modelOf(config: string | null): string {
  if (!config) return "미계측";
  const match = /Model(\d+)/.exec(config);
  return match ? `Model${match[1]}` : config;
}

const MODEL_COLORS = ["#1D4ED8", "#059669", "#B45309"];
const LOT_PALETTE = ["#1D4ED8", "#059669", "#B45309", "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#DC2626"];

function buildTraces(points: ScatterPoint[], mode: ColorMode) {
  if (mode === "default") {
    const inBand = points.filter((p) => p.in_band);
    const outBand = points.filter((p) => !p.in_band);
    return [
      { x: outBand.map((p) => p.x), y: outBand.map((p) => p.y), mode: "markers", type: "scatter", name: "밴드 밖", marker: { color: "#93C5FD", size: 7, opacity: 0.45 }, customdata: outBand.map((p) => p.lot_wafer_id) },
      { x: inBand.map((p) => p.x), y: inBand.map((p) => p.y), mode: "markers", type: "scatter", name: "밴드 안 (Q1~Q3)", marker: { color: "#1D4ED8", size: 9, opacity: 0.85 }, customdata: inBand.map((p) => p.lot_wafer_id) },
    ];
  }
  if (mode === "alarm") {
    const normal = points.filter((p) => p.in_range);
    const alarm = points.filter((p) => !p.in_range);
    return [
      { x: normal.map((p) => p.x), y: normal.map((p) => p.y), mode: "markers", type: "scatter", name: "정상범위 내", marker: { color: "#1D4ED8", size: 7, opacity: 0.7 }, customdata: normal.map((p) => p.lot_wafer_id) },
      { x: alarm.map((p) => p.x), y: alarm.map((p) => p.y), mode: "markers", type: "scatter", name: "정상범위 밖", marker: { color: "#F59E0B", size: 8, opacity: 0.85, line: { color: "#B45309", width: 1 } }, customdata: alarm.map((p) => p.lot_wafer_id) },
    ];
  }
  if (mode === "config_model") {
    const groups = new Map<string, ScatterPoint[]>();
    for (const point of points) {
      const key = modelOf(point.config);
      groups.set(key, [...(groups.get(key) ?? []), point]);
    }
    return [...groups.entries()].map(([key, group], index) => ({
      x: group.map((p) => p.x),
      y: group.map((p) => p.y),
      mode: "markers",
      type: "scatter",
      name: key,
      marker: {
        color: MODEL_COLORS[index % MODEL_COLORS.length],
        size: group.map((p) => (p.in_band ? 8 : 3.5)),
        opacity: group.map((p) => (p.in_band ? 0.85 : 0.3)),
      },
      customdata: group.map((p) => p.lot_wafer_id),
    }));
  }
  // lot
  const groups = new Map<string, ScatterPoint[]>();
  for (const point of points) {
    const key = point.lot_id ?? "미상";
    groups.set(key, [...(groups.get(key) ?? []), point]);
  }
  return [...groups.entries()].map(([key, group], index) => ({
    x: group.map((p) => p.x),
    y: group.map((p) => p.y),
    mode: "markers",
    type: "scatter",
    name: key,
    showlegend: false,
    marker: {
      color: LOT_PALETTE[index % LOT_PALETTE.length],
      size: group.map((p) => (p.in_band ? 8 : 3.5)),
      opacity: group.map((p) => (p.in_band ? 0.85 : 0.3)),
    },
    customdata: group.map((p) => p.lot_wafer_id),
  }));
}

function buildScatterSpec(data: ScreeningScatterResponse, mode: ColorMode) {
  // A non-significant factor gets no Q1/Q3 band, normal-range, or optimal-center
  // lines -- those are derived thresholds and drawing them here would imply an
  // evidence-backed range that doesn't actually exist for this factor.
  const shapes: Record<string, unknown>[] = data.significant
    ? [
        { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.y_q1, y1: data.y_q1, line: { color: "#E5484D", width: 1.5 } },
        { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.y_q3, y1: data.y_q3, line: { color: "#E5484D", width: 1.5 } },
      ]
    : [];
  const annotations: Record<string, unknown>[] = data.significant
    ? [
        { xref: "paper", x: 1, yref: "y", y: data.y_q1, text: `Q1 (${data.y_q1.toFixed(2)})`, showarrow: false, xanchor: "left", font: { color: "#E5484D", size: 11 } },
        { xref: "paper", x: 1, yref: "y", y: data.y_q3, text: `Q3 (${data.y_q3.toFixed(2)})`, showarrow: false, xanchor: "left", font: { color: "#E5484D", size: 11 } },
      ]
    : [];
  if (data.significant && data.normal_range.lo != null) {
    shapes.push({ type: "line", x0: data.normal_range.lo, x1: data.normal_range.lo, yref: "paper", y0: 0, y1: 1, line: { color: "#6E6E73", dash: "dot", width: 1.5 } });
    annotations.push({ x: data.normal_range.lo, yref: "paper", y: 0, text: data.normal_range.lo.toFixed(1), showarrow: false, yanchor: "top", font: { size: 11, weight: 700 } });
  }
  if (data.significant && data.normal_range.hi != null) {
    shapes.push({ type: "line", x0: data.normal_range.hi, x1: data.normal_range.hi, yref: "paper", y0: 0, y1: 1, line: { color: "#6E6E73", dash: "dot", width: 1.5 } });
    annotations.push({ x: data.normal_range.hi, yref: "paper", y: 0, text: data.normal_range.hi.toFixed(1), showarrow: false, yanchor: "top", font: { size: 11, weight: 700 } });
  }
  if (data.significant && data.optimal_center != null) {
    shapes.push({ type: "line", x0: data.optimal_center, x1: data.optimal_center, yref: "paper", y0: 0, y1: 1, line: { color: "#B45309", dash: "dash", width: 1.5 } });
    annotations.push({ x: data.optimal_center, yref: "paper", y: 1, text: `최적 ${data.optimal_center.toFixed(1)}`, showarrow: false, yanchor: "bottom", font: { color: "#B45309", size: 11 } });
  }

  const profileTrace = data.bins.length
    ? [{
        x: data.bins.map((b) => b.x_mean),
        y: data.bins.map((b) => b.y_mean),
        mode: "lines+markers",
        type: "scatter",
        name: "분위구간 평균",
        line: { color: "#1D1D1F", width: 2 },
      }]
    : [];

  return {
    data: [...buildTraces(data.points, mode), ...profileTrace],
    layout: {
      shapes,
      annotations,
      xaxis: { title: { text: data.axis.x_label } },
      yaxis: { title: { text: data.axis.y_label } },
      margin: { t: 20 },
      dragmode: "select",
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
  const [screening, setScreening] = useState<ScreeningResponse | null>(null);
  const [activeTarget, setActiveTarget] = useState(searchParams.get("target") || "Y1");
  const [scatterByFeature, setScatterByFeature] = useState<Record<string, ScreeningScatterResponse>>({});
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  const [selectedWafer, setSelectedWafer] = useState<ScatterPoint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [pendingScrollFeature, setPendingScrollFeature] = useState<string | null>(null);
  const [quickLook, setQuickLook] = useState<{ target: string; feature: string } | null>(null);
  const [quickLookData, setQuickLookData] = useState<ScreeningScatterResponse | null>(null);
  const [quickLookError, setQuickLookError] = useState("");
  const initialDeepLinkHandled = useRef(false);

  const loadScreening = useCallback(async (id: string) => {
    setLoading(true);
    setError("");
    setScatterByFeature({});
    try {
      const response = await getScreening(id);
      setScreening(response);
      const firstWithFactors = response.targets.find((t) => !t.no_significant_factor)?.target;
      if (firstWithFactors && !response.targets.find((t) => t.target === activeTarget && !t.no_significant_factor)) {
        setActiveTarget(firstWithFactors);
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "스크리닝 결과를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadScreening(datasetId), 0);
    return () => window.clearTimeout(timer);
  }, [datasetId, loadScreening]);

  const activeResult = screening?.targets.find((t) => t.target === activeTarget) ?? null;

  useEffect(() => {
    if (!activeResult || activeResult.no_significant_factor) return;
    let cancelled = false;
    (async () => {
      for (const factor of activeResult.factors) {
        if (scatterByFeature[factor.feature]) continue;
        try {
          const data = await getScreeningScatter(datasetId, activeTarget, factor.feature);
          if (!cancelled) setScatterByFeature((current) => ({ ...current, [factor.feature]: data }));
        } catch {
          // Skip a single failed factor rather than blocking the whole tab.
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeResult, datasetId, activeTarget]);

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
    setActiveTarget(target);
    updateUrl(target, feature);
    setQuickLook(null);
    setQuickLookData(null);
    setQuickLookError("");
    // "Selected" (rendered as a regular factor card) and "FDR-significant"
    // are not the same set -- e.g. a factor can pass FDR but still lose the
    // 80% cumulative cutoff to a stronger factor for the same target. Only
    // an actually-rendered card can be scrolled to; anything else opens the
    // ad-hoc quick-look card instead (which independently reports its own
    // significance for the banner/band-lines).
    const targetResult = screening?.targets.find((t) => t.target === target);
    const isSelected = Boolean(targetResult?.factors.some((f) => f.feature === feature));
    if (isSelected) {
      setPendingScrollFeature(feature);
    } else {
      setPendingScrollFeature(null);
      setQuickLook({ target, feature });
    }
  }

  function handleHeatmapSelect(selection: HeatmapCellSelection) {
    openFactor(selection.target, selection.feature);
  }

  // Deep-link support: `?target=&feature=` (e.g. from a heatmap cell click
  // in a previous visit, or a shared link) resolves once the screening
  // result for the dataset is known.
  useEffect(() => {
    if (initialDeepLinkHandled.current || !screening) return;
    initialDeepLinkHandled.current = true;
    const featureFromUrl = searchParams.get("feature");
    const targetFromUrl = searchParams.get("target");
    if (!featureFromUrl || !targetFromUrl) return;
    const timer = window.setTimeout(() => openFactor(targetFromUrl, featureFromUrl), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screening]);

  useEffect(() => {
    if (!pendingScrollFeature || !activeResult) return;
    const timer = window.setTimeout(() => {
      if (activeResult.no_significant_factor || !activeResult.factors.some((f) => f.feature === pendingScrollFeature)) {
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
  }, [pendingScrollFeature, activeResult]);

  useEffect(() => {
    if (!quickLook) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const result = await getScreeningScatter(datasetId, quickLook.target, quickLook.feature);
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

  const noConfigSelected = useMemo(
    () => screening?.targets.every((t) => t.factors.every((f) => f.kind !== "Config")) ?? true,
    [screening],
  );

  return (
    <DashboardShell activeItem="원인 분석">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">ROOT CAUSE</span>
        <h1>원인 분석</h1>
        <p>선정 인자별 산점도로 정상범위가 어떻게 도출됐는지 확인합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(220px,1fr) minmax(180px,1fr)" }}>
          <DatasetSelector label="분석 데이터셋" value={datasetId} onChange={setDatasetId} />
          <div className="fieldGroup">
            <span>Color By</span>
            <select value={colorMode} onChange={(event) => setColorMode(event.target.value as ColorMode)}>
              <option value="default">기본 (파랑 2단계)</option>
              <option value="config_model">Config 모델별</option>
              <option value="lot">LOT별</option>
              <option value="alarm">알람 여부</option>
            </select>
          </div>
        </div>
        {error && <p className="errorMessage">{error}</p>}
      </section>

      <CorrelationHeatmap datasetId={datasetId} onSelectCell={handleHeatmapSelect} />

      {noConfigSelected && (
        <section className="messageBox">Config(장비 설정) 30개 중 통계적으로 유의한 인자는 0개입니다.</section>
      )}

      <div className="targetSegmentBar">
        {TARGETS.map((target) => {
          const result = screening?.targets.find((t) => t.target === target);
          const count = result?.factors.length ?? 0;
          return (
            <button
              key={target}
              type="button"
              className={`targetSegment ${activeTarget === target ? "active" : ""} ${count === 0 ? "empty" : ""}`}
              onClick={() => selectTarget(target)}
            >
              {target}
              <span className="countBadge">{count}</span>
            </button>
          );
        })}
      </div>

      {loading && <p className="emptyMessage">불러오는 중…</p>}

      {quickLook && (
        <article id="heatmapQuickLook" className="resultCard factorChartCard">
          <div className="factorChartMeta">
            <div>
              <span className="sectionLabel">히트맵에서 선택</span>
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
          {!quickLookError && quickLookData && !quickLookData.significant && (
            <p className="heatmapSignificanceBanner">
              이 인자는 통계적으로 유의하지 않습니다 (q = {quickLookData.q_value < 0.001 ? quickLookData.q_value.toExponential(2) : quickLookData.q_value.toFixed(4)}).
              참고용으로만 확인하고, Q1/Q3·정상범위 기준선은 근거가 없어 표시하지 않습니다.
            </p>
          )}
          {quickLookData ? (
            <PlotlyChart spec={buildScatterSpec(quickLookData, colorMode)} height={420} />
          ) : !quickLookError ? (
            <p className="emptyMessage">불러오는 중…</p>
          ) : null}
        </article>
      )}

      {activeResult?.no_significant_factor && (
        <section className="resultCard">
          <p className="emptyMessage">
            {activeTarget}에 대해 BH-FDR을 통과한 인자가 없어 정상범위를 산출할 수 없습니다. 아래 참고용 인자는 통계적으로 유의하지 않으므로 원인으로 해석하지 마세요.
          </p>
        </section>
      )}

      {activeResult && !activeResult.no_significant_factor && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {activeResult.factors.map((factor, index) => {
            const data = scatterByFeature[factor.feature];
            return (
              <article className="resultCard factorChartCard" id={`factor-${factor.feature}`} key={factor.feature}>
                <div className="factorChartMeta">
                  <div>
                    <span className="sectionLabel">{index + 1}위 · ε² {factor.eps2.toFixed(3)}</span>
                    <h2>{factor.feature} ({KIND_LABEL[factor.kind]})</h2>
                  </div>
                  {data && (
                    <small>
                      n={data.n} · ε²={factor.eps2.toFixed(3)} · q={factor.q_value < 0.001 ? factor.q_value.toExponential(2) : factor.q_value.toFixed(4)} · {SHAPE_LABEL[factor.relation_shape]}
                      {factor.optimal_center != null && ` (최적 ${factor.optimal_center.toFixed(1)})`}
                      {data.normal_range.fallback_applied && <span className="referenceOnlyBadge" style={{ marginLeft: 8 }}>축소 적용 (1~99%)</span>}
                    </small>
                  )}
                </div>
                {factor.kind === "Config" ? (
                  <p className="emptyMessage">범주형 인자는 박스플롯 뷰가 준비 중입니다.</p>
                ) : data ? (
                  <PlotlyChart spec={buildScatterSpec(data, colorMode)} height={480} />
                ) : (
                  <p className="emptyMessage">불러오는 중…</p>
                )}
                {data && (
                  <ScatterClickCapture points={data.points} onSelect={setSelectedWafer} />
                )}
              </article>
            );
          })}
        </div>
      )}

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>이 분석은 해당 인자가 계측된 wafer만 대상으로 합니다. R은 전체의 15%, D는 5%입니다. 미계측 wafer로의 일반화는 보장되지 않습니다.</li>
          <li>ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행/후행 관계나 교락 인자는 반영되지 않았습니다.</li>
          <li>U자 인자는 값이 높아서/낮아서 나쁜 게 아니라 최적 중심에서 이탈한 만큼 나쁩니다. 조치 방향은 현재 값이 중심보다 높은지 낮은지에 따라 달라집니다.</li>
        </ul>
      </section>

      {selectedWafer && (
        <WaferDetailPopover point={selectedWafer} target={activeTarget} onClose={() => setSelectedWafer(null)} />
      )}
    </DashboardShell>
  );
}

// PlotlyChart doesn't currently forward click events; this lightweight list
// lets users pick a wafer by id without waiting on deeper Plotly wiring.
function ScatterClickCapture({ points, onSelect }: { points: ScatterPoint[]; onSelect: (point: ScatterPoint) => void }) {
  const outliers = points.filter((p) => !p.in_range).slice(0, 12);
  if (outliers.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <small style={{ color: "var(--text-secondary)" }}>정상범위 밖 wafer (클릭하여 상세 확인): </small>
      {outliers.map((point) => (
        <button
          key={point.lot_wafer_id}
          type="button"
          className="referenceOnlyToggle"
          style={{ marginRight: 6, marginTop: 6 }}
          onClick={() => onSelect(point)}
        >
          {point.lot_wafer_id}
        </button>
      ))}
    </div>
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
