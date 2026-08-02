"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AnalysisHistorySelector from "@/components/AnalysisHistorySelector";
import DashboardSectionState from "@/components/DashboardSectionState";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import StatusBadge from "@/components/StatusBadge";
import StatusCard from "@/components/StatusCard";
import { getAnalysisHistory, getDashboardOverview } from "@/lib/api";
import {
  createEmptyOverview,
  overviewFromHistory,
  resolveOverviewSelection,
} from "@/lib/overview";
import type {
  AnalysisHistorySummary,
  AnalysisOverviewResponse,
  DashboardSectionState as SectionState,
  DashboardState,
  OverviewCauseItem,
  OverviewKpi,
} from "@/types/data";


type BarDatum = { label: string; value: number | null; unit?: string };

function formatNumber(value: number | null, digits = 2): string {
  return value === null
    ? "-"
    : value.toLocaleString("ko-KR", { maximumFractionDigits: digits });
}

function formatDate(value: string | null): string {
  if (!value) return "시각 정보 없음";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "시각 정보 없음" : date.toLocaleString("ko-KR");
}

function causeLabel(item: OverviewCauseItem | undefined, fallback = "-"): string {
  if (!item) return fallback;
  if (item.display_name) return item.display_name;
  if (item.feature) return item.feature;
  if (item.equipment) return item.equipment;
  if (item.chamber) return item.chamber;
  if (item.step !== null && item.step !== undefined) return `Step ${item.step}`;
  return fallback;
}

function causeValue(item: OverviewCauseItem): number | null {
  return item.mean_abs_shap ?? item.impact ?? item.score ?? item.path_score ?? null;
}

function sectionState(
  loading: boolean,
  error: string,
  available: boolean,
  artifactStatus: AnalysisOverviewResponse["source"]["artifact_status"],
): SectionState {
  if (loading) return "loading";
  if (error) return "error";
  if (!available && (artifactStatus === "missing" || artifactStatus === "corrupted")) return "unavailable";
  return available ? "ready" : "empty";
}

function sectionMessage(
  state: SectionState,
  emptyMessage: string,
  error: string,
  artifactStatus: AnalysisOverviewResponse["source"]["artifact_status"],
): string {
  if (state === "loading") return "분석 결과를 불러오는 중입니다.";
  if (state === "error") return error || "데이터를 불러오지 못했습니다.";
  if (state === "unavailable") {
    return artifactStatus === "corrupted"
      ? "분석 결과 파일이 손상되어 이 항목을 표시할 수 없습니다."
      : "분석 결과 파일이 없어 이 항목을 표시할 수 없습니다.";
  }
  return emptyMessage;
}

function MetricBars({
  data,
  state,
  message,
  onRetry,
}: {
  data: BarDatum[];
  state: SectionState;
  message: string;
  onRetry: () => void;
}) {
  const available = data.filter((item) => item.value !== null);
  if (state !== "ready" || available.length === 0) {
    const fallbackState = state === "ready" ? "empty" : state;
    return (
      <DashboardSectionState
        state={fallbackState}
        message={message}
        onRetry={fallbackState === "error" ? onRetry : undefined}
      />
    );
  }
  const maximum = Math.max(...available.map((item) => Math.abs(item.value ?? 0)));
  return (
    <div className="overviewBarChart" role="img" aria-label="실제 분석 값 막대그래프">
      {available.map((item) => {
        const width = maximum > 0 ? Math.abs(item.value ?? 0) / maximum * 100 : 0;
        return (
          <div className="overviewBarRow" key={item.label}>
            <span title={item.label}>{item.label}</span>
            <div className="overviewBarTrack" aria-hidden="true">
              <span style={{ width: `${width}%` }} />
            </div>
            <strong>{formatNumber(item.value)}{item.unit ?? ""}</strong>
          </div>
        );
      })}
    </div>
  );
}

function sourceStatus(source: AnalysisOverviewResponse["source"]): {
  label: string;
  tone: "success" | "info" | "warning" | "danger" | "neutral";
} {
  if (source.status === "completed") return { label: "완료", tone: "success" };
  if (source.status === "partial") return { label: "일부 결과", tone: "warning" };
  if (source.status === "artifact_missing") return { label: "결과 파일 누락", tone: "warning" };
  if (source.status === "artifact_corrupted") return { label: "결과 파일 손상", tone: "danger" };
  if (source.status === "failed") return { label: "불러오기 실패", tone: "danger" };
  if (source.type === "empty") return { label: "이력 없음", tone: "neutral" };
  return { label: source.status, tone: "info" };
}

export default function Home() {
  const [historyItems, setHistoryItems] = useState<AnalysisHistorySummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const selectedAnalysisIdRef = useRef<string | null>(null);
  const [selectionLabel, setSelectionLabel] = useState<"최근 원인 분석" | "선택한 원인 분석">("최근 원인 분석");
  const [selectionNotice, setSelectionNotice] = useState("");
  const [overview, setOverview] = useState<AnalysisOverviewResponse | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState("");
  const [overviewRequestVersion, setOverviewRequestVersion] = useState(0);

  const persistSelection = useCallback((analysisId: string | null, selectedByUser: boolean) => {
    selectedAnalysisIdRef.current = analysisId;
    setSelectedAnalysisId(analysisId);
    setSelectionLabel(selectedByUser ? "선택한 원인 분석" : "최근 원인 분석");
    const url = new URL(window.location.href);
    if (analysisId) {
      url.searchParams.set("analysis_id", analysisId);
      try {
        window.sessionStorage.setItem("last_overview_analysis_id", analysisId);
      } catch {
        // The URL remains the durable source when browser storage is unavailable.
      }
    } else {
      url.searchParams.delete("analysis_id");
      try {
        window.sessionStorage.removeItem("last_overview_analysis_id");
      } catch {
        // Storage availability must not remove the dashboard layout.
      }
    }
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }, []);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await getAnalysisHistory({ limit: 200, sort: "newest" });
      const urlAnalysisId = new URLSearchParams(window.location.search).get("analysis_id");
      let resolvedItems = response.items;
      if (urlAnalysisId && !resolvedItems.some((item) => item.analysis_id === urlAnalysisId)) {
        const targeted = await getAnalysisHistory({ search: urlAnalysisId, limit: 1, sort: "newest" });
        const exact = targeted.items.find((item) => item.analysis_id === urlAnalysisId);
        if (exact) resolvedItems = [exact, ...resolvedItems];
      }
      setHistoryItems(resolvedItems);
      let sessionAnalysisId: string | null = null;
      try {
        sessionAnalysisId = window.sessionStorage.getItem("last_overview_analysis_id");
      } catch {
        sessionAnalysisId = null;
      }
      const decision = resolveOverviewSelection(resolvedItems, {
        urlAnalysisId,
        currentAnalysisId: selectedAnalysisIdRef.current,
        sessionAnalysisId,
      });
      if (decision.invalidUrlAnalysisId) {
        setSelectionNotice(
          `선택한 분석 이력(${decision.invalidUrlAnalysisId})을 찾을 수 없어 최근 완료 이력으로 전환했습니다.`,
        );
      } else {
        setSelectionNotice("");
      }
      const selectedByUser = Boolean(
        !decision.invalidUrlAnalysisId
        && decision.analysisId
        && (decision.analysisId === urlAnalysisId || decision.analysisId === sessionAnalysisId),
      );
      persistSelection(decision.analysisId, selectedByUser);
    } catch (requestError) {
      setHistoryError(
        requestError instanceof Error
          ? requestError.message
          : "원인 분석 이력 목록을 불러오지 못했습니다.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, [persistSelection]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadHistory(); }, 0);
    return () => window.clearTimeout(timer);
  }, [loadHistory]);

  const selectedHistory = useMemo(
    () => historyItems.find((item) => item.analysis_id === selectedAnalysisId) ?? null,
    [historyItems, selectedAnalysisId],
  );

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timer = window.setTimeout(() => {
      if (!selectedAnalysisId) {
        setOverview(null);
        setOverviewError("");
        setOverviewLoading(false);
        return;
      }
      setOverview(null);
      setOverviewError("");
      setOverviewLoading(true);
      getDashboardOverview(selectedAnalysisId, controller.signal)
        .then((response) => {
          if (active) setOverview(response);
        })
        .catch((requestError: unknown) => {
          if (!active || (requestError instanceof DOMException && requestError.name === "AbortError")) return;
          setOverviewError(
            requestError instanceof Error
              ? requestError.message
              : "선택한 원인 분석 결과를 불러오지 못했습니다.",
          );
        })
        .finally(() => {
          if (active) setOverviewLoading(false);
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [selectedAnalysisId, overviewRequestVersion]);

  const retryOverview = useCallback(() => {
    if (selectedAnalysisIdRef.current) setOverviewRequestVersion((value) => value + 1);
  }, []);

  const display = overview
    ?? (selectedHistory ? overviewFromHistory(selectedHistory) : createEmptyOverview());
  const source = display.source;
  const sourcePresentation = sourceStatus(source);
  const analysisBase = source.analysis_id
    ? `/root-cause?analysis_id=${encodeURIComponent(source.analysis_id)}`
    : null;
  const detailError = overviewError || (!selectedAnalysisId ? historyError : "");

  let dashboardState: DashboardState;
  if (historyLoading && historyItems.length === 0) dashboardState = "loading-history";
  else if (historyError && historyItems.length === 0) dashboardState = "api-error";
  else if (!selectedAnalysisId) dashboardState = "history-empty";
  else if (overviewLoading) dashboardState = "loading-analysis";
  else if (overviewError) dashboardState = "api-error";
  else if (source.status === "partial") dashboardState = "partial";
  else if (source.artifact_status === "missing") dashboardState = "artifact-missing";
  else if (source.artifact_status === "corrupted") dashboardState = "artifact-corrupted";
  else dashboardState = "ready";

  const kpis: OverviewKpi[] = [
    {
      label: "평균 예측 수율",
      value: display.summary.average_predicted_yield,
      unit: "%",
      detail: "선택한 분석 이력의 실제 평균값",
      tone: "normal",
    },
    {
      label: "위험 Lot",
      value: display.summary.risk_lot_count,
      detail: "Warning 또는 Critical Lot",
      tone: "warning",
    },
    {
      label: "Critical Wafer",
      value: display.summary.critical_count,
      detail: "Critical 기준에 해당하는 Wafer",
      tone: "danger",
    },
    {
      label: "Warning Wafer",
      value: display.summary.warning_count,
      detail: "Warning 기준에 해당하는 Wafer",
      tone: "warning",
    },
    {
      label: "모델 R²",
      value: display.model_metrics.r2,
      detail: "저장된 모델 평가 R²",
    },
    {
      label: "RMSE",
      value: display.model_metrics.rmse,
      detail: "저장된 모델 평가 RMSE",
    },
  ];
  const kpiLoading = dashboardState === "loading-history" || dashboardState === "loading-analysis";
  const kpiError = dashboardState === "api-error";

  const failureRateData = Object.entries(display.multi_y.failure_rates).map(([label, value]) => ({
    label,
    value,
    unit: "%",
  }));
  const failBitData = Object.entries(display.multi_y.fail_bit_counts).map(([label, value]) => ({ label, value }));
  const causeData = display.causes.top_features.map((item) => ({
    label: causeLabel(item),
    value: causeValue(item),
  }));
  const paretoData = display.pareto.map((item) => ({
    label: causeLabel(item),
    value: item.impact ?? item.score ?? null,
  }));
  const riskData: BarDatum[] = [
    { label: "Normal", value: display.summary.normal_count },
    { label: "Warning", value: display.summary.warning_count },
    { label: "Critical", value: display.summary.critical_count },
  ];
  const riskAvailable = riskData.some((item) => item.value !== null);

  const summaryState = sectionState(overviewLoading, detailError, display.availability.summary, source.artifact_status);
  const multiYState = sectionState(overviewLoading, detailError, display.availability.multi_y, source.artifact_status);
  const causesState = sectionState(overviewLoading, detailError, display.availability.causes, source.artifact_status);
  const paretoState = sectionState(overviewLoading, detailError, display.availability.pareto, source.artifact_status);
  const riskState = sectionState(overviewLoading, detailError, riskAvailable, source.artifact_status);
  const riskLotState = sectionState(overviewLoading, detailError, display.availability.risk_lots, source.artifact_status);
  const riskWaferState = sectionState(overviewLoading, detailError, display.availability.risk_wafers, source.artifact_status);
  const relationshipState = sectionState(overviewLoading, detailError, display.availability.relationships, source.artifact_status);

  return (
    <div className="appShell">
      <Sidebar />
      <div className="contentShell">
        <Header />
        <main id="overview" className="mainContent overviewDashboard" data-dashboard-state={dashboardState}>
          <section className="overviewToolbar" aria-label="원인 분석 이력 선택">
            <div>
              <span className="sectionLabel">Analysis source</span>
              <h2>{selectionLabel}</h2>
              <p>원인 분석 이력을 선택하면 페이지를 새로고침하지 않고 모든 지표가 갱신됩니다.</p>
            </div>
            <AnalysisHistorySelector
              items={historyItems}
              selectedAnalysisId={selectedAnalysisId}
              loading={historyLoading}
              error={historyError}
              onSelect={(analysisId) => {
                setSelectionNotice("");
                persistSelection(analysisId, true);
              }}
              onRetry={() => void loadHistory()}
            />
          </section>

          {selectionNotice && <section className="overviewNotice" role="status">{selectionNotice}</section>}
          {historyError && historyItems.length === 0 && (
            <section className="overviewErrorBanner" role="alert">
              <span><strong>분석 이력 목록을 불러오지 못했습니다.</strong><small>{historyError}</small></span>
              <button className="button secondary" type="button" onClick={() => void loadHistory()}>목록 다시 시도</button>
            </section>
          )}
          {overviewError && (
            <section className="overviewErrorBanner" role="alert">
              <span><strong>선택한 분석 결과를 불러오지 못했습니다.</strong><small>{overviewError}</small></span>
              <button className="button secondary" type="button" onClick={retryOverview}>분석 다시 시도</button>
            </section>
          )}

          <section className="riskSummaryBanner" aria-labelledby="overview-source-title">
            <div className="riskSummaryMessage">
              <span className="riskSummaryIcon" aria-hidden="true"><span /></span>
              <div>
                <div className="overviewSourceBadges">
                  <StatusBadge label={selectionLabel} tone={source.type === "analysis" ? "info" : "neutral"} dot={false} />
                  <StatusBadge label={sourcePresentation.label} tone={sourcePresentation.tone} dot={false} />
                </div>
                <h2 id="overview-source-title">{source.source_filename ?? "원인 분석 이력을 선택해 주세요"}</h2>
                <p>
                  {source.model_name ?? source.model_id ?? "모델 정보 없음"}
                  {source.created_at ? ` · ${formatDate(source.completed_at ?? source.created_at)}` : ""}
                </p>
                <p className="overviewSourceId">{source.analysis_id ?? "analysis_id 없음"}</p>
                {analysisBase ? (
                  <a className="overviewSourceLink" href={analysisBase}>원인 분석 상세 보기 →</a>
                ) : (
                  <a className="overviewSourceLink" href="/root-cause">원인 분석 실행 →</a>
                )}
              </div>
            </div>
            <dl className="riskSummaryMetrics">
              <div><dt>Wafer</dt><dd>{formatNumber(display.summary.wafer_count, 0)}</dd></div>
              <div><dt>Lot</dt><dd>{formatNumber(display.summary.lot_count, 0)}</dd></div>
              <div><dt>저장 상태</dt><dd>{sourcePresentation.label}</dd></div>
            </dl>
          </section>

          <section className="overviewKpiSection" aria-labelledby="overview-kpi-title">
            <div className="sectionHeading">
              <div><span className="sectionLabel">Selected analysis</span><h2 id="overview-kpi-title">핵심 지표</h2></div>
              <p>0은 실제 값이며, 제공되지 않은 값만 -로 표시합니다.</p>
            </div>
            <div className="cardGrid kpiGrid">
              {kpis.map((kpi) => (
                <StatusCard
                  key={kpi.label}
                  label={kpi.label}
                  value={kpiLoading ? "…" : formatNumber(kpi.value, kpi.label.includes("Wafer") || kpi.label.includes("Lot") ? 0 : 3)}
                  unit={!kpiLoading && kpi.value !== null ? kpi.unit : undefined}
                  detail={kpiLoading
                    ? "분석 결과를 불러오는 중"
                    : kpiError
                      ? "분석 결과를 불러오지 못함"
                      : kpi.value === null
                        ? "현재 분석 이력에서 제공되지 않음"
                        : kpi.detail}
                  tone={kpi.tone}
                />
              ))}
            </div>
          </section>

          <section className="overviewChartGrid" aria-label="분석 시각화">
            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Yield series</span><h2>수율 분포 또는 추이</h2></div></div>
              <p className="chartDescription">Snapshot에 실제 시계열 또는 분포가 있을 때만 표시합니다.</p>
              <DashboardSectionState
                state={summaryState === "loading" || summaryState === "error" || summaryState === "unavailable" ? summaryState : "empty"}
                message={summaryState === "loading"
                  ? "수율 시각화를 준비하는 중입니다."
                  : summaryState === "error"
                    ? detailError
                    : "현재 분석 이력에는 수율 분포·시계열 데이터가 없습니다."}
                onRetry={summaryState === "error" ? retryOverview : undefined}
              />
            </article>

            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Y1–Y5</span><h2>Failure Rate</h2></div></div>
              <p className="chartDescription">실제 Multi-Y 분석에 저장된 Target별 평균입니다.</p>
              <MetricBars
                data={failureRateData}
                state={failureRateData.length ? multiYState : multiYState === "ready" ? "empty" : multiYState}
                message={sectionMessage(multiYState, "현재 분석 이력에는 Y1~Y5 Failure Rate가 없습니다.", detailError, source.artifact_status)}
                onRetry={retryOverview}
              />
            </article>

            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Y6–Y10</span><h2>Fail Bit Count</h2></div></div>
              <p className="chartDescription">실제 Multi-Y 분석에 저장된 Target별 평균입니다.</p>
              <MetricBars
                data={failBitData}
                state={failBitData.length ? multiYState : multiYState === "ready" ? "empty" : multiYState}
                message={sectionMessage(multiYState, "현재 분석 이력에는 Y6~Y10 Fail Bit Count가 없습니다.", detailError, source.artifact_status)}
                onRetry={retryOverview}
              />
            </article>

            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Root causes</span><h2>주요 원인 영향도</h2></div></div>
              <p className="chartDescription">저장된 SHAP 또는 영향도 순위를 표시합니다.</p>
              <MetricBars
                data={causeData}
                state={causeData.length ? causesState : causesState === "ready" ? "empty" : causesState}
                message={sectionMessage(causesState, "현재 분석 이력에는 주요 원인 데이터가 없습니다.", detailError, source.artifact_status)}
                onRetry={retryOverview}
              />
            </article>

            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Pareto</span><h2>수율 악화 Pareto 분석</h2></div></div>
              <p className="chartDescription">저장된 실제 영향도 기준 상위 항목입니다.</p>
              <MetricBars
                data={paretoData}
                state={paretoData.length ? paretoState : paretoState === "ready" ? "empty" : paretoState}
                message={sectionMessage(paretoState, "이 분석 이력에는 Pareto 데이터가 없습니다.", detailError, source.artifact_status)}
                onRetry={retryOverview}
              />
            </article>

            <article className="surfaceCard overviewChartCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Risk distribution</span><h2>위험도 분포</h2></div></div>
              <p className="chartDescription">Normal, Warning, Critical의 실제 Wafer 수입니다.</p>
              <MetricBars
                data={riskData}
                state={riskState}
                message={sectionMessage(riskState, "현재 분석 이력에는 위험도 분포가 없습니다.", detailError, source.artifact_status)}
                onRetry={retryOverview}
              />
            </article>
          </section>

          <section className="surfaceCard overviewSectionCard" aria-labelledby="cause-summary-title">
            <div className="sectionHeading compact"><div><span className="sectionLabel">Cause summary</span><h2 id="cause-summary-title">주요 원인 결과</h2></div></div>
            {causesState === "loading" || causesState === "error" || causesState === "unavailable" ? (
              <DashboardSectionState
                state={causesState}
                message={sectionMessage(causesState, "주요 원인 결과가 없습니다.", detailError, source.artifact_status)}
                onRetry={causesState === "error" ? retryOverview : undefined}
              />
            ) : (
              <dl className="overviewCauseSummary">
                <div><dt>Top Failure Target</dt><dd>{analysisBase && display.causes.top_failure_target ? <a href={`${analysisBase}&tab=targets&target=${encodeURIComponent(display.causes.top_failure_target)}`}>{display.causes.top_failure_target}</a> : display.causes.top_failure_target ?? "-"}</dd></div>
                <div><dt>Top Feature</dt><dd>{causeLabel(display.causes.top_features[0])}</dd></div>
                <div><dt>Top Step</dt><dd>{causeLabel(display.causes.top_steps[0])}</dd></div>
                <div><dt>Top Equipment</dt><dd>{causeLabel(display.causes.top_equipment[0])}</dd></div>
                <div><dt>Top Chamber</dt><dd>{causeLabel(display.causes.top_chambers[0])}</dd></div>
              </dl>
            )}
          </section>

          <section className="dashboardAnalysisGrid overviewRiskGrid" aria-label="위험 Lot 및 Wafer">
            <article className="surfaceCard overviewSectionCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Risk lots</span><h2>위험 Lot Top 5</h2></div></div>
              <div className="tableWrap historyTableScroll"><table><thead><tr><th>Lot</th><th>Wafer</th><th>평균 수율</th><th>Critical</th><th>Warning</th></tr></thead><tbody>
                {riskLotState === "ready" ? display.risk_lots.slice(0, 5).map((item, index) => (
                  <tr key={item.lot_id ?? String(index)}>
                    <td>{analysisBase && item.lot_id ? <a className="overviewSourceLink" href={`${analysisBase}&tab=lot&lot_id=${encodeURIComponent(item.lot_id)}`}>{item.lot_id}</a> : item.lot_id ?? "-"}</td>
                    <td>{formatNumber(item.wafer_count, 0)}</td>
                    <td>{formatNumber(item.average_predicted_yield)}{item.average_predicted_yield === null ? "" : "%"}</td>
                    <td>{formatNumber(item.danger_count, 0)}</td>
                    <td>{formatNumber(item.warning_count, 0)}</td>
                  </tr>
                )) : <tr><td colSpan={5}><DashboardSectionState compact state={riskLotState} message={sectionMessage(riskLotState, "현재 분석 이력에 위험 Lot 데이터가 없습니다.", detailError, source.artifact_status)} onRetry={riskLotState === "error" ? retryOverview : undefined} /></td></tr>}
              </tbody></table></div>
            </article>

            <article className="surfaceCard overviewSectionCard">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Risk wafers</span><h2>위험 Wafer Top 5</h2></div></div>
              <div className="tableWrap historyTableScroll"><table><thead><tr><th>Wafer</th><th>예측 수율</th><th>위험도</th><th>주요 원인</th></tr></thead><tbody>
                {riskWaferState === "ready" ? display.risk_wafers.slice(0, 5).map((item, index) => {
                  const identifier = item.identifier === null ? null : String(item.identifier);
                  return (
                    <tr key={identifier ?? String(index)}>
                      <td>{analysisBase && identifier ? <a className="overviewSourceLink" href={`${analysisBase}&tab=wafer&wafer_id=${encodeURIComponent(identifier)}`}>{identifier}</a> : identifier ?? "-"}</td>
                      <td>{formatNumber(item.predicted_value ?? item.prediction)}{item.predicted_value === null && item.prediction === null ? "" : "%"}</td>
                      <td>{item.risk_level ?? "-"}</td>
                      <td>{item.top_harmful_features.slice(0, 2).join(", ") || item.top_step || "-"}</td>
                    </tr>
                  );
                }) : <tr><td colSpan={4}><DashboardSectionState compact state={riskWaferState} message={sectionMessage(riskWaferState, "현재 분석 이력에 위험 Wafer 데이터가 없습니다.", detailError, source.artifact_status)} onRetry={riskWaferState === "error" ? retryOverview : undefined} /></td></tr>}
              </tbody></table></div>
            </article>
          </section>

          <section className="surfaceCard overviewSectionCard" aria-labelledby="relationship-title">
            <div className="sectionHeading compact">
              <div><span className="sectionLabel">Relationships</span><h2 id="relationship-title">관계·통계 요약</h2></div>
              {analysisBase && <a className="overviewSourceLink" href={`${analysisBase}&tab=relationships`}>전체 관계 보기 →</a>}
            </div>
            <div className="tableWrap historyTableScroll"><table><thead><tr><th>변수 관계</th><th>Pearson</th><th>Spearman</th><th>p-value</th><th>FDR</th><th>Effect Size</th><th>Sample</th><th>방향</th></tr></thead><tbody>
              {relationshipState === "ready" ? display.relationships.map((item, index) => {
                const relation = item.relation
                  ?? [item.response, item.defect, item.equipment].filter(Boolean).join(" → ")
                  ?? item.feature
                  ?? "-";
                const pValue = item.pearson_p_value ?? item.spearman_p_value;
                const fdr = item.pearson_fdr_p_value ?? item.spearman_fdr_p_value;
                return (
                  <tr key={`${relation}-${index}`}>
                    <td>{item.feature && item.target ? `${item.feature} → ${item.target}` : relation || "-"}</td>
                    <td>{formatNumber(item.pearson, 4)}</td>
                    <td>{formatNumber(item.spearman, 4)}</td>
                    <td>{formatNumber(pValue, 4)}</td>
                    <td>{formatNumber(fdr, 4)}</td>
                    <td>{formatNumber(item.effect_size ?? item.path_score, 4)}</td>
                    <td>{formatNumber(item.valid_count, 0)}</td>
                    <td>{item.direction ?? item.interpretation ?? "-"}</td>
                  </tr>
                );
              }) : <tr><td colSpan={8}><DashboardSectionState compact state={relationshipState} message={sectionMessage(relationshipState, "현재 분석 이력에는 관계·통계 결과가 없습니다.", detailError, source.artifact_status)} onRetry={relationshipState === "error" ? retryOverview : undefined} /></td></tr>}
            </tbody></table></div>
          </section>

          <section className="surfaceCard overviewSectionCard" aria-labelledby="warning-title">
            <div className="sectionHeading compact"><div><span className="sectionLabel">Warnings</span><h2 id="warning-title">분석 참고 사항</h2></div></div>
            {overviewLoading ? (
              <DashboardSectionState state="loading" message="참고 사항을 불러오는 중입니다." />
            ) : overviewError ? (
              <DashboardSectionState state="error" message={overviewError} onRetry={retryOverview} />
            ) : display.warnings.length ? (
              <div className="warningScrollList">{display.warnings.map((warning) => <p className="warningMessage" key={warning}>{warning}</p>)}</div>
            ) : (
              <DashboardSectionState state="empty" message="현재 분석 이력에 추가 경고가 없습니다." />
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
