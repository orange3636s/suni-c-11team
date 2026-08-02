"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Header from "@/components/Header";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import ModelSelector from "@/components/models/ModelSelector";
import SelectedModelSummary from "@/components/models/SelectedModelSummary";
import OperationProgress from "@/components/OperationProgress";
import Sidebar from "@/components/Sidebar";
import useElapsedTime from "@/hooks/useElapsedTime";
import {
  analyzeRelationships,
  deleteAnalysisHistory,
  getAnalysisHistory,
  getAnalysisHistoryDetail,
} from "@/lib/api";
import type {
  AnalysisHistorySummary,
  AnalysisResult,
  ExplainOptions,
  ExplainResponse,
  ModelSummary,
  RelationshipAnalysisResponse,
  RelationshipFeature,
  RelationshipPath,
  ReportResponse,
  LocalContributionItem,
  NumericStatistic,
  CategoricalStatistic,
} from "@/types/data";

const DEFAULT_OPTIONS: ExplainOptions = {
  max_rows: 500,
  top_n: 10,
  per_wafer_top_n: 5,
};
const DEFAULT_THRESHOLDS = { warning_threshold: 90, danger_threshold: 85 };

type WaferSort =
  | "risk-desc"
  | "risk-asc"
  | "id-asc"
  | "id-desc"
  | "prediction-desc"
  | "prediction-asc";

type RankingGroup =
  | "all" | "r" | "d" | "config";
type WorkspaceTab = "target" | "lot" | "wafer" | "relationships" | "report";
const WORKSPACE_TABS: [WorkspaceTab, string][] = [
  ["target", "Target별 원인"],
  ["lot", "Lot별 원인"],
  ["wafer", "Wafer별 원인"],
  ["relationships", "관계·통계"],
  ["report", "분석 보고서"],
];

function normalizeWorkspaceTab(value: string | null): WorkspaceTab {
  if (value === "overview" || value === "targets") return "target";
  return WORKSPACE_TABS.some(([tab]) => tab === value)
    ? (value as WorkspaceTab)
    : "target";
}

function formatNumber(value: number): string {
  return value.toLocaleString("ko-KR", {
    maximumFractionDigits: 5,
  });
}

function formatDateTime(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ko-KR");
}

function formatFeatureLabel(value: string): string {
  const match = value.match(/^Step_?(\d+)_?(R|D|EQ)(?:_(.+))?$/i);
  if (!match) {
    return value.length > 24 ? `${value.slice(0, 22)}…` : value;
  }
  const [, step, rawType, parameter] = match;
  const typeLabel =
    rawType.toUpperCase() === "R"
      ? "Response"
      : rawType.toUpperCase() === "D"
        ? "Delta"
        : "Equipment";
  const suffix = parameter
    ? ` · ${parameter.replaceAll("_", " ")}`
    : "";
  const formatted = `Step ${step} · ${typeLabel}${suffix}`;
  return formatted.length > 30 ? `${formatted.slice(0, 28)}…` : formatted;
}

function riskLabel(risk: string | null): string {
  if (risk === "danger") return "위험";
  if (risk === "warning") return "주의";
  if (risk === "normal") return "정상";
  return "-";
}

function firstNonEmpty<T>(...values: Array<T[] | null | undefined>): T[] {
  return values.find((items) => Array.isArray(items) && items.length > 0) ?? [];
}

function formatContributionDirection(direction: string | null | undefined): string {
  const labels: Record<string, string> = {
    yield_down: "수율 악화",
    yield_up: "수율 개선",
    defect_up: "불량률 증가",
    defect_down: "불량률 감소",
    fail_rate_up: "불량률 증가",
    fail_rate_down: "불량률 감소",
    count_up: "Fail Bit Count 증가",
    count_down: "Fail Bit Count 감소",
    model_contribution: "모델 기여",
    positive: "양의 연관",
    negative: "음의 연관",
    neutral: "중립",
    insufficient: "근거 부족",
  };
  return direction ? (labels[direction] ?? direction) : "-";
}

function relationshipStrengthLabel(value: string | null | undefined): string {
  const labels: Record<string, string> = {
    strong: "강함",
    moderate: "보통",
    weak: "약함",
    very_weak: "매우 약함",
    insufficient: "계산 불가",
  };
  return value ? (labels[value] ?? value) : "-";
}

function paretoTitle(target: string): string {
  if (target === "Y") return "수율 악화 Pareto 분석";
  if (["Y1", "Y2", "Y3", "Y4", "Y5"].includes(target)) return `${target} 불량률 증가 Pareto 분석`;
  return `${target} Fail Bit Count 증가 Pareto 분석`;
}

function paretoImpactLabel(target: string): string {
  if (target === "Y") return "수율 악화";
  if (["Y1", "Y2", "Y3", "Y4", "Y5"].includes(target)) return "불량률 증가";
  return "Fail Bit Count 증가";
}

export default function RootCausePage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [relationships, setRelationships] =
    useState<RelationshipAnalysisResponse | null>(null);
  const [rankingGroup, setRankingGroup] =
    useState<RankingGroup>("all");
  const [selectedTarget, setSelectedTarget] = useState("Y");
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>(() => {
    if (typeof window === "undefined") return "target";
    return normalizeWorkspaceTab(
      new URLSearchParams(window.location.search).get("tab"),
    );
  });
  const [selectedPath, setSelectedPath] = useState(0);
  const [selectedWafer, setSelectedWafer] = useState(0);
  const [waferSearch, setWaferSearch] = useState("");
  const [waferSort, setWaferSort] = useState<WaferSort>("risk-desc");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [analysisRunKey, setAnalysisRunKey] = useState(0);
  const waferRowRefs = useRef(new Map<number, HTMLTableRowElement>());
  const [activeView, setActiveView] = useState<"new" | "history">("new");
  const [historyItems, setHistoryItems] = useState<AnalysisHistorySummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [restoredHistory, setRestoredHistory] = useState<AnalysisHistorySummary | null>(null);
  const [restoredAnalysis, setRestoredAnalysis] = useState<AnalysisResult | null>(null);
  const [restoredReport, setRestoredReport] = useState<ReportResponse | null>(null);
  const { formattedElapsed: formattedAnalysisElapsed } = useElapsedTime({
    running: loading,
    resetKey: analysisRunKey,
  });

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const analysisId = params.get("analysis_id") ?? sessionStorage.getItem("last_analysis_id");
    if (!analysisId) return;
    void getAnalysisHistoryDetail(analysisId).then((detail) => {
      setRestoredHistory(detail.metadata);
      setModelId(detail.metadata.model_id ?? "");
      const restoredResponse = detail.artifact?.response ?? null;
      const analysisSnapshot = restoredResponse?.analysis_result
        ?? detail.artifact?.analysis_result
        ?? null;
      const reportSnapshot = restoredResponse?.report_snapshot
        ?? detail.artifact?.report_snapshot
        ?? null;
      if (!restoredResponse && !analysisSnapshot && !reportSnapshot) {
        setError("저장된 분석 이력의 상세 Snapshot이 없거나 손상되었습니다.");
        return;
      }
      setRelationships(restoredResponse);
      setResult(restoredResponse?.explanation ?? null);
      setRestoredAnalysis(analysisSnapshot);
      setRestoredReport(reportSnapshot);
      setSelectedTarget(
        restoredResponse?.target
          ?? analysisSnapshot?.target.name
          ?? reportSnapshot?.model.target
          ?? detail.metadata.default_target
          ?? "Y",
      );
      const waferId = params.get("wafer_id");
      if (waferId && restoredResponse) {
        const waferIndex = restoredResponse.explanation.wafer_explanations.findIndex(
          (wafer) => String(wafer.identifier) === waferId,
        );
        if (waferIndex >= 0) setSelectedWafer(waferIndex);
      }
      sessionStorage.setItem("last_analysis_id", analysisId);
    }).catch((requestError) => {
      setError(requestError instanceof Error ? requestError.message : "분석 이력을 복원하지 못했습니다.");
      sessionStorage.removeItem("last_analysis_id");
    });
  }, []);

  async function loadHistory() {
    setHistoryLoading(true); setHistoryError("");
    try { setHistoryItems((await getAnalysisHistory()).items); }
    catch (requestError) { setHistoryError(requestError instanceof Error ? requestError.message : "분석 이력을 불러오지 못했습니다."); }
    finally { setHistoryLoading(false); }
  }

  async function openHistory(item: AnalysisHistorySummary) {
    try {
      const detail = await getAnalysisHistoryDetail(item.analysis_id);
      const restoredResponse = detail.artifact?.response ?? null;
      const analysisSnapshot = restoredResponse?.analysis_result
        ?? detail.artifact?.analysis_result
        ?? null;
      const reportSnapshot = restoredResponse?.report_snapshot
        ?? detail.artifact?.report_snapshot
        ?? null;
      if (!restoredResponse && !analysisSnapshot && !reportSnapshot) {
        throw new Error("저장된 분석 Snapshot을 읽을 수 없습니다.");
      }
      setRelationships(restoredResponse);
      setResult(restoredResponse?.explanation ?? null);
      setRestoredAnalysis(analysisSnapshot);
      setRestoredReport(reportSnapshot);
      setSelectedPath(0);
      setSelectedWafer(0);
      setSelectedTarget(
        restoredResponse?.target
          ?? analysisSnapshot?.target.name
          ?? reportSnapshot?.model.target
          ?? detail.metadata.default_target
          ?? "Y",
      );
      setRestoredHistory(detail.metadata); setModelId(detail.metadata.model_id ?? ""); setActiveView("new");
      const url = new URL(window.location.href); url.searchParams.set("analysis_id", item.analysis_id); window.history.replaceState({}, "", url);
      sessionStorage.setItem("last_analysis_id", item.analysis_id);
    } catch (requestError) { setHistoryError(requestError instanceof Error ? requestError.message : "분석 이력을 열지 못했습니다."); }
  }

  async function removeHistory(item: AnalysisHistorySummary) {
    if (!window.confirm("SHAP, Pareto, 통계 및 분석 보고서 Snapshot이 삭제됩니다. 연결된 예측은 유지됩니다.")) return;
    try {
      await deleteAnalysisHistory(item.analysis_id);
      if (restoredHistory?.analysis_id === item.analysis_id) {
        setRelationships(null);
        setResult(null);
        setRestoredAnalysis(null);
        setRestoredReport(null);
        setRestoredHistory(null);
        sessionStorage.removeItem("last_analysis_id");
      }
      await loadHistory();
    } catch (requestError) {
      setHistoryError(requestError instanceof Error ? requestError.message : "분석 이력을 삭제하지 못했습니다.");
    }
  }

  function selectWorkspaceTab(tab: WorkspaceTab) {
    setWorkspaceTab(tab);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", tab);
    window.history.replaceState({}, "", url);
  }

  const selectedModel = models.find((model) => model.model_id === modelId);
  const responseTarget = relationships?.target
    ?? restoredAnalysis?.target.name
    ?? restoredReport?.model.target;
  const targetOptions = useMemo(() => {
    const available = selectedModel?.available_targets ?? [];
    const values = available.length
      ? available
      : selectedModel?.target
        ? [selectedModel.target]
        : responseTarget
          ? [responseTarget]
          : ["Y"];
    return [...new Set(values)].sort((left, right) =>
      left.localeCompare(right, "en", { numeric: true }),
    );
  }, [responseTarget, selectedModel]);
  const activeTarget = targetOptions.includes(selectedTarget)
    ? selectedTarget
    : targetOptions.includes("Y")
      ? "Y"
      : targetOptions[0] ?? "Y";
  const commonAnalysis = relationships?.analysis_result ?? restoredAnalysis;
  const reportSnapshot = relationships?.report_snapshot ?? restoredReport;
  const historyLotAnalysis = commonAnalysis?.lot_analysis
    ?? reportSnapshot?.lot_analysis
    ?? null;
  const historyRelationshipPaths = firstNonEmpty(
    commonAnalysis?.relationships,
    reportSnapshot?.relationship_analysis?.relationship_paths,
  );
  const commonStatistics = commonAnalysis?.statistics;
  const reportStatistics = reportSnapshot?.relationship_analysis?.statistics
    ?? reportSnapshot?.target_analysis?.statistics;
  const historyStatistics = commonStatistics
    && (commonStatistics.numeric.length || commonStatistics.categorical.length)
    ? commonStatistics
    : reportStatistics;

  const selected = result?.wafer_explanations[selectedWafer];
  const waferLotIds = useMemo(() => Array.from(new Set(
    (result?.wafer_explanations ?? []).map((wafer) => lotIdOf(wafer.identifier, wafer.lot_id)).filter(Boolean),
  )).sort((left, right) => left.localeCompare(right, "ko", { numeric: true })), [result]);
  const selectedLotId = selected ? lotIdOf(selected.identifier, selected.lot_id) : (waferLotIds[0] ?? "");
  const selectedMultiY = commonAnalysis?.multi_y.wafer_results.find(
    (row) => String(row.identifier) === String(selected?.identifier),
  );
  const shapRankingData = useMemo(() => {
    const serverRows = relationships?.rankings?.shap?.[rankingGroup];
    if (serverRows?.length) return serverRows;
    const canonical = (value: string) => {
      const normalized = value.trim().toLowerCase().replaceAll("_", " ");
      if (normalized === "eq") return "equipment";
      if (["missing", "indicator", "observed", "measurement pattern"].includes(normalized)) return "measurement";
      return normalized;
    };
    const legacyRows = (result?.global_importance ?? [])
      .filter((item) => rankingGroup === "all" || (rankingGroup === "config"
        ? ["config", "model", "equipment", "chamber"].includes(canonical(item.parameter_type))
        : canonical(item.parameter_type) === rankingGroup))
      .map((item): RelationshipFeature => ({
        rank: item.rank,
        feature: item.feature,
        display_name: item.feature,
        step: Number(item.step.replace("Step", "")) || null,
        group: item.parameter_type,
        ranking_basis: "Mean absolute SHAP value",
        score: item.mean_abs_shap,
        signed_association: null,
        direction: item.direction,
        valid_count: null,
        missing_count: null,
        missing_rate: null,
        category_count: null,
        is_categorical: ["Config", "Model", "Equipment", "Chamber"].includes(item.parameter_type),
      }));
    if (rankingGroup !== "config") return legacyRows;
    const byStep = new Map<string, RelationshipFeature>();
    legacyRows.forEach((row) => {
      const stepLabel = row.step === null ? "unknown" : `Step${row.step}`;
      const feature = row.step === null ? row.feature : `${stepLabel}_Config`;
      const existing = byStep.get(feature);
      if (existing) {
        existing.score = (existing.score ?? 0) + (row.score ?? 0);
        return;
      }
      byStep.set(feature, {
        ...row,
        feature,
        display_name: row.step === null ? row.display_name : `Step ${row.step} · Config`,
        group: "Config",
        direction: "model_contribution",
        is_categorical: true,
      });
    });
    return [...byStep.values()].sort(
      (left, right) => (right.score ?? 0) - (left.score ?? 0),
    );
  }, [rankingGroup, relationships, result]);
  const correlationRankingData = useMemo(() => {
    const serverRows = relationships?.rankings?.correlation?.[rankingGroup] ?? [];
    if (!relationships || (rankingGroup !== "all" && rankingGroup !== "config")) {
      return serverRows;
    }
    const categoricalRows: RelationshipFeature[] = relationships.statistics.categorical
      .filter((row) => row.target === relationships.target)
      .map((row, index) => {
        const evidence = [row.welch_anova, row.anova, row.kruskal]
          .find((test) => test.p_value !== null) ?? row.anova;
        const stepMatch = row.feature.match(/^Step(\d+)_/i);
        return {
          rank: index + 1,
          feature: row.feature,
          display_name: formatFeatureLabel(row.feature),
          step: stepMatch ? Number(stepMatch[1]) : null,
          group: "Config",
          ranking_basis: "Categorical effect size",
          score: row.effect_size,
          signed_association: null,
          direction: "neutral",
          valid_count: row.valid_count,
          missing_count: row.excluded_count,
          missing_rate: 1 - row.coverage,
          category_count: row.category_count,
          is_categorical: true,
          p_value: evidence.p_value,
          fdr_p_value: evidence.fdr_p_value,
          effect_size: row.effect_size,
        };
      });
    if (rankingGroup === "config") {
      return serverRows.length ? serverRows : categoricalRows;
    }
    const combined = new Map<string, RelationshipFeature>();
    [...serverRows, ...categoricalRows].forEach((row) => {
      if (!combined.has(row.feature)) combined.set(row.feature, row);
    });
    return [...combined.values()].sort(
      (left, right) => (right.score ?? -1) - (left.score ?? -1),
    );
  }, [rankingGroup, relationships]);
  const sortedWafers = useMemo(() => {
    const wafers = result?.wafer_explanations.map((wafer, index) => ({
      wafer,
      originalIndex: index,
    })) ?? [];
    const riskScore = (risk: string | null) =>
      risk === "danger" ? 2 : risk === "warning" ? 1 : 0;

    return wafers.sort((left, right) => {
      if (waferSort === "risk-desc") {
        return (
          riskScore(right.wafer.risk_level) -
          riskScore(left.wafer.risk_level)
        );
      }
      if (waferSort === "risk-asc") {
        return (
          riskScore(left.wafer.risk_level) -
          riskScore(right.wafer.risk_level)
        );
      }
      if (waferSort === "prediction-desc") {
        return right.wafer.prediction - left.wafer.prediction;
      }
      if (waferSort === "prediction-asc") {
        return left.wafer.prediction - right.wafer.prediction;
      }

      const comparison = String(left.wafer.identifier).localeCompare(
        String(right.wafer.identifier),
        "ko",
        { numeric: true, sensitivity: "base" },
      );
      return waferSort === "id-desc" ? -comparison : comparison;
    });
  }, [result, waferSort]);

  function handleFile(selectedFile?: File) {
    setResult(null);
    setRelationships(null);
    setRestoredAnalysis(null);
    setRestoredReport(null);
    setError("");
    if (selectedFile && !selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    setFile(selectedFile ?? null);
  }

  function handleWaferSearch(value: string) {
    setWaferSearch(value);
    const query = value.trim().toLocaleLowerCase("ko");
    if (!query) return;

    const match = sortedWafers.find(({ wafer }) =>
      String(wafer.identifier).toLocaleLowerCase("ko").includes(query),
    );
    if (!match) return;

    selectWafer(match.originalIndex);
  }

  function selectWafer(index: number, scroll = true) {
    setSelectedWafer(index);
    const identifier = result?.wafer_explanations[index]?.identifier;
    if (identifier !== undefined) {
      setWaferSearch(String(identifier));
      localStorage.setItem("root-cause-recent-wafer", String(identifier));
    }
    if (!scroll) return;
    requestAnimationFrame(() => {
      waferRowRefs.current.get(index)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
  }

  function handleQuickWaferSelect(value: string) {
    setWaferSearch(value);
    const matchIndex =
      result?.wafer_explanations.findIndex(
        (wafer) => String(wafer.identifier) === value,
      ) ?? -1;
    if (matchIndex >= 0) selectWafer(matchIndex);
  }

  function handleLotSelect(lotId: string) {
    const matchIndex = result?.wafer_explanations.findIndex(
      (wafer) => lotIdOf(wafer.identifier, wafer.lot_id) === lotId,
    ) ?? -1;
    if (matchIndex >= 0) selectWafer(matchIndex, false);
  }

  async function runAnalysis() {
    if (!file || !modelId || loading) return;
    setAnalysisRunKey((current) => current + 1);
    setLoading(true);
    setError("");
    try {
      const linkedPredictionId = new URLSearchParams(window.location.search).get("prediction_id");
      const response = await analyzeRelationships(
        file,
        modelId,
        DEFAULT_OPTIONS,
        "pearson",
        "wafer_observed_only",
        DEFAULT_THRESHOLDS,
        activeTarget,
        linkedPredictionId,
      );
      setRelationships(response);
      setResult(response.explanation);
      setRestoredAnalysis(null);
      setRestoredReport(null);
      setSelectedTarget(response.target ?? response.explanation.model.target);
      const recentWafer = localStorage.getItem("root-cause-recent-wafer");
      const recentIndex = response.explanation.wafer_explanations.findIndex(
        (wafer) => String(wafer.identifier) === recentWafer,
      );
      setSelectedWafer(recentIndex >= 0 ? recentIndex : 0);
      setSelectedPath(0);
      setWaferSearch("");
      setWaferSort("risk-desc");
      setRestoredHistory(null);
      if (response.analysis_id) {
        const url = new URL(window.location.href); url.searchParams.set("analysis_id", response.analysis_id); window.history.replaceState({}, "", url);
        sessionStorage.setItem("last_analysis_id", response.analysis_id);
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "불량 원인 분석 중 오류가 발생했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar activeItem="불량 원인 분석" />
      <div className="contentShell">
        <Header />
        <main className="mainContent rootCausePage">
          <section className="intro">
            <div>
              <span className="eyebrow">Explainable AI</span>
              <h1>불량 원인 분석</h1>
              <p>
                저장된 모델의 예측을 공정 단계·파라미터·Wafer별로
                분해합니다.
              </p>
            </div>
          </section>

          <div className="trainingViewTabs" role="tablist" aria-label="불량 원인 분석 보기">
            <button className={activeView === "new" ? "active" : ""} type="button" role="tab" aria-selected={activeView === "new"} onClick={() => setActiveView("new")}>새 분석</button>
            <button className={activeView === "history" ? "active" : ""} type="button" role="tab" aria-selected={activeView === "history"} onClick={() => { setActiveView("history"); void loadHistory(); }}>분석 이력</button>
          </div>

          {activeView === "new" ? <>
          <section className="resultCard analysisControls">
            <div className="fieldGroup analysisFileField">
              <label htmlFor="analysis-file">분석 CSV</label>
              <CsvUploadPanel
                id="analysis-file"
                file={file}
                onFileSelect={handleFile}
                disabled={loading}
                compact
              />
            </div>
            <div className="fieldGroup">
              <label htmlFor="analysis-model">학습 모델</label>
              <ModelSelector
                value={modelId}
                disabled={loading}
                onValueChange={(nextModelId) => {
                  setModelId(nextModelId);
                  setResult(null);
                  setRelationships(null);
                  setRestoredAnalysis(null);
                  setRestoredReport(null);
                  setRestoredHistory(null);
                  sessionStorage.removeItem("last_analysis_id");
                  const url = new URL(window.location.href);
                  url.searchParams.delete("analysis_id");
                  window.history.replaceState({}, "", url);
                  const nextModel = models.find((model) => model.model_id === nextModelId);
                  const nextTargets = nextModel?.available_targets?.length
                    ? nextModel.available_targets
                    : nextModel?.target
                      ? [nextModel.target]
                      : ["Y"];
                  setSelectedTarget(nextTargets.includes("Y") ? "Y" : nextTargets[0]);
                  setError("");
                }}
                onModelsChange={(nextModels) => setModels(nextModels)}
                ariaLabel="불량 원인 분석 모델 선택"
              />
            </div>
            <div className="fieldGroup analysisTargetField">
              <label htmlFor="analysis-target">분석 Target</label>
              <select
                id="analysis-target"
                value={activeTarget}
                disabled={loading || !modelId}
                onChange={(event) => setSelectedTarget(event.target.value)}
              >
                {targetOptions.map((target) => (
                  <option key={target} value={target}>{target}</option>
                ))}
              </select>
            </div>
            <SelectedModelSummary model={selectedModel} />
            <div className="uploadActions">
              <button
                className="button primary"
                type="button"
                disabled={!file || !modelId || !selectedModel || loading}
                data-loading={loading}
                aria-busy={loading}
                onClick={() => void runAnalysis()}
              >
                {loading ? (
                  <OperationProgress
                    message="불량 원인 분석 중…"
                    timeLabel="추론 시간"
                    formattedElapsed={formattedAnalysisElapsed}
                  />
                ) : "불량 원인 분석"}
              </button>
            </div>
            {!models.length && (
              <p className="emptyMessage">
                호환 가능한 학습 모델이 없습니다. 먼저 <a href="/training">모델 학습</a>을 진행해 주세요.
              </p>
            )}
            {modelId && !selectedModel && !loading && (
              <p className="emptyMessage">이 이력의 모델은 현재 모델 목록에 없습니다. 저장된 분석은 조회할 수 있지만 새 분석은 실행할 수 없습니다.</p>
            )}
            {error && <p className="errorMessage">{error}</p>}
          </section>

            {restoredHistory && <div className="historyRestoreBanner"><div><strong>저장된 불량 원인 분석</strong><span>{formatDateTime(restoredHistory.created_at)} · {restoredHistory.source_filename}</span></div><div className="historyRowActions">{restoredHistory.prediction_id ? <a className="button secondary" href={`/prediction?prediction_id=${encodeURIComponent(restoredHistory.prediction_id)}`}>예측 결과 보기</a> : <span>연결된 예측 없음</span>}<button className="button secondary" type="button" onClick={() => { setResult(null); setRelationships(null); setRestoredAnalysis(null); setRestoredReport(null); setRestoredHistory(null); const url = new URL(window.location.href); url.searchParams.delete("analysis_id"); window.history.replaceState({}, "", url); }}>새 분석</button></div></div>}

          <nav className="workspaceTabs" aria-label="불량 원인 분석 워크스페이스">
            {WORKSPACE_TABS.map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={workspaceTab === value ? "active" : ""}
                aria-current={workspaceTab === value ? "page" : undefined}
                onClick={() => selectWorkspaceTab(value)}
              >
                {label}
              </button>
            ))}
          </nav>

          {!result && !commonAnalysis && !reportSnapshot && (
            <section className="resultCard">
              <p className="emptyMessage">
                {workspaceTab === "report"
                  ? "불량 원인 분석 이력을 선택하거나 새 분석을 실행해 주세요."
                  : "예측 결과 또는 분석할 데이터를 먼저 선택해 주세요."}
              </p>
            </section>
          )}

          {!result && (commonAnalysis || reportSnapshot) && (
            <>
              {workspaceTab === "target" && <>
                <AnalysisOverview analysis={commonAnalysis} />
                <HistoryTargetSnapshot snapshot={reportSnapshot?.target_analysis ?? null} group={rankingGroup} />
              </>}
              {workspaceTab === "lot" && (commonAnalysis || historyLotAnalysis) && <LotAnalysis
                analysis={commonAnalysis}
                explanation={null}
                lotAnalysis={historyLotAnalysis}
              />}
              {workspaceTab === "lot" && !commonAnalysis && !historyLotAnalysis && <section className="resultCard"><p className="emptyMessage">이 Legacy 분석 이력에는 Lot Snapshot이 없습니다.</p></section>}
              {workspaceTab === "wafer" && commonAnalysis && <HistoryWaferAnalysis analysis={commonAnalysis} />}
              {workspaceTab === "wafer" && !commonAnalysis && <section className="resultCard"><p className="emptyMessage">이 Legacy 분석 이력에는 Wafer Snapshot이 없습니다.</p></section>}
              {workspaceTab === "relationships" && (historyRelationshipPaths.length > 0 || historyStatistics) && <>
                <PathSection paths={historyRelationshipPaths} selectedIndex={selectedPath} onSelect={setSelectedPath} confidenceCriteria={{}} />
                <StatisticsSection statistics={historyStatistics} />
              </>}
              {workspaceTab === "relationships" && historyRelationshipPaths.length === 0 && !historyStatistics && <section className="resultCard"><p className="emptyMessage">이 Legacy 분석 이력에는 관계·통계 Snapshot이 없습니다.</p></section>}
              {workspaceTab === "report" && <AnalysisReport report={reportSnapshot} analysis={commonAnalysis} relationships={relationships} />}
            </>
          )}

          {result && (
            <>
              {workspaceTab === "target" && <>
              <section className="resultCard">
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">분석 요약</span>
                    <h2>
                      {result.analysis_summary.analyzed_rows.toLocaleString()}
                      개 행 설명
                    </h2>
                  </div>
                  <p>
                    {result.model.model_name} ·{" "}
                    {result.analysis_summary.explanation_method}
                    {result.analysis_summary.is_fallback
                      ? " (모델 독립 대체 방식)"
                      : ""}
                  </p>
                </div>
                <p className="analysisDisclaimer">
                  본 분석은 머신러닝 모델의 예측 기여도를 설명합니다. SHAP
                  값이 높은 feature가 실제 불량의 직접 원인임을 확정하는 것은
                  아닙니다.
                </p>
              </section>
              <AnalysisOverview analysis={commonAnalysis} />
              </>}

              {relationships && (
                <>
                  {workspaceTab === "target" && <div className="relationshipToolbar targetFeatureToolbar">
                    <div><span className="sectionLabel">Feature Group</span><strong>분석 범위</strong></div>
                    <SegmentedControl
                      options={[["all", "전체"], ["r", "R"], ["d", "D"], ["config", "Config"]]}
                      value={rankingGroup}
                      onChange={(value) => setRankingGroup(value as RankingGroup)}
                    />
                  </div>}
                  {workspaceTab === "target" && <section className="resultCard relationshipSection">
                    <div className="sectionHeading compact">
                      <div>
                        <span className="sectionLabel">Model Contribution · SHAP</span>
                        <h2>{relationships.target} 모델 기여도</h2>
                      </div>
                      <p>
                        Ranking basis:{" "}
                        {shapRankingData[0]?.ranking_basis ?? "데이터 없음"}
                      </p>
                    </div>
                    {shapRankingData.length ? (
                      <RankingChart data={shapRankingData} />
                    ) : (
                      <p className="emptyMessage">
                        실제 모델 기여도 데이터가 없습니다.
                      </p>
                    )}
                  </section>}

                  {workspaceTab === "target" && <section className="resultCard relationshipSection">
                    <div className="sectionHeading compact"><div><span className="sectionLabel">Observed Relationship</span><h2>{relationships.target} 데이터 관계 강도</h2></div><p>Pearson · Spearman · p-value · FDR</p></div>
                    {correlationRankingData.length ? <><RankingChart data={correlationRankingData} /><StatisticalEvidenceTable data={correlationRankingData} /></> : <p className="emptyMessage">선택 Group에서 계산 가능한 관계 데이터가 없습니다.</p>}
                    <p className="analysisDisclaimer">관계 강도와 통계 검정은 관측 데이터의 연관성을 나타내며, SHAP 모델 기여도 또는 인과관계와 같지 않습니다.</p>
                  </section>}

                  {workspaceTab === "target" && <TargetRelationshipStatistics statistics={relationships.statistics} target={relationships.target} group={rankingGroup} />}

                  {workspaceTab === "target" && <ParetoSection analysis={relationships} rows={shapRankingData} />}

                  {workspaceTab === "lot" && commonAnalysis && <LotAnalysis analysis={commonAnalysis} explanation={result} lotAnalysis={relationships.lot_analysis} />}

                  {workspaceTab === "relationships" && <PathSection
                    paths={relationships.relationship_paths}
                    selectedIndex={selectedPath}
                    onSelect={setSelectedPath}
                    confidenceCriteria={relationships.confidence_criteria}
                  />}
                  {workspaceTab === "relationships" && <StatisticsSection statistics={relationships.statistics} />}
                </>
              )}

              {workspaceTab === "wafer" && <section className="resultCard">
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">개별 설명</span>
                    <h2>Wafer별 기여 변수</h2>
                  </div>
                  <div className="waferSelectorGroup"><label className="compactField">
                    <span>Lot</span>
                    <select value={selectedLotId} onChange={(event) => handleLotSelect(event.target.value)}>
                      {waferLotIds.map((lot) => <option key={lot} value={lot}>{lot}</option>)}
                    </select>
                  </label><label className="waferQuickSelector">
                    <span>LOT_WAFER_ID</span>
                    <input
                      type="search"
                      list="root-cause-wafer-options"
                      value={waferSearch || String(selected?.identifier ?? "")}
                      onChange={(event) =>
                        handleQuickWaferSelect(event.target.value)
                      }
                      placeholder="Wafer 검색"
                    />
                    <datalist id="root-cause-wafer-options">
                      {result.wafer_explanations.map((wafer, index) => (
                        <option
                          key={`${String(wafer.identifier)}-${index}`}
                          value={String(wafer.identifier)}
                        />
                      ))}
                    </datalist>
                  </label></div>
                </div>
                {selected && <WaferDetailAnalysis selected={selected} selectedMultiY={selectedMultiY} wafers={result.wafer_explanations} target={result.model.target} />}
                <div className="waferListTools">
                  <label className="waferSearch">
                    <span className="visuallyHidden">
                      LOT_WAFER_ID 검색
                    </span>
                    <input
                      type="search"
                      placeholder="LOT_WAFER_ID 검색"
                      value={waferSearch}
                      onChange={(event) =>
                        handleWaferSearch(event.target.value)
                      }
                    />
                  </label>
                  <label className="waferSort">
                    <span className="visuallyHidden">LOT 목록 정렬</span>
                    <select
                      value={waferSort}
                      onChange={(event) =>
                        setWaferSort(event.target.value as WaferSort)
                      }
                    >
                      <option value="risk-desc">위험도 높은 순</option>
                      <option value="risk-asc">위험도 낮은 순</option>
                      <option value="id-asc">LOT_WAFER_ID 오름차순</option>
                      <option value="id-desc">LOT_WAFER_ID 내림차순</option>
                      <option value="prediction-desc">예측값 높은 순</option>
                      <option value="prediction-asc">예측값 낮은 순</option>
                    </select>
                  </label>
                </div>
                <div
                  className="tableWrapper waferListScroll"
                  aria-label={`${result.identifier_column} 목록`}
                >
                  <table className="dataTable">
                    <thead>
                      <tr>
                        <th>{result.identifier_column}</th>
                        <th>예측값</th>
                        <th>위험도</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedWafers.map(({ wafer, originalIndex }) => (
                        <tr
                          ref={(node) => {
                            if (node) {
                              waferRowRefs.current.set(originalIndex, node);
                            } else {
                              waferRowRefs.current.delete(originalIndex);
                            }
                          }}
                          className={
                            selectedWafer === originalIndex ? "selectedRow" : ""
                          }
                          key={`${String(wafer.identifier)}-${originalIndex}`}
                          onClick={() => selectWafer(originalIndex, false)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              selectWafer(originalIndex, false);
                            }
                          }}
                          tabIndex={0}
                          role="button"
                          aria-label={`${String(wafer.identifier)} 상세 기여도 ${
                            selectedWafer === originalIndex
                              ? "선택됨"
                              : "보기"
                          }`}
                        >
                          <td>{String(wafer.identifier)}</td>
                          <td>{formatNumber(wafer.prediction)}</td>
                          <td>
                            <span
                              className={`riskBadge ${
                                wafer.risk_level ?? "normal"
                              }`}
                            >
                              {riskLabel(wafer.risk_level)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>}

              {workspaceTab === "report" && (
                <AnalysisReport report={reportSnapshot} analysis={commonAnalysis} relationships={relationships} />
              )}
            </>
          )}
          </> : (
            <section className="resultCard historyCard" aria-labelledby="analysis-history-title">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Analysis History</span><h2 id="analysis-history-title">분석 이력</h2></div><button className="button secondary" type="button" onClick={() => void loadHistory()}>새로고침</button></div>
              {historyLoading ? <p className="emptyMessage">분석 이력을 불러오는 중입니다.</p> : historyError ? <div className="retryMessage"><p className="errorMessage">{historyError}</p><button className="button secondary" type="button" onClick={() => void loadHistory()}>다시 시도</button></div> : !historyItems.length ? <p className="emptyMessage">저장된 불량 원인 분석 이력이 없습니다.</p> : <div className="tableWrap historyTableScroll"><table><thead><tr><th>생성 시각</th><th>파일명</th><th>모델</th><th>연결 예측</th><th>Wafer</th><th>Lot</th><th>Target</th><th>상태</th><th>작업</th></tr></thead><tbody>{historyItems.map((item) => <tr key={item.analysis_id}><td>{formatDateTime(item.created_at)}</td><td>{item.source_filename ?? "데이터 없음"}</td><td>{item.model_name_snapshot ?? item.model_id ?? "삭제된 모델"}</td><td>{item.prediction_id ?? "연결 없음"}</td><td>{item.row_count ?? "데이터 없음"}</td><td>{item.lot_count ?? "데이터 없음"}</td><td>{item.default_target ?? "데이터 없음"}</td><td>{item.status}</td><td><div className="historyRowActions"><button className="button secondary" type="button" onClick={() => void openHistory(item)}>상세 보기</button>{item.prediction_id && <a className="button secondary" href={`/prediction?prediction_id=${encodeURIComponent(item.prediction_id)}`}>예측 보기</a>}<button className="button danger" type="button" onClick={() => void removeHistory(item)}>삭제</button></div></td></tr>)}</tbody></table></div>}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

function contributionTitle(target: string, harmful: boolean): string {
  if (target === "Y") return harmful ? "수율 악화" : "수율 개선";
  if (["Y1", "Y2", "Y3", "Y4", "Y5"].includes(target)) {
    return harmful ? "불량률 증가" : "불량률 감소";
  }
  return harmful ? "Fail Bit Count 증가" : "Fail Bit Count 감소";
}

function optionalNumber(value: unknown, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function recordNumber(value: unknown, key: string): number | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const candidate = Reflect.get(value, key);
  return typeof candidate === "number" && Number.isFinite(candidate)
    ? candidate
    : undefined;
}

function FailureBreakdown({ analysis }: { analysis: AnalysisResult }) {
  const rates = Object.entries(analysis.multi_y.failure_rate_averages);
  const counts = Object.entries(analysis.multi_y.fail_bit_count_averages);
  const finiteRates = rates.flatMap(([, value]) => typeof value === "number" && Number.isFinite(value) ? [value] : []);
  const finiteCounts = counts.flatMap(([, value]) => typeof value === "number" && Number.isFinite(value) ? [value] : []);
  const maxRate = Math.max(...finiteRates, 1);
  const maxCount = Math.max(...finiteCounts, 1);
  return (
    <div className="failureBreakdownGrid">
      <article className="surfaceCard">
        <h3>Y1~Y5 Failure Breakdown</h3>
        {rates.length ? rates.map(([target, value]) => (
          <div className="failureBarRow" key={target}>
            <span>{target}</span>
            <div><i style={{ width: `${typeof value === "number" && Number.isFinite(value) ? (value / maxRate) * 100 : 0}%` }} /></div>
            <strong>{typeof value === "number" && Number.isFinite(value) ? `${optionalNumber(value)}%` : "-"}</strong>
          </div>
        )) : <p className="emptyMessage">Y1~Y5 모델이 모두 준비되지 않아 Derived Y를 계산할 수 없습니다.</p>}
      </article>
      <article className="surfaceCard">
        <h3>Y6~Y10 Fail Bit Count</h3>
        {counts.length ? counts.map(([target, value]) => (
          <div className="failureBarRow count" key={target}>
            <span>{target}</span>
            <div><i style={{ width: `${typeof value === "number" && Number.isFinite(value) ? (value / maxCount) * 100 : 0}%` }} /></div>
            <strong>{optionalNumber(value, 0)}</strong>
          </div>
        )) : <p className="emptyMessage">사용 가능한 Fail Bit Count 모델이 없습니다.</p>}
      </article>
    </div>
  );
}

function AnalysisOverview({ analysis }: { analysis: AnalysisResult | null }) {
  if (!analysis) return <section className="resultCard"><p className="emptyMessage">저장된 이력에 Target KPI Snapshot이 없습니다.</p></section>;
  const testValue = analysis.metrics.test;
  const evaluationValue = analysis.metrics.evaluation_summary;
  return (
    <>
      <section className="resultCard">
        <div className="sectionHeading compact"><div><span className="sectionLabel">Target KPI</span><h2>분석 정보</h2></div><p>{formatDateTime(analysis.created_at)}</p></div>
        <div className="analysisContextGrid">
          <div><span>모델</span><strong>{analysis.model.model_name ?? "-"}</strong></div>
          <div><span>Schema</span><strong>{analysis.model.schema_version ?? "-"}</strong></div>
          <div><span>Target</span><strong>{analysis.target.label}</strong></div>
          <div><span>분석 행</span><strong>{analysis.dataset.row_count.toLocaleString()}</strong></div>
          <div><span>Test R²</span><strong>{optionalNumber(recordNumber(testValue, "r2"))}</strong></div>
          <div><span>RMSE</span><strong>{optionalNumber(recordNumber(testValue, "rmse"))}</strong></div>
          <div><span>MAE</span><strong>{optionalNumber(recordNumber(testValue, "mae"))}</strong></div>
          <div><span>Generalization Gap</span><strong>{optionalNumber(recordNumber(evaluationValue, "generalization_gap"))}</strong></div>
        </div>
      </section>
      <section className="resultCard">
        <div className="sectionHeading compact"><div><span className="sectionLabel">Hybrid Multi-Y</span><h2>공통 예측 요약</h2></div></div>
        <div className="analysisContextGrid">
          <div><span>Direct Y</span><strong>{optionalNumber(analysis.multi_y.average_direct_y)}%</strong></div>
          <div><span>Derived Y</span><strong>{optionalNumber(analysis.multi_y.average_derived_y)}%</strong></div>
          <div><span>Ensemble Y</span><strong>{optionalNumber(analysis.multi_y.average_ensemble_y)}%</strong></div>
          <div><span>Ensemble α</span><strong>{optionalNumber(analysis.multi_y.ensemble_weight)}</strong></div>
          <div><span>Critical Wafer</span><strong>{analysis.risk.critical_count}</strong></div>
          <div><span>Warning Wafer</span><strong>{analysis.risk.warning_count}</strong></div>
        </div>
        <FailureBreakdown analysis={analysis} />
      </section>
      <section className="resultCard">
        <div className="sectionHeading compact"><div><span className="sectionLabel">Measurement Quality</span><h2>측정 품질</h2></div></div>
        <div className="analysisContextGrid">
          <div><span>R Coverage</span><strong>{optionalNumber(analysis.data_quality.r_measurement_coverage * 100)}%</strong></div>
          <div><span>D Coverage</span><strong>{optionalNumber(analysis.data_quality.d_measurement_coverage * 100)}%</strong></div>
          <div><span>Config Completeness</span><strong>{optionalNumber(analysis.data_quality.config_completeness_rate * 100)}%</strong></div>
          <div><span>Target Consistency</span><strong>{analysis.data_quality.target_consistency_rate === null ? "-" : `${optionalNumber(analysis.data_quality.target_consistency_rate * 100)}%`}</strong></div>
        </div>
      </section>
    </>
  );
}

function HistoryTargetSnapshot({
  snapshot,
  group,
}: {
  snapshot: NonNullable<ReportResponse["target_analysis"]> | null;
  group: RankingGroup;
}) {
  if (!snapshot) {
    return <section className="resultCard"><p className="emptyMessage">이 Legacy 분석 이력에는 Target SHAP·관계 Snapshot이 없습니다.</p></section>;
  }
  const target = snapshot.target
    ?? snapshot.statistics.numeric[0]?.target
    ?? snapshot.statistics.categorical[0]?.target
    ?? "Target";
  const shapRows = snapshot.rankings.shap[group] ?? [];
  const relationshipRows = snapshot.rankings.correlation[group] ?? [];
  return <>
    <section className="resultCard relationshipSection"><div className="sectionHeading compact"><div><span className="sectionLabel">Saved Model Contribution · SHAP</span><h2>{target} 모델 기여도</h2></div><p>Analysis History Snapshot</p></div>{shapRows.length ? <RankingChart data={shapRows} /> : <p className="emptyMessage">선택 Group의 SHAP Snapshot이 없습니다.</p>}</section>
    <section className="resultCard relationshipSection"><div className="sectionHeading compact"><div><span className="sectionLabel">Saved Observed Relationship</span><h2>{target} 데이터 관계 강도</h2></div><p>Analysis History Snapshot</p></div>{relationshipRows.length ? <><RankingChart data={relationshipRows} /><StatisticalEvidenceTable data={relationshipRows} /></> : <p className="emptyMessage">선택 Group의 관계 Ranking Snapshot이 없습니다.</p>}</section>
    <TargetRelationshipStatistics statistics={snapshot.statistics} target={target} group={group} />
    <HistoryParetoSnapshot pareto={snapshot.pareto} target={target} />
  </>;
}

function HistoryParetoSnapshot({
  pareto,
  target,
}: {
  pareto: RelationshipAnalysisResponse["pareto"];
  target: string;
}) {
  if (!pareto.features.length) {
    return <section className="resultCard relationshipSection"><div className="sectionHeading compact"><div><span className="sectionLabel">Saved Cumulative Impact</span><h2>{paretoTitle(target)}</h2></div></div><p className="emptyMessage">저장된 Pareto Snapshot이 없습니다.</p></section>;
  }
  return <section className="resultCard relationshipSection">
    <div className="sectionHeading compact"><div><span className="sectionLabel">Saved Cumulative Impact</span><h2>{paretoTitle(target)}</h2></div><p>{pareto.ranking_basis || "Analysis History Snapshot"}</p></div>
    <div className="paretoSummary"><div><span>우선 검토 변수</span><strong>{pareto.required_feature_count}개</strong></div><div><span>누적 영향도</span><strong>{optionalNumber(pareto.cumulative_contribution * 100, 1)}%</strong></div><div><span>전체 변수</span><strong>{pareto.total_feature_count}개</strong></div></div>
    {pareto.features.length ? <div className="compactChart"><ResponsiveContainer width="100%" height={320}><ComposedChart data={pareto.features.slice(0, 20)} margin={{ top: 8, right: 28, bottom: 68, left: 8 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 6" vertical={false} /><XAxis dataKey="display_name" angle={-35} textAnchor="end" interval={0} height={82} axisLine={false} tickLine={false} /><YAxis yAxisId="impact" axisLine={false} tickLine={false} /><YAxis yAxisId="share" orientation="right" domain={[0, 1]} axisLine={false} tickLine={false} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} /><ReferenceLine yAxisId="share" y={0.8} stroke="var(--text-muted)" strokeDasharray="4 5" /><Bar yAxisId="impact" dataKey="impact" fill="#0072C6" radius={[6, 6, 0, 0]} /><Line yAxisId="share" dataKey="cumulative_share" stroke="var(--accent)" dot={false} strokeWidth={2} /></ComposedChart></ResponsiveContainer></div> : <p className="emptyMessage">저장된 Pareto Snapshot이 없습니다.</p>}
  </section>;
}

function HistoryWaferAnalysis({ analysis }: { analysis: AnalysisResult }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const wafers = analysis.wafer_explanations;
  const selected = wafers[selectedIndex] ?? wafers[0];
  if (!selected) {
    return <section className="resultCard"><p className="emptyMessage">이 Legacy 분석 이력에는 Wafer 기여도 Snapshot이 없습니다.</p></section>;
  }
  const resolvedIndex = wafers.indexOf(selected);
  const selectedMultiY = analysis.multi_y.wafer_results.find(
    (row) => String(row.identifier) === String(selected.identifier),
  );
  return <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Saved Wafer Explanation</span><h2>Wafer별 기여 변수</h2></div><label className="compactField"><span>Wafer</span><select value={resolvedIndex} onChange={(event) => setSelectedIndex(Number(event.target.value))}>{wafers.map((wafer, index) => <option key={`${String(wafer.identifier)}-${index}`} value={index}>{String(wafer.identifier)}</option>)}</select></label></div><WaferDetailAnalysis selected={selected} selectedMultiY={selectedMultiY} wafers={wafers} target={analysis.target.name} /></section>;
}

function downloadSnapshot(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function AnalysisReport({ report, analysis, relationships }: { report: ReportResponse | null; analysis: AnalysisResult | null; relationships: RelationshipAnalysisResponse | null }) {
  if (!report && !analysis) {
    return <section className="resultCard"><p className="emptyMessage">불량 원인 분석 이력을 선택하거나 새 분석을 실행해 주세요.</p></section>;
  }
  const analysisId = analysis?.analysis_id ?? report?.analysis_id ?? "analysis";
  const summary = report?.executive_summary;
  const reportTarget = report?.target_analysis;
  const reportRelationships = report?.relationship_analysis;
  const shapRows = firstNonEmpty(
    relationships?.rankings.shap.all,
    reportTarget?.rankings.shap.all,
  );
  const topFeatures = report?.top_features.length
    ? report.top_features.map((item) => ({
        feature: item.feature,
        value: item.mean_abs_shap,
      }))
    : shapRows.map((item) => ({ feature: item.feature, value: item.score }));
  const topSteps = report?.top_steps.length
    ? report.top_steps.map((item) => ({ step: item.step, value: item.mean_abs_shap }))
    : Array.from(shapRows.reduce((steps, item) => {
        const step = item.step === null ? null : `Step${item.step}`;
        if (step && typeof item.score === "number" && Number.isFinite(item.score)) {
          steps.set(step, (steps.get(step) ?? 0) + item.score);
        }
        return steps;
      }, new Map<string, number>())).map(([step, value]) => ({ step, value }));
  const parameterTypes = report?.parameter_type_summary ?? [];
  const riskWafers = firstNonEmpty(report?.top_risk_wafers, analysis?.risk_wafers);
  const lotSummary = firstNonEmpty(report?.lot_summary, analysis?.lot_summary);
  const numericStatistics = firstNonEmpty(
    relationships?.statistics.numeric,
    reportRelationships?.statistics.numeric,
    reportTarget?.statistics.numeric,
  );
  const categoricalStatistics = firstNonEmpty(
    relationships?.statistics.categorical,
    reportRelationships?.statistics.categorical,
    reportTarget?.statistics.categorical,
  );
  const lotCauses = firstNonEmpty(
    relationships?.lot_analysis?.lots,
    report?.lot_analysis?.lots,
  );
  const averagePrediction = summary?.average_predicted_yield ?? analysis?.multi_y.average_ensemble_y ?? analysis?.multi_y.average_direct_y;
  const failureAverages = analysis?.multi_y.failure_rate_averages ?? {};
  const lotFailureCounts = lotCauses.reduce((counts, lot) => {
    const target = lot.top_causes.failure_target ?? lot.top_failure_target;
    if (target) counts.set(target, (counts.get(target) ?? 0) + 1);
    return counts;
  }, new Map<string, number>());
  const topFailure = Object.entries(failureAverages)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]))
    .sort((left, right) => (right[1] ?? -Infinity) - (left[1] ?? -Infinity))[0]?.[0]
    ?? [...lotFailureCounts].sort((left, right) => right[1] - left[1])[0]?.[0]
    ?? "-";
  const reportWaferRows = riskWafers.length
    ? riskWafers.map((row) => ({
        identifier: row.identifier,
        prediction: row.predicted_value,
        risk_level: row.risk_level,
        cause: row.top_harmful_features[0] ?? null,
      }))
    : lotCauses.flatMap((lot) => lot.wafer_list)
        .filter((row) => row.risk_level === "danger" || row.risk_level === "warning")
        .sort((left, right) => (right.risk_level === "danger" ? 2 : 1) - (left.risk_level === "danger" ? 2 : 1))
        .slice(0, 20)
        .map((row) => ({
          identifier: row.identifier,
          prediction: row.prediction,
          risk_level: row.risk_level,
          cause: row.top_feature,
        }));
  const reportLotRows = lotSummary.length
    ? lotSummary.map((row) => ({
        lot_id: row.lot_id,
        wafer_count: row.wafer_count,
        average: row.average_predicted_yield,
        critical_count: row.danger_count,
        cause: lotCauses.find((lot) => lot.lot_id === row.lot_id)?.top_causes.feature
          ?? row.top_harmful_feature,
      }))
    : lotCauses.map((lot) => ({
        lot_id: lot.lot_id,
        wafer_count: lot.wafer_count,
        average: lot.average_predicted_value,
        critical_count: lot.critical_wafer_count,
        cause: lot.top_causes.feature,
      }));
  const hasExplicitConfig = parameterTypes.some((item) => item.parameter_type.toLowerCase() === "config");
  const contributionScores = parameterTypes.reduce((scores, item) => {
    const rawGroup = item.parameter_type.toLowerCase();
    const group = rawGroup === "r" ? "R"
      : rawGroup === "d" ? "D"
        : rawGroup === "config" || ["model", "equipment", "chamber", "eq"].includes(rawGroup) ? "Config"
          : null;
    if (!group || (group === "Config" && hasExplicitConfig && rawGroup !== "config")) return scores;
    scores.set(group, (scores.get(group) ?? 0) + item.mean_abs_shap);
    return scores;
  }, new Map<string, number>());
  if (!contributionScores.size) {
    const rankings = relationships?.rankings.shap ?? reportTarget?.rankings.shap;
    (["r", "d", "config"] as const).forEach((group) => {
      const score = (rankings?.[group] ?? []).reduce((sum, item) => sum + (item.score ?? 0), 0);
      if (score > 0) contributionScores.set(group === "config" ? "Config" : group.toUpperCase(), score);
    });
  }
  const contributionTotal = [...contributionScores.values()].reduce((sum, value) => sum + value, 0);
  const groupContributions = [...contributionScores].map(([group, value]) => ({
    group,
    ratio: contributionTotal > 0 ? value / contributionTotal : 0,
  }));
  function exportJson() {
    downloadSnapshot(
      `analysis_${analysisId}.json`,
      JSON.stringify({ analysis, report, relationships }, null, 2),
      "application/json",
    );
  }
  function exportHtml() {
    const body = document.getElementById("analysis-report")?.outerHTML ?? "";
    const html = `<!doctype html><html lang="ko"><meta charset="utf-8"><title>분석 보고서</title><style>body{font-family:system-ui;padding:32px;color:#1d1d1f}section{margin:20px 0;padding:20px;border:1px solid #ddd;border-radius:16px}table{width:100%;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><body>${body}</body></html>`;
    downloadSnapshot(`analysis_report_${analysisId}.html`, html, "text/html;charset=utf-8");
  }
  return (
    <div id="analysis-report" className="operationalReport">
      <section className="resultCard reportHero">
        <div><span className="sectionLabel">Analysis Report</span><h2>불량 원인 분석 보고서</h2><p>{analysis?.dataset.filename ?? report?.filename ?? "-"} · {analysis?.model.model_name ?? report?.model.model_name ?? "-"} · {analysis?.target.label ?? report?.model.target ?? "-"}</p></div>
        <div className="reportExportActions">
          <button className="button secondary" type="button" onClick={exportJson}>JSON</button>
          <button className="button secondary" type="button" onClick={exportHtml}>HTML</button>
          <button className="button primary" type="button" onClick={() => window.print()}>Print / Save PDF</button>
        </div>
      </section>
      <section className="resultCard">
        <div className="sectionHeading compact"><div><span className="sectionLabel">분석 정보</span><h2>핵심 요약</h2></div><p>{formatDateTime(report?.created_at ?? analysis?.created_at)}</p></div>
        <div className="reportKpiGrid improved">
          <div><span>Wafer 수</span><strong>{summary?.total_wafers ?? analysis?.dataset.row_count ?? "-"}</strong></div>
          <div><span>Lot 수</span><strong>{reportLotRows.length || "-"}</strong></div>
          <div><span>평균 예측</span><strong>{optionalNumber(averagePrediction)}</strong></div>
          <div><span>Critical</span><strong>{summary?.danger_count ?? analysis?.risk.critical_count ?? "-"}</strong></div>
          <div><span>Warning</span><strong>{summary?.warning_count ?? analysis?.risk.warning_count ?? "-"}</strong></div>
          <div><span>주요 Failure Target</span><strong>{topFailure}</strong></div>
          <div><span>주요 Feature</span><strong>{topFeatures[0]?.feature ?? "-"}</strong></div>
          <div><span>주요 Config</span><strong>{lotCauses.find((lot) => lot.top_causes.config)?.top_causes.config ?? "-"}</strong></div>
        </div>
        {analysis && <FailureBreakdown analysis={analysis} />}
      </section>
      <section className="reportThreeColumn">
        <article className="resultCard"><h3>Target별 SHAP Top</h3>{topFeatures.length ? <ol>{topFeatures.slice(0, 10).map((item) => <li key={item.feature}><span>{item.feature}</span><strong>{optionalNumber(item.value, 4)}</strong></li>)}</ol> : <p className="emptyMessage">저장된 SHAP 요약이 없습니다.</p>}</article>
        <article className="resultCard"><h3>주요 Step</h3>{topSteps.length ? <ol>{topSteps.slice(0, 10).map((item) => <li key={item.step}><span>{item.step}</span><strong>{optionalNumber(item.value, 4)}</strong></li>)}</ol> : <p className="emptyMessage">저장된 Step 요약이 없습니다.</p>}</article>
        <article className="resultCard"><h3>R/D/Config 기여</h3>{groupContributions.length ? <ol>{groupContributions.map((item) => <li key={item.group}><span>{item.group}</span><strong>{optionalNumber(item.ratio * 100)}%</strong></li>)}</ol> : <p className="emptyMessage">저장된 Group 기여 요약이 없습니다.</p>}</article>
      </section>
      <section className="reportTwoColumn">
        <article className="resultCard reportScrollCard"><h3>Wafer별 위험 Top</h3>{reportWaferRows.length ? <div className="tableWrapper compactReportScroll"><table><thead><tr><th>ID</th><th>예측</th><th>위험도</th><th>주요 원인</th></tr></thead><tbody>{reportWaferRows.map((row) => <tr key={String(row.identifier)}><td>{String(row.identifier)}</td><td>{optionalNumber(row.prediction)}</td><td>{riskLabel(row.risk_level)}</td><td>{row.cause ?? "-"}</td></tr>)}</tbody></table></div> : <p className="emptyMessage">저장된 Wafer 원인 결과가 없습니다.</p>}</article>
        <article className="resultCard reportScrollCard"><h3>Lot별 위험 및 주요 원인</h3>{reportLotRows.length ? <div className="tableWrapper compactReportScroll"><table><thead><tr><th>LOT</th><th>Wafer</th><th>평균</th><th>Critical</th><th>원인</th></tr></thead><tbody>{reportLotRows.slice(0, 20).map((row) => <tr key={row.lot_id}><td>{row.lot_id}</td><td>{row.wafer_count ?? "-"}</td><td>{optionalNumber(row.average)}</td><td>{row.critical_count ?? "-"}</td><td>{row.cause ?? "-"}</td></tr>)}</tbody></table></div> : <p className="emptyMessage">저장된 Lot 결과가 없습니다.</p>}</article>
      </section>
      <section className="resultCard"><h3>관계·통계</h3>{numericStatistics.length || categoricalStatistics.length ? <><StatisticalNumericTable rows={numericStatistics.slice(0, 12)} /><StatisticalCategoricalTable rows={categoricalStatistics.slice(0, 12)} /></> : <p className="emptyMessage">저장된 관계·통계 결과가 없습니다.</p>}</section>
      <section className="resultCard"><h3>분석 방법과 제한사항</h3>{(report?.methodology_notes ?? analysis?.methodology.notes ?? []).length ? <ul>{(report?.methodology_notes ?? analysis?.methodology.notes ?? []).map((note) => <li key={note}>{note}</li>)}</ul> : <p className="emptyMessage">저장된 방법론 설명이 없습니다.</p>}</section>
    </div>
  );
}

function SegmentedControl({
  options,
  value,
  onChange,
}: {
  options: [string, string][];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="segmentedControl">
      {options.map(([option, label]) => (
        <button
          key={option}
          type="button"
          className={value === option ? "active" : ""}
          aria-pressed={value === option}
          onClick={() => onChange(option)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function groupColor(group: string): string {
  if (group === "R") return "var(--chart-primary)";
  if (group === "D") return "var(--warning)";
  return "var(--chart-secondary)";
}

function RankingChart({ data }: { data: RelationshipFeature[] }) {
  return (
    <div className="rankingLayout">
      <ContributionBarChart data={data} />
      <div className="tableWrapper rankingTable">
        <table className="dataTable">
          <thead>
            <tr>
              <th>순위</th><th>변수</th><th>유형</th><th>점수</th>
              <th>방향</th><th>유효 표본</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={item.feature}>
                <td>{index + 1}</td>
                <td title={item.feature}>{item.display_name}</td>
                <td>{item.group}</td>
                <td>{item.score === null ? "-" : formatNumber(item.score)}</td>
                <td>{formatContributionDirection(item.direction)}</td>
                <td>{item.valid_count?.toLocaleString() ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ContributionBarChart({ data }: { data: RelationshipFeature[] }) {
  return <div className="chartCanvas relationshipChart" role="img"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} layout="vertical" margin={{ top: 8, right: 18, bottom: 4, left: 28 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 5" horizontal={false} /><XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: "var(--chart-axis)" }} /><YAxis type="category" dataKey="display_name" width={138} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: "var(--chart-axis)" }} tickFormatter={formatFeatureLabel} /><Tooltip content={({ active, payload }) => { if (!active || !payload?.[0]) return null; const item = payload[0].payload as RelationshipFeature; return <div className="chartTooltip"><strong>{item.feature}</strong><span>유형: {item.group}</span><span>점수: {optionalNumber(item.score, 5)}</span><span>방향: {formatContributionDirection(item.direction)}</span><span>유효 표본: {item.valid_count?.toLocaleString() ?? "SHAP 기준"}</span></div>; }} /><Bar dataKey="score" radius={[0, 6, 6, 0]} animationDuration={220}>{data.map((item) => <Cell key={item.feature} fill={groupColor(item.group)} />)}</Bar></BarChart></ResponsiveContainer></div>;
}

function StatisticalEvidenceTable({ data }: { data: RelationshipFeature[] }) {
  if (!data.some((row) => row.p_value != null || row.effect_size != null)) return null;
  return <div className="tableWrapper compactEvidenceTable"><table><thead><tr><th>Feature</th><th>표본</th><th>연관</th><th>p-value</th><th>FDR</th><th>Effect Size</th></tr></thead><tbody>
    {data.slice(0, 12).map((row) => <tr key={row.feature}><td>{formatFeatureLabel(row.display_name)}</td><td>{row.valid_count ?? "-"}</td><td>{optionalNumber(row.signed_association, 4)}</td><td>{optionalNumber(row.p_value, 5)}</td><td>{optionalNumber(row.fdr_p_value, 5)}</td><td>{optionalNumber(row.effect_size, 4)}</td></tr>)}
  </tbody></table><p className="metricNotice">p-value와 FDR은 통계적 연관성의 참고 지표이며, Effect Size와 표본 수를 함께 검토해야 합니다. 인과관계를 의미하지 않습니다.</p></div>;
}

function StatisticsSection({ statistics }: {
  statistics: RelationshipAnalysisResponse["statistics"] | undefined;
}) {
  const numeric = statistics?.numeric ?? [];
  const categorical = statistics?.categorical ?? [];
  const [group, setGroup] = useState<RankingGroup>("all");
  const numericGroup = (row: NumericStatistic): RankingGroup | "other" => {
    const raw = row.group?.toLowerCase();
    if (raw === "r" || raw === "d") return raw;
    if (/^Step\d+_R\d+$/i.test(row.feature)) return "r";
    if (/^Step\d+_D\d+$/i.test(row.feature)) return "d";
    return "other";
  };
  const filteredNumeric = group === "all" ? numeric : numeric.filter((row) => numericGroup(row) === group);
  const filteredCategorical = group === "all" || group === "config" ? categorical : [];
  if (!numeric.length && !categorical.length) {
    return <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Statistical Analysis</span><h2>관계·통계 검정</h2></div></div><p className="emptyMessage">유효 표본 또는 지원되는 Feature가 없어 통계 검정을 계산할 수 없습니다.</p></section>;
  }
  return <section className="resultCard relationshipSection">
    <div className="sectionHeading compact"><div><span className="sectionLabel">Statistical Analysis</span><h2>관계·통계 검정</h2></div><p>실제 관측 표본 · 다중 검정 FDR 보정</p></div>
    <div className="relationshipToolbar"><SegmentedControl options={[["all", "전체"], ["r", "R"], ["d", "D"], ["config", "Config"]]} value={group} onChange={(value) => setGroup(value as RankingGroup)} /></div>
    {filteredNumeric.length > 0 && <><RelationshipStatisticChart numeric={filteredNumeric[0]} /><StatisticalNumericTable rows={filteredNumeric} /></>}
    {filteredCategorical.length > 0 && <><RelationshipStatisticChart categorical={filteredCategorical[0]} /><StatisticalCategoricalTable rows={filteredCategorical} /></>}
    {!filteredNumeric.length && !filteredCategorical.length && <p className="emptyMessage">선택 Group에서 계산 가능한 통계 결과가 없습니다.</p>}
    <p className="analysisDisclaimer">p-value와 FDR은 연관성의 참고 지표입니다. Effect Size, 표본 수, Coverage를 함께 검토해야 하며 인과관계를 의미하지 않습니다.</p>
  </section>;
}

function TargetRelationshipStatistics({ statistics, target, group }: { statistics: RelationshipAnalysisResponse["statistics"]; target: string; group: RankingGroup }) {
  const numeric = (statistics.numeric ?? []).filter((row) => {
    if (row.target !== target) return false;
    if (group === "all") return true;
    if (group === "r") return row.group?.toLowerCase() === "r" || /^Step\d+_R\d+$/i.test(row.feature);
    if (group === "d") return row.group?.toLowerCase() === "d" || /^Step\d+_D\d+$/i.test(row.feature);
    return false;
  });
  const categorical = group === "all" || group === "config"
    ? (statistics.categorical ?? []).filter((row) => row.target === target)
    : [];
  return <section className="resultCard relationshipSection"><div className="sectionHeading compact"><div><span className="sectionLabel">Relationship Evidence</span><h2>{target} 관계 강도 및 통계</h2></div><p>약한 관계와 0에 가까운 값도 포함</p></div>{numeric.length ? <><RelationshipStatisticChart numeric={numeric[0]} /><StatisticalNumericTable rows={numeric} /></> : null}{categorical.length ? <><RelationshipStatisticChart categorical={categorical[0]} /><StatisticalCategoricalTable rows={categorical} /></> : null}{!numeric.length && !categorical.length ? <p className="emptyMessage">유효 표본 또는 Config 범주가 없어 이 Group의 관계를 계산할 수 없습니다.</p> : null}</section>;
}

function RelationshipStatisticChart({ numeric, categorical }: { numeric?: NumericStatistic; categorical?: CategoricalStatistic }) {
  if (numeric) {
    const points = numeric.scatter_data ?? [];
    return <div className="statisticsChartBlock"><div className="sectionHeading compact subsectionHeading"><div><span className="sectionLabel">Scatter Plot</span><h3>{numeric.feature} vs {numeric.target}</h3></div><p>Pearson {optionalNumber(numeric.pearson, 4)} · Spearman {optionalNumber(numeric.spearman, 4)} · n={numeric.valid_count.toLocaleString()}</p></div>{points.length >= 2 ? <ScatterPanel title={`${numeric.feature} vs ${numeric.target}`} data={points} xLabel={numeric.feature} yLabel={numeric.target} xDescription="Feature value" yDescription="Target value" /> : <p className="emptyMessage">시각화할 유효 점이 부족합니다. 통계값은 전체 유효 표본 기준입니다.</p>}</div>;
  }
  if (categorical) {
    const groups = (categorical.category_summary ?? categorical.boxplot_data ?? []).flatMap((item): BoxSummary[] => {
      const { median, q1, q3, whisker_min: whiskerMin, whisker_max: whiskerMax } = item;
      if (
        typeof median !== "number" || !Number.isFinite(median)
        || typeof q1 !== "number" || !Number.isFinite(q1)
        || typeof q3 !== "number" || !Number.isFinite(q3)
        || typeof whiskerMin !== "number" || !Number.isFinite(whiskerMin)
        || typeof whiskerMax !== "number" || !Number.isFinite(whiskerMax)
      ) return [];
      return [{ label: item.category, count: item.count, median, q1, q3, whiskerMin, whiskerMax, outliers: item.outliers ?? [], outlierCount: item.outlier_count }];
    });
    return <div className="statisticsChartBlock"><div className="sectionHeading compact subsectionHeading"><div><span className="sectionLabel">Config Box Plot</span><h3>{categorical.feature} vs {categorical.target}</h3></div><p>Effect Size {optionalNumber(categorical.effect_size, 4)} · n={categorical.valid_count.toLocaleString()}</p></div>{groups.length ? <BoxPlotGraphic groups={groups} variable={categorical.feature} /> : <p className="emptyMessage">범주별 분포 Snapshot이 없습니다.</p>}</div>;
  }
  return null;
}

function StatisticalNumericTable({ rows }: { rows: NumericStatistic[] }) {
  return <div className="statisticsBlock"><h3>수치형 관계</h3><div className="tableWrapper"><table><thead><tr><th>관계</th><th>Feature</th><th>Target</th><th>Pearson</th><th>p-value</th><th>FDR</th><th>Spearman</th><th>p-value</th><th>FDR</th><th>Effect Size</th><th>방향</th><th>강도</th><th>표본</th><th>Coverage</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${row.relation}-${row.feature}-${index}`}><td>{row.relation}</td><td>{formatFeatureLabel(row.feature)}</td><td>{row.target}</td><td>{optionalNumber(row.pearson, 4)}</td><td>{optionalNumber(row.pearson_p_value, 5)}</td><td>{optionalNumber(row.pearson_fdr_p_value, 5)}</td><td>{optionalNumber(row.spearman, 4)}</td><td>{optionalNumber(row.spearman_p_value, 5)}</td><td>{optionalNumber(row.spearman_fdr_p_value, 5)}</td><td>{optionalNumber(row.effect_size, 4)}</td><td>{formatContributionDirection(row.direction)}</td><td>{relationshipStrengthLabel(row.strength)}</td><td>{row.valid_count.toLocaleString()}</td><td>{optionalNumber(row.coverage * 100, 1)}%</td></tr>)}</tbody></table></div></div>;
}

function StatisticalCategoricalTable({ rows }: { rows: CategoricalStatistic[] }) {
  return <div className="statisticsBlock"><h3>범주형 Config 관계</h3><div className="tableWrapper"><table><thead><tr><th>Feature</th><th>Target</th><th>Test</th><th>Statistic</th><th>p-value</th><th>FDR</th><th>Effect Size</th><th>범주</th><th>표본</th><th>Coverage</th></tr></thead><tbody>{rows.flatMap((row, index) => {
    const tests: Array<[string, CategoricalStatistic["anova"]]> = [
      ["ANOVA", row.anova],
      ["Welch ANOVA", row.welch_anova],
      ["Kruskal-Wallis", row.kruskal],
    ];
    return tests.map(([name, result]) => <tr key={`${row.feature}-${name}-${index}`}><td>{formatFeatureLabel(row.feature)}</td><td>{row.target}</td><td>{name}</td><td>{optionalNumber(result.statistic, 5)}</td><td>{optionalNumber(result.p_value, 5)}</td><td>{optionalNumber(result.fdr_p_value, 5)}</td><td>{optionalNumber(row.effect_size, 4)}</td><td>{row.category_count}</td><td>{row.valid_count.toLocaleString()}</td><td>{optionalNumber(row.coverage * 100, 1)}%</td></tr>);
  })}</tbody></table></div></div>;
}

function WaferDetailAnalysis({ selected, selectedMultiY, wafers, target }: {
  selected: ExplainResponse["wafer_explanations"][number];
  selectedMultiY: AnalysisResult["multi_y"]["wafer_results"][number] | undefined;
  wafers: ExplainResponse["wafer_explanations"];
  target: string;
}) {
  const allContributions = [...selected.top_negative_contributors, ...selected.top_positive_contributors]
    .sort((left, right) => Math.abs(right.shap_value) - Math.abs(left.shap_value));
  const topCause = (type: string) => allContributions.find(
    (row) => row.parameter_type.toLowerCase() === type.toLowerCase(),
  )?.feature ?? "-";
  const activeLot = lotIdOf(selected.identifier, selected.lot_id);
  const allValues = wafers.map((wafer) => wafer.prediction);
  const lotValues = wafers
    .filter((wafer) => lotIdOf(wafer.identifier, wafer.lot_id) === activeLot)
    .map((wafer) => wafer.prediction);
  const boxes = [
    summarizeValues("전체", allValues),
    summarizeValues(activeLot || "선택 Lot", lotValues),
  ].filter((row): row is BoxSummary => row !== null).map((row) => ({ ...row, selectedValue: selected.prediction }));
  const featureRows: RelationshipFeature[] = allContributions.slice(0, 12).map((row, index) => ({
    rank: index + 1,
    feature: row.feature,
    display_name: row.feature,
    step: Number(row.step.replace("Step", "")) || null,
    group: row.parameter_type,
    ranking_basis: "Absolute local SHAP contribution",
    score: Math.abs(row.shap_value),
    signed_association: row.shap_value,
    direction: row.shap_value < 0 ? "yield_down" : "yield_up",
    valid_count: 1,
    missing_count: null,
    missing_rate: null,
    category_count: null,
    is_categorical: ["Equipment", "Chamber", "Model"].includes(row.parameter_type),
  }));
  return <>
    <div className="analysisContextGrid waferSummaryGrid">
      <div><span>선택 Wafer</span><strong>{String(selected.identifier)}</strong></div>
      <div><span>예측값</span><strong>{formatNumber(selected.prediction)}</strong></div>
      <div><span>기준값</span><strong>{formatNumber(selected.base_value)}</strong></div>
      <div><span>위험도</span><strong>{riskLabel(selected.risk_level)}</strong></div>
      <div><span>Lot</span><strong>{activeLot || "-"}</strong></div>
    </div>
    <LocalPareto rows={selected.top_negative_contributors} title={paretoTitle(target)} impactLabel={paretoImpactLabel(target)} />
    <div className="sectionHeading compact subsectionHeading"><div><span className="sectionLabel">Feature Importance</span><h3>선택 Wafer 주요 Feature</h3></div><p>SHAP Contribution 절댓값 기준</p></div>
    {featureRows.length ? <><ContributionBarChart data={featureRows} /><div className="analysisContextGrid"><div><span>Top Feature</span><strong>{allContributions[0]?.feature ?? "-"}</strong></div><div><span>Top Step</span><strong>{allContributions[0]?.step ?? "-"}</strong></div><div><span>Top R</span><strong>{topCause("R")}</strong></div><div><span>Top D</span><strong>{topCause("D")}</strong></div><div><span>Top Equipment</span><strong>{topCause("Equipment") !== "-" ? topCause("Equipment") : topCause("EQ")}</strong></div><div><span>Top Chamber</span><strong>{topCause("Chamber")}</strong></div></div></> : <p className="emptyMessage">선택 Wafer의 Feature Importance가 없습니다.</p>}
    <div className="contributionGrid"><ContributionList title={contributionTitle(target, true)} rows={selected.top_negative_contributors} field="harmful_contribution" /><ContributionList title={contributionTitle(target, false)} rows={selected.top_positive_contributors} field="beneficial_contribution" /></div>
    {selectedMultiY && <div className="analysisContextGrid"><div><span>Direct Y</span><strong>{optionalNumber(selectedMultiY.direct_y)}%</strong></div><div><span>Derived Y</span><strong>{optionalNumber(selectedMultiY.derived_y)}%</strong></div><div><span>Ensemble Y</span><strong>{optionalNumber(selectedMultiY.ensemble_y)}%</strong></div><div><span>Direct–Derived Gap</span><strong>{optionalNumber(selectedMultiY.direct_derived_gap)}%p</strong></div>{Object.entries(selectedMultiY.failure_rates).map(([target, value]) => <div key={target}><span>{target} Fail Rate</span><strong>{optionalNumber(value)}%</strong></div>)}{Object.entries(selectedMultiY.fail_bit_counts).map(([target, value]) => <div key={target}><span>{target} Fail Bit Count</span><strong>{optionalNumber(value, 0)}</strong></div>)}</div>}
    <div className="sectionHeading compact subsectionHeading"><div><span className="sectionLabel">Comparison Box Plot</span><h3>전체·선택 Lot 예측 분포</h3></div><p>선택 Wafer Point Overlay</p></div>
    {boxes.length ? <BoxPlotGraphic groups={boxes} variable={`Predicted ${target}`} /> : <p className="emptyMessage">Box Plot을 만들 유효 표본이 없습니다.</p>}
  </>;
}

function LocalPareto({ rows, title, context = "선택 Wafer 1장 · SHAP 방향 기여", impactLabel = "수율 악화" }: { rows: LocalContributionItem[]; title: string; context?: string; impactLabel?: string }) {
  const data = useMemo(() => {
    const positive = rows.map((row) => ({ name: formatFeatureLabel(row.feature), impact: Math.max(row.harmful_contribution, 0) })).filter((row) => row.impact > 0).sort((a, b) => b.impact - a.impact);
    const total = positive.reduce((sum, row) => sum + row.impact, 0);
    return positive.map((row, index) => ({
      ...row,
      share: total > 0 ? row.impact / total : 0,
      cumulative: total > 0
        ? positive.slice(0, index + 1).reduce((sum, item) => sum + item.impact, 0) / total
        : 0,
    }));
  }, [rows]);
  if (!data.length) return <p className="emptyMessage">선택 Wafer의 악화 방향 기여 데이터가 없습니다.</p>;
  return <section className="localPareto"><div className="sectionHeading compact"><div><span className="sectionLabel">Pareto</span><h3>{title}</h3></div><p>{context}</p></div><div className="compactChart"><ResponsiveContainer width="100%" height={300}><ComposedChart data={data.slice(0, 20)} margin={{ top: 8, right: 28, bottom: 68, left: 8 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 6" vertical={false} /><XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} height={82} axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "var(--chart-axis)" }} /><YAxis yAxisId="impact" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "var(--chart-axis)" }} /><YAxis yAxisId="share" orientation="right" domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "var(--chart-axis)" }} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} /><Tooltip content={({ active, payload }) => { if (!active || !payload?.[0]) return null; const item = payload[0].payload as { name: string; impact: number; share: number; cumulative: number }; return <div className="chartTooltip"><strong>{item.name}</strong><span>{impactLabel}: {formatNumber(item.impact)}</span><span>기여 비율: {(item.share * 100).toFixed(1)}%</span><span>누적 비율: {(item.cumulative * 100).toFixed(1)}%</span></div>; }} /><Bar yAxisId="impact" dataKey="impact" fill="var(--pareto-bar)" radius={[6, 6, 0, 0]} /><Line yAxisId="share" dataKey="cumulative" stroke="var(--accent)" dot={false} strokeWidth={2} /><ReferenceLine yAxisId="share" y={0.8} stroke="var(--text-muted)" strokeDasharray="4 5" label={{ value: "80%", position: "insideTopRight", fill: "var(--text-muted)", fontSize: 11 }} /></ComposedChart></ResponsiveContainer></div><p className="analysisDisclaimer">SHAP 결과는 모델 기반 설명이며 실제 인과관계를 확정하지 않습니다.</p></section>;
}

function lotIdOf(identifier: unknown, explicitLotId?: string | null): string {
  if (typeof explicitLotId === "string" && explicitLotId.trim()) {
    return explicitLotId.trim();
  }
  const value = String(identifier ?? "").trim();
  const match = value.match(/^(.*?)[_-]?(?:WAFER|WF|W)[_-]?\d+$/i);
  return match?.[1]?.replace(/[_-]+$/, "") ?? "";
}

type LotFeatureView = {
  feature: string; display_name: string; group: string; step: string;
  mean_signed_shap: number; mean_abs_shap: number;
  adverse_contribution: number; improvement_contribution: number;
  sample_count: number; coverage: number;
};
type LotParetoView = {
  feature: string; display_name: string; group: string;
  adverse_contribution: number; share: number; cumulative_share: number;
  sample_count: number; coverage: number;
};
type LotView = {
  lot_id: string; wafer_count: number; analyzed_wafer_count: number;
  shap_coverage: number; average_predicted_value: number | null;
  minimum_predicted_value: number | null; maximum_predicted_value: number | null;
  risk_extreme_predicted_value: number | null;
  risk_extreme_direction: "minimum" | "maximum" | null;
  critical_wafer_count: number;
  warning_wafer_count: number; normal_wafer_count: number;
  average_confidence: number | null; top_failure_target: string | null;
  feature_importance: Record<RankingGroup, LotFeatureView[]>;
  pareto: Record<RankingGroup, LotParetoView[]>;
  wafer_list: Array<{ identifier: unknown; prediction: number | null; risk_level: string | null; confidence: number | null; top_feature: string | null; top_step: string | null; top_config: string | null }>;
  top_causes: { feature: string | null; step: string | null; config: string | null; failure_target: string | null };
};

type ApiLotCauseItem = NonNullable<RelationshipAnalysisResponse["lot_analysis"]>["lots"][number];

function apiLotView(lot: ApiLotCauseItem): LotView {
  const waferCount = lot.wafer_count ?? lot.wafer_list.length;
  const analyzedWaferCount = lot.analyzed_wafer_count
    ?? lot.wafer_list.filter((wafer) => wafer.shap_available === true).length;
  const riskCount = (risk: "danger" | "warning" | "normal") =>
    lot.wafer_list.filter((wafer) => wafer.risk_level === risk).length;
  const featureGroup = (group: RankingGroup): LotFeatureView[] =>
    lot.feature_importance[group].map((row) => ({
      feature: row.feature,
      display_name: row.display_name,
      group: row.group,
      step: row.step,
      mean_signed_shap: row.mean_signed_shap,
      mean_abs_shap: row.mean_abs_shap,
      adverse_contribution: row.adverse_contribution,
      improvement_contribution: row.improvement_contribution,
      sample_count: row.sample_count,
      coverage: row.coverage,
    }));
  const paretoGroup = (group: RankingGroup): LotParetoView[] =>
    lot.pareto[group].map((row) => ({
      feature: row.feature,
      display_name: row.display_name,
      group: row.group,
      adverse_contribution: row.adverse_contribution,
      share: row.share,
      cumulative_share: row.cumulative_share,
      sample_count: row.sample_count,
      coverage: row.coverage,
    }));
  return {
    lot_id: lot.lot_id,
    wafer_count: waferCount,
    analyzed_wafer_count: analyzedWaferCount,
    shap_coverage: lot.shap_coverage
      ?? (waferCount ? analyzedWaferCount / waferCount : 0),
    average_predicted_value: lot.average_predicted_value,
    minimum_predicted_value: lot.minimum_predicted_value,
    maximum_predicted_value: lot.maximum_predicted_value,
    risk_extreme_predicted_value: lot.risk_extreme_predicted_value,
    risk_extreme_direction: lot.risk_extreme_direction,
    critical_wafer_count: lot.critical_wafer_count ?? riskCount("danger"),
    warning_wafer_count: lot.warning_wafer_count ?? riskCount("warning"),
    normal_wafer_count: lot.normal_wafer_count ?? riskCount("normal"),
    average_confidence: lot.average_confidence,
    top_failure_target: lot.top_failure_target,
    feature_importance: {
      all: featureGroup("all"),
      r: featureGroup("r"),
      d: featureGroup("d"),
      config: featureGroup("config"),
    },
    pareto: {
      all: paretoGroup("all"),
      r: paretoGroup("r"),
      d: paretoGroup("d"),
      config: paretoGroup("config"),
    },
    wafer_list: lot.wafer_list.map((wafer) => ({
      identifier: wafer.identifier,
      prediction: wafer.prediction,
      risk_level: wafer.risk_level,
      confidence: wafer.confidence,
      top_feature: wafer.top_feature,
      top_step: wafer.top_step,
      top_config: wafer.top_config,
    })),
    top_causes: lot.top_causes,
  };
}

function legacyLotViews(
  analysis: AnalysisResult,
  wafersSnapshot: ExplainResponse["wafer_explanations"],
): LotView[] {
  const ids = new Set([
    ...(analysis.lot_summary ?? []).map((row) => row.lot_id),
    ...wafersSnapshot.map((row) => lotIdOf(row.identifier, row.lot_id)),
  ]);
  return [...ids].filter(Boolean).map((lotId) => {
    const wafers = wafersSnapshot.filter((row) => lotIdOf(row.identifier, row.lot_id) === lotId);
    const summary = analysis.lot_summary.find((row) => row.lot_id === lotId);
    const features = new Map<string, LotFeatureView & { wafers: Set<string> }>();
    wafers.forEach((wafer) => {
      [...wafer.top_negative_contributors, ...wafer.top_positive_contributors].forEach((item) => {
        const config = ["model", "equipment", "chamber", "eq"].includes(item.parameter_type.toLowerCase());
        const feature = config && item.step !== "unknown" ? `${item.step}_Config` : item.feature;
        const group = config ? "Config" : item.parameter_type.toUpperCase();
        const existing = features.get(feature) ?? { feature, display_name: feature, group, step: item.step, mean_signed_shap: 0, mean_abs_shap: 0, adverse_contribution: 0, improvement_contribution: 0, sample_count: 0, coverage: 0, wafers: new Set<string>() };
        existing.mean_signed_shap += item.shap_value;
        existing.mean_abs_shap += Math.abs(item.shap_value);
        existing.adverse_contribution += item.harmful_contribution;
        existing.improvement_contribution += item.beneficial_contribution;
        existing.wafers.add(String(wafer.identifier));
        features.set(feature, existing);
      });
    });
    const ranked = [...features.values()].map((item) => {
      const sampleCount = item.wafers.size;
      return { ...item, mean_signed_shap: item.mean_signed_shap / Math.max(sampleCount, 1), mean_abs_shap: item.mean_abs_shap / Math.max(sampleCount, 1), adverse_contribution: item.adverse_contribution / Math.max(sampleCount, 1), improvement_contribution: item.improvement_contribution / Math.max(sampleCount, 1), sample_count: sampleCount, coverage: sampleCount / Math.max(summary?.wafer_count ?? wafers.length, 1) };
    }).sort((left, right) => right.mean_abs_shap - left.mean_abs_shap).map((item): LotFeatureView => ({
      feature: item.feature,
      display_name: item.display_name,
      group: item.group,
      step: item.step,
      mean_signed_shap: item.mean_signed_shap,
      mean_abs_shap: item.mean_abs_shap,
      adverse_contribution: item.adverse_contribution,
      improvement_contribution: item.improvement_contribution,
      sample_count: item.sample_count,
      coverage: item.coverage,
    }));
    const grouped = { all: ranked, r: ranked.filter((item) => item.group === "R"), d: ranked.filter((item) => item.group === "D"), config: ranked.filter((item) => item.group === "Config") };
    const buildPareto = (rows: LotFeatureView[]): LotParetoView[] => {
      const ordered = [...rows].sort((left, right) => right.adverse_contribution - left.adverse_contribution);
      const total = ordered.reduce((sum, row) => sum + Math.max(row.adverse_contribution, 0), 0);
      let cumulative = 0;
      return ordered.map((row) => {
        const share = total > 0
          ? Math.max(row.adverse_contribution, 0) / total
          : 0;
        cumulative += share;
        return { ...row, share, cumulative_share: cumulative };
      });
    };
    const pareto: Record<RankingGroup, LotParetoView[]> = {
      all: buildPareto(grouped.all),
      r: buildPareto(grouped.r),
      d: buildPareto(grouped.d),
      config: buildPareto(grouped.config),
    };
    const predictions = wafers.map((row) => row.prediction).filter(Number.isFinite);
    const minimumPrediction = predictions.length ? Math.min(...predictions) : null;
    const maximumPrediction = predictions.length ? Math.max(...predictions) : null;
    const target = analysis.target.name;
    return { lot_id: lotId, wafer_count: summary?.wafer_count ?? wafers.length, analyzed_wafer_count: wafers.length, shap_coverage: wafers.length / Math.max(summary?.wafer_count ?? wafers.length, 1), average_predicted_value: summary?.average_predicted_yield ?? (predictions.length ? predictions.reduce((sum, value) => sum + value, 0) / predictions.length : null), minimum_predicted_value: minimumPrediction, maximum_predicted_value: maximumPrediction, risk_extreme_predicted_value: target === "Y" ? minimumPrediction : maximumPrediction, risk_extreme_direction: target === "Y" ? "minimum" : "maximum", critical_wafer_count: summary?.danger_count ?? wafers.filter((row) => row.risk_level === "danger").length, warning_wafer_count: summary?.warning_count ?? wafers.filter((row) => row.risk_level === "warning").length, normal_wafer_count: summary?.normal_count ?? wafers.filter((row) => row.risk_level === "normal").length, average_confidence: null, top_failure_target: null, feature_importance: grouped, pareto, wafer_list: wafers.map((row) => ({ identifier: row.identifier, prediction: row.prediction, risk_level: row.risk_level, confidence: null, top_feature: row.top_negative_contributors[0]?.feature ?? null, top_step: row.top_negative_contributors[0]?.step ?? null, top_config: null })), top_causes: { feature: ranked[0]?.feature ?? summary?.top_harmful_feature ?? null, step: ranked[0]?.step ?? summary?.top_harmful_step ?? null, config: grouped.config[0]?.feature ?? null, failure_target: null } };
  });
}

function LotAnalysis({ analysis, explanation, lotAnalysis }: { analysis: AnalysisResult | null; explanation: ExplainResponse | null; lotAnalysis: RelationshipAnalysisResponse["lot_analysis"] | undefined }) {
  const lots = useMemo(() => {
    if (lotAnalysis?.lots.length) {
      return lotAnalysis.lots
        .filter((lot) => lot.wafer_count !== null || lot.wafer_list.length > 0)
        .map(apiLotView);
    }
    if (!analysis) return [];
    const wafersSnapshot = explanation?.wafer_explanations ?? analysis.wafer_explanations;
    return legacyLotViews(analysis, wafersSnapshot);
  }, [analysis, explanation, lotAnalysis]);
  const [selectedLot, setSelectedLot] = useState(() => typeof window === "undefined" ? "" : new URLSearchParams(window.location.search).get("lot_id") ?? "");
  const [lotSearch, setLotSearch] = useState("");
  const [lotSort, setLotSort] = useState<"risk" | "id" | "source">("risk");
  const [lotGroup, setLotGroup] = useState<RankingGroup>("all");
  const visibleLots = useMemo(() => lots.map((lot, index) => ({ lot, index })).filter(({ lot }) => lot.lot_id.toLocaleLowerCase("ko").includes(lotSearch.trim().toLocaleLowerCase("ko"))).sort((left, right) => lotSort === "source" ? left.index - right.index : lotSort === "id" ? left.lot.lot_id.localeCompare(right.lot.lot_id, "ko", { numeric: true }) : right.lot.critical_wafer_count - left.lot.critical_wafer_count || right.lot.warning_wafer_count - left.lot.warning_wafer_count || left.lot.lot_id.localeCompare(right.lot.lot_id, "ko", { numeric: true })), [lotSearch, lotSort, lots]);
  const activeLot = lots.find((lot) => lot.lot_id === selectedLot) ?? visibleLots[0]?.lot ?? lots[0];
  if (!activeLot) return <section className="resultCard"><p className="emptyMessage">원본 Lot_ID 또는 안전하게 추출할 수 있는 Lot 식별자가 없습니다.</p></section>;
  const featureRows: RelationshipFeature[] = (activeLot.feature_importance[lotGroup] ?? []).slice(0, 20).map((row, index) => ({ rank: index + 1, feature: row.feature, display_name: row.display_name, step: Number(row.step.replace("Step", "")) || null, group: row.group, ranking_basis: "Lot mean absolute SHAP", score: row.mean_abs_shap, signed_association: row.mean_signed_shap, direction: row.mean_signed_shap > 0 ? "positive" : row.mean_signed_shap < 0 ? "negative" : "neutral", valid_count: row.sample_count, missing_count: activeLot.wafer_count - row.sample_count, missing_rate: 1 - row.coverage, category_count: null, is_categorical: row.group === "Config" }));
  const lotValues = activeLot.wafer_list.map((row) => row.prediction).filter((value): value is number => value !== null && Number.isFinite(value));
  const allValues = lots.flatMap((lot) => lot.wafer_list.map((row) => row.prediction)).filter((value): value is number => value !== null && Number.isFinite(value));
  const boxes = [summarizeValues(activeLot.lot_id, lotValues), summarizeValues("전체", allValues)].filter((row): row is BoxSummary => row !== null);
  const lotTarget = explanation?.model.target ?? lotAnalysis?.target ?? analysis?.target.name ?? "Target";
  const selectLot = (lotId: string) => { setSelectedLot(lotId); const url = new URL(window.location.href); url.searchParams.set("lot_id", lotId); window.history.replaceState({}, "", url); };
  return <>
    <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Lot Selector</span><h2>Lot별 원인</h2></div><p>{lots.length.toLocaleString()}개 Lot</p></div><div className="lotWorkspace"><div className="lotSelectorPanel"><div className="lotSelectorTools"><input type="search" value={lotSearch} onChange={(event) => setLotSearch(event.target.value)} placeholder="Lot_ID 검색" /><select value={lotSort} onChange={(event) => setLotSort(event.target.value as "risk" | "id" | "source")}><option value="risk">위험도 순</option><option value="id">ID 순</option><option value="source">분석 순서</option></select></div><div className="lotSelectorList">{visibleLots.map(({ lot }) => <button key={lot.lot_id} type="button" className={lot.lot_id === activeLot.lot_id ? "active" : ""} onClick={() => selectLot(lot.lot_id)} title={lot.lot_id}><strong>{lot.lot_id}</strong><span>Critical {lot.critical_wafer_count} · Wafer {lot.wafer_count}</span></button>)}</div></div><div className="analysisContextGrid lotSummaryGrid"><div><span>Lot_ID</span><strong title={activeLot.lot_id}>{activeLot.lot_id}</strong></div><div><span>Wafer 수</span><strong>{activeLot.wafer_count}</strong></div><div><span>{lotTarget === "Y" ? "평균 예측 수율" : "평균 예측값"}</span><strong>{optionalNumber(activeLot.average_predicted_value)}</strong></div><div><span>{activeLot.risk_extreme_direction === "maximum" ? "최대 위험 예측값" : "최저 예측 수율"}</span><strong>{optionalNumber(activeLot.risk_extreme_predicted_value)}</strong></div><div><span>Critical</span><strong>{activeLot.critical_wafer_count}</strong></div><div><span>Warning</span><strong>{activeLot.warning_wafer_count}</strong></div><div><span>Normal</span><strong>{activeLot.normal_wafer_count}</strong></div><div><span>평균 Confidence</span><strong>{activeLot.average_confidence === null ? "-" : `${optionalNumber(activeLot.average_confidence * 100)}%`}</strong></div><div><span>주요 Failure Target</span><strong>{activeLot.top_failure_target ?? "-"}</strong></div><div><span>SHAP Coverage</span><strong>{optionalNumber(activeLot.shap_coverage * 100)}%</strong></div></div></div></section>
    <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Lot Feature Importance</span><h2>선택 Lot 모델 기여도</h2></div><SegmentedControl options={[["all", "전체"], ["r", "R"], ["d", "D"], ["config", "Config"]]} value={lotGroup} onChange={(value) => setLotGroup(value as RankingGroup)} /></div>{featureRows.length ? <RankingChart data={featureRows} /> : <p className="emptyMessage">이 Lot의 {lotGroup === "config" ? "Config " : ""}SHAP 집계 데이터가 없습니다.</p>}<LotParetoChart rows={activeLot.pareto[lotGroup] ?? []} target={lotTarget} /><div className="analysisContextGrid"><div><span>Top Feature</span><strong>{activeLot.top_causes.feature ?? "-"}</strong></div><div><span>Top Step</span><strong>{activeLot.top_causes.step ?? "-"}</strong></div><div><span>Top Config</span><strong>{activeLot.top_causes.config ?? "-"}</strong></div><div><span>Top Failure Target</span><strong>{activeLot.top_causes.failure_target ?? "-"}</strong></div></div></section>
    <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Distribution</span><h2>선택 Lot vs 전체 예측 분포</h2></div></div>{boxes.length && lotValues.length >= 2 ? <BoxPlotGraphic groups={boxes} variable={`Predicted ${lotTarget}`} /> : <p className="emptyMessage">분포를 계산할 유효 표본이 부족합니다.</p>}</section>
    <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Lot Wafers</span><h2>Lot 내 위험 Wafer</h2></div><p>{activeLot.wafer_list.length.toLocaleString()}개</p></div><div className="tableWrapper waferListScroll"><table><thead><tr><th>Wafer</th><th>예측값</th><th>위험도</th><th>Confidence</th><th>Top Feature</th><th>Top Config</th></tr></thead><tbody>{[...activeLot.wafer_list].sort((left, right) => (right.risk_level === "danger" ? 2 : right.risk_level === "warning" ? 1 : 0) - (left.risk_level === "danger" ? 2 : left.risk_level === "warning" ? 1 : 0)).map((wafer) => <tr key={String(wafer.identifier)}><td>{String(wafer.identifier)}</td><td>{optionalNumber(wafer.prediction)}</td><td>{riskLabel(wafer.risk_level)}</td><td>{wafer.confidence === null ? "-" : `${optionalNumber(wafer.confidence * 100)}%`}</td><td>{wafer.top_feature ?? "-"}</td><td>{wafer.top_config ?? "-"}</td></tr>)}</tbody></table></div></section>
  </>;
}

function LotParetoChart({ rows, target }: { rows: LotParetoView[]; target: string }) {
  if (!rows.length) return <p className="emptyMessage">선택 Lot의 악화 방향 Pareto 데이터가 없습니다.</p>;
  return <div className="lotPareto"><div className="sectionHeading compact subsectionHeading"><div><span className="sectionLabel">Pareto</span><h3>{paretoTitle(target)}</h3></div><p>선택 Lot Wafer만 집계</p></div><div className="compactChart"><ResponsiveContainer width="100%" height={300}><ComposedChart data={rows.slice(0, 20)} margin={{ top: 8, right: 28, bottom: 68, left: 8 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 6" vertical={false} /><XAxis dataKey="display_name" angle={-35} textAnchor="end" interval={0} height={82} axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "var(--chart-axis)" }} /><YAxis yAxisId="impact" axisLine={false} tickLine={false} /><YAxis yAxisId="share" orientation="right" domain={[0, 1]} axisLine={false} tickLine={false} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} /><Tooltip content={({ active, payload }) => { if (!active || !payload?.[0]) return null; const item = payload[0].payload as LotParetoView; return <div className="chartTooltip"><strong>{item.feature}</strong><span>{paretoImpactLabel(target)}: {optionalNumber(item.adverse_contribution, 5)}</span><span>기여율: {optionalNumber(item.share * 100)}%</span><span>누적: {optionalNumber(item.cumulative_share * 100)}%</span><span>표본: {item.sample_count.toLocaleString()} · Coverage {optionalNumber(item.coverage * 100)}%</span></div>; }} /><ReferenceLine yAxisId="share" y={0.8} stroke="var(--text-muted)" strokeDasharray="4 5" /><Bar yAxisId="impact" dataKey="adverse_contribution" fill="#0072C6" radius={[6, 6, 0, 0]} /><Line yAxisId="share" dataKey="cumulative_share" stroke="var(--accent)" dot={false} strokeWidth={2} /></ComposedChart></ResponsiveContainer></div></div>;
}

function ParetoSection({
  analysis,
  rows,
}: {
  analysis: RelationshipAnalysisResponse;
  rows?: RelationshipFeature[];
}) {
  const pareto = useMemo(() => {
    if (!rows) return analysis.pareto;
    const ranked = [...rows]
      .filter((row) => Number.isFinite(row.score) && (row.score ?? 0) >= 0)
      .sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
    const total = ranked.reduce((sum, row) => sum + (row.score ?? 0), 0);
    const shares = ranked.map((row) => total > 0 ? (row.score ?? 0) / total : 0);
    const cumulativeShares = shares.map((_, index) => shares.slice(0, index + 1).reduce((sum, share) => sum + share, 0));
    const thresholdIndex = cumulativeShares.findIndex((share) => share >= 0.8);
    const required = thresholdIndex >= 0 ? thresholdIndex + 1 : 0;
    const features = ranked.map((row, index) => {
      const impact = row.score ?? 0;
      const share = shares[index];
      const cumulativeShare = cumulativeShares[index];
      return { ...row, impact, share, cumulative_share: cumulativeShare, within_threshold: cumulativeShare <= 0.8 || index + 1 === required };
    });
    const count = (key: string) => ranked.filter((row) => row.group.toLowerCase() === key).length;
    return {
      ...analysis.pareto,
      features,
      required_feature_count: required,
      cumulative_contribution: features[required - 1]?.cumulative_share ?? 0,
      total_feature_count: features.length,
      total_impact: total,
      group_counts: { R: count("r"), D: count("d"), EQ: count("eq") + count("equipment") },
    };
  }, [analysis.pareto, rows]);
  if (!pareto.features.length) {
    return <section className="resultCard relationshipSection"><div className="sectionHeading compact"><div><span className="sectionLabel">Cumulative Impact</span><h2>{paretoTitle(analysis.target)}</h2></div><p>Ranking basis: {pareto.ranking_basis}</p></div><p className="emptyMessage">Pareto를 계산할 실제 기여도 데이터가 없습니다.</p></section>;
  }
  const uiGroupCounts = pareto.features.reduce((counts, row) => {
    const group = row.group.toLowerCase();
    if (group === "r") counts.R += 1;
    else if (group === "d") counts.D += 1;
    else if (["config", "model", "equipment", "eq", "chamber"].includes(group)) counts.Config += 1;
    return counts;
  }, { R: 0, D: 0, Config: 0 });
  return (
    <section className="resultCard relationshipSection">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">Cumulative Impact</span>
          <h2>{paretoTitle(analysis.target)}</h2>
        </div>
        <p>Ranking basis: {pareto.ranking_basis}</p>
      </div>
      <div className="paretoSummary">
        <div><span>우선 검토 변수</span><strong>{pareto.required_feature_count}개</strong></div>
        <div><span>누적 영향도</span><strong>{(pareto.cumulative_contribution * 100).toFixed(1)}%</strong></div>
        <div><span>전체 변수</span><strong>{pareto.total_feature_count}개</strong></div>
        <div>
          <span>상위 그룹 구성</span>
          <strong>R {uiGroupCounts.R} · D {uiGroupCounts.D} · Config {uiGroupCounts.Config}</strong>
        </div>
      </div>
      {pareto.features.length ? (
        <div className="paretoScroll">
          <div
            className="paretoChart"
            style={{ width: Math.max(760, pareto.features.length * 52) }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={pareto.features} margin={{ top: 8, right: 24, bottom: 12, left: 8 }}>
                <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 6" vertical={false} />
                <XAxis
                  dataKey="display_name"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: "var(--chart-axis)" }}
                  tickFormatter={(value) => String(value).replace("Step ", "S")}
                />
                <YAxis yAxisId="impact" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: "var(--chart-axis)" }} />
                <YAxis
                  yAxisId="share"
                  orientation="right"
                  domain={[0, 1]}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
                  tick={{ fontSize: 12, fill: "var(--chart-axis)" }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.[0]) return null;
                    const item = payload[0].payload as { display_name: string; feature: string; impact: number; share: number; cumulative_share: number };
                    return <div className="chartTooltip"><strong>{item.display_name || item.feature}</strong><span>{paretoImpactLabel(analysis.target)}: {formatNumber(item.impact)}</span><span>기여 비율: {(item.share * 100).toFixed(1)}%</span><span>누적 비율: {(item.cumulative_share * 100).toFixed(1)}%</span></div>;
                  }}
                />
                <ReferenceLine yAxisId="share" y={0.8} stroke="var(--text-muted)" strokeDasharray="4 5" label={{ value: "80%", position: "insideTopRight", fill: "var(--text-muted)", fontSize: 11 }} />
                <Bar yAxisId="impact" dataKey="impact" fill="var(--pareto-bar)" radius={[6, 6, 0, 0]}>
                  {pareto.features.map((item) => (
                    <Cell
                      key={item.feature}
                      fill="var(--pareto-bar)"
                      fillOpacity={item.within_threshold ? 1 : 0.42}
                    />
                  ))}
                </Bar>
                <Line
                  yAxisId="share"
                  type="monotone"
                  dataKey="cumulative_share"
                  stroke="var(--accent)"
                  dot={false}
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <p className="emptyMessage">누적 영향도를 계산할 데이터가 없습니다.</p>
      )}
      <p className="analysisDisclaimer">{pareto.caveat}</p>
    </section>
  );
}

function associationValue(value: AssociationSummaryLike): string {
  if (!value) return "-";
  const metric = value.eta_squared ?? value.pearson;
  return metric === null || metric === undefined ? "-" : metric.toFixed(3);
}

type AssociationSummaryLike = RelationshipPath["r_d"];

function PathSection({
  paths,
  selectedIndex,
  onSelect,
  confidenceCriteria,
}: {
  paths: RelationshipPath[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  confidenceCriteria: Record<string, string>;
}) {
  const [pathSearch, setPathSearch] = useState("");
  const selected = paths[selectedIndex];
  const confidenceDescriptions = [
    confidenceCriteria.sufficient ? `충분(${confidenceCriteria.sufficient})` : null,
    confidenceCriteria.caution ? `주의(${confidenceCriteria.caution})` : null,
    confidenceCriteria.insufficient ? `부족(${confidenceCriteria.insufficient})` : null,
  ].filter((item): item is string => item !== null);
  const visiblePaths = useMemo(
    () =>
      paths
        .map((path, index) => ({ path, index }))
        .filter(({ path }) => {
          const query = pathSearch.trim().toLocaleLowerCase("ko");
          if (!query) return true;
          return [
            `Step ${path.step}`,
            path.response,
            path.equipment,
            path.defect,
          ].some((value) =>
            String(value ?? "").toLocaleLowerCase("ko").includes(query),
          );
        })
        .sort((left, right) => right.path.path_score - left.path.path_score),
    [pathSearch, paths],
  );
  return (
    <section className="resultCard relationshipSection pathUnifiedCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">Relationship Analysis</span>
            <h2>Relationship Path</h2>
          </div>
          <p>Path relevance score · 인과 효과가 아닌 탐색 점수</p>
        </div>
        <div className="relationshipToolbar pathToolbar">
          <label className="waferSearch">
            <span className="visuallyHidden">Relationship Path 검색</span>
            <input
              type="search"
              value={pathSearch}
              onChange={(event) => setPathSearch(event.target.value)}
              placeholder="Step, Response, Equipment, Defect 검색"
            />
          </label>
          <span className="pathSortLabel">Score 높은 순</span>
        </div>
        {paths.length ? (
          <div className="tableWrapper relationshipPathScroll">
            <table className="dataTable pathTable">
              <thead><tr>
                <th>Rank</th><th>Step</th><th>Response</th><th>Equipment</th>
                <th>Defect</th><th>R→D</th><th>EQ→D</th><th>D→Y</th>
                <th>SHAP</th><th>Score</th><th>Sample</th><th>Confidence</th>
              </tr></thead>
              <tbody>
                {visiblePaths.map(({ path, index }) => (
                  <tr
                    key={path.step}
                    className={selectedIndex === index ? "selectedRow" : ""}
                    onClick={() => onSelect(index)}
                  >
                    <td>{path.rank}</td><td>Step {path.step}</td>
                    <td>{path.response ?? "-"}</td><td>{path.equipment ?? "-"}</td>
                    <td>{path.defect}</td><td>{associationValue(path.r_d)}</td>
                    <td>{associationValue(path.eq_d)}</td><td>{associationValue(path.d_y)}</td>
                    <td>{formatNumber(path.shap_importance)}</td>
                    <td>{path.path_score.toFixed(3)}</td>
                    <td>{path.valid_count.toLocaleString()}</td>
                    <td><span className={`confidenceBadge ${path.confidence}`}>{path.confidence}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="emptyMessage">동일 Step의 연관 경로를 구성할 데이터가 없습니다.</p>}
        {paths.length > 0 && visiblePaths.length === 0 && (
          <p className="emptyMessage">검색 조건과 일치하는 경로가 없습니다.</p>
        )}
        {confidenceDescriptions.length > 0 && <p className="chartDescription">
          Confidence 기준: {confidenceDescriptions.join(" · ")}
        </p>}
        {selected && <PathDetail path={selected} />}
    </section>
  );
}

function PathDetail({ path }: { path: RelationshipPath }) {
  return (
    <div className="relationshipDetailBlock">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">Selected Path</span>
          <h2>Step {path.step} 관계 패널</h2>
        </div>
      </div>
      <div className="pathDetailGrid">
        <ScatterPanel title={`${path.response ?? "R"} vs ${path.defect}`} data={path.r_vs_d} xLabel={path.response ?? "R"} yLabel={path.defect} xDescription="Response Value" yDescription="Defect Measurement" />
        {path.r_vs_y?.length ? <ScatterPanel title={`${path.response ?? "R"} vs Y`} data={path.r_vs_y} xLabel={path.response ?? "R"} yLabel="Final Yield Y" xDescription="Response Value" yDescription="Final Yield (%)" /> : null}
        <EquipmentPanel title={`${path.equipment ?? "EQ"} vs ${path.defect}`} data={path.eq_vs_d} />
        <ScatterPanel title={`${path.defect} vs Y`} data={path.d_vs_y} xLabel={path.defect} yLabel="Final Yield Y" xDescription="Defect Value" yDescription="Final Yield (%)" />
      </div>
      <YieldBoxPlot path={path} />
    </div>
  );
}

type BoxSummary = {
  label: string;
  count: number;
  median: number;
  q1: number;
  q3: number;
  whiskerMin: number;
  whiskerMax: number;
  outliers: number[];
  outlierCount: number;
  selectedValue?: number;
};

function quantile(values: number[], ratio: number): number {
  if (values.length === 1) return values[0];
  const position = (values.length - 1) * ratio;
  const lower = Math.floor(position);
  const remainder = position - lower;
  return values[lower] + (values[lower + 1] - values[lower]) * remainder;
}

function summarizeValues(label: string, rawValues: number[]): BoxSummary | null {
  const values = rawValues.filter(Number.isFinite).sort((left, right) => left - right);
  if (!values.length) return null;
  const q1 = quantile(values, 0.25);
  const median = quantile(values, 0.5);
  const q3 = quantile(values, 0.75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;
  const inliers = values.filter((value) => value >= lowerFence && value <= upperFence);
  const outliers = values.filter((value) => value < lowerFence || value > upperFence);
  return {
    label,
    count: values.length,
    median,
    q1,
    q3,
    whiskerMin: inliers[0] ?? values[0],
    whiskerMax: inliers[inliers.length - 1] ?? values[values.length - 1],
    outliers,
    outlierCount: outliers.length,
  };
}

function numericBoxGroups(
  points: { x: number; y: number }[] | undefined,
): BoxSummary[] {
  const sorted = (points ?? [])
    .filter(({ x, y }) => Number.isFinite(x) && Number.isFinite(y))
    .sort((left, right) => left.x - right.x);
  if (!sorted.length) return [];
  const groupCount = Math.min(4, sorted.length);
  return Array.from({ length: groupCount }, (_, groupIndex) => {
    const start = Math.floor((groupIndex * sorted.length) / groupCount);
    const end = Math.floor(((groupIndex + 1) * sorted.length) / groupCount);
    const group = sorted.slice(start, end);
    const first = group[0]?.x ?? 0;
    const last = group[group.length - 1]?.x ?? first;
    return summarizeValues(
      first === last
        ? formatNumber(first)
        : `${formatNumber(first)}–${formatNumber(last)}`,
      group.map(({ y }) => y),
    );
  }).filter((summary): summary is BoxSummary => summary !== null);
}

function YieldBoxPlot({ path }: { path: RelationshipPath }) {
  const [tab, setTab] = useState<"R" | "D" | "EQ">("R");
  const groups = useMemo(() => {
    if (tab === "R") return numericBoxGroups(path.r_vs_y);
    if (tab === "D") return numericBoxGroups(path.d_vs_y);
    return (path.eq_vs_y ?? []).map((item) => ({
      label: item.equipment,
      count: item.count,
      median: item.median,
      q1: item.q1,
      q3: item.q3,
      whiskerMin: item.whisker_min ?? item.minimum,
      whiskerMax: item.whisker_max ?? item.maximum,
      outliers: item.outliers ?? [],
      outlierCount: item.outlier_count ?? item.outliers?.length ?? 0,
    }));
  }, [path, tab]);
  const variable =
    tab === "R"
      ? path.response ?? "Response"
      : tab === "D"
        ? path.defect
        : path.equipment ?? "Equipment";

  return (
    <section className="yieldBoxPlotSection">
      <div className="boxPlotHeading">
        <div>
          <span className="sectionLabel">Yield Distribution</span>
          <h3>{variable} · Yield Box Plot</h3>
          <p>Median, IQR, 1.5×IQR whisker, outlier와 표본 수를 실제 분석 데이터로 계산합니다.</p>
        </div>
        <SegmentedControl
          options={[["R", "R"], ["D", "D"], ["EQ", "EQ"]]}
          value={tab}
          onChange={(value) => setTab(value as "R" | "D" | "EQ")}
        />
      </div>
      {groups.length ? (
        <>
          <BoxPlotGraphic groups={groups} variable={variable} />
          <div className="boxStatGrid">
            {groups.map((group) => (
              <article key={group.label} className="boxStatItem">
                <strong title={group.label}>{group.label}</strong>
                <span>Median {formatNumber(group.median)}</span>
                <span>IQR {formatNumber(group.q3 - group.q1)}</span>
                <span>Outlier {group.outlierCount.toLocaleString()} · n={group.count.toLocaleString()}</span>
              </article>
            ))}
          </div>
        </>
      ) : (
        <p className="emptyMessage">
          선택 경로의 {tab} 변수와 Yield를 함께 비교할 유효 데이터가 없습니다.
        </p>
      )}
    </section>
  );
}

function BoxPlotGraphic({
  groups,
  variable,
}: {
  groups: BoxSummary[];
  variable: string;
}) {
  const allValues = groups.flatMap((group) => [
    group.whiskerMin,
    group.whiskerMax,
    ...group.outliers,
    ...(group.selectedValue === undefined ? [] : [group.selectedValue]),
  ]);
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = rawMax === rawMin ? Math.max(Math.abs(rawMax) * 0.05, 1) : (rawMax - rawMin) * 0.08;
  const min = rawMin - padding;
  const max = rawMax + padding;
  const width = Math.max(720, groups.length * 110);
  const height = 330;
  const plotTop = 24;
  const plotBottom = 270;
  const scaleY = (value: number) =>
    plotBottom - ((value - min) / (max - min)) * (plotBottom - plotTop);
  const ticks = Array.from({ length: 5 }, (_, index) => min + ((max - min) * index) / 4);

  return (
    <div className="boxPlotScroll" role="img" aria-label={`${variable}별 Yield box plot`}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ minWidth: width }}>
        <text className="boxAxisTitle" x="16" y="18">Final Yield Y</text>
        {ticks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={tick}>
              <line className="boxGridLine" x1="66" x2={width - 16} y1={y} y2={y} />
              <text className="boxTick" x="58" y={y + 4} textAnchor="end">{formatNumber(tick)}</text>
            </g>
          );
        })}
        {groups.map((group, index) => {
          const slot = (width - 90) / groups.length;
          const x = 74 + slot * index + slot / 2;
          const boxWidth = Math.min(52, slot * 0.58);
          const tooltip = `${group.label}\nMedian: ${formatNumber(group.median)}\nIQR: ${formatNumber(group.q3 - group.q1)}\nOutlier: ${group.outlierCount}\nn: ${group.count}`;
          return (
            <g key={`${group.label}-${index}`}>
              <title>{tooltip}</title>
              <line className="boxWhisker" x1={x} x2={x} y1={scaleY(group.whiskerMax)} y2={scaleY(group.whiskerMin)} />
              <line className="boxWhisker" x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={scaleY(group.whiskerMax)} y2={scaleY(group.whiskerMax)} />
              <line className="boxWhisker" x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={scaleY(group.whiskerMin)} y2={scaleY(group.whiskerMin)} />
              <rect
                className="boxShape"
                x={x - boxWidth / 2}
                y={scaleY(group.q3)}
                width={boxWidth}
                height={Math.max(2, scaleY(group.q1) - scaleY(group.q3))}
                rx="5"
              />
              <line className="boxMedian" x1={x - boxWidth / 2} x2={x + boxWidth / 2} y1={scaleY(group.median)} y2={scaleY(group.median)} />
              {group.outliers.slice(0, 30).map((outlier, outlierIndex) => (
                <circle className="boxOutlier" key={`${outlier}-${outlierIndex}`} cx={x} cy={scaleY(outlier)} r="3" />
              ))}
              {group.selectedValue !== undefined && <circle className="boxSelectedPoint" cx={x} cy={scaleY(group.selectedValue)} r="6"><title>선택 Wafer: {formatNumber(group.selectedValue)}</title></circle>}
              <text className="boxCategory" x={x} y="294" textAnchor="middle">{group.label.length > 14 ? `${group.label.slice(0, 12)}…` : group.label}</text>
              <text className="boxCount" x={x} y="312" textAnchor="middle">n={group.count}</text>
            </g>
          );
        })}
        <text className="boxAxisTitle" x={width / 2} y="328" textAnchor="middle">{variable}</text>
      </svg>
    </div>
  );
}

function ScatterPanel({ title, data, xLabel, yLabel, xDescription, yDescription }: {
  title: string;
  data: { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  xDescription: string;
  yDescription: string;
}) {
  const statistics = useMemo(() => {
    if (data.length < 2) return { correlation: null, trend: [] as { x: number; y: number }[] };
    const meanX = data.reduce((sum, point) => sum + point.x, 0) / data.length;
    const meanY = data.reduce((sum, point) => sum + point.y, 0) / data.length;
    const covariance = data.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0);
    const sumX = data.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0);
    const sumY = data.reduce((sum, point) => sum + (point.y - meanY) ** 2, 0);
    if (sumX <= 0 || sumY <= 0) return { correlation: null, trend: [] as { x: number; y: number }[] };
    const slope = covariance / sumX;
    const intercept = meanY - slope * meanX;
    const xValues = data.map((point) => point.x);
    const minimum = Math.min(...xValues);
    const maximum = Math.max(...xValues);
    return {
      correlation: covariance / Math.sqrt(sumX * sumY),
      trend: [{ x: minimum, y: slope * minimum + intercept }, { x: maximum, y: slope * maximum + intercept }],
    };
  }, [data]);
  return (
    <article className="relationshipPanel">
      <div className="relationshipPanelHeading"><h3>{title}</h3><span>r = {optionalNumber(statistics.correlation, 3)}</span></div>
      {data.length ? (
        <div className="detailChart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 10, right: 18, bottom: 42, left: 28 }}>
              <CartesianGrid stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                dataKey="x"
                name={xLabel}
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: `${xLabel} (${xDescription})`, position: "insideBottom", offset: -24, fill: "var(--chart-axis)" }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={yLabel}
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: `${yLabel} (${yDescription})`, angle: -90, position: "insideLeft", offset: -16, fill: "var(--chart-axis)" }}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                formatter={(value, name) => [formatNumber(Number(value)), String(name)]}
              />
              <Legend verticalAlign="top" height={28} />
              <Scatter name={`${yLabel} by ${xLabel}`} data={data} fill="var(--chart-primary)" fillOpacity={0.72} />
              {statistics.trend.length > 0 && <Line name="회귀선" data={statistics.trend} dataKey="y" stroke="var(--chart-critical)" strokeWidth={2} dot={false} isAnimationActive={false} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="emptyMessage">표시할 유효 데이터가 없습니다.</p>}
    </article>
  );
}

function EquipmentPanel({ title, data }: {
  title: string; data: RelationshipPath["eq_vs_d"];
}) {
  return (
    <article className="relationshipPanel">
      <h3>{title}</h3>
      {data.length ? (
        <div className="detailChart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 18, bottom: 42, left: 28 }}>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="equipment"
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: "Equipment category", position: "insideBottom", offset: -24, fill: "var(--chart-axis)" }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: "Defect value", angle: -90, position: "insideLeft", offset: -16, fill: "var(--chart-axis)" }}
              />
              <Tooltip formatter={(value, name) => [formatNumber(Number(value)), String(name)]} />
              <Legend verticalAlign="top" height={28} />
              <Bar dataKey="q3" fill="var(--chart-primary-soft)" name="Q3" />
              <Bar dataKey="median" fill="var(--chart-secondary)" name="중앙값" />
              <Line dataKey="mean" stroke="#a96208" name="평균" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="emptyMessage">범주형 Equipment 분포 데이터가 없습니다.</p>}
    </article>
  );
}

function ContributionList({
  title,
  rows,
  field,
}: {
  title: string;
  rows: ExplainResponse["wafer_explanations"][number]["top_negative_contributors"];
  field: "harmful_contribution" | "beneficial_contribution";
}) {
  return (
    <div className={field === "harmful_contribution" ? "contributionPanel degradation" : "contributionPanel improvement"}>
      <h3>{title}</h3>
      <p className="chartDescription">
        값이 클수록 해당 Wafer의 {title} 방향 영향이 큽니다.
      </p>
      <div
        className="localChart"
        role="img"
        aria-label={`${title} feature 영향도 막대 차트`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows.map((row) => ({
              name: row.feature,
              value: row[field],
            }))}
            layout="vertical"
            margin={{ left: 12 }}
          >
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={112}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
              tickFormatter={formatFeatureLabel}
            />
            <Tooltip
              formatter={(value) => [
                formatNumber(Number(value)),
                title,
              ]}
              contentStyle={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                background: "var(--chart-tooltip)",
                color: "var(--text-primary)",
                boxShadow: "var(--shadow-elevated)",
              }}
            />
            <Bar
              dataKey="value"
              fill={
                field === "harmful_contribution" ? "var(--yield-degradation)" : "var(--yield-improvement)"
              }
              radius={[0, 6, 6, 0]}
              animationDuration={220}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ol className="contributionList">
        {rows.map((row) => (
          <li key={row.feature}>
            <div>
              <strong title={row.feature}>{formatFeatureLabel(row.feature)}</strong>
              <span>
                {row.step} · {row.parameter_type} · 값{" "}
                {String(row.value ?? "-")}
              </span>
            </div>
            <b>{formatNumber(row[field])}</b>
          </li>
        ))}
      </ol>
    </div>
  );
}
