"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LabelProps, LegendPayload } from "recharts";

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import ModelSelector, { isModelUsable } from "@/components/models/ModelSelector";
import SelectedModelSummary from "@/components/models/SelectedModelSummary";
import OperationProgress from "@/components/OperationProgress";
import PreprocessingSummary from "@/components/PreprocessingSummary";
import Sidebar from "@/components/Sidebar";
import useElapsedTime from "@/hooks/useElapsedTime";
import {
  deletePredictionHistory,
  downloadPredictions,
  getPredictionHistory,
  getPredictionHistoryDetail,
  predictCsv,
} from "@/lib/api";
import type {
  ModelSummary,
  PredictionHistorySummary,
  PredictionResponse,
  PredictionRow,
  PredictionThresholds,
} from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const DEFAULT_THRESHOLDS: PredictionThresholds = {
  warning_threshold: 90,
  danger_threshold: 85,
};
const DEFAULT_MOVING_AVERAGE_WINDOW = 5;
const RESULT_PAGE_SIZE = 50;
const MOVING_AVERAGE_WINDOWS = Array.from({ length: 25 }, (_, index) => index + 1);

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function restoreArtifactRows(
  response: PredictionResponse,
  rows: PredictionRow[] | undefined,
): PredictionResponse {
  if (!rows?.length) return response;
  return {
    ...response,
    predictions: rows,
    truncated: rows.length < response.summary.total_rows,
    preview_row_count: rows.length,
  };
}

export function calculateMovingAverage(
  values: unknown[],
  windowSize: number,
): Array<number | null> {
  return values.map((_, index) => {
    const windowValues = values
      .slice(Math.max(0, index - windowSize + 1), index + 1)
      .map(finiteNumber)
      .filter((value): value is number => value !== null);
    if (!windowValues.length) return null;
    return (
      windowValues.reduce((sum, value) => sum + value, 0) /
      windowValues.length
    );
  });
}

function ThresholdLabel({
  viewBox,
  label,
  value,
  color,
  yShift,
}: LabelProps & {
  label: string;
  value: number;
  color: string;
  yShift: number;
}) {
  if (!viewBox || !("x" in viewBox)) return null;
  const x = viewBox.x + viewBox.width + 10;
  const y = viewBox.y + yShift;
  return (
    <g aria-hidden="true">
      <line x1={x - 7} y1={viewBox.y} x2={x} y2={y} stroke={color} />
      <rect
        x={x}
        y={y - 10}
        width={88}
        height={20}
        rx={9}
        className="thresholdLabelSurface"
      />
      <text x={x + 7} y={y + 3.5} fill={color} fontSize={10} fontWeight={650}>
        {label} {value.toFixed(1)}%
      </text>
    </g>
  );
}

type RiskFilter = "all" | "normal" | "warning" | "danger";
type TrendSeries =
  | "actual"
  | "predicted"
  | "predictedYieldMean"
  | "predictedYieldMovingAverage"
  | "warning"
  | "critical";
type ResultSort =
  | "prediction-desc"
  | "prediction-asc"
  | "id-asc"
  | "id-desc"
  | "lot-asc"
  | "lot-desc"
  | "wafer-asc"
  | "wafer-desc";

function formatMetric(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(4);
}

function formatPrediction(value: unknown): string {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function formatYieldPercent(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : `${value.toFixed(2)}%`;
}

function riskLabel(value: unknown): string {
  if (value === "normal") return "정상";
  if (value === "warning") return "주의";
  if (value === "danger") return "위험";
  return "-";
}

function naturalCompare(left: unknown, right: unknown): number {
  return String(left ?? "").localeCompare(String(right ?? ""), "ko", {
    numeric: true,
    sensitivity: "base",
  });
}

function identifierText(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  return text || null;
}

function waferSlot(value: unknown): number | null {
  const text = identifierText(value);
  if (!text) return null;
  if (/^\d+$/.test(text)) return Number(text);
  const match = text.match(/(?:WAFER|WF|W)?[_-]?(\d+)$/i);
  return match ? Number(match[1]) : null;
}

function parseCombinedIdentifier(value: unknown) {
  const combined = identifierText(value);
  const match = combined?.match(
    /^(.*?)[_-]?(?:WAFER|WF|W)[_-]?(\d+)$/i,
  );
  if (!match) return { lot: null, wafer: null, slot: null };
  const lot = match[1].replace(/[_-]+$/, "") || null;
  const slot = Number(match[2]);
  return { lot, wafer: `W${String(slot).padStart(2, "0")}`, slot };
}

export function canonicalPredictionIdentifiers(
  row: PredictionRow,
  identifierColumn: string,
) {
  let combined =
    identifierText(row.Lot_Wafer_ID) ??
    identifierText(row.lot_wafer_id) ??
    (identifierColumn === "row_id" ? null : identifierText(row[identifierColumn]));
  const parsed = parseCombinedIdentifier(combined);
  const lot =
    identifierText(row.Lot_ID) ?? identifierText(row.lot_id) ?? parsed.lot;
  const sourceWafer =
    identifierText(row.Wafer_ID) ?? identifierText(row.wafer_id);
  const slot =
    waferSlot(row.Wafer_Slot) ??
    waferSlot(row.wafer_slot) ??
    waferSlot(sourceWafer) ??
    parsed.slot;
  const wafer =
    sourceWafer ??
    (slot === null ? null : `W${String(slot).padStart(2, "0")}`) ??
    parsed.wafer;
  if (!combined && lot && wafer) combined = `${lot}_${wafer}`;
  return {
    combined: combined ?? "-",
    lot: lot ?? "-",
    wafer: wafer ?? "-",
    slot,
  };
}

export default function PredictionPage() {
  const [modelWarnings, setModelWarnings] = useState<string[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [thresholds, setThresholds] =
    useState<PredictionThresholds>(DEFAULT_THRESHOLDS);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const selectedModel = models.find((model) => model.model_id === selectedModelId && isModelUsable(model));
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionRunKey, setPredictionRunKey] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [searchText, setSearchText] = useState("");
  const [resultSort, setResultSort] =
    useState<ResultSort>("prediction-desc");
  const [resultPage, setResultPage] = useState(0);
  const [movingAverageWindow, setMovingAverageWindow] = useState(
    DEFAULT_MOVING_AVERAGE_WINDOW,
  );
  const [hiddenTrendSeries, setHiddenTrendSeries] = useState<
    Set<TrendSeries>
  >(new Set());
  const resultTableRef = useRef<HTMLDivElement>(null);
  const [activeView, setActiveView] = useState<"new" | "history">("new");
  const [historyItems, setHistoryItems] = useState<PredictionHistorySummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [restoredHistory, setRestoredHistory] = useState<PredictionHistorySummary | null>(null);
  const { formattedElapsed: formattedPredictionElapsed } = useElapsedTime({
    running: isPredicting,
    resetKey: predictionRunKey,
  });

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const predictionId = params.get("prediction_id") ?? sessionStorage.getItem("last_prediction_id");
    if (!predictionId) return;
    void getPredictionHistoryDetail(predictionId).then((detail) => {
      if (!detail.artifact?.response) return;
      setResult(restoreArtifactRows(detail.artifact.response, detail.artifact.rows));
      setRestoredHistory(detail.metadata);
      setSelectedModelId(detail.metadata.model_id ?? "");
      sessionStorage.setItem("last_prediction_id", predictionId);
    }).catch(() => sessionStorage.removeItem("last_prediction_id"));
  }, []);

  async function loadHistory() {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      setHistoryItems((await getPredictionHistory()).items);
    } catch (requestError) {
      setHistoryError(requestError instanceof Error ? requestError.message : "예측 이력을 불러오지 못했습니다.");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function openHistory(item: PredictionHistorySummary) {
    setHistoryError("");
    try {
      const detail = await getPredictionHistoryDetail(item.prediction_id);
      if (!detail.artifact?.response) throw new Error("저장된 예측 상세 Artifact를 읽을 수 없습니다.");
      setResult(restoreArtifactRows(detail.artifact.response, detail.artifact.rows));
      setResultPage(0);
      setRestoredHistory(detail.metadata);
      setSelectedModelId(detail.metadata.model_id ?? "");
      setActiveView("new");
      const url = new URL(window.location.href);
      url.searchParams.set("prediction_id", item.prediction_id);
      window.history.replaceState({}, "", url);
      sessionStorage.setItem("last_prediction_id", item.prediction_id);
    } catch (requestError) {
      setHistoryError(requestError instanceof Error ? requestError.message : "예측 이력을 열지 못했습니다.");
    }
  }

  async function removeHistory(item: PredictionHistorySummary) {
    if (!window.confirm("저장된 Wafer별 예측 결과가 삭제됩니다. 연결된 원인 분석은 유지됩니다.")) return;
    await deletePredictionHistory(item.prediction_id);
    if (restoredHistory?.prediction_id === item.prediction_id) {
      setResult(null);
      setRestoredHistory(null);
      sessionStorage.removeItem("last_prediction_id");
    }
    await loadHistory();
  }


  function selectFile(selectedFile?: File) {
    setResult(null);
    setResultPage(0);
    setError("");
    if (!selectedFile) {
      setFile(null);
      return;
    }
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setFile(null);
      setError("파일 크기는 20MB 이하여야 합니다.");
      return;
    }
    setFile(selectedFile);
  }

  async function handlePredict() {
    if (!file || !selectedModelId || isPredicting) return;
    setError("");
    setResult(null);
    setResultPage(0);
    setPredictionRunKey((current) => current + 1);
    setIsPredicting(true);
    try {
      const response = await predictCsv(file, selectedModelId, thresholds);
      setResult(response);
      setRestoredHistory(null);
      if (response.prediction_id) {
        const url = new URL(window.location.href);
        url.searchParams.set("prediction_id", response.prediction_id);
        window.history.replaceState({}, "", url);
        sessionStorage.setItem("last_prediction_id", response.prediction_id);
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "수율 예측 중 오류가 발생했습니다.",
      );
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleDownload() {
    if (!file || !selectedModelId || isDownloading) return;
    setError("");
    setIsDownloading(true);
    try {
      const blob = await downloadPredictions(
        file,
        selectedModelId,
        thresholds,
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `predictions_${selectedModelId}.csv`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "예측 결과를 다운로드하지 못했습니다.",
      );
    } finally {
      setIsDownloading(false);
    }
  }

  const filteredRows = useMemo(() => {
    if (!result) return [];
    const normalizedSearch = searchText.trim().toLowerCase();
    return result.predictions
      .filter((row) => {
        const matchesRisk =
          riskFilter === "all" || row.risk_level === riskFilter;
        const identifiers = canonicalPredictionIdentifiers(
          row,
          result.identifier_column,
        );
        const searchTarget = [
          identifiers.combined,
          identifiers.lot,
          identifiers.wafer,
        ].join(" ").toLowerCase();
        return (
          matchesRisk &&
          (!normalizedSearch || searchTarget.includes(normalizedSearch))
        );
      });
  }, [result, riskFilter, searchText]);

  const sortedRows = useMemo(() => {
    if (!result) return [];
    const predictionColumn = `predicted_${result.model.target}`;
    return [...filteredRows].sort((left, right) => {
      if (!resultSort.startsWith("prediction")) {
        const leftIds = canonicalPredictionIdentifiers(left, result.identifier_column);
        const rightIds = canonicalPredictionIdentifiers(right, result.identifier_column);
        const field = resultSort.startsWith("lot")
          ? "lot"
          : resultSort.startsWith("wafer")
            ? "wafer"
            : "combined";
        const comparison = naturalCompare(leftIds[field], rightIds[field]);
        return resultSort.endsWith("desc") ? -comparison : comparison;
      }
      const difference =
        Number(left[predictionColumn]) - Number(right[predictionColumn]);
      return resultSort === "prediction-asc" ? difference : -difference;
    });
  }, [filteredRows, result, resultSort]);

  const resultPageCount = Math.max(
    1,
    Math.ceil(sortedRows.length / RESULT_PAGE_SIZE),
  );
  const activeResultPage = Math.min(resultPage, resultPageCount - 1);
  const displayedRows = useMemo(
    () => sortedRows.slice(
      activeResultPage * RESULT_PAGE_SIZE,
      (activeResultPage + 1) * RESULT_PAGE_SIZE,
    ),
    [activeResultPage, sortedRows],
  );

  function changeResultPage(direction: -1 | 1) {
    setResultPage((page) =>
      Math.max(0, Math.min(resultPageCount - 1, page + direction)),
    );
    resultTableRef.current?.scrollTo({ top: 0 });
  }

  const trendData = useMemo(() => {
    if (!result || !filteredRows.length) return [];
    const predictedKey = `predicted_${result.model.target}`;
    const actualKey = `actual_${result.model.target}`;
    const predictedValues = filteredRows.map((row) => row[predictedKey]);
    const validPredictedValues = result.predictions
      .map((row) => row[predictedKey])
      .map(finiteNumber)
      .filter((value): value is number => value !== null);
    const predictedYieldMean = validPredictedValues.length
      ? validPredictedValues.reduce((sum, value) => sum + value, 0) /
        validPredictedValues.length
      : null;
    const predictedYieldMovingAverage = calculateMovingAverage(
      predictedValues,
      movingAverageWindow,
    );
    const completeTrend = filteredRows.map((row, index) => {
      const predicted = finiteNumber(row[predictedKey]);
      const identifiers = canonicalPredictionIdentifiers(
        row,
        result.identifier_column,
      );
      return {
        index: index + 1,
        identifier: identifiers.combined === "-" ? String(index + 1) : identifiers.combined,
        predicted,
        predictedYieldMean,
        predictedYieldMovingAverage: predictedYieldMovingAverage[index],
        actual:
          typeof row[actualKey] === "number"
            ? Number(row[actualKey])
            : undefined,
        risk: String(row.risk_level ?? "normal"),
      };
    });
    const sampleEvery = Math.max(
      1,
      Math.ceil(completeTrend.length / 80),
    );
    return completeTrend.filter(
      (_, index) =>
        index % sampleEvery === 0 || index === completeTrend.length - 1,
    );
  }, [filteredRows, movingAverageWindow, result]);

  const thresholdsNeedSeparation =
    Math.abs(thresholds.warning_threshold - thresholds.danger_threshold) < 2;
  const hasActualTrend = trendData.some((item) => item.actual !== undefined);
  const trendLegendPayload: LegendPayload[] = [
    ...(hasActualTrend
      ? [{ value: "실제 수율", dataKey: "actual", color: "var(--chart-actual)", type: "line" as const }]
      : []),
    { value: "예측 수율", dataKey: "predicted", color: "var(--chart-primary)", type: "line" as const },
    { value: "예측 수율 평균", dataKey: "predictedYieldMean", color: "var(--chart-mean)", type: "plainline" as const },
    { value: "예측 수율 이동 평균", dataKey: "predictedYieldMovingAverage", color: "var(--chart-moving-average)", type: "line" as const },
    { value: "Warning", dataKey: "warning", color: "var(--chart-warning)", type: "plainline" as const },
    { value: "Critical", dataKey: "critical", color: "var(--chart-critical)", type: "plainline" as const },
  ].map((entry) => ({
    ...entry,
    inactive: hiddenTrendSeries.has(entry.dataKey as TrendSeries),
  }));

  function toggleTrendSeries(entry: LegendPayload) {
    const series = entry.dataKey as TrendSeries;
    setHiddenTrendSeries((current) => {
      const next = new Set(current);
      if (next.has(series)) next.delete(series);
      else next.add(series);
      return next;
    });
  }

  const trendDashPattern: Partial<Record<TrendSeries, string>> = {
    predictedYieldMean: "2 5",
    warning: "4 5",
    critical: "7 3 2 3",
  };

  const visibleWarnings = [...modelWarnings, ...(result?.warnings ?? [])];

  function handleResultSort(nextSort: ResultSort) {
    setResultSort(nextSort);
    setResultPage(0);
    resultTableRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }

  const diagnostics = useMemo(() => {
    if (!result) return { points: [], histogram: [] };
    const predictedKey = `predicted_${result.model.target}`;
    const actualKey = `actual_${result.model.target}`;
    const points = result.predictions.flatMap((row) => {
      const actual = row[actualKey];
      const predicted = row[predictedKey];
      return typeof actual === "number" && typeof predicted === "number"
        ? [{ actual, predicted, error: predicted - actual }]
        : [];
    });
    if (!points.length) return { points, histogram: [] };
    const errors = points.map((point) => point.error);
    const minimum = Math.min(...errors);
    const maximum = Math.max(...errors);
    const binCount = Math.min(12, Math.max(5, Math.ceil(Math.sqrt(errors.length))));
    const width = maximum === minimum ? 1 : (maximum - minimum) / binCount;
    const histogram = Array.from({ length: binCount }, (_, index) => ({
      label: `${(minimum + index * width).toFixed(1)}`,
      count: 0,
    }));
    errors.forEach((errorValue) => {
      const index = Math.min(
        binCount - 1,
        Math.max(0, Math.floor((errorValue - minimum) / width)),
      );
      histogram[index].count += 1;
    });
    return { points, histogram };
  }, [result]);

  const ensembleInfo = result?.predictions[0] as (PredictionRow & {
    final_strategy?: string;
    ensemble_used?: boolean;
    base_model_count?: number;
    model_agreement?: { available?: boolean; prediction_spread?: number | null };
  }) | undefined;

  const predictionColumn = result
    ? `predicted_${result.model.target}`
    : "predicted_Y";
  const actualColumn = result
    ? `actual_${result.model.target}`
    : "actual_Y";

  return (
    <div className="appShell">
      <Sidebar activeItem="수율 예측" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro">
            <span className="eyebrow">머신러닝 추론</span>
            <h1>수율 예측</h1>
            <p>
              저장된 학습 모델로 신규 공정 데이터의 목표값을 예측합니다.
            </p>
          </section>

          <div className="trainingViewTabs" role="tablist" aria-label="수율 예측 보기">
            <button className={activeView === "new" ? "active" : ""} type="button" role="tab" aria-selected={activeView === "new"} onClick={() => setActiveView("new")}>새 예측</button>
            <button className={activeView === "history" ? "active" : ""} type="button" role="tab" aria-selected={activeView === "history"} onClick={() => { setActiveView("history"); void loadHistory(); }}>예측 이력</button>
          </div>

          {activeView === "new" ? (
            <>
          <section className="uploadCard" aria-labelledby="prediction-form-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">예측 설정</span>
                <h2 id="prediction-form-title">수율 예측</h2>
              </div>
              <p>학습 당시 feature 순서와 전처리 규칙을 적용합니다.</p>
            </div>

            <div className="predictionControls">
              <div className="fieldGroup modelField">
                <label htmlFor="prediction-model">학습 모델</label>
                <ModelSelector
                  value={selectedModelId}
                  disabled={isPredicting}
                  onValueChange={(nextModelId, reason) => {
                    setSelectedModelId(nextModelId);
                    if (reason !== "reconcile") setResult(null);
                  }}
                  onModelsChange={(nextModels, warnings) => {
                    setModels(nextModels);
                    setModelWarnings(warnings);
                  }}
                  ariaLabel="수율 예측 모델 선택"
                />
              </div>
              <div className="fieldGroup">
                <label htmlFor="warning-threshold">정상 기준</label>
                <input
                  id="warning-threshold"
                  type="number"
                  step="0.1"
                  value={thresholds.warning_threshold}
                  onChange={(event) =>
                    setThresholds((current) => ({
                      ...current,
                      warning_threshold: Number(event.target.value),
                    }))
                  }
                />
              </div>
              <div className="fieldGroup">
                <label htmlFor="danger-threshold">위험 기준</label>
                <input
                  id="danger-threshold"
                  type="number"
                  step="0.1"
                  value={thresholds.danger_threshold}
                  onChange={(event) =>
                    setThresholds((current) => ({
                      ...current,
                      danger_threshold: Number(event.target.value),
                    }))
                  }
                />
              </div>
            </div>
            <div className="predictionInputStack">
              <SelectedModelSummary model={selectedModel} />

              <CsvUploadPanel
                id="prediction-file"
                file={file}
                onFileSelect={selectFile}
                disabled={isPredicting}
                compact
                title="예측할 CSV를 드래그하거나 클릭하여 선택하세요."
              />
            </div>

            {error && <div className="messageBox error" role="alert">{error}</div>}
            {visibleWarnings.length > 0 && (
              <div className="trainingWarnings predictionWarnings">
                <strong>주의사항 {visibleWarnings.length}건</strong>
                <ul>
                  {visibleWarnings.map(
                    (warning) => <li key={warning}>{warning}</li>,
                  )}
                </ul>
              </div>
            )}

            <div className="uploadActions">
              <button
                className="button primary"
                type="button"
                disabled={!file || !selectedModelId || isPredicting}
                data-loading={isPredicting}
                aria-busy={isPredicting}
                onClick={handlePredict}
              >
                {isPredicting ? (
                  <OperationProgress
                    message="수율을 예측하고 있습니다…"
                    timeLabel="추론 시간"
                    formattedElapsed={formattedPredictionElapsed}
                  />
                ) : "수율 예측 실행"}
              </button>
            </div>
          </section>

          {result && (
            <>
              <section className="resultCard" aria-labelledby="prediction-summary-title">
                {restoredHistory && (
                  <div className="historyRestoreBanner">
                    <div><strong>저장된 예측 결과</strong><span>{new Date(restoredHistory.created_at).toLocaleString("ko-KR")} · {restoredHistory.source_filename}</span></div>
                    <div className="historyRowActions">
                      <button className="button secondary" type="button" onClick={() => { setResult(null); setRestoredHistory(null); const url = new URL(window.location.href); url.searchParams.delete("prediction_id"); window.history.replaceState({}, "", url); }}>새 예측</button>
                      <a className="button secondary" href={`/root-cause?prediction_id=${encodeURIComponent(restoredHistory.prediction_id)}&model_id=${encodeURIComponent(restoredHistory.model_id ?? "")}`}>불량 원인 분석 열기</a>
                    </div>
                  </div>
                )}
                <div className="resultHeader">
                  <div>
                    <span className="sectionLabel">예측 요약</span>
                    <h2 id="prediction-summary-title">
                      {result.model.target} · {result.model.model_name}
                    </h2>
                  </div>
                  <button
                    className="button secondary"
                    type="button"
                    onClick={handleDownload}
                    disabled={isDownloading}
                    data-loading={isDownloading}
                    aria-busy={isDownloading}
                  >
                    {isDownloading ? "다운로드 중..." : "CSV 다운로드"}
                  </button>
                </div>
                <div className="predictionKpiGrid">
                  <div><span>분석 Wafer 수</span><strong>{result.summary.total_rows}</strong></div>
                  <div><span>평균 예측 수율</span><strong>{result.summary.average_prediction.toFixed(2)}</strong></div>
                  <div className="normalKpi"><span>정상 Wafer</span><strong>{result.summary.normal_count}</strong></div>
                  <div className="warningKpi"><span>주의 Wafer</span><strong>{result.summary.warning_count}</strong></div>
                  <div className="dangerKpi"><span>위험 Wafer</span><strong>{result.summary.danger_count}</strong></div>
                  {ensembleInfo?.final_strategy && <div><span>Final Strategy</span><strong>{ensembleInfo.final_strategy}</strong></div>}
                  {ensembleInfo?.ensemble_used !== undefined && <div><span>모델 구성</span><strong>{ensembleInfo.ensemble_used ? `${ensembleInfo.base_model_count ?? "-"}-Model Ensemble` : "Single Model"}</strong></div>}
                  {ensembleInfo?.model_agreement?.available && <div title="Model Agreement는 앙상블 구성 모델 간 예측 차이를 나타내는 참고 지표입니다."><span>Ensemble Agreement · spread</span><strong>{formatMetric(ensembleInfo.model_agreement.prediction_spread)}</strong></div>}
                  {result.summary.evaluation && (
                    <>
                      <div><span>R²</span><strong>{formatMetric(result.summary.evaluation.r2)}</strong></div>
                      <div><span>RMSE</span><strong>{formatMetric(result.summary.evaluation.rmse)}</strong></div>
                      <div><span>MAE</span><strong>{formatMetric(result.summary.evaluation.mae)}</strong></div>
                    </>
                  )}
                </div>
                <PreprocessingSummary summary={result.preprocessing} />
              </section>

              <section
                className="resultCard predictionTrendCard"
                aria-labelledby="prediction-trend-title"
              >
                <div className="sectionHeading compact predictionTrendHeading">
                  <div>
                    <span className="sectionLabel">Yield trend</span>
                    <h2 id="prediction-trend-title">수율 예측 추이</h2>
                  </div>
                  <div className="movingAverageControl">
                    <label htmlFor="moving-average-window">이동평균 구간</label>
                    <select
                      id="moving-average-window"
                      value={movingAverageWindow}
                      onChange={(event) =>
                        setMovingAverageWindow(Number(event.target.value))
                      }
                    >
                      {MOVING_AVERAGE_WINDOWS.map((windowSize) => (
                        <option key={windowSize} value={windowSize}>
                          {windowSize}개
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div
                  className="predictionTrendCanvas"
                  role="img"
                  aria-label={`Wafer 원본 순서별 실제 수율, 예측 수율, 예측 수율 평균과 최근 ${movingAverageWindow}개 Wafer 이동 평균 선 차트`}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={trendData}
                      margin={{ top: 10, right: 112, bottom: 8, left: 2 }}
                    >
                      <CartesianGrid
                        stroke="var(--chart-grid)"
                        strokeDasharray="3 5"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="index"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                        label={{
                          value: "Wafer sequence",
                          position: "insideBottom",
                          offset: -2,
                          fontSize: 10,
                          fill: "var(--chart-axis)",
                        }}
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                        tickFormatter={(value) => `${Number(value).toFixed(0)}%`}
                        width={44}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.[0]) return null;
                          const item = payload[0].payload as (typeof trendData)[number];
                          return (
                            <div className="chartTooltip">
                              <strong>{item.identifier}</strong>
                              <span>
                                예측 수율: {formatYieldPercent(item.predicted)}
                              </span>
                              {item.actual !== undefined && (
                                <span>
                                  실제 수율: {formatYieldPercent(item.actual)}
                                </span>
                              )}
                              <span>
                                예측 수율 평균: {formatYieldPercent(item.predictedYieldMean)}
                              </span>
                              <span>
                                예측 수율 이동 평균({movingAverageWindow}): {formatYieldPercent(item.predictedYieldMovingAverage)}
                              </span>
                              <span>이동평균 구간: {movingAverageWindow}개 Wafer</span>
                              <span>위험도: {riskLabel(item.risk)}</span>
                            </div>
                          );
                        }}
                      />
                      <Legend
                        verticalAlign="top"
                        align="right"
                        iconType="plainline"
                        wrapperStyle={{
                          fontSize: 11,
                          color: "var(--chart-axis)",
                        }}
                        content={() => (
                          <div className="trendLegend" aria-label="차트 계열 표시 설정">
                            {trendLegendPayload.map((entry) => {
                              const series = entry.dataKey as TrendSeries;
                              const isVisible = !hiddenTrendSeries.has(series);
                              return (
                                <button
                                  key={series}
                                  type="button"
                                  aria-pressed={isVisible}
                                  onClick={() => toggleTrendSeries(entry)}
                                >
                                  <svg width="24" height="10" aria-hidden="true">
                                    <line
                                      x1="1"
                                      y1="5"
                                      x2="23"
                                      y2="5"
                                      stroke={entry.color}
                                      strokeWidth={series === "predicted" || series === "predictedYieldMovingAverage" ? 3 : 2}
                                      strokeDasharray={trendDashPattern[series]}
                                    />
                                  </svg>
                                  {entry.value}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      />
                      {!hiddenTrendSeries.has("warning") && (
                        <ReferenceLine
                          y={thresholds.warning_threshold}
                          stroke="var(--chart-warning)"
                          strokeDasharray="4 5"
                          label={
                            <ThresholdLabel
                              label="Warning"
                              value={thresholds.warning_threshold}
                              color="var(--chart-warning)"
                              yShift={thresholdsNeedSeparation ? -12 : 0}
                            />
                          }
                        />
                      )}
                      {!hiddenTrendSeries.has("critical") && (
                        <ReferenceLine
                          y={thresholds.danger_threshold}
                          stroke="var(--chart-critical)"
                          strokeDasharray="7 3 2 3"
                          label={
                            <ThresholdLabel
                              label="Critical"
                              value={thresholds.danger_threshold}
                              color="var(--chart-critical)"
                              yShift={thresholdsNeedSeparation ? 12 : 0}
                            />
                          }
                        />
                      )}
                      {hasActualTrend && (
                        <Line
                          type="monotone"
                          dataKey="actual"
                          name="실제 수율"
                          stroke="var(--chart-actual)"
                          strokeWidth={2.1}
                          dot={false}
                          connectNulls
                          animationDuration={220}
                          hide={hiddenTrendSeries.has("actual")}
                        />
                      )}
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="예측 수율"
                        stroke="var(--chart-primary)"
                        strokeWidth={2.8}
                        animationDuration={220}
                        hide={hiddenTrendSeries.has("predicted")}
                        dot={(dotProps) => {
                          const point = dotProps.payload as (typeof trendData)[number];
                          const isRisk = point.risk !== "normal";
                          return (
                            <circle
                              key={`${point.identifier}-${point.index}`}
                              cx={dotProps.cx}
                              cy={dotProps.cy}
                              r={isRisk ? 3.5 : 1.8}
                              fill={
                                point.risk === "danger"
                                  ? "#b33a46"
                                  : point.risk === "warning"
                                    ? "#a96208"
                                    : "var(--chart-primary)"
                              }
                              stroke="var(--surface)"
                              strokeWidth={isRisk ? 1.5 : 0}
                            />
                          );
                        }}
                      />
                      <Line
                        type="linear"
                        dataKey="predictedYieldMean"
                        name="예측 수율 평균"
                        stroke="var(--chart-mean)"
                        strokeWidth={1.8}
                        strokeDasharray="2 5"
                        dot={false}
                        animationDuration={220}
                        hide={hiddenTrendSeries.has("predictedYieldMean")}
                      />
                      <Line
                        type="monotone"
                        dataKey="predictedYieldMovingAverage"
                        name="예측 수율 이동 평균"
                        stroke="var(--chart-moving-average)"
                        strokeWidth={2.7}
                        dot={false}
                        animationDuration={220}
                        hide={hiddenTrendSeries.has("predictedYieldMovingAverage")}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  </div>
              </section>

              {diagnostics.points.length > 0 && (
                <section className="predictionDiagnosticsGrid">
                  <article className="resultCard">
                    <div className="sectionHeading compact">
                      <div>
                        <span className="sectionLabel">Model fit</span>
                        <h2>Actual vs Predicted</h2>
                      </div>
                    </div>
                    <div className="predictionDiagnosticCanvas">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 16, bottom: 20, left: 8 }}>
                          <CartesianGrid stroke="var(--chart-grid)" />
                          <XAxis type="number" dataKey="actual" name="Actual" tick={{ fontSize: 10, fill: "var(--chart-axis)" }} />
                          <YAxis type="number" dataKey="predicted" name="Predicted" tick={{ fontSize: 10, fill: "var(--chart-axis)" }} />
                          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                          <Scatter data={diagnostics.points} fill="var(--chart-primary)" fillOpacity={0.72} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </article>
                  <article className="resultCard">
                    <div className="sectionHeading compact">
                      <div>
                        <span className="sectionLabel">Prediction error</span>
                        <h2>예측 오차 Histogram</h2>
                      </div>
                    </div>
                    <div className="predictionDiagnosticCanvas">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={diagnostics.histogram}>
                          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                          <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--chart-axis)" }} />
                          <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "var(--chart-axis)" }} />
                          <Tooltip />
                          <Bar dataKey="count" name="Wafer 수" fill="#647185" radius={[5, 5, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </article>
                </section>
              )}

              <section className="resultCard" aria-labelledby="prediction-table-title">
                <div className="resultHeader predictionTableHeader">
                  <div>
                    <span className="sectionLabel">Wafer별 결과</span>
                    <h2 id="prediction-table-title">예측 데이터</h2>
                  </div>
                  <div className="tableTools">
                    <input
                      type="search"
                      placeholder="Lot/Wafer ID 검색"
                      aria-label="Lot_Wafer_ID, Lot_ID 또는 Wafer_ID 검색"
                      value={searchText}
                      onChange={(event) => {
                        setSearchText(event.target.value);
                        setResultPage(0);
                        resultTableRef.current?.scrollTo({ top: 0 });
                      }}
                    />
                    <select
                      aria-label="위험 상태 필터"
                      value={riskFilter}
                      onChange={(event) => {
                        setRiskFilter(event.target.value as RiskFilter);
                        setResultPage(0);
                        resultTableRef.current?.scrollTo({ top: 0 });
                      }}
                    >
                      <option value="all">전체 위험 상태</option>
                      <option value="normal">정상</option>
                      <option value="warning">주의</option>
                      <option value="danger">위험</option>
                    </select>
                    <select
                      aria-label="Wafer 결과 정렬"
                      value={resultSort}
                      onChange={(event) =>
                        handleResultSort(event.target.value as ResultSort)
                      }
                    >
                      <option value="prediction-desc">예측 수율 높은 순</option>
                      <option value="prediction-asc">예측 수율 낮은 순</option>
                      <option value="id-asc">ID 오름차순</option>
                      <option value="id-desc">ID 내림차순</option>
                      <option value="lot-asc">Lot_ID 오름차순</option>
                      <option value="lot-desc">Lot_ID 내림차순</option>
                      <option value="wafer-asc">Wafer_ID 오름차순</option>
                      <option value="wafer-desc">Wafer_ID 내림차순</option>
                    </select>
                  </div>
                </div>
                {result.truncated && (
                  <p className="predictionPreviewNotice" role="status">
                    응답 미리보기 {result.predictions.length.toLocaleString()}행 / 전체 {result.summary.total_rows.toLocaleString()}행입니다.
                    전체 결과는 저장된 예측 이력 상세 또는 CSV 다운로드에서 확인할 수 있습니다.
                  </p>
                )}
                <div
                  className="tableWrap predictionResultScroll"
                  ref={resultTableRef}
                >
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Lot_Wafer_ID</th>
                        <th className="secondaryColumn" scope="col">Lot_ID</th>
                        <th className="secondaryColumn" scope="col">Wafer_ID</th>
                        <th className="secondaryColumn" scope="col">Wafer_Slot</th>
                        <th scope="col">예측값</th>
                        <th scope="col">위험 상태</th>
                        <th scope="col">Confidence</th>
                        {result.predictions.some((row) => actualColumn in row) && (
                          <>
                            <th scope="col">실제값</th>
                            <th scope="col">절대 오차</th>
                          </>
                        )}
                        <th className="secondaryColumn" scope="col">모델</th>
                        <th className="secondaryColumn" scope="col">분석 상태</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayedRows.map((row: PredictionRow, index) => {
                        const identifiers = canonicalPredictionIdentifiers(
                          row,
                          result.identifier_column,
                        );
                        return (
                        <tr key={`${identifiers.combined}-${index}`}>
                          <td className="identifierCell" title={identifiers.combined}>{identifiers.combined}</td>
                          <td className="secondaryColumn identifierCell" title={identifiers.lot}>{identifiers.lot}</td>
                          <td className="secondaryColumn identifierCell" title={identifiers.wafer}>{identifiers.wafer}</td>
                          <td className="secondaryColumn">{identifiers.slot ?? "-"}</td>
                          <td>{formatPrediction(row[predictionColumn])}</td>
                          <td>
                            <span className={`riskBadge ${String(row.risk_level ?? "")}`}>
                              {riskLabel(row.risk_level)}
                            </span>
                          </td>
                          <td>{identifierText(row.confidence) ?? "-"}</td>
                          {result.predictions.some((item) => actualColumn in item) && (
                            <>
                              <td>{formatPrediction(row[actualColumn])}</td>
                              <td>{formatPrediction(row.absolute_error)}</td>
                            </>
                          )}
                          <td className="secondaryColumn">
                            {result.model.model_name}
                          </td>
                          <td className="secondaryColumn">분석 완료</td>
                        </tr>
                        );
                      })}
                      {!displayedRows.length && (
                        <tr>
                          <td
                            className="emptyTableCell"
                            colSpan={
                              result.predictions.some(
                                (item) => actualColumn in item,
                              )
                                ? 11
                                : 9
                            }
                          >
                            예측 결과가 없습니다. 필터 조건을 확인하세요.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                {sortedRows.length > RESULT_PAGE_SIZE && (
                  <nav className="tablePagination" aria-label="Wafer 예측 결과 페이지">
                    <button
                      className="button secondary"
                      type="button"
                      disabled={activeResultPage === 0}
                      onClick={() => changeResultPage(-1)}
                    >
                      이전
                    </button>
                    <span aria-live="polite" aria-atomic="true">
                      {activeResultPage + 1} / {resultPageCount} · 총 {sortedRows.length.toLocaleString()}행
                    </span>
                    <button
                      className="button secondary"
                      type="button"
                      disabled={activeResultPage >= resultPageCount - 1}
                      onClick={() => changeResultPage(1)}
                    >
                      다음
                    </button>
                  </nav>
                )}
              </section>
            </>
          )}
            </>
          ) : (
            <section className="resultCard historyCard" aria-labelledby="prediction-history-title">
              <div className="sectionHeading compact"><div><span className="sectionLabel">Prediction History</span><h2 id="prediction-history-title">예측 이력</h2></div><button className="button secondary" type="button" onClick={() => void loadHistory()}>새로고침</button></div>
              {historyLoading ? <p className="emptyMessage">예측 이력을 불러오는 중입니다.</p> : historyError ? <div className="retryMessage"><p className="errorMessage">{historyError}</p><button className="button secondary" type="button" onClick={() => void loadHistory()}>다시 시도</button></div> : !historyItems.length ? <p className="emptyMessage">저장된 수율 예측 이력이 없습니다.</p> : (
                <div className="tableWrap historyTableScroll"><table><thead><tr><th>생성 시각</th><th>파일명</th><th>모델</th><th>Wafer</th><th>Lot</th><th>평균 수율</th><th>Critical</th><th>Warning</th><th>상태</th><th>작업</th></tr></thead><tbody>{historyItems.map((item) => { const summary = item.summary ?? {}; return <tr key={item.prediction_id}><td>{new Date(item.created_at).toLocaleString("ko-KR")}</td><td>{item.source_filename ?? "데이터 없음"}</td><td>{item.model_name_snapshot ?? item.model_id ?? "삭제된 모델"}</td><td>{item.row_count ?? "데이터 없음"}</td><td>{item.lot_count ?? "데이터 없음"}</td><td>{typeof summary.average_predicted_yield === "number" ? summary.average_predicted_yield.toFixed(2) : "데이터 없음"}</td><td>{String(summary.critical_count ?? "데이터 없음")}</td><td>{String(summary.warning_count ?? "데이터 없음")}</td><td>{item.status}</td><td><div className="historyRowActions"><button type="button" className="button secondary" onClick={() => void openHistory(item)}>상세 보기</button><a className="button secondary" href={`/root-cause?prediction_id=${encodeURIComponent(item.prediction_id)}&model_id=${encodeURIComponent(item.model_id ?? "")}`}>원인 분석</a><button type="button" className="button danger" onClick={() => void removeHistory(item)}>삭제</button></div></td></tr>; })}</tbody></table></div>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
