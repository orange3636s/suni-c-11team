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

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import {
  downloadPredictions,
  getModels,
  predictCsv,
} from "@/lib/api";
import type {
  ModelSummary,
  PredictionResponse,
  PredictionRow,
  PredictionThresholds,
} from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const DEFAULT_THRESHOLDS: PredictionThresholds = {
  warning_threshold: 95,
  danger_threshold: 90,
};

type RiskFilter = "all" | "normal" | "warning" | "danger";
type ResultSort =
  | "prediction-desc"
  | "prediction-asc"
  | "id-asc"
  | "id-desc";

function formatMetric(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(4);
}

function formatPrediction(value: unknown): string {
  return typeof value === "number" ? value.toFixed(2) : "-";
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

function identifierParts(value: unknown): { lot: string; wafer: string } {
  const identifier = String(value ?? "");
  const match = identifier.match(/^(.*?)[_-](?:WF|WAFER)[_-]?(\d+)$/i);
  return match
    ? { lot: match[1], wafer: match[2] }
    : { lot: "-", wafer: "-" };
}

export default function PredictionPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelWarnings, setModelWarnings] = useState<string[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [thresholds, setThresholds] =
    useState<PredictionThresholds>(DEFAULT_THRESHOLDS);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [searchText, setSearchText] = useState("");
  const [resultSort, setResultSort] =
    useState<ResultSort>("prediction-desc");
  const resultTableRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;
    async function loadModels() {
      try {
        const response = await getModels();
        if (!mounted) return;
        setModels(response.models);
        setModelWarnings(response.warnings);
        const defaultModel =
          response.models.find((model) => model.target === "Y") ??
          response.models[0];
        const requestedModelId = new URLSearchParams(
          window.location.search,
        ).get("model_id");
        const requestedModel = response.models.find(
          (model) => model.model_id === requestedModelId,
        );
        setSelectedModelId(
          requestedModel?.model_id ?? defaultModel?.model_id ?? "",
        );
      } catch (requestError) {
        if (mounted) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "모델 목록을 불러오지 못했습니다.",
          );
        }
      } finally {
        if (mounted) setIsLoadingModels(false);
      }
    }
    void loadModels();
    return () => {
      mounted = false;
    };
  }, []);

  const selectedModel = models.find(
    (model) => model.model_id === selectedModelId,
  );

  function selectFile(selectedFile?: File) {
    setResult(null);
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
    setIsPredicting(true);
    try {
      setResult(
        await predictCsv(file, selectedModelId, thresholds),
      );
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
        const identifier = String(
          row[result.identifier_column] ?? "",
        ).toLowerCase();
        return (
          matchesRisk &&
          (!normalizedSearch || identifier.includes(normalizedSearch))
        );
      });
  }, [result, riskFilter, searchText]);

  const displayedRows = useMemo(() => {
    if (!result) return [];
    const predictionColumn = `predicted_${result.model.target}`;
    return [...filteredRows].sort((left, right) => {
      if (resultSort === "id-asc" || resultSort === "id-desc") {
        const comparison = naturalCompare(
          left[result.identifier_column],
          right[result.identifier_column],
        );
        return resultSort === "id-desc" ? -comparison : comparison;
      }
      const difference =
        Number(left[predictionColumn]) - Number(right[predictionColumn]);
      return resultSort === "prediction-asc" ? difference : -difference;
    });
  }, [filteredRows, result, resultSort]);

  const trendData = useMemo(() => {
    if (!result || !filteredRows.length) return [];
    const predictedKey = `predicted_${result.model.target}`;
    const actualKey = `actual_${result.model.target}`;
    const predictedValues = filteredRows.map((row) =>
      Number(row[predictedKey]),
    );
    const overallMean =
      predictedValues.reduce((sum, value) => sum + value, 0) /
      predictedValues.length;
    let cumulativeTotal = 0;
    const completeTrend = filteredRows.map((row, index) => {
      const predicted = Number(row[predictedKey]);
      cumulativeTotal += predicted;
      return {
        index: index + 1,
        identifier: String(row[result.identifier_column] ?? index + 1),
        predicted,
        overallMean,
        cumulativeMean: cumulativeTotal / (index + 1),
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
  }, [filteredRows, result]);

  const resultSortLabel = {
    "prediction-desc": "예측 수율 높은 순",
    "prediction-asc": "예측 수율 낮은 순",
    "id-asc": "LOT_WAFER_ID 오름차순",
    "id-desc": "LOT_WAFER_ID 내림차순",
  }[resultSort];

  function handleResultSort(nextSort: ResultSort) {
    setResultSort(nextSort);
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

  if (!isLoadingModels && models.length === 0) {
    return (
      <div className="appShell">
        <Sidebar activeItem="수율 예측" />
        <div className="contentShell">
          <Header />
          <main className="mainContent uploadPage">
            <section className="emptyModelState">
              <span className="eyebrow">수율 예측</span>
              <h1>사용 가능한 학습 모델이 없습니다.</h1>
              <p>먼저 모델 학습 페이지에서 모델을 생성하세요.</p>
              {error && <div className="messageBox error">{error}</div>}
              <a className="button primary linkButton" href="/training">
                모델 학습 페이지로 이동
              </a>
            </section>
          </main>
        </div>
      </div>
    );
  }

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
            <h1>Wafer 수율 예측</h1>
            <p>
              저장된 학습 모델로 신규 공정 데이터의 목표값을 예측합니다.
            </p>
          </section>

          <section className="uploadCard" aria-labelledby="prediction-form-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">예측 설정</span>
                <h2 id="prediction-form-title">모델과 신규 CSV</h2>
              </div>
              <p>학습 당시 feature 순서와 전처리 규칙을 적용합니다.</p>
            </div>

            <div className="predictionControls">
              <div className="fieldGroup modelField">
                <label htmlFor="prediction-model">학습 모델</label>
                <select
                  id="prediction-model"
                  value={selectedModelId}
                  onChange={(event) => {
                    setSelectedModelId(event.target.value);
                    setResult(null);
                  }}
                >
                  {models.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.target} · {model.model_name} ·{" "}
                      {new Date(model.created_at).toLocaleString("ko-KR")}
                    </option>
                  ))}
                </select>
                {selectedModel && (
                  <span className="fieldHint">
                    Test R² {formatMetric(selectedModel.test_metrics.r2)} ·
                    RMSE {formatMetric(selectedModel.test_metrics.rmse)} ·
                    Feature {selectedModel.feature_count}개
                  </span>
                )}
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

            <CsvUploadPanel
              id="prediction-file"
              file={file}
              onFileSelect={selectFile}
              disabled={isPredicting}
              compact
              title="예측할 CSV를 드래그하거나 클릭하여 선택하세요."
            />

            {error && <div className="messageBox error" role="alert">{error}</div>}
            {[...modelWarnings, ...(result?.warnings ?? [])].length > 0 && (
              <div className="trainingWarnings">
                <strong>주의사항</strong>
                <ul>
                  {[...modelWarnings, ...(result?.warnings ?? [])].map(
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
                {isPredicting ? "예측 중..." : "수율 예측 실행"}
              </button>
            </div>
          </section>

          {result && (
            <>
              <section className="resultCard" aria-labelledby="prediction-summary-title">
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
                  {result.summary.evaluation && (
                    <>
                      <div><span>R²</span><strong>{formatMetric(result.summary.evaluation.r2)}</strong></div>
                      <div><span>RMSE</span><strong>{formatMetric(result.summary.evaluation.rmse)}</strong></div>
                      <div><span>MAE</span><strong>{formatMetric(result.summary.evaluation.mae)}</strong></div>
                    </>
                  )}
                </div>
              </section>

              <section
                className="resultCard predictionTrendCard"
                aria-labelledby="prediction-trend-title"
              >
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">Yield trend</span>
                    <h2 id="prediction-trend-title">Wafer 수율 예측 추이</h2>
                  </div>
                  <span className="fieldHint">
                    최대 80개 지점으로 균등 샘플링
                  </span>
                </div>
                <p className="chartDescription">
                  현재 필터에 포함된 전체 Wafer의 평균과 누적 평균을 함께
                  표시합니다. 누적 평균은 필터 적용 후 원본 CSV 행 순서
                  기준이며 시간 추세를 의미하지 않습니다.
                </p>
                <div
                  className="predictionTrendCanvas"
                  role="img"
                  aria-label="Wafer 순서별 실제 수율과 예측 수율 선 차트"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={trendData}
                      margin={{ top: 10, right: 22, bottom: 8, left: 2 }}
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
                                예측 수율: {item.predicted.toFixed(2)}%
                              </span>
                              {item.actual !== undefined && (
                                <span>
                                  실제 수율: {item.actual.toFixed(2)}%
                                </span>
                              )}
                              <span>
                                전체 평균: {item.overallMean.toFixed(2)}%
                              </span>
                              <span>
                                누적 평균: {item.cumulativeMean.toFixed(2)}%
                              </span>
                              <span>상태: {riskLabel(item.risk)}</span>
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
                      />
                      <ReferenceLine
                        y={thresholds.warning_threshold}
                        stroke="#a96208"
                        strokeDasharray="5 5"
                        label={{
                          value: "Warning",
                          fill: "#a96208",
                          fontSize: 10,
                          position: "insideTopRight",
                        }}
                      />
                      <ReferenceLine
                        y={thresholds.danger_threshold}
                        stroke="#b33a46"
                        strokeDasharray="5 5"
                        label={{
                          value: "Critical",
                          fill: "#b33a46",
                          fontSize: 10,
                          position: "insideBottomRight",
                        }}
                      />
                      {trendData.some(
                        (item) => item.actual !== undefined,
                      ) && (
                        <Line
                          type="monotone"
                          dataKey="actual"
                          name="실제 수율"
                          stroke="#7d8796"
                          strokeWidth={1.5}
                          dot={false}
                          connectNulls
                          animationDuration={220}
                        />
                      )}
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="예측 수율"
                        stroke="var(--chart-primary)"
                        strokeWidth={2.2}
                        animationDuration={220}
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
                        dataKey="overallMean"
                        name="전체 예측 수율 평균"
                        stroke="var(--chart-secondary)"
                        strokeWidth={1.4}
                        strokeDasharray="6 5"
                        dot={false}
                        animationDuration={220}
                      />
                      <Line
                        type="monotone"
                        dataKey="cumulativeMean"
                        name="누적 예측 수율 평균"
                        stroke="var(--warning)"
                        strokeWidth={1.8}
                        strokeDasharray="3 3"
                        dot={false}
                        animationDuration={220}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  </div>
                  <p className="trendOrderNote">
                    표 정렬: {resultSortLabel} · 차트 누적 평균: 원본 CSV 행 순서
                  </p>
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
                      placeholder="Lot_Wafer_ID 검색"
                      aria-label="Wafer 또는 LOT 식별자 검색"
                      value={searchText}
                      onChange={(event) => setSearchText(event.target.value)}
                    />
                    <select
                      aria-label="위험 상태 필터"
                      value={riskFilter}
                      onChange={(event) =>
                        setRiskFilter(event.target.value as RiskFilter)
                      }
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
                    </select>
                  </div>
                </div>
                <div
                  className="tableWrap predictionResultScroll"
                  ref={resultTableRef}
                >
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">{result.identifier_column}</th>
                        <th className="secondaryColumn" scope="col">Lot ID</th>
                        <th className="secondaryColumn" scope="col">Wafer ID</th>
                        <th scope="col">예측값</th>
                        <th scope="col">위험 상태</th>
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
                        const identifier = row[result.identifier_column];
                        const parts = identifierParts(identifier);
                        return (
                        <tr key={`${String(identifier)}-${index}`}>
                          <td>{String(identifier ?? "-")}</td>
                          <td className="secondaryColumn">{parts.lot}</td>
                          <td className="secondaryColumn">{parts.wafer}</td>
                          <td>{formatPrediction(row[predictionColumn])}</td>
                          <td>
                            <span className={`riskBadge ${String(row.risk_level ?? "")}`}>
                              {riskLabel(row.risk_level)}
                            </span>
                          </td>
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
                                ? 9
                                : 7
                            }
                          >
                            예측 결과가 없습니다. 필터 조건을 확인하세요.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
