"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

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
type SortDirection = "asc" | "desc";

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

export default function PredictionPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
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
  const [isDragging, setIsDragging] = useState(false);
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [searchText, setSearchText] = useState("");
  const [sortDirection, setSortDirection] =
    useState<SortDirection>("desc");

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
        setSelectedModelId(defaultModel?.model_id ?? "");
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

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
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

  const displayedRows = useMemo(() => {
    if (!result) return [];
    const predictionColumn = `predicted_${result.model.target}`;
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
      })
      .sort((left, right) => {
        const leftValue = Number(left[predictionColumn]);
        const rightValue = Number(right[predictionColumn]);
        return sortDirection === "asc"
          ? leftValue - rightValue
          : rightValue - leftValue;
      });
  }, [result, riskFilter, searchText, sortDirection]);

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

            <div
              className={`dropZone compactDrop ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
            >
              <input
                ref={fileInputRef}
                className="visuallyHidden"
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileChange}
              />
              <strong>새 CSV를 드래그하거나 클릭하여 선택하세요.</strong>
              <span>{file ? file.name : "최대 20MB"}</span>
            </div>

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
                </div>
              </section>

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
                      value={searchText}
                      onChange={(event) => setSearchText(event.target.value)}
                    />
                    <select
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
                      value={sortDirection}
                      onChange={(event) =>
                        setSortDirection(event.target.value as SortDirection)
                      }
                    >
                      <option value="desc">예측 수율 높은 순</option>
                      <option value="asc">예측 수율 낮은 순</option>
                    </select>
                  </div>
                </div>
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">{result.identifier_column}</th>
                        <th scope="col">예측값</th>
                        <th scope="col">위험 상태</th>
                        {result.predictions.some((row) => actualColumn in row) && (
                          <>
                            <th scope="col">실제값</th>
                            <th scope="col">절대 오차</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {displayedRows.map((row: PredictionRow, index) => (
                        <tr key={`${String(row[result.identifier_column])}-${index}`}>
                          <td>{String(row[result.identifier_column] ?? "-")}</td>
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
                        </tr>
                      ))}
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
