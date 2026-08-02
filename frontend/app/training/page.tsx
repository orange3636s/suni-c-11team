"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import ModelHistoryPanel from "@/components/ModelHistoryPanel";
import PreprocessingSummary from "@/components/PreprocessingSummary";
import Sidebar from "@/components/Sidebar";
import { trainModel } from "@/lib/api";
import { normalizeMetricSummary } from "@/lib/training";
import type { MetricSummary, ModelMetrics, TrainResponse } from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;

function formatMetric(value: number | null): string {
  return value === null ? "-" : value.toFixed(4);
}

function MetricRow({
  label,
  metrics,
}: {
  label: string;
  metrics: ModelMetrics;
}) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td>{formatMetric(metrics.r2)}</td>
      <td>{formatMetric(metrics.rmse)}</td>
      <td>{formatMetric(metrics.mae)}</td>
    </tr>
  );
}

function MetricSummaryChart({ summary }: { summary: MetricSummary }) {
  const data = (["r2", "rmse", "mae"] as const).flatMap((metric) => {
    const aggregate = summary[metric];
    return aggregate ? [{ metric: metric.toUpperCase(), mean: aggregate.mean, std: aggregate.std }] : [];
  });
  if (!data.length) return null;
  return <div className="chartCanvas" role="img" aria-label="교차 검증 Metric Summary"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 16, right: 20, bottom: 8, left: 8 }}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="metric" axisLine={false} tickLine={false} /><YAxis axisLine={false} tickLine={false} /><Tooltip formatter={(value, name, item) => [`${formatMetric(Number(value))} ± ${formatMetric(item.payload.std)}`, String(name)]} /><Bar dataKey="mean" name="평균 ± 표준편차" fill="var(--chart-primary)" radius={[8, 8, 0, 0]} /></BarChart></ResponsiveContainer></div>;
}

export default function TrainingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [activeView, setActiveView] = useState<"new" | "history">("new");
  const cvMetricSummary = normalizeMetricSummary(result);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const params = new URLSearchParams(window.location.search);
      if (params.get("view") === "history" || params.has("model_id")) {
        setActiveView("history");
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  function selectView(view: "new" | "history") {
    setActiveView(view);
    const url = new URL(window.location.href);
    if (view === "history") {
      url.searchParams.set("view", "history");
    } else {
      url.searchParams.delete("view");
      url.searchParams.delete("model_id");
    }
    window.history.replaceState({}, "", url);
  }

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

  async function handleTrain() {
    if (!file || isTraining) return;
    setError("");
    setResult(null);
    setIsTraining(true);
    try {
      setResult(await trainModel(file));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "모델 학습 중 오류가 발생했습니다.",
      );
    } finally {
      setIsTraining(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar activeItem="모델 학습" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro">
            <span className="eyebrow">머신러닝</span>
            <h1>모델 학습</h1>
            <p>
              Hybrid Multi-Y와 Lot 기반 Nested Group K-Fold로 안정적인 모델을 선택합니다.
            </p>
          </section>

          <div className="trainingViewTabs" role="tablist">
            <button
              className={activeView === "new" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={activeView === "new"}
              onClick={() => selectView("new")}
            >
              새 모델 학습
            </button>
            <button
              className={activeView === "history" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={activeView === "history"}
              onClick={() => selectView("history")}
            >
              학습 이력
            </button>
          </div>

          {activeView === "new" ? (
            <>
          <section className="uploadCard" aria-labelledby="training-form-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">학습 설정</span>
                <h2 id="training-form-title">학습 데이터</h2>
              </div>
              <p>Lot_ID 기준 · 재현 시드 42</p>
            </div>

            <div className="fieldGroup trainingFileField">
              <label htmlFor="training-file">학습 CSV</label>
              <CsvUploadPanel
                id="training-file"
                file={file}
                onFileSelect={selectFile}
                disabled={isTraining}
                compact
              />
            </div>

            <div className="trainingControls">
              <div className="hybridTrainingCard structureCard cvProtocolCard" aria-label="Nested Group K-Fold 검증">
                <span>교차 검증</span>
                <strong>Nested Group K-Fold</strong>
                <p>Outer 5-Fold에서 일반화 성능을 평가하고 Inner 3-Fold에서 모델과 앙상블을 선택합니다.</p>
                <small>그룹: Lot_ID · 동일 Lot 교차 혼입 방지 · seed 42</small>
              </div>
              <div className="hybridTrainingCard structureCard" aria-label="학습 구조 Hybrid Multi-Y">
                <span>학습 구조</span>
                <strong>Hybrid Multi-Y</strong>
                <p>Y1~Y5 불량률과 Y6~Y10 Fail Bit Count를 각각 예측하고 Direct Y와 결합해 최종 수율을 산출합니다.</p>
                <small>학습 대상: Final Y · Y1~Y5 · Y6~Y10 · Risk Classification</small>
              </div>
            </div>

            {error && (
              <div className="messageBox error" role="alert">{error}</div>
            )}

            <div className="uploadActions">
              <button
                className="button primary"
                type="button"
                disabled={!file || isTraining}
                data-loading={isTraining}
                aria-busy={isTraining}
                onClick={handleTrain}
              >
                {isTraining ? "모델을 학습하고 있습니다..." : "모델 학습 시작"}
              </button>
            </div>
          </section>

          {result && (
            <>
              <section className="resultCard" aria-labelledby="training-result-title">
                <div className="resultHeader">
                  <div>
                    <span className="sectionLabel">학습 결과</span>
                    <h2 id="training-result-title">{result.target} 최적 모델</h2>
                  </div>
                  <span className="resultBadge success">학습 완료</span>
                </div>

                <div className="trainingSummaryGrid">
                  <div>
                    <span>최적 모델</span>
                    <strong>{result.best_model}</strong>
                  </div>
                  <div>
                    <span>{cvMetricSummary ? "CV R²" : "Test R²"}</span>
                    <strong>{cvMetricSummary ? `${formatMetric(cvMetricSummary.r2?.mean ?? null)} ± ${formatMetric(cvMetricSummary.r2?.std ?? null)}` : formatMetric(result.metrics.test.r2)}</strong>
                  </div>
                  <div>
                    <span>{cvMetricSummary ? "CV RMSE" : "Test RMSE"}</span>
                    <strong>{cvMetricSummary ? `${formatMetric(cvMetricSummary.rmse?.mean ?? null)} ± ${formatMetric(cvMetricSummary.rmse?.std ?? null)}` : formatMetric(result.metrics.test.rmse)}</strong>
                  </div>
                  <div>
                    <span>{cvMetricSummary ? "CV MAE" : "Test MAE"}</span>
                    <strong>{cvMetricSummary ? `${formatMetric(cvMetricSummary.mae?.mean ?? null)} ± ${formatMetric(cvMetricSummary.mae?.std ?? null)}` : formatMetric(result.metrics.test.mae)}</strong>
                  </div>
                  <div>
                    <span>사용 Feature 수</span>
                    <strong>{result.feature_count}</strong>
                  </div>
                </div>
                <PreprocessingSummary summary={result.preprocessing} />

                {result.ensemble && (
                  <div className="trainingWarnings">
                    <strong>앙상블 선택 결과 · {result.ensemble.selected ? result.ensemble.selected_type : "Single Model"}</strong>
                    <p>{result.ensemble.base_models.join(" / ")} · RMSE 개선률 {(100 * (result.ensemble.improvement_over_single.rmse_relative ?? 0)).toFixed(2)}% · Ensemble Agreement 상관 {formatMetric(result.ensemble.agreement.mean_pairwise_correlation)}</p>
                    <div className="tableWrap">
                      <table><thead><tr><th>Target</th><th>선택 유형</th><th>Base Models</th><th>Weight</th><th>개선률</th><th>Fold 안정성</th></tr></thead>
                      <tbody>{Object.entries(result.ensemble.target_configs).map(([targetName, config]) => (
                        <tr key={targetName}><th>{targetName}</th><td>{config.selected_type}</td><td>{config.base_models.join(" / ")}</td><td>{Object.entries(config.weights).map(([name, weight]) => `${name} ${weight.toFixed(2)}`).join(" · ")}</td><td>{(100 * (config.improvement_over_single.rmse_relative ?? 0)).toFixed(2)}%</td><td>{config.fold_rmse_std.toFixed(4)}</td></tr>
                      ))}</tbody></table>
                    </div>
                  </div>
                )}

                <p className="metricNotice">
                  R²가 0 이하이면 기준선보다 예측력이 낮을 수 있습니다.
                </p>

                <div className="splitSummary">
                  <span>Outer 5-Fold</span>
                  <span>Inner 3-Fold</span>
                  <span>Group Lot_ID</span>
                  <span>Seed 42</span>
                </div>

                <div className="previewHeader">
                  <div>
                    <span className="sectionLabel">선정 모델 평가</span>
                    <h3>교차 검증 성능 요약</h3>
                  </div>
                </div>
                {cvMetricSummary ? <><MetricSummaryChart summary={cvMetricSummary} /><div className="tableWrap">
                  <table>
                    <thead><tr><th scope="col">Metric</th><th scope="col">평균</th><th scope="col">표준편차</th></tr></thead>
                    <tbody>{(["r2", "rmse", "mae", "mse"] as const).map((metric) => {
                      const aggregate = cvMetricSummary[metric];
                      if (!aggregate) return null;
                      return <tr key={metric}><th scope="row">{metric.toUpperCase()}</th><td>{formatMetric(aggregate.mean)}</td><td>{formatMetric(aggregate.std)}</td></tr>;
                    })}</tbody>
                  </table>
                </div></> : <p className="emptyMessage">평가 결과가 없습니다.</p>}
                {!cvMetricSummary && <><div className="previewHeader"><div><span className="sectionLabel">Legacy Holdout</span><h3>Train·Validation·Test 성능</h3></div></div><div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">데이터셋</th>
                        <th scope="col">R²</th>
                        <th scope="col">RMSE</th>
                        <th scope="col">MAE</th>
                      </tr>
                    </thead>
                    <tbody>
                      <MetricRow label="Train" metrics={result.metrics.train} />
                      <MetricRow
                        label="Validation"
                        metrics={result.metrics.validation}
                      />
                      <MetricRow label="Test" metrics={result.metrics.test} />
                    </tbody>
                  </table>
                </div></>}
              </section>

              <section className="resultCard" aria-labelledby="comparison-title">
                <div className="resultHeader">
                  <div>
                    <span className="sectionLabel">모델 비교</span>
                    <h2 id="comparison-title">Validation 성능</h2>
                  </div>
                </div>
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">모델</th>
                        <th scope="col">Validation R²</th>
                        <th scope="col">Validation RMSE</th>
                        <th scope="col">Validation MAE</th>
                        <th scope="col">선정 여부</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.model_comparison.map((item) => (
                        <tr key={item.model_name}>
                          <th scope="row">{item.model_name}</th>
                          <td>{formatMetric(item.validation?.r2 ?? null)}</td>
                          <td>{formatMetric(item.validation?.rmse ?? null)}</td>
                          <td>{formatMetric(item.validation?.mae ?? null)}</td>
                          <td>
                            {item.selected
                              ? "선정"
                              : item.status === "failed"
                                ? item.error_message ?? "학습 실패"
                                : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {result.warnings.length > 0 && (
                  <div className="trainingWarnings">
                    <strong>주의사항</strong>
                    <ul>
                      {result.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </>
          )}
            </>
          ) : (
            <section className="resultCard modelHistoryCard">
              <ModelHistoryPanel />
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
