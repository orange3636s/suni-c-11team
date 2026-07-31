"use client";

import type { ChangeEvent } from "react";
import { useRef, useState } from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { trainModel } from "@/lib/api";
import type { ModelMetrics, TrainResponse } from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const TARGETS = ["Y", ...Array.from({ length: 10 }, (_, index) => `Y${index + 1}`)];

function formatFileSize(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

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

export default function TrainingPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("Y");
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState("");
  const [isTraining, setIsTraining] = useState(false);

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

  async function handleTrain() {
    if (!file || isTraining) return;
    setError("");
    setResult(null);
    setIsTraining(true);
    try {
      setResult(await trainModel(file, target));
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
            <h1>회귀 모델 학습</h1>
            <p>
              검증과 전처리를 거친 공정 데이터로 여러 모델을 비교합니다.
            </p>
          </section>

          <section className="uploadCard" aria-labelledby="training-form-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">학습 설정</span>
                <h2 id="training-form-title">데이터와 목표 변수</h2>
              </div>
              <p>Train 64% · Validation 16% · Test 20%</p>
            </div>

            <div className="trainingControls">
              <div className="fieldGroup">
                <label htmlFor="training-target">목표 변수</label>
                <select
                  id="training-target"
                  value={target}
                  onChange={(event) => {
                    setTarget(event.target.value);
                    setResult(null);
                  }}
                  disabled={isTraining}
                >
                  {TARGETS.map((targetName) => (
                    <option key={targetName} value={targetName}>
                      {targetName}
                    </option>
                  ))}
                </select>
              </div>

              <div className="fieldGroup fileField">
                <label htmlFor="training-file">학습 CSV</label>
                <input
                  ref={inputRef}
                  id="training-file"
                  className="visuallyHidden"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleFileChange}
                />
                <button
                  className="fileSelectButton"
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  disabled={isTraining}
                >
                  CSV 파일 선택
                </button>
              </div>
            </div>

            {file && (
              <div className="selectedFile" aria-live="polite">
                <div>
                  <span>선택된 파일</span>
                  <strong>{file.name}</strong>
                </div>
                <span>{formatFileSize(file.size)}</span>
              </div>
            )}

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
                {isTraining ? "모델을 학습하고 있습니다..." : "모델 학습"}
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
                    <span>Test R²</span>
                    <strong>{formatMetric(result.metrics.test.r2)}</strong>
                  </div>
                  <div>
                    <span>Test RMSE</span>
                    <strong>{formatMetric(result.metrics.test.rmse)}</strong>
                  </div>
                  <div>
                    <span>Test MAE</span>
                    <strong>{formatMetric(result.metrics.test.mae)}</strong>
                  </div>
                  <div>
                    <span>사용 Feature 수</span>
                    <strong>{result.feature_count}</strong>
                  </div>
                </div>

                <p className="metricNotice">
                  R²가 0 이하이면 기준선보다 예측력이 낮을 수 있습니다.
                </p>

                <div className="splitSummary">
                  <span>Train {result.split.train_rows}행</span>
                  <span>Validation {result.split.validation_rows}행</span>
                  <span>Test {result.split.test_rows}행</span>
                  <span>
                    {result.split.group_split_used
                      ? "Lot 그룹 분리"
                      : "Random 분리"}
                  </span>
                </div>

                <div className="previewHeader">
                  <div>
                    <span className="sectionLabel">선정 모델 평가</span>
                    <h3>데이터셋별 성능</h3>
                  </div>
                </div>
                <div className="tableWrap">
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
                </div>
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
        </main>
      </div>
    </div>
  );
}
