"use client";

import { useState } from "react";

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import ModelHistoryPanel from "@/components/ModelHistoryPanel";
import Sidebar from "@/components/Sidebar";
import { trainModel } from "@/lib/api";
import type { ModelMetrics, TrainResponse } from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const TARGETS = ["Y", ...Array.from({ length: 10 }, (_, index) => `Y${index + 1}`)];
const DEFAULT_SPLIT = { train: 64, validation: 16, test: 20 };
type SplitKey = keyof typeof DEFAULT_SPLIT;

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
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("Y");
  const [split, setSplit] = useState(DEFAULT_SPLIT);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [activeView, setActiveView] = useState<"new" | "history">("new");

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
      setResult(await trainModel(file, target, split));
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

  function updateSplit(key: SplitKey, rawValue: number) {
    const nextValue = Math.min(90, Math.max(5, Math.round(rawValue)));
    const otherKeys = (
      ["train", "validation", "test"] as SplitKey[]
    ).filter((item) => item !== key);

    setSplit((current) => {
      const remaining = 100 - nextValue;
      const adjustableTotal =
        current[otherKeys[0]] + current[otherKeys[1]] - 10;
      const firstWeight =
        adjustableTotal > 0
          ? (current[otherKeys[0]] - 5) / adjustableTotal
          : 0.5;
      const firstValue = Math.min(
        remaining - 5,
        Math.max(5, 5 + Math.round((remaining - 10) * firstWeight)),
      );

      return {
        ...current,
        [key]: nextValue,
        [otherKeys[0]]: firstValue,
        [otherKeys[1]]: remaining - firstValue,
      };
    });
    setResult(null);
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

          <div className="trainingViewTabs" role="tablist">
            <button
              className={activeView === "new" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={activeView === "new"}
              onClick={() => setActiveView("new")}
            >
              새 모델 학습
            </button>
            <button
              className={activeView === "history" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={activeView === "history"}
              onClick={() => setActiveView("history")}
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
                <h2 id="training-form-title">데이터와 목표 변수</h2>
              </div>
              <p>
                Train {split.train}% · Validation {split.validation}% · Test{" "}
                {split.test}%
              </p>
            </div>

            <fieldset className="splitControls">
              <legend>Dataset Split</legend>
              <p>한 비율을 변경하면 나머지 두 비율이 자동 보정됩니다.</p>
              <div className="splitControlGrid">
                {(
                  [
                    ["train", "Train"],
                    ["validation", "Validation"],
                    ["test", "Test"],
                  ] as [SplitKey, string][]
                ).map(([key, label]) => (
                  <label className="splitControl" key={key}>
                    <span>
                      {label}
                      <strong>{split[key]}%</strong>
                    </span>
                    <input
                      type="range"
                      min="5"
                      max="90"
                      value={split[key]}
                      disabled={isTraining}
                      onChange={(event) =>
                        updateSplit(key, Number(event.target.value))
                      }
                    />
                    <input
                      type="number"
                      min="5"
                      max="90"
                      value={split[key]}
                      disabled={isTraining}
                      aria-label={`${label} 비율`}
                      onChange={(event) =>
                        updateSplit(key, Number(event.target.value))
                      }
                    />
                  </label>
                ))}
              </div>
              <div className="splitTotal" aria-live="polite">
                합계 <strong>{split.train + split.validation + split.test}%</strong>
              </div>
            </fieldset>

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
                <CsvUploadPanel
                  id="training-file"
                  file={file}
                  onFileSelect={selectFile}
                  disabled={isTraining}
                  compact
                />
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
