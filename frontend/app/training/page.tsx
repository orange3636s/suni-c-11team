"use client";

import { useEffect, useState } from "react";

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import ModelHistoryPanel from "@/components/ModelHistoryPanel";
import OperationProgress from "@/components/OperationProgress";
import Sidebar from "@/components/Sidebar";
import useElapsedTime from "@/hooks/useElapsedTime";
import { createTrainingJob, getTrainingJob } from "@/lib/api";
import type {
  TrainingJobResult,
  TrainingJobStatusResponse,
} from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const JOB_POLL_INTERVAL_MS = 2_500;

function formatMetric(value: number | null | undefined): string {
  return value == null ? "-" : value.toFixed(4);
}

export default function TrainingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TrainingJobResult | null>(null);
  const [job, setJob] = useState<TrainingJobStatusResponse | null>(null);
  const [error, setError] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [trainingRunKey, setTrainingRunKey] = useState(0);
  const [activeView, setActiveView] = useState<"new" | "history">("new");
  const { formattedElapsed: formattedTrainingElapsed } = useElapsedTime({
    running: isTraining,
    resetKey: trainingRunKey,
  });

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const params = new URLSearchParams(window.location.search);
      if (params.get("view") === "history" || params.has("model_id")) {
        setActiveView("history");
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!isTraining || !job?.job_id) return;

    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController | null = null;

    const poll = async () => {
      controller = new AbortController();
      try {
        const nextJob = await getTrainingJob(job.job_id, controller.signal);
        if (disposed) return;
        setJob(nextJob);

        if (nextJob.status === "completed") {
          if (nextJob.result) {
            setResult(nextJob.result);
          } else {
            setError("학습은 완료되었지만 결과 요약을 불러오지 못했습니다.");
          }
          setIsTraining(false);
          return;
        }

        if (nextJob.status === "failed" || nextJob.status === "interrupted") {
          setError(
            nextJob.error ||
              (nextJob.status === "interrupted"
                ? "서버 재시작으로 학습이 중단되었습니다. 다시 학습해 주세요."
                : "모델 학습 중 오류가 발생했습니다."),
          );
          setIsTraining(false);
          return;
        }

        timer = window.setTimeout(poll, JOB_POLL_INTERVAL_MS);
      } catch (requestError) {
        if (disposed || (requestError instanceof DOMException && requestError.name === "AbortError")) {
          return;
        }
        setError(
          requestError instanceof Error
            ? requestError.message
            : "학습 상태 확인 중 오류가 발생했습니다.",
        );
        setIsTraining(false);
      }
    };

    timer = window.setTimeout(poll, JOB_POLL_INTERVAL_MS);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
      controller?.abort();
    };
  }, [isTraining, job?.job_id]);

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
    setJob(null);
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
    setJob(null);
    setTrainingRunKey((current) => current + 1);
    setIsTraining(true);
    try {
      const created = await createTrainingJob(file);
      setJob({
        ...created,
        stage: "학습 대기 중",
        progress: 0,
        elapsed_seconds: 0,
        result: null,
        error: null,
      });
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "학습 Job 생성 중 오류가 발생했습니다.",
      );
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
              공정 구조를 자동 감지하고 Lot을 분리한 Multi-Y 모델로 최종 수율을 계산합니다.
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

                <div className="hybridTrainingCard structureCard" aria-label="자동 학습 설정">
                  <span>자동 학습 설정</span>
                  <strong>데이터에 맞춰 안전하게 자동 구성</strong>
                  <p>데이터 구조 자동 탐지 · Group 3-Fold 교차검증 · Multi-Y 순차 학습 · 최종 수율 자동 계산</p>
                  <small>float32 전처리 · Config 순서형 인코딩 · 외부 XGBoost 의존성 없음</small>
                </div>

                {error && <div className="messageBox error" role="alert">{error}</div>}

                {isTraining && job && (
                  <div className="trainingJobStatus">
                    <div>
                      <strong role="status" aria-live="polite" aria-atomic="true">
                        {job.stage || "학습 준비 중"}
                      </strong>
                      <span aria-hidden="true">{Math.round(job.progress)}%</span>
                    </div>
                    <progress
                      aria-label="모델 학습 진행률"
                      max={100}
                      value={Math.max(0, Math.min(100, job.progress))}
                    />
                  </div>
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
                    {isTraining ? (
                      <OperationProgress
                        message="모델을 학습하고 있습니다…"
                        timeLabel="학습 시간"
                        formattedElapsed={formattedTrainingElapsed}
                        detail={job ? `${job.status} · ${Math.round(job.progress)}%` : "Job 생성 중"}
                      />
                    ) : "모델 학습"}
                  </button>
                </div>
              </section>

              {result && (
                <section className="resultCard" aria-labelledby="training-result-title">
                  <div className="resultHeader">
                    <div>
                      <span className="sectionLabel">학습 결과</span>
                      <h2 id="training-result-title">{result.target} 모델 학습 완료</h2>
                    </div>
                    <span className="resultBadge success">저장 완료</span>
                  </div>

                  <div className="trainingSummaryGrid">
                    <div title={result.best_model}>
                      <span>모델</span>
                      <strong>{result.best_model}</strong>
                    </div>
                    <div>
                      <span>Test R²</span>
                      <strong>{formatMetric(result.test_metrics?.r2)}</strong>
                    </div>
                    <div>
                      <span>Test RMSE</span>
                      <strong>{formatMetric(result.test_metrics?.rmse)}</strong>
                    </div>
                    <div>
                      <span>Test MAE</span>
                      <strong>{formatMetric(result.test_metrics?.mae)}</strong>
                    </div>
                    <div>
                      <span>Feature 수</span>
                      <strong>{result.feature_count}</strong>
                    </div>
                  </div>

                  <div className="trainingResultMeta">
                    <span>model_id</span>
                    <code title={result.model_id}>{result.model_id}</code>
                    <span>경고 {result.warning_count}건</span>
                  </div>

                  <div className="uploadActions">
                    <button className="button secondary" type="button" onClick={() => selectView("history")}>
                      학습 이력에서 확인
                    </button>
                  </div>
                </section>
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
