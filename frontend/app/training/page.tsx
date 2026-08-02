"use client";

import { useCallback, useEffect, useState } from "react";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { createTrainingJob, getLatestModel, getTrainingJob, type LatestModelMetadata } from "@/lib/api";

const showMetric = (value?: number | null) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
const showDate = (value?: string | null) => value ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "-";

export default function TrainingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [latest, setLatest] = useState<LatestModelMetadata | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const loadLatest = useCallback(async () => setLatest((await getLatestModel()).latest_model), []);
  useEffect(() => {
    const timer = window.setTimeout(() => { void loadLatest().catch(() => setLatest(null)); }, 0);
    return () => window.clearTimeout(timer);
  }, [loadLatest]);
  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await getTrainingJob(jobId);
        setStage(job.stage); setProgress(job.progress);
        if (job.status === "completed") {
          window.clearInterval(timer); setJobId(null); setFile(null);
          setMessage("Y 최종 수율 모델 학습이 완료되었으며 최신 모델로 저장되었습니다.");
          await loadLatest();
        } else if (job.status === "failed" || job.status === "interrupted") {
          window.clearInterval(timer); setJobId(null);
          setError(job.error || "모델 학습 중 서버 오류가 발생했습니다.");
        }
      } catch (pollError) {
        window.clearInterval(timer); setJobId(null);
        setError(pollError instanceof Error ? pollError.message : "학습 상태를 확인하지 못했습니다.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId, loadLatest]);

  async function train() {
    if (!file || jobId) return;
    setError(""); setMessage(""); setStage("학습 파일을 전송하는 중입니다."); setProgress(0);
    try { setJobId((await createTrainingJob(file)).job_id); }
    catch (trainingError) { setError(trainingError instanceof Error ? trainingError.message : "모델 학습을 시작하지 못했습니다."); }
  }

  const metrics = latest?.metrics?.test;
  return <div className="appShell"><Sidebar activeItem="모델 학습" /><div className="contentShell"><Header /><main className="mainContent uploadPage">
    <section className="uploadIntro pageHeading"><span className="eyebrow">Machine Learning</span><h1>모델 학습</h1><p>CSV의 최종 수율 Y를 직접 학습해 가장 최근 성공 모델로 저장합니다.</p></section>
    <section className="uploadCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Target Y</span><h2>학습 파일</h2></div></div>
      <CsvUploadPanel id="training-file" file={file} onFileSelect={(selected) => setFile(selected ?? null)} disabled={Boolean(jobId)} compact title="Y 컬럼이 포함된 CSV 파일을 선택해주세요." description="Y1~Y10과 식별자 컬럼은 학습 특성에서 자동 제외됩니다." />
      <div className="uploadActions"><button className="button primary" type="button" disabled={!file || Boolean(jobId)} onClick={() => void train()}>{jobId ? "모델 학습 중…" : "모델 학습"}</button></div>
      {jobId && <p className="trainingProgress" role="status">{stage} · {progress}%</p>}{message && <p className="messageBox success" role="status">{message}</p>}{error && <p className="errorMessage" role="alert">{error}</p>}
    </section>
    <section className="resultCard"><div className="sectionHeading compact"><div><span className="sectionLabel">Latest model</span><h2>현재 사용 모델</h2></div></div>
      {latest ? <div className="trainingSummaryGrid"><div><span>모델</span><strong>{latest.model_name || latest.model_id}</strong></div><div><span>Target</span><strong>Y</strong></div><div><span>학습 시각</span><strong>{showDate(latest.trained_at)}</strong></div><div><span>버전</span><strong>{latest.version || latest.model_id}</strong></div><div><span>Test R²</span><strong>{showMetric(metrics?.r2)}</strong></div><div><span>Test RMSE</span><strong>{showMetric(metrics?.rmse)}</strong></div><div><span>Test MAE</span><strong>{showMetric(metrics?.mae)}</strong></div><div><span>학습 행</span><strong>{latest.row_count ?? "-"}</strong></div></div> : <p className="emptyMessage">저장된 학습 모델이 없습니다.</p>}
    </section>
  </main></div></div>;
}
