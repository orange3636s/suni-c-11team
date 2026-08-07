"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { createTrainingJob, getModelPerformance, getTrainingJob, saveTrainingState } from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";

// 지시서 I-2: 학습 탭을 없애고 이 팝업 하나로 축소한다. 넣는 것은 딱
// 셋 -- 3줄 읽기전용 정보, SQL/Refresh 입력, 파일 첨부·수동 학습 실행
// 버튼. 성능 지표·히트맵·전처리 비교 등은 절대 넣지 않는다(원인 분석
// 탭에 히트맵이 이미 있다).
export default function TrainingPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { training, setTraining } = useAnalysisState();
  const [sqlHost, setSqlHost] = useState(training?.sqlHost ?? "");
  const [sqlPort, setSqlPort] = useState(training?.sqlPort ?? "");
  const [refreshMinutes, setRefreshMinutes] = useState(
    training?.refreshIntervalMinutes != null ? String(training.refreshIntervalMinutes) : "",
  );
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // 팝업을 다시 열 때마다 컨텍스트의 최신 저장값으로 되돌린다 -- 다른
  // 탭에서 값이 바뀌었을 수 있으므로 로컬 state를 열 때 한 번 동기화.
  useEffect(() => {
    if (!open) return;
    setSqlHost(training?.sqlHost ?? "");
    setSqlPort(training?.sqlPort ?? "");
    setRefreshMinutes(training?.refreshIntervalMinutes != null ? String(training.refreshIntervalMinutes) : "");
    setError("");
    setMessage("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await getTrainingJob(jobId);
        setStage(job.stage);
        if (job.status === "completed") {
          window.clearInterval(timer);
          setJobId(null);
          setMessage("모델 학습이 완료되었습니다.");
          const performance = await getModelPerformance().catch(() => null);
          if (performance) {
            const dataset = performance.source_filename || "training";
            const nextTraining = {
              dataset,
              createdAt: performance.trained_at ?? new Date().toISOString(),
              performance,
              sqlHost,
              sqlPort,
              refreshIntervalMinutes: refreshMinutes.trim() ? Number(refreshMinutes) : null,
            };
            setTraining({ ...nextTraining, sqlDb: training?.sqlDb ?? "", sqlUser: training?.sqlUser ?? "" });
            void saveTrainingState(dataset, {
              performance,
              sqlHost,
              sqlPort,
              sqlDb: training?.sqlDb ?? "",
              sqlUser: training?.sqlUser ?? "",
              refreshIntervalMinutes: nextTraining.refreshIntervalMinutes,
            }).catch(() => {});
          }
        } else if (job.status === "failed" || job.status === "interrupted") {
          window.clearInterval(timer);
          setJobId(null);
          setError(job.error || "모델 학습 중 서버 오류가 발생했습니다.");
        }
      } catch (pollError) {
        window.clearInterval(timer);
        setJobId(null);
        setError(pollError instanceof Error ? pollError.message : "학습 상태를 확인하지 못했습니다.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  async function saveSettings() {
    setError("");
    setMessage("");
    const dataset = training?.dataset || "training-settings";
    const refreshIntervalMinutes = refreshMinutes.trim() ? Number(refreshMinutes) : null;
    try {
      const performance =
        training?.performance ?? (await getModelPerformance().catch(() => null)) ?? {
          model_id: null,
          trained_at: null,
          source_filename: null,
          targets: [],
          final_yield: null,
          row_count: null,
          feature_count: null,
        };
      await saveTrainingState(dataset, {
        performance,
        sqlHost,
        sqlPort,
        sqlDb: training?.sqlDb ?? "",
        sqlUser: training?.sqlUser ?? "",
        refreshIntervalMinutes,
      });
      setTraining((previous) =>
        previous
          ? { ...previous, sqlHost, sqlPort, refreshIntervalMinutes }
          : {
              dataset,
              createdAt: new Date().toISOString(),
              performance,
              sqlHost,
              sqlPort,
              sqlDb: "",
              sqlUser: "",
              refreshIntervalMinutes,
            },
      );
      setMessage("설정을 저장했습니다.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "설정을 저장하지 못했습니다.");
    }
  }

  async function runManualTraining() {
    if (!file || jobId) return;
    setError("");
    setMessage("");
    setStage("학습 Job을 등록하는 중입니다.");
    try {
      const accepted = await createTrainingJob(file);
      setJobId(accepted.job_id);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "모델 학습을 시작하지 못했습니다.");
    }
  }

  if (!open) return null;

  const performance = training?.performance;
  const isRunning = Boolean(jobId);

  return (
    <div className="settingsPanelBackdrop" onClick={onClose} role="presentation">
      <div className="settingsPanel" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-label="모델 학습">
        <div className="settingsPanelHeader">
          <h2>모델 학습</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={18} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>현재 모델</h3>
            <dl className="trainingInfoList">
              <div>
                <dt>최근 학습</dt>
                <dd>{formatLastRun(performance?.trained_at)}</dd>
              </div>
              <div>
                <dt>학습 파일</dt>
                <dd>{performance?.source_filename ?? "-"}</dd>
              </div>
              <div>
                <dt>데이터 크기</dt>
                <dd>
                  {performance?.row_count != null && performance?.feature_count != null
                    ? `${performance.row_count.toLocaleString()} 행 × ${performance.feature_count.toLocaleString()} 열`
                    : "-"}
                </dd>
              </div>
            </dl>
          </section>

          <section className="settingsSection">
            <h3>SQL 연결</h3>
            <p className="settingsSectionDesc">호스트·포트만 저장합니다. 비밀번호는 서버 환경변수로 관리하며 여기서 입력하지 않습니다.</p>
            <div className="trainingSqlRow">
              <label className="notifyFieldLabel">
                호스트
                <input type="text" value={sqlHost} onChange={(event) => setSqlHost(event.target.value)} placeholder="db.internal" />
              </label>
              <label className="notifyFieldLabel trainingPortField">
                포트
                <input type="text" value={sqlPort} onChange={(event) => setSqlPort(event.target.value)} placeholder="5432" />
              </label>
            </div>
            <label className="notifyFieldLabel">
              Refresh (분마다 최신 데이터를 받아 자동 학습)
              <input
                type="number"
                min={0}
                value={refreshMinutes}
                onChange={(event) => setRefreshMinutes(event.target.value)}
                placeholder="60"
              />
            </label>
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => void saveSettings()}>
                설정 저장
              </button>
            </div>
          </section>

          <section className="settingsSection">
            <h3>수동 학습</h3>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => fileInputRef.current?.click()} disabled={isRunning}>
                파일 첨부{file ? `: ${file.name}` : ""}
              </button>
              <button type="button" className="button primary" onClick={() => void runManualTraining()} disabled={!file || isRunning}>
                {isRunning ? "학습 중…" : "수동 학습 실행"}
              </button>
            </div>
            {isRunning && <p className="trainingProgress" role="status">{stage}</p>}
            {error && <p className="notifyFieldError">{error}</p>}
            {message && <p className="notifyTestResult ok">{message}</p>}
          </section>
        </div>
      </div>
    </div>
  );
}
