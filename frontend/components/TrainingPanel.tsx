"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { createTrainingJob, getModelPerformance, getPromotionHistory, getTrainingJob, saveTrainingState } from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";
import { useFocusTrap } from "@/lib/useFocusTrap";
import type { PromotionEvent } from "@/types/data";

// 지시서 I-2: 학습 탭을 없애고 이 팝업 하나로 축소한다. 넣는 것은 딱
// 셋 -- 3줄 읽기전용 정보, SQL/Refresh 입력, 파일 첨부·수동 학습 실행
// 버튼. 성능 지표·히트맵·전처리 비교 등은 절대 넣지 않는다(원인 분석
// 탭에 히트맵이 이미 있다).
export default function TrainingPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { training, setTraining } = useAnalysisState();
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, open);
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
  // 지시서 §2-2: 승격 여부와 무관한 최근 학습 시도 -- 게이트 미달로
  // 교체되지 않았을 때 "학습은 돌았는데 모델은 그대로"임을 보여준다.
  const [latestPromotion, setLatestPromotion] = useState<PromotionEvent | null>(null);
  // A-4: promotion-history 라우트가 죽어 있어도(예: /models/{model_id}에
  // 가려짐) 조용히 "승격 이력 없음"으로 보이면 게이트 미달 사실이 통째로
  // 숨는다 -- 조회 실패와 "이력 없음"을 구분해서 보여준다.
  const [promotionHistoryError, setPromotionHistoryError] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // D-6: 폴링 이펙트(69행)는 deps가 [jobId]뿐이라 학습이 도는 동안 고친
  // sqlHost/refreshMinutes를 클로저가 시작 시점 값으로 붙잡고 있다 --
  // 완료 시점에 그 옛 값을 서버에 저장해 사용자가 학습 중 고친 값을
  // 덮어써 버렸다. ref로 항상 최신값을 읽는다.
  const sqlHostRef = useRef(sqlHost);
  const sqlPortRef = useRef(sqlPort);
  const refreshMinutesRef = useRef(refreshMinutes);
  const trainingRef = useRef(training);
  useEffect(() => {
    sqlHostRef.current = sqlHost;
  }, [sqlHost]);
  useEffect(() => {
    sqlPortRef.current = sqlPort;
  }, [sqlPort]);
  useEffect(() => {
    refreshMinutesRef.current = refreshMinutes;
  }, [refreshMinutes]);
  useEffect(() => {
    trainingRef.current = training;
  }, [training]);

  function refreshPromotionHistory() {
    getPromotionHistory(1)
      .then((response) => {
        setPromotionHistoryError(false);
        setLatestPromotion(response.items[0] ?? null);
      })
      .catch(() => {
        setLatestPromotion(null);
        setPromotionHistoryError(true);
      });
  }

  // 팝업을 다시 열 때마다 컨텍스트의 최신 저장값으로 되돌린다 -- 다른
  // 탭에서 값이 바뀌었을 수 있으므로 로컬 state를 열 때 한 번 동기화.
  useEffect(() => {
    if (!open) return;
    setSqlHost(training?.sqlHost ?? "");
    setSqlPort(training?.sqlPort ?? "");
    setRefreshMinutes(training?.refreshIntervalMinutes != null ? String(training.refreshIntervalMinutes) : "");
    setError("");
    setMessage("");
    refreshPromotionHistory();
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
          refreshPromotionHistory();
          const performance = await getModelPerformance().catch(() => null);
          if (performance) {
            const dataset = performance.source_filename || "training";
            // D-6: 클로저의 sqlHost/sqlPort/refreshMinutes/training이 아니라
            // ref로 완료 시점의 최신 입력값을 읽는다.
            const latestSqlHost = sqlHostRef.current;
            const latestSqlPort = sqlPortRef.current;
            const latestRefreshMinutes = refreshMinutesRef.current;
            const latestTraining = trainingRef.current;
            const nextTraining = {
              dataset,
              createdAt: performance.trained_at ?? new Date().toISOString(),
              performance,
              sqlHost: latestSqlHost,
              sqlPort: latestSqlPort,
              refreshIntervalMinutes: latestRefreshMinutes.trim() ? Number(latestRefreshMinutes) : null,
            };
            setTraining({ ...nextTraining, sqlDb: latestTraining?.sqlDb ?? "", sqlUser: latestTraining?.sqlUser ?? "" });
            void saveTrainingState(dataset, {
              performance,
              sqlHost: latestSqlHost,
              sqlPort: latestSqlPort,
              sqlDb: latestTraining?.sqlDb ?? "",
              sqlUser: latestTraining?.sqlUser ?? "",
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
      const result = await saveTrainingState(dataset, {
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
      // H-3⑤: 저장 자체는 성공했지만 서버의 스케줄러 반영(reschedule/pause)이
      // 실패하면 다음 자동 수집이 이전 주기로 계속 돈다 -- 조용히 넘기지
      // 않고 별도 안내로 구분한다.
      if (!result.schedule_applied) {
        setError("설정은 저장됐지만 자동 수집 주기 반영에는 실패했습니다. 다시 시도해 주세요.");
      } else {
        setMessage("설정을 저장했습니다.");
      }
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
      <div
        ref={panelRef}
        className="settingsPanel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="모델 학습·자동화"
        tabIndex={-1}
      >
        <div className="settingsPanelHeader">
          <h2>모델 학습·자동화</h2>
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
            {/* 지시서 §2-2: 승격 여부와 무관하게 최근 학습 시도를 보여준다
                -- 게이트 미달로 교체되지 않았다면("학습은 돌았는데 모델은
                그대로") 그 사실이 여기 드러나야 한다. */}
            {latestPromotion && !latestPromotion.promoted && (
              <p className="notifyFieldError">
                게이트 미달: {formatLastRun(latestPromotion.created_at)}에 학습한 모델({latestPromotion.candidate_model_id})이
                기존 모델보다 나빠 승격되지 않았습니다 -- {latestPromotion.reason}
              </p>
            )}
            {latestPromotion?.promoted === 1 && latestPromotion.candidate_model_id === performance?.model_id && (
              <p className="notifyTestResult ok">최근 학습이 게이트를 통과해 승격됐습니다 -- {latestPromotion.reason}</p>
            )}
            {promotionHistoryError && (
              <p className="notifyFieldError">
                승격 이력을 불러오지 못했습니다.{" "}
                <button type="button" className="linkButton" onClick={refreshPromotionHistory}>
                  다시 시도
                </button>
              </p>
            )}
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
