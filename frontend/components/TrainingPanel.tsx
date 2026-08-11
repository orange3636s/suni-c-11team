"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import {
  createTrainingJob,
  getModelPerformance,
  getPromotionHistory,
  getTrainingJob,
  retrainBundled,
  saveTrainingState,
} from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";
import { useFocusTrap } from "@/lib/useFocusTrap";
import type { ModelPerformanceResponse, PromotionEvent, TargetPerformance } from "@/types/data";

// RA-1: 모델 학습·자동화 통합 팝업을 둘로 나눈다 -- 이 팝업은 학습
// 전용(수동 업로드만, RB-3)이다. SQL 연결·refresh time은
// ModelAnalysisPanel.tsx로 옮겼다(RA-2: 자동화는 분석의 입력이므로
// 분석 팝업에 속한다 -- 이 팝업에 다시 두지 않는다).
//
// B-11: ModelAnalysisPanel과 같은 3구획 형태로 맞춘다 -- ① 현재 학습
// 데이터(모델 성능 포함) ② 학습 데이터 변경 ③ 학습 시작. 클래스는
// settingsSection/settingsSectionDesc를 그대로 쓴다.
function formatNextRefresh(createdAtIso: string | null | undefined, refreshIntervalMinutes: number | null | undefined): string | null {
  if (!createdAtIso || !refreshIntervalMinutes) return null;
  const created = new Date(createdAtIso);
  if (Number.isNaN(created.getTime())) return null;
  return formatLastRun(new Date(created.getTime() + refreshIntervalMinutes * 60_000).toISOString());
}

// B-6: R²가 음수이거나 없으면 "-0.00"처럼 오해를 부르는 숫자를 찍지
// 않는다 -- 실제로 -0.0025 같은 값이 나온 이력이 있다. RMSE는 R²의
// 부호와 무관하게 여전히 유효한 오차 크기라 별도로 판단한다.
function formatR2(r2: number | null): string {
  if (r2 == null || r2 < 0) return "학습 데이터에 신호가 없어 평가 불가";
  return `R² ${r2.toFixed(2)}`;
}

function formatRmse(rmse: number | null): string | null {
  if (rmse == null) return null;
  return `RMSE ${rmse.toFixed(2)} %p`;
}

function targetPerformanceLine(t: TargetPerformance): string {
  const parts = [formatR2(t.r2), formatRmse(t.rmse)].filter((part): part is string => part != null);
  return `${t.target} ${parts.join(" · ")}`;
}

// B-6④: "표본 10,000장 · 강함 등급 인자 3/5" -- 있는 값만 쓰고 없는 값은
// 만들지 않는다.
function evidenceLine(performance: ModelPerformanceResponse | null | undefined): string | null {
  if (!performance) return null;
  const parts: string[] = [];
  if (performance.row_count != null) parts.push(`표본 ${performance.row_count.toLocaleString()}장`);
  const withTier = performance.targets.filter((t) => t.confidence_tier != null);
  if (withTier.length > 0) {
    const strongCount = withTier.filter((t) => t.confidence_tier === "strong").length;
    parts.push(`강함 등급 인자 ${strongCount}/${withTier.length}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

export default function TrainingPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { training, setTraining, snapshot } = useAnalysisState();
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, open);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  // 지시서 §2-2: 승격 여부와 무관한 최근 학습 시도 -- 게이트 미달로
  // 교체되지 않았을 때 "학습은 돌았는데 모델은 그대로"임을 보여준다.
  // RB-4: 승격 게이트 자체는 제거됐지만(무조건 교체), 성능 저하 시
  // 경고를 보여주는 데 여전히 같은 이력을 재사용한다.
  const [latestPromotion, setLatestPromotion] = useState<PromotionEvent | null>(null);
  // A-4: promotion-history 라우트가 죽어 있어도(예: /models/{model_id}에
  // 가려짐) 조용히 "승격 이력 없음"으로 보이면 저하 사실이 통째로
  // 숨는다 -- 조회 실패와 "이력 없음"을 구분해서 보여준다.
  const [promotionHistoryError, setPromotionHistoryError] = useState(false);
  // B-10-3: 잡 큐(jobId)와 별개인 동기 재학습 호출 -- "내장 데이터로
  // 재학습" 진행 중임을 나타낸다. isRunning은 둘 중 하나만 있어도 true다.
  const [retraining, setRetraining] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // D-6: 폴링 이펙트(아래)는 deps가 [jobId]뿐이라 학습이 도는 동안 다른
  // 탭(모델 분석 팝업)에서 고친 sqlHost/refreshMinutes를 클로저가 시작
  // 시점 값으로 붙잡고 있다 -- 완료 시점에 그 옛 값을 서버에 저장해
  // 사용자가 학습 중 고친 값을 덮어써 버렸다. ref로 항상 최신값을 읽는다.
  const trainingRef = useRef(training);
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

  // 팝업을 다시 열 때마다 최신 이력으로 되돌린다.
  useEffect(() => {
    if (!open) return;
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

  // 학습 완료(잡 폴링 또는 동기 재학습) 후 공통으로 하는 일 -- 최신
  // 성능을 읽어와 패널 상태·서버 저장을 갱신한다. 잡 큐 경로와
  // 내장-재학습 경로가 이 로직을 그대로 공유한다(새 파이프라인 아님).
  async function applyTrainedPerformance() {
    const performance = await getModelPerformance().catch(() => null);
    if (!performance) return;
    const dataset = performance.source_filename || "training";
    // D-6: 클로저의 training이 아니라 ref로 완료 시점의 최신
    // sqlHost/sqlPort/refreshIntervalMinutes(모델 분석 팝업이
    // 관리)를 읽어 그대로 보존한다 -- 이 팝업은 그 값을 모른다.
    const latestTraining = trainingRef.current;
    const nextTraining = {
      dataset,
      createdAt: performance.trained_at ?? new Date().toISOString(),
      performance,
      sqlHost: latestTraining?.sqlHost ?? "",
      sqlPort: latestTraining?.sqlPort ?? "",
      sqlDb: latestTraining?.sqlDb ?? "",
      sqlUser: latestTraining?.sqlUser ?? "",
      refreshIntervalMinutes: latestTraining?.refreshIntervalMinutes ?? null,
    };
    setTraining(nextTraining);
    void saveTrainingState(dataset, {
      performance,
      sqlHost: nextTraining.sqlHost,
      sqlPort: nextTraining.sqlPort,
      sqlDb: nextTraining.sqlDb,
      sqlUser: nextTraining.sqlUser,
      refreshIntervalMinutes: nextTraining.refreshIntervalMinutes,
    }).catch(() => {});
  }

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
          await applyTrainedPerformance();
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

  async function runManualTraining() {
    if (!file || jobId || retraining) return;
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

  // B-10-3: 학습 쪽 되돌리기는 등록 해제가 아니라 내장 train.CSV로 즉시
  // 재학습하는 것뿐이다(분석 쪽 deactivateDataset과 의미가 다르다) --
  // 서버가 학습을 마칠 때까지 기다리는 동기 호출이라 잡 큐를 타지 않는다.
  async function runBundledRetrain() {
    if (jobId || retraining) return;
    setError("");
    setMessage("");
    setRetraining(true);
    try {
      await retrainBundled();
      setMessage("내장 데이터로 재학습을 완료했습니다.");
      refreshPromotionHistory();
      await applyTrainedPerformance();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "내장 데이터로 재학습하지 못했습니다.");
    } finally {
      setRetraining(false);
    }
  }

  if (!open) return null;

  const performance = training?.performance;
  const isRunning = Boolean(jobId) || retraining;

  return (
    <div className="settingsPanelBackdrop" onClick={onClose} role="presentation">
      <div
        ref={panelRef}
        className="settingsPanel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="모델 학습"
        tabIndex={-1}
      >
        <div className="settingsPanelHeader">
          <h2>모델 학습</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>현재 학습 데이터</h3>
            <dl className="trainingInfoList">
              <div>
                <dt>출처</dt>
                <dd>
                  {performance?.model_id
                    ? training?.dataset === "train"
                      ? "내장 train.csv"
                      : "수동 업로드"
                    : "내장 train.csv"}
                </dd>
              </div>
              <div>
                <dt>소스 파일</dt>
                <dd>{performance?.source_filename ?? "-"}</dd>
              </div>
              <div>
                <dt>학습 시각</dt>
                <dd>{formatLastRun(performance?.trained_at)}</dd>
              </div>
              <div>
                <dt>데이터 크기</dt>
                <dd>
                  {performance?.row_count != null && performance?.feature_count != null
                    ? `${performance.row_count.toLocaleString()}행 × ${performance.feature_count.toLocaleString()}열`
                    : "-"}
                </dd>
              </div>
              <div>
                <dt>모델 버전</dt>
                <dd className="trainingChampionId" title={snapshot?.model.champion_version ?? undefined}>
                  {snapshot?.model.champion_version ?? "-"}
                </dd>
              </div>
            </dl>

            {/* B-6: "모드별 R²" -> "모델 성능" -- 최종 수율(Y) 성능을 맨
                위에, 타깃별은 R²·RMSE를 함께 보여준다(R²만으로는 오차
                크기를 알 수 없다). */}
            <p className="sectionLabel" style={{ marginTop: 14 }}>모델 성능</p>
            <dl className="trainingInfoList">
              <div>
                <dt>최종 수율(Y)</dt>
                <dd>
                  {performance?.final_yield
                    ? [formatR2(performance.final_yield.r2), formatRmse(performance.final_yield.rmse)]
                        .filter((part): part is string => part != null)
                        .join(" · ")
                    : "-"}
                </dd>
              </div>
            </dl>
            {performance?.targets && performance.targets.length > 0 && (
              <p className="sectionCaption" style={{ margin: "6px 0 0" }}>
                {performance.targets.map(targetPerformanceLine).join(" / ")}
              </p>
            )}
            {evidenceLine(performance) && <p className="sectionCaption" style={{ margin: "2px 0 0" }}>{evidenceLine(performance)}</p>}

            {/* RB-4: 승격 게이트를 제거했으므로(무조건 교체) 여기 문구도
                "게이트 미달"이 아니라 "교체됨 + 성능 변화"로 바뀐다.
                latestPromotion.promoted는 이제 항상 true지만, reason에
                성능 저하 여부가 담겨 있으면 그대로 보여준다(RB-4 "교체는
                하되 침묵하지 마라"). */}
            {latestPromotion && latestPromotion.promoted === 1 && (
              <p className={latestPromotion.reason.includes("저하") ? "notifyFieldError" : "notifyTestResult ok"}>
                모델 {latestPromotion.candidate_model_id}로 교체됨 · {formatLastRun(latestPromotion.created_at)} · {latestPromotion.reason}
              </p>
            )}
            {promotionHistoryError && (
              <p className="notifyFieldError">
                교체 이력을 불러오지 못했습니다.{" "}
                <button type="button" className="linkButton" onClick={refreshPromotionHistory}>
                  다시 시도
                </button>
              </p>
            )}
          </section>

          <section className="settingsSection">
            <h3>학습 데이터 변경</h3>
            <p className="settingsSectionDesc">파일을 선택하면 학습 대상으로 지정됩니다. [모델 학습 시작]을 눌러야 실제 학습이 실행됩니다.</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => fileInputRef.current?.click()} disabled={isRunning}>
                파일 선택
              </button>
              <button
                type="button"
                className="button secondary"
                onClick={() => void runBundledRetrain()}
                disabled={isRunning}
                title="내장 train.CSV로 즉시 재학습합니다. 수 초~수십 초 걸립니다."
              >
                {retraining ? "학습 중…" : "내장 데이터로 재학습"}
              </button>
            </div>
            {file && <p className="sectionCaption" style={{ margin: "6px 0 0" }}>선택한 파일: {file.name}</p>}
          </section>

          <section className="settingsSection">
            <h3>학습 시작</h3>
            <div className="notifyFormActions">
              <button type="button" className="button primary" onClick={() => void runManualTraining()} disabled={!file || isRunning}>
                {isRunning ? "학습 중…" : "모델 학습 시작"}
              </button>
            </div>
            {isRunning && <p className="trainingProgress" role="status">{stage || "내장 데이터로 재학습 중…"}</p>}
            {error && <p className="notifyFieldError">{error}</p>}
            {message && <p className="notifyTestResult ok">{message}</p>}
          </section>
        </div>
      </div>
    </div>
  );
}
