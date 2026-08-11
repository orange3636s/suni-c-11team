"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { activateDataset, ApiResponseError, deactivateDataset, fetchFromDb, triggerRefresh, uploadDataset } from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";
import { useFocusTrap } from "@/lib/useFocusTrap";

// "모델 분석" 팝업 -- 네 화면(모니터링 홈·Config별 트리맵·원인
// 분석·수율 예측)을 한 번에 갱신하는 [분석 시작]의 유일한 진입점이다.
// SQL 연결·refresh time·자동화 on/off는 "알림·자동화 설정"(SettingsPanel)
// 이 담당한다 -- 이 팝업은 "어떤 데이터로 분석할지"만 다룬다.
//
// 데이터 소스 등록(파일 선택/데이터베이스에서 불러오기)과 실제 분석
// 실행([분석 시작])은 별개다 -- 등록은 즉시 반영되지만, 네 화면 계산은
// 사용자가 [분석 시작]을 눌러야 시작된다.

function sourceLabel(mode: "sql" | "fallback" | "manual" | undefined): string {
  if (mode === "manual") return "수동 등록 (업로드 또는 데이터베이스)";
  if (mode === "fallback") return "내장 test_remove_y.CSV";
  if (mode === "sql") return "자동(SQL)";
  return "-";
}

// src/automation/refresh.py STAGE_LABELS와 같은 문구 -- 실패 단계를
// 사용자에게 보여줄 때 원시 키("pareto") 대신 이 라벨을 쓴다.
const STAGE_LABEL_KO: Record<string, string> = {
  resolve: "데이터 확인",
  hydrate_eval: "모델 추론 (분석 데이터)",
  fmea: "데이터 한계 진단",
  action_priority: "조치 우선순위 (학습 데이터)",
  treemap_warmup: "Config별 트리맵",
  pareto: "원인 분석",
  yield_prediction: "수율 예측",
  save: "저장",
};

function measuredStatusLabel(provenance: { measured_rows: number; predicted_rows: number; mixed_rows: number } | null | undefined): string {
  if (!provenance) return "-";
  const { measured_rows, predicted_rows, mixed_rows } = provenance;
  const total = measured_rows + predicted_rows + mixed_rows;
  if (total === 0) return "-";
  if (predicted_rows === total) return "Y1~Y5 전부 결측 (모델 예측)";
  if (measured_rows === total) return "전부 실측";
  return `일부 실측 (실측 ${measured_rows.toLocaleString()}장 · 혼재 ${mixed_rows.toLocaleString()}장 · 전부 예측 ${predicted_rows.toLocaleString()}장)`;
}

// 등록 직후(아직 [분석 시작]을 누르기 전이거나 파이프라인이 도는 중)에는
// snapshot이 이 등록을 반영하기 전이므로, 업로드/DB 조회 응답이 이미
// 들고 있는 정보로 "현재 분석 데이터" 블록을 즉시 보여준다.
type PendingRegistration = {
  datasetId: string;
  filename: string;
  rowCount: number | null;
  setAt: string;
};

export default function ModelAnalysisPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { snapshot, manualEvalOverride, notifications, refreshRunning, analysisProgress, refreshSnapshotNow, lastRun } = useAnalysisState();
  const [pending, setPending] = useState<PendingRegistration | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, open);

  const [registerError, setRegisterError] = useState("");
  const [registerMessage, setRegisterMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [fetchingDb, setFetchingDb] = useState(false);
  const [reverting, setReverting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [startError, setStartError] = useState("");

  useEffect(() => {
    if (!open) return;
    setRegisterError("");
    setRegisterMessage("");
    setStartError("");
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  // 새 스냅샷이 도착하면(이 등록을 반영했든, 그 사이 다른 소스가
  // 활성화됐든) pending 표시를 접는다.
  useEffect(() => {
    if (!pending) return;
    if (manualEvalOverride?.dataset_id === pending.datasetId) {
      // 아직 이 등록 그대로다 -- 계속 pending으로 보여준다.
      return;
    }
    setPending(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manualEvalOverride?.dataset_id]);

  if (!open) return null;

  const source = snapshot?.source;
  const provenance = snapshot?.analysis.target_provenance;
  const dbConfigured = Boolean(
    notifications.automation.sql_host && notifications.automation.sql_port && notifications.automation.sql_db && notifications.automation.sql_user,
  );

  async function registerDataset(result: { dataset_id?: string | null; row_count?: number | null; success?: boolean; blocking_errors?: string[] }, filenameFallback: string) {
    if (result.success === false) {
      setRegisterError((result.blocking_errors ?? []).join(" ") || "등록이 거부되었습니다.");
      return;
    }
    if (!result.dataset_id) {
      setRegisterError("등록에 실패했습니다.");
      return;
    }
    setPending({ datasetId: result.dataset_id, filename: filenameFallback, rowCount: result.row_count ?? null, setAt: new Date().toISOString() });
    await activateDataset(result.dataset_id);
    setRegisterMessage("분석 데이터로 등록했습니다. [분석 시작]을 눌러 네 화면을 갱신하세요.");
    refreshSnapshotNow();
  }

  async function handleFileChosen(chosen: File | null) {
    if (!chosen) return;
    setRegisterError("");
    setRegisterMessage("");
    setUploading(true);
    try {
      const result = await uploadDataset(chosen);
      await registerDataset(result, chosen.name);
    } catch (failure) {
      setRegisterError(failure instanceof Error ? failure.message : "업로드하지 못했습니다.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleFetchFromDb() {
    setRegisterError("");
    setRegisterMessage("");
    setFetchingDb(true);
    try {
      const result = await fetchFromDb();
      await registerDataset(result, result.dataset_id ?? "db_fetch.csv");
    } catch (failure) {
      if (failure instanceof ApiResponseError && failure.status === 404) {
        setRegisterError("가져올 새 데이터가 없습니다.");
      } else {
        setRegisterError(failure instanceof Error ? failure.message : "데이터베이스에서 가져오지 못했습니다.");
      }
    } finally {
      setFetchingDb(false);
    }
  }

  // B-10-2: 상단 배너(ManualModeBanner, 제거됨)의 되돌리기 로직을 그대로
  // 옮긴다 -- 등록만 지울 뿐(SC-3과 분리) 분석을 자동으로 다시 실행하지
  // 않는다는 동작은 그대로다.
  async function handleRevert() {
    setRegisterError("");
    setReverting(true);
    try {
      await deactivateDataset();
      refreshSnapshotNow();
    } catch {
      setRegisterError("되돌리지 못했습니다.");
    } finally {
      setReverting(false);
    }
  }

  async function handleStartAnalysis() {
    setStartError("");
    try {
      await triggerRefresh();
      refreshSnapshotNow();
    } catch (failure) {
      setStartError(
        failure instanceof ApiResponseError && failure.status === 409
          ? "이미 분석이 진행 중입니다."
          : failure instanceof Error
            ? failure.message
            : "분석을 시작하지 못했습니다.",
      );
    }
  }

  const progressLabel = analysisProgress ? `분석 진행 중… (${analysisProgress.index}/${analysisProgress.total}) ${analysisProgress.stage}` : refreshRunning ? "분석 진행 중…" : null;
  // 작업지시(Config 하이드레이션 실패 수정) T4: "triggered: true"를 받은
  // 뒤에도 백그라운드 실행이 조용히 실패할 수 있다 -- lastRun.status가
  // "failed"면 실패 단계·사유를 그대로 보여준다. 실패해도 마지막 정상
  // 스냅샷은 그대로 보존되므로, "지금 화면은 이전 결과"임을 함께 안내한다
  // (최근 실패와 혼동하지 않도록 분리해서 표시).
  const lastRunFailed = !refreshRunning && lastRun?.status === "failed";

  return (
    <div className="settingsPanelBackdrop" onClick={onClose} role="presentation">
      <div
        ref={panelRef}
        className="settingsPanel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="모델 분석"
        tabIndex={-1}
      >
        <div className="settingsPanelHeader">
          <h2>모델 분석</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>현재 분석 데이터</h3>
            <dl className="trainingInfoList">
              <div>
                <dt>출처</dt>
                <dd>{pending ? "수동 등록 (반영 대기 중)" : sourceLabel(source?.mode)}</dd>
              </div>
              <div>
                <dt>파일명</dt>
                <dd>{pending ? pending.filename : (source?.eval_dataset_filename ?? source?.eval_dataset ?? "-")}</dd>
              </div>
              <div>
                <dt>등록 시각</dt>
                <dd>{pending ? formatLastRun(pending.setAt) : snapshot ? formatLastRun(snapshot.created_at) : "-"}</dd>
              </div>
              <div>
                <dt>데이터 크기</dt>
                <dd>
                  {pending
                    ? pending.rowCount != null
                      ? `${pending.rowCount.toLocaleString()}행`
                      : "-"
                    : source?.row_count != null
                      ? `${source.row_count.toLocaleString()}행`
                      : "-"}
                </dd>
              </div>
              <div>
                <dt>실측 상태</dt>
                <dd>{pending ? "분석 시작 후 확인 가능" : measuredStatusLabel(provenance)}</dd>
              </div>
              <div>
                <dt>모델</dt>
                <dd className="trainingChampionId" title={snapshot?.model.champion_version ?? undefined}>
                  {snapshot?.model.champion_version ?? "-"}
                </dd>
              </div>
            </dl>
            {snapshot && snapshot.errors.length > 0 && (
              <p className="notifyFieldError">최근 분석 오류: {snapshot.errors.join(" · ")}</p>
            )}
          </section>

          <section className="settingsSection">
            <h3>분석 데이터 변경</h3>
            <p className="settingsSectionDesc">
              파일을 선택하거나 데이터베이스에서 불러오면 즉시 분석 데이터로 등록됩니다 -- 다시 바꿀 때까지 유지됩니다.
              되돌린 결과를 보려면 [분석 시작]을 눌러야 합니다.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(event) => void handleFileChosen(event.target.files?.[0] ?? null)}
            />
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => fileInputRef.current?.click()} disabled={uploading || fetchingDb}>
                {uploading ? "업로드 중…" : "파일 선택"}
              </button>
              <button
                type="button"
                className="button secondary"
                onClick={() => void handleFetchFromDb()}
                disabled={uploading || fetchingDb || !dbConfigured}
                title={dbConfigured ? undefined : "알림·자동화 설정에서 서버를 먼저 등록하세요"}
              >
                {fetchingDb ? "가져오는 중…" : "데이터베이스에서 불러오기"}
              </button>
              <button
                type="button"
                className="button secondary"
                onClick={() => void handleRevert()}
                disabled={!manualEvalOverride || reverting}
                title={manualEvalOverride ? undefined : "되돌릴 수동 등록이 없습니다"}
              >
                {reverting ? "되돌리는 중…" : "내장 데이터로 되돌리기"}
              </button>
            </div>
            {registerError && <p className="notifyFieldError">{registerError}</p>}
            {registerMessage && <p className="notifyTestResult ok">{registerMessage}</p>}
          </section>

          <section className="settingsSection">
            <h3>분석 시작</h3>
            <p className="settingsSectionDesc">
              등록된 데이터로 네 화면(모니터링 홈·Config별 트리맵·원인 분석·수율 예측)을 한 번에 갱신합니다. 서버 지연으로
              화면이 비었을 때도 이 버튼으로 복구할 수 있습니다.
            </p>
            <div className="notifyFormActions">
              <button type="button" className="button primary" onClick={() => void handleStartAnalysis()} disabled={refreshRunning}>
                {refreshRunning ? "분석 중…" : "분석 시작"}
              </button>
            </div>
            {progressLabel && <p className="settingsSectionDesc">{progressLabel} · 팝업을 닫아도 계속됩니다.</p>}
            {startError && <p className="notifyFieldError">{startError}</p>}
            {lastRunFailed && (
              <p className="notifyFieldError">
                최근 분석 실패
                {lastRun?.failed_stage && ` (${STAGE_LABEL_KO[lastRun.failed_stage] ?? lastRun.failed_stage} 단계)`}
                {": "}
                {lastRun?.error_message ?? "알 수 없는 오류"}
                {snapshot && " · 지금 화면에 보이는 결과는 이전 분석 그대로입니다."}
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
