"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { activateDataset, saveTrainingState, uploadDataset } from "@/lib/api";
import { formatLastRun } from "@/lib/timeFormat";
import { useFocusTrap } from "@/lib/useFocusTrap";

// RA-1/RA-2/RC-1: "모델 학습·자동화" 통합 팝업을 둘로 나누면서 생긴
// 새 팝업. SQL 연결·refresh time은 분석 데이터를 가져오기 위한
// 것이므로(자동화는 분석의 입력) 여기(모델 분석)에 둔다 -- 독립
// 자동화 팝업은 만들지 않는다(RA-2 "하지 말 것").
//
// SQL 연결/refresh 저장은 기존 `saveTrainingState`(POST
// /api/state/training)를 그대로 재사용한다 -- 이 값들이 서버에 저장되는
// 위치(state 테이블의 "training" 슬롯)는 바뀌지 않았고, 어느 팝업이
// 읽고 쓰는지만 바뀌었다.
//
// "현재 분석 데이터"는 자동 갱신 스냅샷(RefreshSnapshot)에서 그대로
// 읽는다 -- 실측 상태(measured_rows/predicted_rows/mixed_rows)는 이미
// target_provenance가 들고 있으므로 새 API가 필요 없다.
function formatNextRefresh(createdAtIso: string | null | undefined, refreshIntervalMinutes: number | null | undefined): string | null {
  if (!createdAtIso || !refreshIntervalMinutes) return null;
  const created = new Date(createdAtIso);
  if (Number.isNaN(created.getTime())) return null;
  return formatLastRun(new Date(created.getTime() + refreshIntervalMinutes * 60_000).toISOString());
}

function sourceLabel(mode: "sql" | "fallback" | "manual" | undefined): string {
  if (mode === "sql") return "자동(SQL)";
  if (mode === "manual") return "수동 업로드";
  if (mode === "fallback") return "내장 test.csv";
  return "-";
}

function measuredStatusLabel(provenance: { measured_rows: number; predicted_rows: number; mixed_rows: number } | null | undefined): string {
  if (!provenance) return "-";
  const { measured_rows, predicted_rows, mixed_rows } = provenance;
  const total = measured_rows + predicted_rows + mixed_rows;
  if (total === 0) return "-";
  if (predicted_rows === total) return "y1~y5 전부 결측 (전부 예측)";
  if (measured_rows === total) return "전부 실측";
  return `일부 실측 (실측 ${measured_rows.toLocaleString()}장 · 혼재 ${mixed_rows.toLocaleString()}장 · 전부 예측 ${predicted_rows.toLocaleString()}장)`;
}

export default function ModelAnalysisPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { training, setTraining, snapshot } = useAnalysisState();
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, open);
  const [sqlHost, setSqlHost] = useState(training?.sqlHost ?? "");
  const [sqlPort, setSqlPort] = useState(training?.sqlPort ?? "");
  const [sqlDb, setSqlDb] = useState(training?.sqlDb ?? "");
  const [sqlUser, setSqlUser] = useState(training?.sqlUser ?? "");
  const [refreshMinutes, setRefreshMinutes] = useState(
    training?.refreshIntervalMinutes != null ? String(training.refreshIntervalMinutes) : "",
  );
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [recentCycleMinutes, setRecentCycleMinutes] = useState<number | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setSqlHost(training?.sqlHost ?? "");
    setSqlPort(training?.sqlPort ?? "");
    setSqlDb(training?.sqlDb ?? "");
    setSqlUser(training?.sqlUser ?? "");
    setRefreshMinutes(training?.refreshIntervalMinutes != null ? String(training.refreshIntervalMinutes) : "");
    setError("");
    setMessage("");
    setUploadError("");
    setUploadMessage("");
    // RC-1: "최근 사이클 소요 12분" -- 실제로 재려면 run_refresh_pipeline이
    // 소요 시간을 스냅샷에 기록해야 하는데, 아직 어디에도 그 계측이
    // 없다(RF-2 조사: record_run은 정의만 있고 호출부가 없다). 추측값을
    // 보여주지 않기 위해 지금은 항상 null -- RC 그룹에서 실제 계측을
    // 추가한다.
    setRecentCycleMinutes(null);
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

  async function saveSettings() {
    setError("");
    setMessage("");
    const dataset = training?.dataset || "training-settings";
    const refreshIntervalMinutes = refreshMinutes.trim() ? Number(refreshMinutes) : null;
    try {
      const performance = training?.performance ?? {
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
        sqlDb,
        sqlUser,
        refreshIntervalMinutes,
      });
      setTraining((previous) =>
        previous
          ? { ...previous, sqlHost, sqlPort, sqlDb, sqlUser, refreshIntervalMinutes }
          : {
              dataset,
              createdAt: new Date().toISOString(),
              performance,
              sqlHost,
              sqlPort,
              sqlDb,
              sqlUser,
              refreshIntervalMinutes,
            },
      );
      if (!result.schedule_applied) {
        setError("설정은 저장됐지만 자동 수집 주기 반영에는 실패했습니다. 다시 시도해 주세요.");
      } else {
        setMessage("설정을 저장했습니다.");
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "설정을 저장하지 못했습니다.");
    }
  }

  async function runManualAnalysis() {
    if (!file || uploading) return;
    setUploadError("");
    setUploadMessage("");
    setUploading(true);
    try {
      const result = await uploadDataset(file);
      if (!result.success) {
        setUploadError(result.blocking_errors.join(" ") || "업로드가 거부되었습니다.");
        return;
      }
      if (result.dataset_id) {
        await activateDataset(result.dataset_id);
        setUploadMessage("업로드한 파일을 분석 데이터로 반영했습니다.");
        setFile(null);
      }
    } catch (failure) {
      setUploadError(failure instanceof Error ? failure.message : "모델 분석을 시작하지 못했습니다.");
    } finally {
      setUploading(false);
    }
  }

  if (!open) return null;

  const source = snapshot?.source;
  const provenance = snapshot?.analysis.target_provenance;
  const refreshIntervalNumber = refreshMinutes.trim() ? Number(refreshMinutes) : null;
  // RC-1: "주기가 소요 시간보다 짧으면 경고" -- 정확한 사이클 소요 시간
  // 계측이 아직 없어(위 recentCycleMinutes) 항상 null이면 경고를 내지
  // 않는다(추측으로 경고하지 않는다).
  const cycleTooShort =
    recentCycleMinutes != null && refreshIntervalNumber != null && refreshIntervalNumber > 0 && refreshIntervalNumber < recentCycleMinutes;

  return (
    <div className="settingsPanelBackdrop" onClick={onClose} role="presentation">
      <div
        ref={panelRef}
        className="settingsPanel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="모델 분석·자동화"
        tabIndex={-1}
      >
        <div className="settingsPanelHeader">
          <h2>모델 분석·자동화</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>SQL 연결</h3>
            <p className="settingsSectionDesc">호스트·포트만 저장합니다. 비밀번호는 서버 환경변수(DB_PASSWORD)로 설정합니다.</p>
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
            <div className="trainingSqlRow">
              <label className="notifyFieldLabel">
                DB명
                <input type="text" value={sqlDb} onChange={(event) => setSqlDb(event.target.value)} placeholder="suni_prod" />
              </label>
              <label className="notifyFieldLabel">
                사용자명
                <input type="text" value={sqlUser} onChange={(event) => setSqlUser(event.target.value)} placeholder="suni_reader" />
              </label>
            </div>
            <label className="notifyFieldLabel">
              Refresh Time (분마다 최신 데이터를 받아 분석)
              <input
                type="number"
                min={0}
                value={refreshMinutes}
                onChange={(event) => setRefreshMinutes(event.target.value)}
                placeholder="60"
              />
            </label>
            {cycleTooShort && (
              <p className="notifyFieldError">
                현재 주기({refreshIntervalNumber}분)가 최근 소요({recentCycleMinutes}분)보다 짧아 사이클이 스킵될 수 있습니다.
              </p>
            )}
            <p className="settingsSectionDesc">
              연결 상태{" "}
              <span className={`trainingAutomationStatus${source?.mode === "sql" ? "" : " offline"}`}>
                <span className="sidebarStatusDot" aria-hidden="true" />
                {sourceLabel(source?.mode)}
              </span>
              {snapshot?.source && " · 마지막 스캔 "}
              {snapshot && formatLastRun(snapshot.created_at)}
            </p>
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => void saveSettings()}>
                연결 테스트 및 설정 저장
              </button>
            </div>
            {error && <p className="notifyFieldError">{error}</p>}
            {message && <p className="notifyTestResult ok">{message}</p>}
          </section>

          <section className="settingsSection">
            <h3>현재 분석 데이터</h3>
            <dl className="trainingInfoList">
              <div>
                <dt>출처</dt>
                <dd>{sourceLabel(source?.mode)}</dd>
              </div>
              <div>
                <dt>소스 파일</dt>
                <dd>{source?.eval_dataset_filename ?? source?.eval_dataset ?? "-"}</dd>
              </div>
              <div>
                <dt>분석 시각</dt>
                <dd>{snapshot ? formatLastRun(snapshot.created_at) : "-"}</dd>
                {formatNextRefresh(snapshot?.created_at, training?.refreshIntervalMinutes) && (
                  <dd className="settingsSectionDesc">다음 갱신 {formatNextRefresh(snapshot?.created_at, training?.refreshIntervalMinutes)}</dd>
                )}
              </div>
              <div>
                <dt>데이터 크기</dt>
                <dd>{source?.row_count != null ? `${source.row_count.toLocaleString()}행` : "-"}</dd>
              </div>
              <div>
                <dt>실측 상태</dt>
                <dd>{measuredStatusLabel(provenance)}</dd>
              </div>
              <div>
                <dt>사용 모델</dt>
                <dd className="trainingChampionId" title={snapshot?.model.champion_version ?? undefined}>
                  {snapshot?.model.champion_version ?? "-"}
                </dd>
              </div>
            </dl>
            {snapshot && snapshot.errors.length > 0 && (
              <p className="notifyFieldError">최근 갱신 오류: {snapshot.errors.join(" · ")}</p>
            )}
          </section>

          <section className="settingsSection">
            <h3>수동 모델 분석</h3>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <div className="notifyFormActions">
              <button type="button" className="button secondary" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                파일 선택{file ? `: ${file.name}` : ""}
              </button>
              <button type="button" className="button primary" onClick={() => void runManualAnalysis()} disabled={!file || uploading}>
                {uploading ? "분석 중…" : "CSV 업로드"}
              </button>
            </div>
            {uploadError && <p className="notifyFieldError">{uploadError}</p>}
            {uploadMessage && <p className="notifyTestResult ok">{uploadMessage}</p>}
          </section>
        </div>
      </div>
    </div>
  );
}
