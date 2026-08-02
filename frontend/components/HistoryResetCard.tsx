"use client";

import { useEffect, useRef, useState } from "react";

import { getHistoryResetSummary, resetAllHistory } from "@/lib/api";
import type { HistoryResetResponse, HistoryResetSummary } from "@/types/data";


const REQUIRED_CONFIRMATION = "초기화";

type HistoryResetCardProps = {
  onResetComplete: () => void;
};

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "이력 초기화 중 서버 오류가 발생했습니다.";
}

function deletedMessage(response: HistoryResetResponse): string {
  return [
    `모델 ${response.deleted.models.toLocaleString("ko-KR")}개`,
    `예측 이력 ${response.deleted.prediction_histories.toLocaleString("ko-KR")}개`,
    `분석 이력 ${response.deleted.analysis_histories.toLocaleString("ko-KR")}개`,
  ].join(", ");
}

export default function HistoryResetCard({ onResetComplete }: HistoryResetCardProps) {
  const [summary, setSummary] = useState<HistoryResetSummary | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLElement | null>(null);
  const confirmationRef = useRef<HTMLInputElement | null>(null);
  const resettingRef = useRef(false);

  useEffect(() => {
    resettingRef.current = resetting;
  }, [resetting]);

  useEffect(() => {
    if (!dialogOpen) return;
    const previousOverflow = document.body.style.overflow;
    const trigger = triggerRef.current;
    document.body.style.overflow = "hidden";
    const focusTimer = window.setTimeout(() => confirmationRef.current?.focus(), 0);

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !resettingRef.current) {
        event.preventDefault();
        setDialogOpen(false);
        setConfirmation("");
        setError("");
        return;
      }
      if (event.key !== "Tab") return;
      const dialog = dialogRef.current;
      const focusable = Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          "button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
        ) ?? [],
      );
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!dialog?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      trigger?.focus();
    };
  }, [dialogOpen]);

  async function openDialog() {
    if (summaryLoading || resetting) return;
    setSummaryLoading(true);
    setError("");
    setNotice("");
    try {
      setSummary(await getHistoryResetSummary());
      setConfirmation("");
      setDialogOpen(true);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSummaryLoading(false);
    }
  }

  function closeDialog() {
    if (resettingRef.current) return;
    setDialogOpen(false);
    setConfirmation("");
    setError("");
  }

  async function permanentlyReset() {
    if (confirmation !== REQUIRED_CONFIRMATION || resettingRef.current) return;
    resettingRef.current = true;
    setResetting(true);
    setError("");

    let response: HistoryResetResponse;
    try {
      response = await resetAllHistory();
      if (response.success !== true) {
        throw new Error("이력 초기화 중 서버 오류가 발생했습니다.");
      }
    } catch (requestError) {
      setError(errorMessage(requestError));
      resettingRef.current = false;
      setResetting(false);
      return;
    }

    onResetComplete();
    setNotice(
      `모델 학습, 수율 예측, 원인 분석 이력이 모두 초기화되었습니다. ${deletedMessage(response)}를 삭제했습니다.`,
    );
    setSummary(null);
    setConfirmation("");
    resettingRef.current = false;
    setResetting(false);
    setDialogOpen(false);
  }

  return (
    <>
      <section
        className="surfaceCard overviewSectionCard historyResetCard"
        aria-labelledby="history-reset-card-title"
      >
        <div className="historyResetCardCopy">
          <span className="sectionLabel">Data management</span>
          <h2 id="history-reset-card-title">데이터 및 이력 초기화</h2>
          <p>
            서버에 저장된 모델 학습·수율 예측·원인 분석 이력을 모두 삭제합니다.
            이 작업은 되돌릴 수 없습니다.
          </p>
        </div>
        <div className="historyResetActions">
          <button
            ref={triggerRef}
            className="button historyResetDestructive"
            type="button"
            disabled={summaryLoading || resetting}
            data-loading={summaryLoading ? "true" : undefined}
            aria-haspopup="dialog"
            aria-controls="history-reset-dialog"
            onClick={() => void openDialog()}
          >
            {summaryLoading ? "삭제 대상을 확인하는 중…" : "모든 이력 초기화"}
          </button>
        </div>
        {notice && (
          <div className="historyResetNotice success" role="status" aria-live="polite">
            {notice}
          </div>
        )}
        {error && !dialogOpen && (
          <div className="historyResetNotice error" role="alert">{error}</div>
        )}
      </section>

      {dialogOpen && summary && (
        <div
          className="historyResetBackdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeDialog();
          }}
        >
          <section
            id="history-reset-dialog"
            ref={dialogRef}
            className="historyResetDialog"
            role="dialog"
            aria-modal="true"
            aria-busy={resetting}
            aria-labelledby="history-reset-dialog-title"
            aria-describedby="history-reset-dialog-description history-reset-warning"
          >
            <header>
              <span className="sectionLabel">Permanent deletion</span>
              <h2 id="history-reset-dialog-title">모든 분석 이력을 초기화하시겠습니까?</h2>
            </header>
            <p id="history-reset-dialog-description">다음 데이터가 영구적으로 삭제됩니다.</p>
            <ul className="historyResetSummary">
              <li>저장 모델 및 학습 이력 {summary.model_count.toLocaleString("ko-KR")}개</li>
              <li>수율 예측 이력 {summary.prediction_history_count.toLocaleString("ko-KR")}개</li>
              <li>원인 분석 이력 {summary.analysis_history_count.toLocaleString("ko-KR")}개</li>
              <li>연결된 상세 분석 결과</li>
            </ul>
            <p id="history-reset-warning" className="historyResetWarning">
              이 작업은 되돌릴 수 없습니다.
            </p>
            <label className="historyResetConfirmation" htmlFor="history-reset-confirmation">
              <span>계속하려면 <strong>초기화</strong>를 입력해 주세요.</span>
              <input
                ref={confirmationRef}
                id="history-reset-confirmation"
                type="text"
                value={confirmation}
                disabled={resetting}
                autoComplete="off"
                spellCheck={false}
                onChange={(event) => setConfirmation(event.target.value)}
              />
            </label>
            {error && <div className="historyResetNotice error" role="alert">{error}</div>}
            <div className="historyResetDialogActions">
              <button
                className="button secondary"
                type="button"
                disabled={resetting}
                onClick={closeDialog}
              >
                취소
              </button>
              <button
                className="button historyResetDestructive"
                type="button"
                disabled={confirmation !== REQUIRED_CONFIRMATION || resetting}
                data-loading={resetting ? "true" : undefined}
                onClick={() => void permanentlyReset()}
              >
                {resetting ? "이력을 초기화하고 있습니다…" : "영구 삭제"}
              </button>
            </div>
            <span className="srOnly" role="status" aria-live="polite">
              {resetting ? "이력을 초기화하고 있습니다…" : ""}
            </span>
          </section>
        </div>
      )}
    </>
  );
}
