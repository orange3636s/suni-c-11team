"use client";

import { useMemo, useRef, useState } from "react";

import StatusBadge from "@/components/StatusBadge";
import type { AnalysisHistorySummary } from "@/types/data";


type AnalysisHistorySelectorProps = {
  items: AnalysisHistorySummary[];
  selectedAnalysisId: string | null;
  loading: boolean;
  error: string;
  onSelect: (analysisId: string) => void;
  onRetry: () => void;
};

function safeDate(value: string | null | undefined): string {
  if (!value) return "시각 정보 없음";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "시각 정보 없음" : date.toLocaleString("ko-KR");
}

function statusPresentation(status: AnalysisHistorySummary["status"]): {
  label: string;
  tone: "success" | "info" | "warning" | "danger" | "neutral";
} {
  if (status === "completed") return { label: "완료", tone: "success" };
  if (status === "partial") return { label: "일부 결과", tone: "warning" };
  if (status === "artifact_missing") return { label: "결과 파일 누락", tone: "warning" };
  if (status === "artifact_corrupted") return { label: "결과 파일 손상", tone: "danger" };
  if (status === "failed") return { label: "실패", tone: "danger" };
  if (status === "running") return { label: "진행 중", tone: "info" };
  return { label: status, tone: "neutral" };
}

function summaryNumber(item: AnalysisHistorySummary, key: string): number | null {
  const direct = key === "average_predicted_yield"
    ? item.average_predicted_yield
    : key === "critical_count"
      ? item.critical_count
      : undefined;
  if (typeof direct === "number" && Number.isFinite(direct)) return direct;
  const value = item.summary?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export default function AnalysisHistorySelector({
  items,
  selectedAnalysisId,
  loading,
  error,
  onSelect,
  onRetry,
}: AnalysisHistorySelectorProps) {
  const [search, setSearch] = useState("");
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const selected = items.find((item) => item.analysis_id === selectedAnalysisId) ?? null;
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase("ko-KR");
    if (!query) return items;
    return items.filter((item) => [
      item.analysis_id,
      item.source_filename,
      item.model_name,
      item.model_name_snapshot,
      item.model_id,
    ].some((value) => value?.toLocaleLowerCase("ko-KR").includes(query)));
  }, [items, search]);

  const summaryLabel = loading
    ? "분석 이력 불러오는 중"
    : error
      ? "분석 이력 조회 실패"
      : selected
        ? `${selected.source_filename ?? "파일명 없음"} · ${safeDate(selected.completed_at ?? selected.created_at)}`
        : "분석 이력 선택";

  return (
    <details className="analysisHistorySelector" ref={detailsRef}>
      <summary title={summaryLabel}>
        <span>
          <small>분석 이력 선택</small>
          <strong>{summaryLabel}</strong>
        </span>
        <span className="selectorChevron" aria-hidden="true">⌄</span>
      </summary>
      <div className="analysisHistoryPopover">
        <label className="analysisHistorySearch">
          <span className="srOnly">분석 이력 검색</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="파일명, 모델명, analysis_id 검색"
            autoComplete="off"
          />
        </label>

        {loading ? (
          <div className="selectorState">분석 이력을 불러오는 중입니다.</div>
        ) : error ? (
          <div className="selectorState error">
            <span>{error}</span>
            <button className="button secondary" type="button" onClick={onRetry}>다시 시도</button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="selectorState">검색 조건에 맞는 분석 이력이 없습니다.</div>
        ) : (
          <div className="analysisHistoryOptions" role="listbox" aria-label="불량 원인 분석 이력">
            {filtered.map((item) => {
              const presentation = statusPresentation(item.status);
              const isSelected = item.analysis_id === selectedAnalysisId;
              const isUnavailable = item.status === "running" || item.status === "failed";
              const averageYield = summaryNumber(item, "average_predicted_yield");
              const criticalCount = summaryNumber(item, "critical_count");
              return (
                <button
                  key={item.analysis_id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  disabled={isUnavailable}
                  className={`analysisHistoryOption${isSelected ? " selected" : ""}`}
                  title={`${item.analysis_id}\n${item.source_filename ?? "파일명 없음"}\n${item.model_name ?? item.model_name_snapshot ?? item.model_id ?? "모델 정보 없음"}`}
                  onClick={() => {
                    onSelect(item.analysis_id);
                    if (detailsRef.current) detailsRef.current.open = false;
                  }}
                >
                  <span className="analysisHistoryOptionTop">
                    <time>{safeDate(item.completed_at ?? item.created_at)}</time>
                    <StatusBadge label={presentation.label} tone={presentation.tone} dot={false} />
                  </span>
                  <strong>{item.source_filename ?? "파일명 없음"}</strong>
                  <span className="analysisHistoryOptionMeta">
                    {item.model_name ?? item.model_name_snapshot ?? item.model_id ?? "모델 정보 없음"}
                  </span>
                  {(averageYield !== null || criticalCount !== null) && (
                    <span className="analysisHistoryOptionMetrics">
                      {averageYield !== null && <span>평균 {averageYield.toLocaleString("ko-KR", { maximumFractionDigits: 2 })}%</span>}
                      {criticalCount !== null && <span>Critical {criticalCount.toLocaleString("ko-KR")}</span>}
                    </span>
                  )}
                  <span className="analysisHistoryOptionId">{item.analysis_id}</span>
                  {isSelected && <span className="analysisHistoryCheck" aria-label="선택됨">✓</span>}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </details>
  );
}
