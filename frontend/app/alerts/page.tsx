"use client";

import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import CurrentDatasetLabel from "@/components/CurrentDatasetLabel";
import {
  HScrollTableBody,
  NoSearchResults,
  TableCaption,
  TableToolbar,
  useTableSearchSort,
} from "@/components/DataTablePanel";
import DashboardShell from "@/components/DashboardShell";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import { LastRunNote, TrainingAnalysisDataNote } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import { getAlertsRanking } from "@/lib/api";
import type { AlertCandidate, AlertCellColor, AlertRankingResponse } from "@/types/data";

// RE그룹: 알림기록 판정 -- y(=100 − Σ Y1~Y5) 오름차순 상위 N건. 정렬은
// 이 하나뿐이다("하지 말 것: 신뢰도로 정렬하거나 신뢰도 하한으로
// 후보를 거르지 마라"). 신뢰도(RC-4)는 판단 재료로 표시만 한다.

const DEFAULT_TOP_N = 10;
const LOW_RELIABILITY_THRESHOLD = 40;
const FAIL_TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

function cellClassName(cell: AlertCellColor | undefined): string {
  if (!cell) return "alertCell";
  if (cell.shade === "measured") return "alertCell measured";
  if (cell.direction == null) return "alertCell gray";
  return `alertCell ${cell.direction} ${cell.shade}`;
}

function directionText(cell: AlertCellColor): string | null {
  if (cell.direction == null || cell.optimal_center == null || cell.factor_value == null) return null;
  const side = cell.factor_value > cell.optimal_center ? "오른쪽" : "왼쪽";
  const effect = cell.direction === "red" ? "악화" : "개선";
  return `꼭짓점 ${cell.optimal_center.toFixed(1)} ${side} → 증가 시 ${effect}`;
}

function cellTooltip(target: string, valuePct: number | undefined, cell: AlertCellColor | undefined): string {
  if (!cell) return target;
  if (cell.shade === "measured") return `${target}  실측 ${valuePct != null ? valuePct.toFixed(2) : "-"}%p`;
  const lines = [`${target}  예측 ${valuePct != null ? valuePct.toFixed(2) : "-"}%p`];
  if (cell.feature) {
    lines.push(`인자 ${cell.feature}${cell.factor_value != null ? ` = ${cell.factor_value.toFixed(1)} (계측)` : " (미계측)"}`);
  }
  if (cell.contribution_pct != null) {
    const dir = directionText(cell);
    lines.push(`파레토 기여율 ${cell.contribution_pct.toFixed(1)}%${dir ? ` · ${dir}` : ""}`);
  }
  return lines.join("\n");
}

function downloadCsv(data: AlertRankingResponse) {
  const header = ["LOT_WF_ID", "LOT", "y", ...FAIL_TARGETS, "신뢰도", "주요 원인", "사유"];
  const rows = data.candidates.map((c) => [
    c.lot_wafer_id,
    c.lot_id ?? "",
    c.y.toFixed(2),
    ...FAIL_TARGETS.map((t) => (c.y_components[t] != null ? c.y_components[t].toFixed(2) : "")),
    String(c.reliability),
    c.primary_target,
    c.reason,
  ]);
  const csv = [header, ...rows].map((line) => line.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([`﻿${csv}`], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `alerts_${data.eval_dataset_id}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function AlertsPage() {
  return (
    <DashboardShell activeItem="수율 예측">
      <AlertsContent />
    </DashboardShell>
  );
}

function AlertsContent() {
  const { snapshot, alarms, training } = useAnalysisState();
  const { setAnalysisPanelOpen } = usePanelState();
  const [topN, setTopN] = useState(DEFAULT_TOP_N);
  const [data, setData] = useState<AlertRankingResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // RD-2: 데이터셋은 모델 분석이 정한다 -- 이 화면은 스냅샷/공유 상태가
  // 가리키는 값을 읽기만 한다.
  const trainDataset = snapshot?.source.train_dataset ?? alarms?.trainDataset ?? "train";
  const evalDataset = snapshot?.source.eval_dataset ?? alarms?.evalDataset ?? "test";

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getAlertsRanking(trainDataset, evalDataset, topN)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((failure) => {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "알림 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [trainDataset, evalDataset, topN]);

  const { search, setSearch, sort, setSort, sorted } = useTableSearchSort<AlertCandidate>(
    data?.candidates ?? [],
    (item) => `${item.lot_wafer_id} ${item.lot_id ?? ""} ${item.primary_target}`,
    [
      { value: "y_asc", label: "y 오름차순 (기본)", compare: (a, b) => a.y - b.y },
      { value: "reliability_desc", label: "신뢰도 높은 순", compare: (a, b) => b.reliability - a.reliability },
    ],
    "y_asc",
    (a, b) => a.lot_wafer_id.localeCompare(b.lot_wafer_id),
  );

  const zeroReliabilityInList = (data?.candidates ?? []).some((c) => c.reliability === 0);

  return (
    <div className="rcPage">
      <div className="pageHeading">
        <h1>수율 예측</h1>
        <p className="sectionCaption">
          수율(y) 낮은 순입니다. 신뢰도는 실측 모드는 1.0, 예측 모드는 핵심 인자의 파레토 기여율만큼 인정해 합산한 값입니다.
        </p>
        {data && <LastRunNote createdAt={snapshot?.created_at} />}
        <TrainingAnalysisDataNote
          trainFilename={training?.performance?.source_filename ?? null}
          evalFilename={snapshot?.source?.eval_dataset_filename ?? null}
        />
        <FallbackModeBadge />
      </div>

      <section className="resultCard">
        <div className="rcControlBar">
          <CurrentDatasetLabel label="예측 대상" datasetId={evalDataset} onOpenAnalysisPanel={() => setAnalysisPanelOpen(true)} />
          <label className="fieldGroup">
            <span>표시 건수</span>
            <input
              type="number"
              min={1}
              max={100}
              value={topN}
              onChange={(event) => setTopN(Math.max(1, Math.min(100, Number(event.target.value) || DEFAULT_TOP_N)))}
            />
          </label>
          <button type="button" className="button secondary" onClick={() => data && downloadCsv(data)} disabled={!data}>
            CSV 내보내기
          </button>
        </div>
      </section>

      {loading ? (
        <p className="emptyMessage">불러오는 중…</p>
      ) : error ? (
        <section className="resultCard">
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">알림 목록을 불러오지 못했습니다 — {error}</p>
            </div>
          </div>
        </section>
      ) : !data || data.candidates.length === 0 ? (
        <section className="resultCard">
          <p className="emptyMessage">알림 후보가 없습니다.</p>
        </section>
      ) : (
        <>
          <section className="resultCard">
            <TableToolbar
              search={search}
              onSearchChange={setSearch}
              sort={sort}
              onSortChange={setSort}
              sortOptions={[
                { value: "y_asc", label: "y 오름차순 (기본)" },
                { value: "reliability_desc", label: "신뢰도 높은 순" },
              ]}
              placeholder="LOT/Wafer 검색"
            />
            <HScrollTableBody rows={10} minWidth={1180}>
              <table className="dataTable">
                <thead>
                  <tr>
                    <th>LOT_WF_ID</th>
                    <th>LOT</th>
                    <th>y</th>
                    {FAIL_TARGETS.map((t) => (
                      <th key={t}>{t}</th>
                    ))}
                    <th title="실측 1.0 + 예측 시 핵심 인자 파레토 기여율 합산 x 20">신뢰도</th>
                    <th>주요 원인</th>
                    <th>사유</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.length === 0 ? (
                    <tr>
                      <td colSpan={10}>
                        <NoSearchResults onClear={() => setSearch("")} />
                      </td>
                    </tr>
                  ) : (
                    sorted.map((c) => (
                      <tr key={c.lot_wafer_id}>
                        <td className="data">{c.lot_wafer_id}</td>
                        <td>{c.lot_id ?? "-"}</td>
                        <td className="numCol">{c.y.toFixed(1)}%</td>
                        {FAIL_TARGETS.map((t) => (
                          <td key={t} className={cellClassName(c.cells[t])} title={cellTooltip(t, c.y_components[t], c.cells[t])}>
                            {c.y_components[t] != null ? `${c.y_components[t].toFixed(2)}%p` : "-"}
                          </td>
                        ))}
                        <td className={`alertReliabilityCell${c.reliability < LOW_RELIABILITY_THRESHOLD ? " low" : ""}`}>
                          {c.reliability}
                        </td>
                        <td>{c.primary_target}</td>
                        <td className="alertReasonCell">{c.reason}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </HScrollTableBody>
            <TableCaption total={data.candidates.length} shown={sorted.length} />
          </section>

          <section className="resultCard alertsSummaryCard">
            <p>
              상위 {data.candidates.length}건 평균 신뢰도 {data.summary.mean_reliability.toFixed(0)} · {LOW_RELIABILITY_THRESHOLD} 미만{" "}
              {data.summary.below_threshold_count}건 · 0점 {data.summary.zero_reliability_count}건
            </p>
            {/* RE-3: "이 경고를 빼지 마라" -- 신뢰도 0인 wafer가 상위 목록에
                들어오면 근거가 사실상 모델 평균에 가깝다는 사실을 밝힌다. */}
            {zeroReliabilityInList && (
              <p className="notifyFieldError">신뢰도 0 — 선정 인자가 하나도 계측되지 않아 모델 평균에 가까운 값입니다.</p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
