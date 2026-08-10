"use client";

import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import CurrentDatasetLabel from "@/components/CurrentDatasetLabel";
import { HScrollTableBody, TableCaption, TableToolbar, useTableSearchSort, type SortOption } from "@/components/DataTablePanel";
import DashboardShell from "@/components/DashboardShell";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import { LastRunNote, TrainingAnalysisDataNote } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import { getYieldPrediction } from "@/lib/api";
import type { AlertCellColor, YieldCandidate, YieldPredictionResponse, YieldReliabilityInfo } from "@/types/data";

// VA~VE: 수율 예측 -- 이 모델은 순위는 맞지만 값은 못 맞춘다(R² 0.12,
// 상위 20장 적중 95%). 그래서 이 화면은 "순위 도구"이지 예측값 표시
// 도구가 아니다 -- Y 열을 강조하지 않고, 미계측 웨이퍼는 순위에서
// 뺀다(AUC 0.509 -- 무작위와 같다).

const FAIL_TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
const DEFAULT_VISIBLE = 10;
const RANK_TOOLTIP = "상위 20장 안의 순서는 예측 오차 범위 내에서 바뀔 수 있습니다";

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

// VA-4: "Step18_R1 (82.5%)" -- 폴백으로 하위 인자를 쓰면 기여율이 낮게
// 표시되므로 사용자가 근거 강도를 즉시 안다. 인자가 없으면 "—"(무채색).
function coreFactorCell(candidate: YieldCandidate, target: string) {
  const cell = candidate.core_factors[target];
  if (!cell || !cell.feature || cell.contribution_pct == null) {
    return <span className="ypCoreFactorEmpty">—</span>;
  }
  const fallback = cell.rank_used != null && cell.rank_used > 1;
  return (
    <span title={fallback ? `${cell.rank_used}위 인자로 폴백됨 (1위 인자 미계측)` : undefined}>
      {cell.feature} <span className="ypCoreFactorPct">({cell.contribution_pct.toFixed(1)}%)</span>
    </span>
  );
}

// VC-1: 신뢰도 = (기여율 20% 이상 인자가 계측된 타깃 수) / 5.
function reliabilityClassName(count: number): string {
  if (count === 0) return "ypReliabilityCell zero";
  if (count === 1) return "ypReliabilityCell low";
  return "ypReliabilityCell";
}

// VC-2: 숫자만으로는 어느 인자가 빠졌는지 모른다 -- 계측/미계측 타깃과
// 그 인자명을 툴팁으로 보여준다.
function reliabilityTooltip(info: YieldReliabilityInfo): string {
  const lines = [`계측 ${info.count}/5`];
  if (info.measured.length) lines.push(`${info.measured.map((m) => `${m.target} ${m.feature}`).join(" · ")} 계측`);
  if (info.unmeasured.length) lines.push(`${info.unmeasured.map((m) => `${m.target} ${m.feature}`).join(" · ")} 미계측`);
  return lines.join("\n");
}

function downloadCsv(data: YieldPredictionResponse, rows: YieldCandidate[]) {
  const header = ["순위", "LOT_WF_ID", "LOT", ...FAIL_TARGETS.map((t) => `${t}핵심인자`), ...FAIL_TARGETS, "Y", "신뢰도", "권장사항"];
  const csvRows = rows.map((c, index) => [
    String(index + 1),
    c.lot_wafer_id,
    c.lot_id ?? "",
    ...FAIL_TARGETS.map((t) => {
      const cell = c.core_factors[t];
      return cell?.feature && cell.contribution_pct != null ? `${cell.feature} (${cell.contribution_pct.toFixed(1)}%)` : "";
    }),
    ...FAIL_TARGETS.map((t) => (c.y_components[t] != null ? c.y_components[t].toFixed(2) : "")),
    c.y.toFixed(2),
    `${c.reliability.count}/5`,
    c.recommendation.text.replace(/\n/g, " / "),
  ]);
  const csv = [header, ...csvRows].map((line) => line.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([`﻿${csv}`], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `yield_prediction_${data.eval_dataset_id}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

const SORT_OPTIONS: SortOption<YieldCandidate>[] = [
  { value: "y_asc", label: "Y 낮은 순", compare: (a, b) => a.y - b.y },
  { value: "lot_asc", label: "LOT_WF_ID 오름차순", compare: (a, b) => a.lot_wafer_id.localeCompare(b.lot_wafer_id) },
  { value: "lot_desc", label: "LOT_WF_ID 내림차순", compare: (a, b) => b.lot_wafer_id.localeCompare(a.lot_wafer_id) },
  { value: "reliability_desc", label: "신뢰도 순", compare: (a, b) => b.reliability.count - a.reliability.count },
  ...FAIL_TARGETS.map((target) => ({
    // VB-5: "Yn 높은 순"은 불량률이 높은 순(손실이 크다) -- Y(수율)와
    // 방향이 반대다.
    value: `${target.toLowerCase()}_desc`,
    label: `${target} 높은 순`,
    compare: (a: YieldCandidate, b: YieldCandidate) => (b.y_components[target] ?? 0) - (a.y_components[target] ?? 0),
  })),
];

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
  const [data, setData] = useState<YieldPredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [showUnmeasured, setShowUnmeasured] = useState(false);

  const trainDataset = snapshot?.source.train_dataset ?? alarms?.trainDataset ?? "train";
  const evalDataset = snapshot?.source.eval_dataset ?? alarms?.evalDataset ?? "test";

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getYieldPrediction(trainDataset, evalDataset)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((failure) => {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "수율 예측 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [trainDataset, evalDataset]);

  const { search, setSearch, sort, setSort, sorted } = useTableSearchSort<YieldCandidate>(
    data?.candidates ?? [],
    (item) => `${item.lot_wafer_id} ${item.lot_id ?? ""}`,
    SORT_OPTIONS,
    "y_asc",
    (a, b) => a.y - b.y,
  );

  const searching = search.trim().length > 0;
  // VB-2/VB-4: 기본은 상위 10, "전체 보기"로 확장. 검색 중에는 상위 10
  // 제한을 해제한다(찾는 웨이퍼가 100위여도 나와야 한다).
  const visible = searching || expanded ? sorted : sorted.slice(0, DEFAULT_VISIBLE);

  const fallback = data?.fallback_summary;
  const noneRatio = fallback && fallback.total_combinations > 0 ? (fallback.none_count / fallback.total_combinations) * 100 : null;

  return (
    <div className="rcPage">
      <div className="pageHeading">
        <h1>수율 예측</h1>
        <p className="sectionCaption">
          이 모델은 상위 20장 적중률 95%로 순위는 정확하지만, 예측값 자체의 오차는 큽니다(R² 0.12). 절대값이 아니라 검토 우선순위로
          활용하세요.
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
          <button type="button" className="button secondary" onClick={() => data && downloadCsv(data, sorted)} disabled={!data}>
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
              <p className="analysisErrorMessage">수율 예측 목록을 불러오지 못했습니다 — {error}</p>
            </div>
          </div>
        </section>
      ) : !data || data.candidates.length === 0 ? (
        <section className="resultCard">
          <p className="emptyMessage">판정 가능한 wafer가 없습니다.</p>
        </section>
      ) : (
        <>
          <section className="resultCard">
            <TableToolbar
              search={search}
              onSearchChange={setSearch}
              sort={sort}
              onSortChange={setSort}
              sortOptions={SORT_OPTIONS}
              placeholder="LOT_WF_ID 검색"
            />
            <HScrollTableBody rows={10} minWidth={1640}>
              <table className="dataTable ypTable">
                <thead>
                  <tr>
                    <th className="ypColRank">순위</th>
                    <th className="ypColWafer">LOT_WF_ID</th>
                    <th>LOT</th>
                    {FAIL_TARGETS.map((t) => (
                      <th key={`${t}-factor`} className="ypColCoreFactor">
                        {t}핵심인자
                      </th>
                    ))}
                    {FAIL_TARGETS.map((t) => (
                      <th key={t}>{t}</th>
                    ))}
                    <th title="합산 예측 수율 -- 절대값 정확도가 낮습니다(R² 0.12)">Y</th>
                    <th title="기여율 20% 이상 인자가 계측된 타깃 수 / 5">신뢰도</th>
                    <th className="ypColRecommendation">권장사항</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.length === 0 ? (
                    <tr>
                      <td colSpan={16}>
                        <div className="emptyMessage tableEmptySearch">
                          <span>일치하는 wafer가 없습니다 — &quot;{search}&quot;</span>
                          <button type="button" onClick={() => setSearch("")}>
                            검색어 지우기
                          </button>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    visible.map((c, index) => (
                      <tr key={c.lot_wafer_id}>
                        <td className="ypColRank data" title={RANK_TOOLTIP}>
                          {index + 1}
                        </td>
                        <td className="ypColWafer data">{c.lot_wafer_id}</td>
                        <td>{c.lot_id ?? "-"}</td>
                        {FAIL_TARGETS.map((t) => (
                          <td key={`${t}-factor`} className="ypColCoreFactor">
                            {coreFactorCell(c, t)}
                          </td>
                        ))}
                        {FAIL_TARGETS.map((t) => (
                          <td key={t} className={cellClassName(c.cells[t])} title={cellTooltip(t, c.y_components[t], c.cells[t])}>
                            {c.y_components[t] != null ? `${c.y_components[t].toFixed(2)}%p` : "-"}
                          </td>
                        ))}
                        {/* VB-3: Y(합산값)에는 색을 쓰지 않는다. */}
                        <td className="numCol">{c.y.toFixed(2)}%</td>
                        <td className={reliabilityClassName(c.reliability.count)} title={reliabilityTooltip(c.reliability)}>
                          {c.reliability.count}/5
                        </td>
                        <td className="ypColRecommendation">
                          {c.recommendation.text ? (
                            <span className="ypRecommendationText" title={c.recommendation.text}>
                              {c.recommendation.text}
                            </span>
                          ) : (
                            <span className="ypCoreFactorEmpty">—</span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </HScrollTableBody>
            <div className="ypTableFooter">
              <TableCaption total={sorted.length} shown={visible.length} />
              {!searching && sorted.length > DEFAULT_VISIBLE && (
                <button type="button" className="referenceOnlyToggle" onClick={() => setExpanded((prev) => !prev)}>
                  {expanded ? "접기" : `전체 ${sorted.length}건 보기`}
                </button>
              )}
            </div>
            {fallback && (
              <p className="tableCaption">
                핵심 인자 폴백 -- 1위 계측 {fallback.rank_counts["1"] ?? 0}건 · 전체 미계측 {fallback.none_count}건
                {noneRatio != null ? ` (${noneRatio.toFixed(0)}%)` : ""} · 웨이퍼 x 타깃 {fallback.total_combinations}조합 기준
              </p>
            )}
          </section>

          {data.unmeasured_count > 0 && (
            <section className="resultCard">
              <p>
                미계측 웨이퍼 {data.unmeasured_count}건 -- 핵심 인자와 실측값이 모두 없어 판정 근거가 없는 wafer입니다(AUC 0.509, 무작위와
                같음). 판정 목록에서 제외됩니다.
              </p>
              <button type="button" className="referenceOnlyToggle" onClick={() => setShowUnmeasured((prev) => !prev)}>
                {showUnmeasured ? "접기" : "목록 보기"}
              </button>
              {showUnmeasured && <p className="ypUnmeasuredList">{data.unmeasured_wafer_ids.join(", ")}</p>}
            </section>
          )}
        </>
      )}
    </div>
  );
}
