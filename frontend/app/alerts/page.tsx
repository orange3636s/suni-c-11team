"use client";

import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { HScrollTableBody, TableCaption, TableToolbar, useTableSearchSort, type SortOption } from "@/components/DataTablePanel";
import DashboardShell from "@/components/DashboardShell";
import { PageHeaderMeta } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import { dispatchYieldUpdateNotification, getNotificationSettings, getYieldPrediction } from "@/lib/api";
import { DISPLAY_CONTRIBUTION_THRESHOLD_PCT } from "@/lib/chartSelection";
import type {
  AlertCellColor,
  DispatchResponse,
  NotificationSettingsSummary,
  YieldCandidate,
  YieldPredictionResponse,
  YieldReliabilityInfo,
} from "@/types/data";

// 수율 예측 -- 이 모델은 순위는 맞지만 값은 못 맞춘다(R² 0.12,
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
  // Y1~Y5는 비율(불량률) 값이지 편차가 아니므로 %가 맞다 --
  // %p는 편차·기대효과·감소량 같은 "차이" 값에만 쓴다.
  if (cell.shade === "measured") return `${target}  실측 ${valuePct != null ? valuePct.toFixed(2) : "-"}%`;
  const lines = [`${target}  예측 ${valuePct != null ? valuePct.toFixed(2) : "-"}%`];
  if (cell.feature) {
    lines.push(`인자 ${cell.feature}${cell.factor_value != null ? ` = ${cell.factor_value.toFixed(1)} (계측)` : " (미계측)"}`);
  }
  if (cell.contribution_pct != null) {
    const dir = directionText(cell);
    lines.push(`파레토 기여율 ${cell.contribution_pct.toFixed(1)}%${dir ? ` · ${dir}` : ""}`);
  }
  return lines.join("\n");
}

// 기여율 구간별 색 강도 -- DISPLAY_CONTRIBUTION_THRESHOLD_PCT(10%)가 색
// 유무의 경계와 일치한다(신뢰도 카운트 기준과 동일한 상수를 공유).
function coreFactorTierClass(pct: number): string {
  if (pct >= 60) return "ypCoreFactorTierStrong";
  if (pct >= 20) return "ypCoreFactorTierMedium";
  if (pct >= DISPLAY_CONTRIBUTION_THRESHOLD_PCT) return "ypCoreFactorTierLight";
  return "ypCoreFactorTierMuted";
}

// "Step18_R1 (82.5%)" -- 폴백으로 하위 인자를 쓰면 기여율이 낮게
// 표시되므로 사용자가 근거 강도를 즉시 안다. 후보 인자 자체가 없으면
// "—"(무채색). 1~5위가 전부 미계측이면 1위 인자를 회색 + "미계측"으로
// 보여준다 -- 빈칸 대신 "이 인자를 계측하면 예측이 정확해진다"는 조치
// 가능한 정보를 준다.
function coreFactorCell(candidate: YieldCandidate, target: string) {
  const cell = candidate.core_factors[target];
  if (!cell || !cell.feature || cell.contribution_pct == null) {
    return <span className="ypCoreFactorEmpty">—</span>;
  }
  if (!cell.measured) {
    return (
      <span className="ypCoreFactorUnmeasured" title="계측되지 않아 예측에 사용하지 않았습니다. 이 인자를 계측하면 정확도가 올라갑니다.">
        {cell.feature} <span className="ypCoreFactorPct">({cell.contribution_pct.toFixed(1)}%)</span> 미계측
      </span>
    );
  }
  const fallback = cell.rank_used != null && cell.rank_used > 1;
  return (
    <span
      className={coreFactorTierClass(cell.contribution_pct)}
      title={fallback ? `${cell.rank_used}위 인자로 폴백됨 (1위 인자 미계측)` : undefined}
    >
      {cell.feature} <span className="ypCoreFactorPct">({cell.contribution_pct.toFixed(1)}%)</span>
    </span>
  );
}

// 신뢰도 = (기여율 10% 이상 인자가 계측된 타깃 수) / 5.
function reliabilityClassName(count: number): string {
  if (count === 0) return "ypReliabilityCell zero";
  if (count === 1) return "ypReliabilityCell low";
  return "ypReliabilityCell";
}

// 숫자만으로는 어느 인자가 빠졌는지 모른다 -- 계측/미계측 타깃과
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
      if (!cell?.feature || cell.contribution_pct == null) return "";
      const suffix = cell.measured ? "" : " 미계측";
      return `${cell.feature} (${cell.contribution_pct.toFixed(1)}%)${suffix}`;
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
    // "Yn 높은 순"은 불량률이 높은 순(손실이 크다) -- Y(수율)와
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
  const { snapshot, alarms } = useAnalysisState();
  const { setAnalysisPanelOpen } = usePanelState();
  const [data, setData] = useState<YieldPredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [showUnmeasured, setShowUnmeasured] = useState(false);
  const [notifyDialogOpen, setNotifyDialogOpen] = useState(false);
  const [notifySettings, setNotifySettings] = useState<NotificationSettingsSummary | null>(null);
  const [notifySending, setNotifySending] = useState(false);
  const [notifyResult, setNotifyResult] = useState<DispatchResponse | null>(null);
  const [notifyError, setNotifyError] = useState("");

  const trainDataset = snapshot?.source.train_dataset ?? alarms?.trainDataset ?? "train";
  const evalDataset = snapshot?.source.eval_dataset ?? alarms?.evalDataset ?? "test";

  // 다이얼로그를 열 때마다 연결 상태를 새로 읽는다 -- 설정 패널에서
  // 방금 채널을 끊었을 수도 있으니 캐시된 값을 재사용하지 않는다.
  function openNotifyDialog() {
    setNotifyDialogOpen(true);
    setNotifyResult(null);
    setNotifyError("");
    setNotifySettings(null);
    getNotificationSettings()
      .then(setNotifySettings)
      .catch(() => setNotifyError("채널 연결 상태를 불러오지 못했습니다."));
  }

  function closeNotifyDialog() {
    setNotifyDialogOpen(false);
  }

  function confirmNotifySend() {
    setNotifySending(true);
    setNotifyError("");
    dispatchYieldUpdateNotification(trainDataset, evalDataset)
      .then(setNotifyResult)
      .catch((failure) => setNotifyError(failure instanceof Error ? failure.message : "발송 요청에 실패했습니다."))
      .finally(() => setNotifySending(false));
  }

  const notifyChannelsConnected = Boolean(
    notifySettings && (notifySettings.slack.connected || notifySettings.telegram.connected || notifySettings.gmail.connected),
  );

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
  // 기본은 상위 10, "전체 보기"로 확장. 검색 중에는 상위 10
  // 제한을 해제한다(찾는 웨이퍼가 100위여도 나와야 한다).
  const visible = searching || expanded ? sorted : sorted.slice(0, DEFAULT_VISIBLE);

  const fallback = data?.fallback_summary;
  const noneRatio = fallback && fallback.total_combinations > 0 ? (fallback.none_count / fallback.total_combinations) * 100 : null;
  const rank1Count = fallback?.rank_counts["1"] ?? 0;
  const fallbackRankCount = fallback
    ? Object.entries(fallback.rank_counts)
        .filter(([rank]) => rank !== "1")
        .reduce((sum, [, count]) => sum + count, 0)
    : 0;

  return (
    <div className="rcPage">
      <div className="pageHeading">
        <h1>수율 예측</h1>
        <p className="sectionCaption">
          이 모델은 상위 20장 적중률 95%로 순위는 정확하지만, 예측값 자체의 오차는 큽니다(R² 0.12). 절대값이 아니라 검토 우선순위로
          활용하세요.
        </p>
        <PageHeaderMeta />
      </div>

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
          <p className="emptyMessage">
            분석 결과가 없습니다. 모델 분석에서 분석을 시작하세요.{" "}
            <button type="button" className="button secondary sm" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
          </p>
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
              extra={
                <>
                  <button type="button" className="button secondary" onClick={() => data && downloadCsv(data, sorted)} disabled={!data}>
                    CSV 내보내기
                  </button>
                  <button type="button" className="button secondary" onClick={openNotifyDialog} disabled={!data}>
                    알림 전송
                  </button>
                </>
              }
            />
            <HScrollTableBody rows={10} minWidth={1900}>
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
                    <th title="기여율 10% 이상 인자가 계측된 타깃 수 / 5">신뢰도</th>
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
                            {c.y_components[t] != null ? `${c.y_components[t].toFixed(2)}%` : "-"}
                          </td>
                        ))}
                        {/* Y(합산값)에는 색을 쓰지 않는다. */}
                        <td className="numCol">{c.y.toFixed(2)}%</td>
                        <td className={reliabilityClassName(c.reliability.count)} title={reliabilityTooltip(c.reliability)}>
                          {c.reliability.count}/5
                        </td>
                        <td className="ypColRecommendation colNoTruncate">
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
                {fallback.total_combinations.toLocaleString()}개 조합 ({data?.total_wafers.toLocaleString() ?? "-"} wafer x{" "}
                {FAIL_TARGETS.length} 모드) 중 1위 인자 계측 {rank1Count.toLocaleString()}건 · 2~5위 폴백 {fallbackRankCount.toLocaleString()}건 ·
                전부 미계측 {fallback.none_count.toLocaleString()}건{noneRatio != null ? ` (${noneRatio.toFixed(0)}%)` : ""}
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

      {notifyDialogOpen && (
        <div className="notifyDialogBackdrop" onClick={closeNotifyDialog}>
          <div className="notifyDialogCard" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="notifyDialogHeader">
              <h2>알림 전송</h2>
              <button type="button" className="compareModalClose" onClick={closeNotifyDialog} aria-label="닫기">
                ×
              </button>
            </div>
            {notifyResult ? (
              notifyResult.skipped ? (
                <p className="analysisErrorMessage">발송하지 않았습니다 — {notifyResult.reason}</p>
              ) : (
                <div className="notifyDialogResult">
                  <p>발송 완료 ({notifyResult.sent_count ?? 0}건)</p>
                  <ul className="notifyDialogResultList">
                    {Object.entries(notifyResult.results ?? {}).map(([channel, item]) => (
                      <li key={channel}>
                        {channel}: {item.ok ? "성공" : `실패 — ${item.error ?? "알 수 없는 오류"}`}
                      </li>
                    ))}
                  </ul>
                </div>
              )
            ) : !notifySettings ? (
              <p className="emptyMessage">{notifyError || "채널 연결 상태를 불러오는 중…"}</p>
            ) : (
              <>
                <p className="notifyDialogChannels">
                  연결된 채널:{" "}
                  {[
                    notifySettings.slack.connected ? "Slack" : null,
                    notifySettings.telegram.connected ? "Telegram" : null,
                    notifySettings.gmail.connected ? "Gmail" : null,
                  ]
                    .filter(Boolean)
                    .join(", ") || "없음"}
                </p>
                <p className="notifyDialogChannels">내용: 수율 하위 10건 + 타깃별 상위 3건</p>
                <p className="notifyDialogChannels">대상: {evalDataset}</p>
                {!notifyChannelsConnected && <p className="analysisErrorMessage">연결된 채널이 없어 보낼 수 없습니다.</p>}
                {notifyError && <p className="analysisErrorMessage">{notifyError}</p>}
                <div className="notifyDialogActions">
                  <button type="button" className="button secondary" onClick={closeNotifyDialog}>
                    취소
                  </button>
                  <button
                    type="button"
                    className="button primary"
                    onClick={confirmNotifySend}
                    disabled={notifySending || !notifyChannelsConnected}
                  >
                    {notifySending ? "전송 중…" : "전송"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
