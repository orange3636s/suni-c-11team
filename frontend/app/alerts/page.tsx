"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import {
  HScrollTableBody,
  NoSearchResults,
  ScrollTableBody,
  TableCaption,
  TableToolbar,
  useTableSearchSort,
  type SortOption,
} from "@/components/DataTablePanel";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import { usePanelState } from "@/components/PanelStateProvider";
import { niceTicks } from "@/lib/niceTicks";
import { getAlarmSummary, getAlarms, getRecommendations, saveAlarmsState } from "@/lib/api";
import type {
  AlarmItem,
  AlarmSummaryResponse,
  FactorBand,
  MeasurementBiasSummary,
  RecommendationItem,
} from "@/types/data";

const SEVERITY_LABEL: Record<string, string> = { low: "낮음", medium: "중간", high: "높음" };
const RANGE_UNBOUNDED_LO = "-∞";
const RANGE_UNBOUNDED_HI = "+∞";

function alarmExplainMessage(item: AlarmItem): string {
  const [lo, hi] = item.normal_range;
  const rangeText = `${lo != null ? lo.toFixed(1) : RANGE_UNBOUNDED_LO}~${hi != null ? hi.toFixed(1) : RANGE_UNBOUNDED_HI}`;
  return (
    `알람: ${item.lot_wafer_id} · ${item.feature} = ${item.value.toFixed(1)} (관리한계 ${rangeText}) · ${item.target}\n` +
    "이 알람에 대해 설명해 주세요."
  );
}

function recommendationExplainMessage(item: RecommendationItem): string {
  const [lo, hi] = item.recommended_range;
  const rangeText = `${lo.toFixed(1)}~${hi.toFixed(1)}`;
  return (
    `개선 권장: ${item.lot_wafer_id} · ${item.feature} = ${item.value.toFixed(1)} (권장 구간 ${rangeText}) · ${item.target}\n` +
    "이 항목에 대해 설명해 주세요."
  );
}
const SEVERITY_RANK: Record<string, number> = { high: 3, medium: 2, low: 1 };
const TAG_LABEL: Record<string, string> = { priority: "우선 권장", recommended: "권장", reference: "참고" };
const TAG_RANK: Record<string, number> = { priority: 3, recommended: 2, reference: 1 };
const DIRECTION_LABEL: Record<string, string> = { down: "↓ 낮추기", up: "↑ 높이기" };

function SeverityBadge({ severity }: { severity: string }) {
  return <span className={`severityBadge severityBadge-${severity}`}>{SEVERITY_LABEL[severity] ?? severity}</span>;
}

function RecommendationTagBadge({ tag }: { tag: string }) {
  return <span className={`recommendationTag tag-${tag}`}>{TAG_LABEL[tag] ?? tag}</span>;
}

function lotCompare(a: string | null, b: string | null): number {
  return (a ?? "").localeCompare(b ?? "", undefined, { numeric: true });
}

function alarmWaferSlot(item: AlarmItem): number {
  return item.wafer_slot ?? 0;
}

function recommendationWaferSlot(item: RecommendationItem): number {
  if (!item.lot_id || !item.lot_wafer_id.startsWith(item.lot_id)) return 0;
  const match = /(\d+)/.exec(item.lot_wafer_id.slice(item.lot_id.length));
  return match ? Number(match[1]) : 0;
}

function alarmTieBreak(a: AlarmItem, b: AlarmItem): number {
  return lotCompare(a.lot_id, b.lot_id) || alarmWaferSlot(a) - alarmWaferSlot(b);
}

function recommendationTieBreak(a: RecommendationItem, b: RecommendationItem): number {
  return lotCompare(a.lot_id, b.lot_id) || recommendationWaferSlot(a) - recommendationWaferSlot(b);
}

const ALARM_SORT_OPTIONS: SortOption<AlarmItem>[] = [
  { value: "severity", label: "심각성", compare: (a, b) => (SEVERITY_RANK[b.severity] ?? 0) - (SEVERITY_RANK[a.severity] ?? 0) },
  { value: "lot_asc", label: "LOT 오름차순", compare: (a, b) => lotCompare(a.lot_id, b.lot_id) },
  { value: "lot_desc", label: "LOT 내림차순", compare: (a, b) => lotCompare(b.lot_id, a.lot_id) },
  { value: "target_asc", label: "타깃 오름차순", compare: (a, b) => a.target.localeCompare(b.target, undefined, { numeric: true }) },
  { value: "target_desc", label: "타깃 내림차순", compare: (a, b) => b.target.localeCompare(a.target, undefined, { numeric: true }) },
  { value: "step_asc", label: "Step 오름차순", compare: (a, b) => a.step - b.step },
  { value: "step_desc", label: "Step 내림차순", compare: (a, b) => b.step - a.step },
];

const RECOMMENDATION_SORT_OPTIONS: SortOption<RecommendationItem>[] = [
  {
    value: "tag",
    label: "태그",
    compare: (a, b) => {
      const rank = (TAG_RANK[b.tag] ?? 0) - (TAG_RANK[a.tag] ?? 0);
      if (rank !== 0) return rank;
      const am = a.expected_improvement_pct != null ? Math.abs(a.expected_improvement_pct) : -1;
      const bm = b.expected_improvement_pct != null ? Math.abs(b.expected_improvement_pct) : -1;
      return bm - am;
    },
  },
  { value: "lot_asc", label: "LOT 오름차순", compare: (a, b) => lotCompare(a.lot_id, b.lot_id) },
  { value: "lot_desc", label: "LOT 내림차순", compare: (a, b) => lotCompare(b.lot_id, a.lot_id) },
  { value: "target_asc", label: "타깃 오름차순", compare: (a, b) => a.target.localeCompare(b.target, undefined, { numeric: true }) },
  { value: "target_desc", label: "타깃 내림차순", compare: (a, b) => b.target.localeCompare(a.target, undefined, { numeric: true }) },
  { value: "step_asc", label: "Step 오름차순", compare: (a, b) => a.step - b.step },
  { value: "step_desc", label: "Step 내림차순", compare: (a, b) => b.step - a.step },
];

export default function AlertsPage() {
  const { analysisDataset, requestChat } = usePanelState();
  // 사전 알람 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- summary/
  // alarms/recommendations live in the shared context now, so tab
  // switching renders instantly with no refetch, and a reload restores
  // the last-fetched result from the server (spec §3-1: 산점도 좌표가
  // 없는 다른 두 결과와 달리, 이 결과는 애초에 좌표를 담지 않으므로
  // 복원이 항상 완전하다 -- 별도의 배경 재요청이 필요 없다).
  const { alarms: alarmsState, setAlarms: setAlarmsState, hydrated } = useAnalysisState();
  const [trainDataset, setTrainDataset] = useState("train");
  const [evalDataset, setEvalDataset] = useState("test");
  const [severityFilter, setSeverityFilter] = useState("");
  const [showReferenceTag, setShowReferenceTag] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [factorBandIndex, setFactorBandIndex] = useState(0);

  const summary = alarmsState?.summary ?? null;
  const alarms = alarmsState?.alarms ?? null;
  const recommendations = alarmsState?.recommendations ?? null;
  // 셀렉터(train/eval) 중 하나라도 표시 중인 결과와 다르면 경고 (spec
  // §4-3/§5-3) -- 자동으로 결과를 지우거나 다시 불러오지 않는다.
  const datasetMismatch = Boolean(
    alarmsState && (alarmsState.trainDataset !== trainDataset || alarmsState.evalDataset !== evalDataset),
  );

  // 심각성 필터는 이미 불러온 alarms.items를 클라이언트에서 거를 뿐, 서버를
  // 다시 부르지 않는다 -- 필터를 바꿔도 "탭 전환 시 API 재호출 금지"와
  // 같은 원칙(불필요한 네트워크 요청 최소화)을 지키면서 즉시 반응한다.
  const severityFilteredAlarmItems = useMemo(() => {
    const items = alarms?.items ?? [];
    return severityFilter ? items.filter((item) => item.severity === severityFilter) : items;
  }, [alarms, severityFilter]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [summaryResponse, alarmsResponse, recommendationsResponse] = await Promise.all([
        getAlarmSummary(trainDataset, evalDataset),
        getAlarms(trainDataset, evalDataset),
        getRecommendations(trainDataset, evalDataset),
      ]);
      setAlarmsState({
        trainDataset,
        evalDataset,
        createdAt: new Date().toISOString(),
        summary: summaryResponse,
        alarms: alarmsResponse,
        recommendations: recommendationsResponse,
      });
      setFactorBandIndex(0);
      // 조회 성공 직후 저장 (spec §3-4) -- 실패해도 방금 불러온 결과는
      // 이미 화면에 반영되어 있다 (spec §3-2).
      void saveAlarmsState(trainDataset, evalDataset, {
        summary: summaryResponse,
        alarms: alarmsResponse,
        recommendations: recommendationsResponse,
      }).catch(() => {});
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "알람 로그를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [trainDataset, evalDataset, setAlarmsState]);

  // 재접속/새로고침, 그리고 탭을 옮겼다 돌아온 경우 모두 이 마운트
  // 이펙트가 처리한다 (spec §4-2/§4-3) -- 셀렉터를 컨텍스트의 결과에 맞춰
  // 한 번만 동기화한다.
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current) return;
    syncedFromRestore.current = true;
    if (!alarmsState) return;
    const timer = window.setTimeout(() => {
      setTrainDataset(alarmsState.trainDataset);
      setEvalDataset(alarmsState.evalDataset);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [hydrated, alarmsState]);

  // Fallback only: nothing restored/cached yet -- load once with
  // whatever the default train/eval selection is, matching this page's
  // original always-auto-load behavior for a genuinely first visit.
  // Never refires just because a selector changes afterward (spec §5-3:
  // "결과를 자동으로 지우지 마라" / "사용자가 실행 버튼을 눌러야 갱신된다"),
  // and never on a tab revisit (checklist §탭 이동 #4).
  const autoLoaded = useRef(false);
  useEffect(() => {
    if (!hydrated || autoLoaded.current) return;
    autoLoaded.current = true;
    if (alarmsState) return;
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, alarmsState]);

  const alarmTable = useTableSearchSort(
    severityFilteredAlarmItems,
    (item) => `${item.lot_wafer_id} ${item.feature} ${item.target}`,
    ALARM_SORT_OPTIONS,
    "severity",
    alarmTieBreak,
  );

  const visibleRecommendationItems = useMemo(() => {
    const items = recommendations?.items ?? [];
    return showReferenceTag ? items : items.filter((item) => item.tag !== "reference");
  }, [recommendations, showReferenceTag]);

  const recommendationTable = useTableSearchSort(
    visibleRecommendationItems,
    (item) => `${item.lot_wafer_id} ${item.feature} ${item.target}`,
    RECOMMENDATION_SORT_OPTIONS,
    "tag",
    recommendationTieBreak,
  );

  const factorBands = summary?.factor_bands ?? [];
  const activeFactorBand = factorBands[factorBandIndex] ?? factorBands[0] ?? null;

  return (
    <DashboardShell activeItem="사전 알람 로그">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">PRE-ALERT LOG</span>
        <h1>사전 알람 로그</h1>
        <p>학습 데이터셋에서 산출한 정상범위를 평가 데이터셋에 적용해 이탈 여부를 판정합니다.</p>
        <LastRunNote createdAt={alarmsState?.createdAt} />
      </section>

      <section className="uploadCard">
        <div className="rcControlBar alarmControlBar">
          <DatasetSelector label="정상범위 산출 (train)" value={trainDataset} onChange={setTrainDataset} />
          <DatasetSelector label="판정 대상 (eval)" value={evalDataset} onChange={setEvalDataset} />
          <div className="fieldGroup">
            <span>심각성</span>
            <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
              <option value="">전체</option>
              <option value="low">낮음</option>
              <option value="medium">중간</option>
              <option value="high">높음</option>
            </select>
          </div>
          <button type="button" className="button primary" disabled={loading} onClick={() => void load()} style={{ alignSelf: "end" }}>
            {loading ? "조회 중…" : summary ? "다시 조회" : "조회"}
          </button>
        </div>
        <DatasetMismatchWarning mismatch={datasetMismatch} />
        {error && <p className="errorMessage">{error}</p>}
      </section>

      {summary ? (
        <>
          <section className="alarmSummaryGrid">
            <AlarmSummaryCard
              label="알람 wafer"
              value={`${summary.counts.alarm}장`}
              aux={pct(summary.counts.alarm, summary.total_wafers)}
              tone="highlight"
              title="관리한계(LCL/UCL) 이탈"
            />
            <AlarmSummaryCard
              label="개선 권장 wafer"
              value={`${summary.counts.out_of_recommended}장`}
              aux={pct(summary.counts.out_of_recommended, summary.total_wafers)}
              tone="neutral"
              title="권장구간 밖 (관리한계 내)"
            />
            <AlarmSummaryCard
              label="정상 wafer"
              value={`${summary.counts.in_recommended}장`}
              aux={pct(summary.counts.in_recommended, summary.total_wafers)}
              tone="good"
              title="권장구간 내"
            />
            <AlarmSummaryCard
              label="판정불가 (미계측) wafer"
              value={`${summary.counts.unmeasured}장`}
              aux={pct(summary.counts.unmeasured, summary.total_wafers)}
              tone="faint"
              title="선정 인자가 하나도 계측되지 않아 판정할 수 없는 wafer"
            />
          </section>
          <p className="alarmSummaryCaption">전체 {summary.total_wafers.toLocaleString()}장 기준</p>

          <ConceptYieldBandCard summary={summary} />

          {activeFactorBand && (
            <FactorYieldBandCard
              bands={factorBands}
              activeIndex={factorBands.indexOf(activeFactorBand)}
              onChange={setFactorBandIndex}
            />
          )}

          <UnmeasuredCard summary={summary} />
        </>
      ) : (
        <section className="alarmSummaryGrid">
          {Array.from({ length: 4 }).map((_, index) => (
            <div className="alarmSummaryCard skeleton" key={index}>
              <div className="alarmSkeletonLine label" />
              <div className="alarmSkeletonLine value" />
            </div>
          ))}
        </section>
      )}
      {!summary && <p className="alarmSummaryNote">원인 분석을 실행하면 집계됩니다</p>}

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">ALARMS</span>
            <h2>알람 목록 ({severityFilteredAlarmItems.length}건)</h2>
          </div>
          {alarms && severityFilteredAlarmItems.length > 0 && (
            <TableToolbar
              search={alarmTable.search}
              onSearchChange={alarmTable.setSearch}
              sort={alarmTable.sort}
              onSortChange={alarmTable.setSort}
              sortOptions={ALARM_SORT_OPTIONS}
              placeholder="Wafer ID · 인자 · 장비 검색"
            />
          )}
        </div>
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {!loading && alarms && severityFilteredAlarmItems.length === 0 && (
          <p className="emptyMessage">조건에 맞는 알람이 없습니다.</p>
        )}
        {!loading && alarms && severityFilteredAlarmItems.length > 0 && alarmTable.sorted.length === 0 && (
          <NoSearchResults onClear={() => alarmTable.setSearch("")} />
        )}
        {!loading && alarms && alarmTable.sorted.length > 0 && (
          <>
            <div className="tableHideOnMobile">
              <HScrollTableBody minWidth={980}>
                <table>
                  <thead>
                    <tr>
                      <th className="col-wafer colNoTruncate">Wafer</th>
                      <th>LOT</th>
                      <th>Step</th>
                      <th>인자</th>
                      <th>타깃</th>
                      <th className="numCol col-value colNoTruncate">값</th>
                      <th className="col-range colNoTruncate">정상범위</th>
                      <th className="numCol col-deviation colNoTruncate">이탈량</th>
                      <th>방향</th>
                      <th className="col-severity colNoTruncate">심각성</th>
                      <th className="numCol" title="해당 행의 타깃(target) 실제 불량률">실측값</th>
                      <th aria-label="해설" />
                    </tr>
                  </thead>
                  <tbody>
                    {alarmTable.sorted.map((item, index) => (
                      <AlarmRow
                        key={`${item.lot_wafer_id}-${item.feature}-${index}`}
                        item={item}
                        onExplain={() => requestChat(alarmExplainMessage(item), "chat")}
                        explainDisabled={!analysisDataset}
                      />
                    ))}
                  </tbody>
                </table>
              </HScrollTableBody>
            </div>
            {/* ≤767px: 카드형 전환 (spec §B-6) -- 같은 alarmTable.sorted를
                공유하므로 데이터/정렬/검색 상태가 둘 사이에서 갈라질 일이 없다. */}
            <div className="alarmCardList">
              {alarmTable.sorted.map((item, index) => (
                <AlarmCard
                  key={`card-${item.lot_wafer_id}-${item.feature}-${index}`}
                  item={item}
                  onExplain={() => requestChat(alarmExplainMessage(item), "chat")}
                  explainDisabled={!analysisDataset}
                />
              ))}
            </div>
            <TableCaption total={alarmTable.sorted.length} shown={Math.min(10, alarmTable.sorted.length)} />
          </>
        )}
        <p className="tableDisclaimer">
          알람은 인자 값이 관리한계(LCL/UCL)를 벗어난 wafer입니다. 관리한계는 학습 데이터의
          인자 분포에서 산출한 값으로, 해당 wafer가 평소와 다른 조건에서 처리되었음을 뜻합니다.
          불량의 원인으로 확정된 것은 아니며, 우선 확인 대상을 좁히는 용도입니다.
        </p>
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">LOT</span>
            <h2>LOT별 알람 집계 (상위)</h2>
          </div>
        </div>
        {summary && summary.top_lots.length > 0 ? (
          <>
            <ScrollTableBody rows={5}>
              <table className="lotSummaryTable">
                <thead><tr><th style={{ width: "70%" }}>LOT</th><th className="numCol" style={{ width: "30%" }}>알람 건수</th></tr></thead>
                <tbody>
                  {summary.top_lots.map((lot) => (
                    <tr key={lot.lot_id}><td>{lot.lot_id}</td><td className="numCol">{lot.alarm_count}</td></tr>
                  ))}
                </tbody>
              </table>
            </ScrollTableBody>
            <TableCaption
              total={summary.top_lots.length}
              shown={Math.min(5, summary.top_lots.length)}
              totalUnit="개 LOT"
              shownUnit="개"
            />
          </>
        ) : (
          <p className="emptyMessage">알람이 발생한 LOT이 없습니다.</p>
        )}
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">RECOMMENDATIONS</span>
            <h2>개선 권장 목록 ({recommendations?.total ?? 0}건)</h2>
          </div>
          {recommendations && recommendations.items.length > 0 && (
            <TableToolbar
              search={recommendationTable.search}
              onSearchChange={recommendationTable.setSearch}
              sort={recommendationTable.sort}
              onSortChange={recommendationTable.setSort}
              sortOptions={RECOMMENDATION_SORT_OPTIONS}
              placeholder="Wafer ID · 인자 · 장비 검색"
              extra={
                <label className="recommendationToggle">
                  <input
                    type="checkbox"
                    checked={showReferenceTag}
                    onChange={(event) => setShowReferenceTag(event.target.checked)}
                  />
                  참고 항목 포함
                </label>
              }
            />
          )}
        </div>
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {!loading && recommendations && recommendations.items.length === 0 && (
          <p className="emptyMessage">권장할 항목이 없습니다.</p>
        )}
        {!loading && recommendations && recommendations.items.length > 0 && recommendationTable.sorted.length === 0 && (
          <NoSearchResults onClear={() => recommendationTable.setSearch("")} />
        )}
        {!loading && recommendations && recommendationTable.sorted.length > 0 && (
          <>
            <div className="tableHideOnMobile">
              <HScrollTableBody minWidth={960}>
                <table>
                  <thead>
                    <tr>
                      <th className="col-wafer colNoTruncate">Wafer</th>
                      <th>LOT</th>
                      <th>Step</th>
                      <th>인자</th>
                      <th>타깃</th>
                      <th className="numCol col-value colNoTruncate">현재값</th>
                      <th className="col-range colNoTruncate">권장 구간</th>
                      <th>이동 방향</th>
                      <th className="numCol">기대 개선</th>
                      <th>태그</th>
                      <th aria-label="해설" />
                    </tr>
                  </thead>
                  <tbody>
                    {recommendationTable.sorted.map((item, index) => (
                      <RecommendationRow
                        key={`${item.lot_wafer_id}-${item.feature}-${index}`}
                        item={item}
                        onExplain={() => requestChat(recommendationExplainMessage(item), "chat")}
                        explainDisabled={!analysisDataset}
                      />
                    ))}
                  </tbody>
                </table>
              </HScrollTableBody>
            </div>
            <div className="alarmCardList">
              {recommendationTable.sorted.map((item, index) => (
                <RecommendationCard
                  key={`card-${item.lot_wafer_id}-${item.feature}-${index}`}
                  item={item}
                  onExplain={() => requestChat(recommendationExplainMessage(item), "chat")}
                  explainDisabled={!analysisDataset}
                />
              ))}
            </div>
            <TableCaption total={recommendationTable.sorted.length} shown={Math.min(10, recommendationTable.sorted.length)} />
          </>
        )}
        <p className="tableDisclaimer">
          알람은 관리한계(LCL/UCL) 이탈을 나타내는 이상 탐지이고, 개선 권장은 권장 구간 이탈을 나타내는 개선 제안입니다. 이미 알람으로 잡힌 wafer는 같은 인자에 대해 중복 집계하지 않습니다.
        </p>
      </section>

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>정상범위는 학습 데이터셋에서 해당 인자 자신의 분포로 산출한 IQR×1.5 관리한계(LCL~UCL)이며, 타깃(Y) 값과는 무관하게 계산됩니다. 인과관계가 아닌 통계적 이탈 판정입니다.</li>
          <li>판정불가 wafer는 선정 인자가 계측되지 않아 판정 자체가 불가능한 것이며, 정상을 의미하지 않습니다.</li>
          <li>개선 권장 구간은 관리한계(LCL/UCL) 안쪽으로 clamp되며, clamp 결과 구간이 사라지면 해당 인자는 권장 목록에서 제외됩니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}

function pct(count: number, total: number): string {
  if (total <= 0) return "0.0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

function AlarmSummaryCard({
  label,
  value,
  aux,
  tone = "default",
  title,
}: {
  label: string;
  value: string;
  aux?: string;
  tone?: "default" | "highlight" | "neutral" | "good" | "faint";
  title?: string;
}) {
  return (
    <div className={`alarmSummaryCard ${tone !== "default" ? `tone-${tone}` : ""}`} title={title}>
      <span className="alarmSummaryLabel" title={label}>{label}</span>
      <div className="alarmSummaryValueRow">
        <strong className="alarmSummaryValue">{value}</strong>
        {aux && <span className="alarmSummaryAux">{aux}</span>}
      </div>
    </div>
  );
}

/* ===================================================================
   카드①②: 구간별 수율/불량률 밴드. 공통 세그먼트+세로선+수치 레이아웃을
   두 카드가 공유하고, 개념도(균등 폭)와 실제 축(값 기준 폭)만 갈린다.
   =================================================================== */

type BandTone = "out" | "outrec" | "inrec";
type BandSegment = { tone: BandTone; label: string; widthPct: number };
type BandLine = { tone: "control" | "recommended"; pct: number };
type BandTick = { pct: number; label: string };

const TONE_LABEL: Record<BandTone, string> = { out: "이탈", outrec: "권장 밖", inrec: "권장 내" };

function formatTickValue(value: number): string {
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function formatDeltaPP(delta: number): string {
  if (!Number.isFinite(delta)) return "-";
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "";
  return `${sign}${Math.abs(delta).toFixed(2)}%p`;
}

function ArrowIcon() {
  return (
    <svg width="20" height="12" viewBox="0 0 20 12" fill="none" aria-hidden="true">
      <path d="M1 6h15" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 1.5 17 6l-5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function BandTrack({
  segments,
  lines,
  ticks,
  lclPct,
  uclPct,
  recLoPct,
  recHiPct,
}: {
  segments: BandSegment[];
  lines: BandLine[];
  ticks?: BandTick[];
  lclPct: number | null;
  uclPct: number | null;
  recLoPct: number | null;
  recHiPct: number | null;
}) {
  return (
    <>
      <div className="yieldBandLabelsRow">
        {lclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${lclPct}%` }}>LCL</span>}
        {recLoPct != null && recHiPct != null && (
          <span
            className="yieldBandRecommendedArrow"
            style={{ left: `${(recLoPct + recHiPct) / 2}%` }}
          >
            ◀─ 권장구간 ─▶
          </span>
        )}
        {uclPct != null && <span className="yieldBandBoundaryLabel" style={{ left: `${uclPct}%` }}>UCL</span>}
      </div>
      <div className="yieldBandTrack">
        {segments.map((seg, index) => (
          <div key={index} className={`yieldBandSeg seg-${seg.tone}`} style={{ width: `${Math.max(seg.widthPct, 0)}%` }}>
            <span>{seg.label}</span>
          </div>
        ))}
        {lines.map((line, index) => (
          <div key={index} className={`yieldBandLine line-${line.tone}`} style={{ left: `${line.pct}%` }} />
        ))}
      </div>
      {ticks && (
        <div className="yieldBandTicks">
          {ticks.map((tick) => (
            <span key={tick.label} className="yieldBandTick" style={{ left: `${tick.pct}%` }}>{tick.label}</span>
          ))}
        </div>
      )}
    </>
  );
}

function BandStatsRow({
  stats,
  arrowTone,
  pctDenominatorLabel,
}: {
  stats: { name: string; tone: BandTone; value: number | null; count: number; pct: number }[];
  arrowTone: "yield" | "defect";
  // 이 %의 분모가 무엇인지 (spec 문구 전수 검토 §A-6-1) -- 요약 카드 상단은
  // 전체 wafer 기준, 이 행은 판정 완료(계측) wafer 기준으로 분모가 서로
  // 다르므로, 같은 count(예: alarm 225장)가 화면마다 다른 %로 보일 때
  // 혼동하지 않도록 분모를 명시한다.
  pctDenominatorLabel?: string;
}) {
  return (
    <div className="yieldBandStats">
      {stats.map((stat, index) => (
        <div key={stat.tone} style={{ display: "contents" }}>
          <div className={`yieldBandStatCol seg-${stat.tone}`}>
            <span className="yieldBandStatName">{stat.name}</span>
            <strong className="yieldBandStatValue">{stat.value != null ? stat.value.toFixed(2) : "-"}</strong>
            <span className="yieldBandStatSub">
              {stat.count.toLocaleString()}장 · {stat.pct.toFixed(1)}%{pctDenominatorLabel ? ` (${pctDenominatorLabel})` : ""}
            </span>
          </div>
          {index < stats.length - 1 && (
            <div className={`yieldBandArrowGap tone-${arrowTone}`}>
              <ArrowIcon />
              <span>
                {stats[index + 1].value != null && stat.value != null
                  ? formatDeltaPP(stats[index + 1].value! - stat.value!)
                  : "-"}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** 카드①: 구간별 평균 수율 -- 개념도. 다섯 구간 폭을 균등하게 그린다(실제
 * 인자 값 축이 아니다) -- 축 눈금·"인자 값" 캡션 없음, 특정 인자를 가리키지
 * 않는다. */
function ConceptYieldBandCard({ summary }: { summary: AlarmSummaryResponse }) {
  const segments: BandSegment[] = [
    { tone: "out", label: "이탈", widthPct: 20 },
    { tone: "outrec", label: "권장 밖", widthPct: 20 },
    { tone: "inrec", label: "권장 내", widthPct: 20 },
    { tone: "outrec", label: "권장 밖", widthPct: 20 },
    { tone: "out", label: "이탈", widthPct: 20 },
  ];
  const lines: BandLine[] = [
    { tone: "control", pct: 20 },
    { tone: "recommended", pct: 40 },
    { tone: "recommended", pct: 60 },
    { tone: "control", pct: 80 },
  ];

  const measured = summary.measured_wafers;
  const stats = [
    { name: TONE_LABEL.out, tone: "out" as const, value: summary.band_yield.alarm, count: summary.counts.alarm, pct: measured > 0 ? (summary.counts.alarm / measured) * 100 : 0 },
    {
      name: TONE_LABEL.outrec,
      tone: "outrec" as const,
      value: summary.band_yield.out_of_recommended,
      count: summary.counts.out_of_recommended,
      pct: measured > 0 ? (summary.counts.out_of_recommended / measured) * 100 : 0,
    },
    {
      name: TONE_LABEL.inrec,
      tone: "inrec" as const,
      value: summary.band_yield.in_recommended,
      count: summary.counts.in_recommended,
      pct: measured > 0 ? (summary.counts.in_recommended / measured) * 100 : 0,
    },
  ];

  return (
    <section className="resultCard yieldBandCard">
      <div className="yieldBandCaptionRow">
        <div className="yieldBandCardTitle"><h3>구간별 평균 수율 (%)</h3></div>
        <span className="yieldBandCardMeta">{measured.toLocaleString()}장 판정 완료</span>
      </div>
      <BandTrack segments={segments} lines={lines} lclPct={20} uclPct={80} recLoPct={40} recHiPct={60} />
      <BandStatsRow stats={stats} arrowTone="yield" pctDenominatorLabel="판정 완료 기준" />
    </section>
  );
}

/** 카드②: 인자별 불량률 -- 선택된 인자의 실제 값 축(실측 min~max) 기준.
 * 단조 인자(하한 없음)는 왼쪽 이탈 영역을 그리지 않는다. */
function FactorYieldBandCard({
  bands,
  activeIndex,
  onChange,
}: {
  bands: FactorBand[];
  activeIndex: number;
  onChange: (index: number) => void;
}) {
  const band = bands[activeIndex] ?? bands[0];
  if (!band) return null;

  const pad = (band.x_max - band.x_min) * 0.04 || 1;
  const domainLoRaw = Math.min(band.x_min, band.lcl ?? band.x_min);
  const domainHiRaw = Math.max(band.x_max, band.ucl ?? band.x_max);
  const domainLo = domainLoRaw - pad;
  const domainHi = domainHiRaw + pad;
  const span = domainHi - domainLo || 1;
  const pctOf = (v: number) => ((v - domainLo) / span) * 100;

  const lclPct = band.lcl != null ? pctOf(band.lcl) : null;
  const uclPct = band.ucl != null ? pctOf(band.ucl) : null;
  const hasRecommended = band.recommended_lo != null && band.recommended_hi != null;
  const recLoPct = hasRecommended ? pctOf(band.recommended_lo as number) : null;
  const recHiPct = hasRecommended ? pctOf(band.recommended_hi as number) : null;

  const segments: BandSegment[] = [];
  const lines: BandLine[] = [];
  let leftEdge = 0;
  if (lclPct != null) {
    segments.push({ tone: "out", label: "이탈", widthPct: lclPct - leftEdge });
    lines.push({ tone: "control", pct: lclPct });
    leftEdge = lclPct;
  }
  const rightEdge = uclPct ?? 100;
  if (recLoPct != null && recHiPct != null) {
    segments.push({ tone: "outrec", label: "권장 밖", widthPct: recLoPct - leftEdge });
    lines.push({ tone: "recommended", pct: recLoPct });
    segments.push({ tone: "inrec", label: "권장 내", widthPct: recHiPct - recLoPct });
    lines.push({ tone: "recommended", pct: recHiPct });
    leftEdge = recHiPct;
  }
  segments.push({ tone: "outrec", label: "권장 밖", widthPct: rightEdge - leftEdge });
  if (uclPct != null) {
    lines.push({ tone: "control", pct: uclPct });
    segments.push({ tone: "out", label: "이탈", widthPct: 100 - uclPct });
  }

  const ticks: BandTick[] = niceTicks([domainLo, domainHi], 6).map((value) => ({
    pct: pctOf(value),
    label: formatTickValue(value),
  }));

  const totalMeasured = band.out_of_control.count + band.out_of_recommended.count + band.in_recommended.count;
  const statsOf = (count: number) => (totalMeasured > 0 ? (count / totalMeasured) * 100 : 0);
  const stats = [
    { name: TONE_LABEL.out, tone: "out" as const, value: band.out_of_control.mean_defect_rate, count: band.out_of_control.count, pct: statsOf(band.out_of_control.count) },
    { name: TONE_LABEL.outrec, tone: "outrec" as const, value: band.out_of_recommended.mean_defect_rate, count: band.out_of_recommended.count, pct: statsOf(band.out_of_recommended.count) },
    { name: TONE_LABEL.inrec, tone: "inrec" as const, value: band.in_recommended.mean_defect_rate, count: band.in_recommended.count, pct: statsOf(band.in_recommended.count) },
  ];

  return (
    <section className="resultCard yieldBandCard">
      <div className="yieldBandCardHeader">
        <div className="yieldBandCardTitle">
          <h3>인자별 불량률 (%)</h3>
          <label className="factorSelectField">
            <select
              className="factorSelect"
              value={activeIndex}
              onChange={(event) => onChange(Number(event.target.value))}
            >
              {bands.map((item, index) => (
                <option key={`${item.feature}-${item.target}`} value={index}>
                  {item.feature} → {item.target}
                </option>
              ))}
            </select>
          </label>
        </div>
        <span className="yieldBandCardMeta">낮을수록 좋음</span>
      </div>
      <BandTrack segments={segments} lines={lines} ticks={ticks} lclPct={lclPct} uclPct={uclPct} recLoPct={recLoPct} recHiPct={recHiPct} />
      <BandStatsRow stats={stats} arrowTone="defect" />
    </section>
  );
}

/** 카드③: 판정불가 -- 톤을 낮춘다(보조 텍스트만, 큰 숫자 없음). 위 카드들과
 * 분리해 90.24가 "최고 구간"으로 오독되지 않게 한다. 비율은 표기하지
 * 않는다(분모가 다르다). p값은 데이터셋별로 갱신되며, 검정이 불가능하면
 * 그 문장을 생략한다. */
/** 인자별 계측 편향 검정 결과를 사람이 읽을 문장으로 바꾼다 (spec 문구 전수
 * 검토 §A-7) -- 전체 wafer를 뭉뚱그린 이전 집계 검정("R/D 계측이 하나도
 * 없는 wafer" vs "하나 이상 계측된 wafer")은 인자별로 보면 실제로 존재하는
 * 편향을 평균 내 지워버릴 수 있어(실측: train.CSV에서 집계 검정은
 * p=0.74로 "편향 없음"이었지만 선정 인자 5개는 전부 q<0.0001로 유의했다),
 * 백엔드가 이미 인자별로 재검정한 요약(`MeasurementBiasSummary`)을 그대로
 *문장으로 옮긴다 -- 여기서 다시 계산하지 않는다. */
function describeMeasurementBias(bias: MeasurementBiasSummary | null): string | null {
  if (!bias) return null;
  if (bias.significant_count === 0) {
    return "계측 대상 선정에 따른 편향은 관측되지 않았습니다.";
  }
  const directionWord = { low: "낮게", high: "높게", mixed: "다르게" }[bias.direction ?? "mixed"];
  const scope =
    bias.significant_count === bias.tested_count
      ? `선정 인자 ${bias.significant_count}개 모두에서`
      : `선정 인자 ${bias.tested_count}개 중 ${bias.significant_count}개에서`;
  return (
    `계측 대상이 무작위로 선정되지 않았을 가능성이 있습니다. ${scope} 계측된 wafer의 불량률이 ` +
    `미계측 wafer보다 ${directionWord} 관측되었습니다. 이 분석 결과를 미계측 wafer로 ` +
    `일반화할 때는 주의가 필요합니다.`
  );
}

function UnmeasuredCard({ summary }: { summary: AlarmSummaryResponse }) {
  const avgYield = summary.band_yield.unmeasured;
  const biasText = describeMeasurementBias(summary.measurement_bias);
  return (
    <section className="unmeasuredCard">
      <p>
        <strong>
          판정불가 {summary.counts.unmeasured.toLocaleString()}장{avgYield != null ? ` · 평균 수율 ${avgYield.toFixed(2)} (%)` : ""}
        </strong>
      </p>
      <p>선정 인자가 계측되지 않아 판정할 수 없는 wafer입니다.</p>
      {biasText && <p>{biasText}</p>}
    </section>
  );
}

function ExplainButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      className="explainButton"
      onClick={onClick}
      disabled={disabled}
      title={disabled ? "원인 분석을 먼저 실행하세요" : "SUNI에게 이 건에 대해 물어보기"}
    >
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
      해설
    </button>
  );
}

function AlarmRow({
  item,
  onExplain,
  explainDisabled,
}: {
  item: AlarmItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  const [lo, hi] = item.normal_range;
  const rangeText = `${lo != null ? lo.toFixed(1) : "-∞"} ~ ${hi != null ? hi.toFixed(1) : "+∞"}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  return (
    <tr>
      <td className="col-wafer colNoTruncate">
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.lot_wafer_id}</Link>
      </td>
      <td>{item.lot_id ?? "-"}</td>
      <td>{item.step}</td>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.feature}</Link>
      </td>
      <td>{item.target}</td>
      <td className="numCol col-value colNoTruncate">{item.value.toFixed(2)}</td>
      <td className="col-range colNoTruncate" title={`train에서 ${item.feature} 자체 분포의 IQR×1.5 관리한계 (Y와 무관)`}>{rangeText}</td>
      <td className="numCol col-deviation colNoTruncate">{item.deviation.toFixed(2)}</td>
      <td>{item.direction === "above" ? "높음" : "낮음"}</td>
      <td className="col-severity colNoTruncate"><SeverityBadge severity={item.severity} /></td>
      <td className="numCol">{item.actual_y != null ? item.actual_y.toFixed(2) : "-"}</td>
      <td><ExplainButton onClick={onExplain} disabled={explainDisabled} /></td>
    </tr>
  );
}

/** ≤767px row-to-card conversion (spec §B-6) -- same data as AlarmRow,
 * compressed into an identifier+badge line and 2 detail lines instead of
 * 12 columns, since a 980px-min-width scrolling table is unusable on a
 * 375px screen. CSS (.tableHideOnMobile / .alarmCardList) decides which
 * of the two renders; both stay mounted so no extra fetch/state is needed. */
function AlarmCard({
  item,
  onExplain,
  explainDisabled,
}: {
  item: AlarmItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  const [lo, hi] = item.normal_range;
  const rangeText = `${lo != null ? lo.toFixed(1) : "-∞"}~${hi != null ? hi.toFixed(1) : "+∞"}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  return (
    <div className="alarmCard">
      <div className="alarmCardTopRow">
        <span className="alarmCardId"><Link href={rootCauseHref}>{item.lot_wafer_id}</Link></span>
        <SeverityBadge severity={item.severity} />
      </div>
      <div className="alarmCardMeta">
        <Link href={rootCauseHref}>{item.feature}</Link> · {item.target} · Step {item.step}
        {item.lot_id && ` · ${item.lot_id}`}
      </div>
      <div className="alarmCardStatsRow">
        <span>값 <b>{item.value.toFixed(1)}</b></span>
        <span>정상 <b>{rangeText}</b></span>
        <span>이탈 <b>{item.deviation >= 0 ? "+" : ""}{item.deviation.toFixed(1)}</b></span>
      </div>
      <div className="alarmCardActions">
        <ExplainButton onClick={onExplain} disabled={explainDisabled} />
      </div>
    </div>
  );
}

function RecommendationRow({
  item,
  onExplain,
  explainDisabled,
}: {
  item: RecommendationItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  const [lo, hi] = item.recommended_range;
  const rangeText = `${lo.toFixed(1)} ~ ${hi.toFixed(1)}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  const improvement = item.expected_improvement_pct != null ? `−${item.expected_improvement_pct.toFixed(0)}%` : "-";
  return (
    <tr>
      <td className="col-wafer colNoTruncate">
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.lot_wafer_id}</Link>
      </td>
      <td>{item.lot_id ?? "-"}</td>
      <td>{item.step}</td>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.feature}</Link>
      </td>
      <td>{item.target}</td>
      <td className="numCol col-value colNoTruncate">{item.value.toFixed(2)}</td>
      <td className="col-range colNoTruncate">{rangeText}</td>
      <td>
        <span className={`recommendationDirection dir-${item.direction}`}>{DIRECTION_LABEL[item.direction] ?? item.direction}</span>
      </td>
      <td className="numCol">{improvement}</td>
      <td><RecommendationTagBadge tag={item.tag} /></td>
      <td><ExplainButton onClick={onExplain} disabled={explainDisabled} /></td>
    </tr>
  );
}

/** ≤767px row-to-card conversion for 개선 권장 목록 -- see AlarmCard. */
function RecommendationCard({
  item,
  onExplain,
  explainDisabled,
}: {
  item: RecommendationItem;
  onExplain: () => void;
  explainDisabled: boolean;
}) {
  const [lo, hi] = item.recommended_range;
  const rangeText = `${lo.toFixed(1)}~${hi.toFixed(1)}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  const improvement = item.expected_improvement_pct != null ? `−${item.expected_improvement_pct.toFixed(0)}%` : "-";
  return (
    <div className="alarmCard">
      <div className="alarmCardTopRow">
        <span className="alarmCardId"><Link href={rootCauseHref}>{item.lot_wafer_id}</Link></span>
        <RecommendationTagBadge tag={item.tag} />
      </div>
      <div className="alarmCardMeta">
        <Link href={rootCauseHref}>{item.feature}</Link> · {item.target} · Step {item.step}
        {item.lot_id && ` · ${item.lot_id}`}
      </div>
      <div className="alarmCardStatsRow">
        <span>값 <b>{item.value.toFixed(1)}</b></span>
        <span>권장 <b>{rangeText}</b></span>
        <span className={`recommendationDirection dir-${item.direction}`}>{DIRECTION_LABEL[item.direction] ?? item.direction}</span>
        <span>기대개선 <b>{improvement}</b></span>
      </div>
      <div className="alarmCardActions">
        <ExplainButton onClick={onExplain} disabled={explainDisabled} />
      </div>
    </div>
  );
}
