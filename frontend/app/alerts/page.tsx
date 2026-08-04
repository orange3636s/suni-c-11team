"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import {
  NoSearchResults,
  ScrollTableBody,
  TableCaption,
  TableToolbar,
  useTableSearchSort,
  type SortOption,
} from "@/components/DataTablePanel";
import { getAlarmSummary, getAlarms, getRecommendations } from "@/lib/api";
import type {
  AlarmItem,
  AlarmListResponse,
  AlarmSummaryResponse,
  RecommendationItem,
  RecommendationListResponse,
} from "@/types/data";

const SEVERITY_LABEL: Record<string, string> = { low: "낮음", medium: "중간", high: "높음" };
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
  const [trainDataset, setTrainDataset] = useState("train");
  const [evalDataset, setEvalDataset] = useState("test");
  const [summary, setSummary] = useState<AlarmSummaryResponse | null>(null);
  const [alarms, setAlarms] = useState<AlarmListResponse | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationListResponse | null>(null);
  const [severityFilter, setSeverityFilter] = useState("");
  const [showReferenceTag, setShowReferenceTag] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [summaryResponse, alarmsResponse, recommendationsResponse] = await Promise.all([
        getAlarmSummary(trainDataset, evalDataset),
        getAlarms(trainDataset, evalDataset, severityFilter || undefined),
        getRecommendations(trainDataset, evalDataset),
      ]);
      setSummary(summaryResponse);
      setAlarms(alarmsResponse);
      setRecommendations(recommendationsResponse);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "알람 로그를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [trainDataset, evalDataset, severityFilter]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const alarmTable = useTableSearchSort(
    alarms?.items ?? [],
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

  const yieldGap = summary?.yield_gap;
  const totalWafers = summary
    ? summary.counts.alarm + summary.counts.normal + summary.counts.unmeasured
    : 0;

  return (
    <DashboardShell activeItem="사전 알람 로그">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">PRE-ALERT LOG</span>
        <h1>사전 알람 로그</h1>
        <p>학습 데이터셋에서 산출한 정상범위를 평가 데이터셋에 적용해 이탈 여부를 판정합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(200px,1fr) minmax(200px,1fr) minmax(140px,.6fr)" }}>
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
        </div>
        {error && <p className="errorMessage">{error}</p>}
      </section>

      {summary ? (
        <section className="alarmSummaryGrid">
          <AlarmSummaryCard
            label="알람 wafer"
            value={`${summary.counts.alarm}장`}
            aux={pct(summary.counts.alarm, totalWafers)}
            tone="highlight"
          />
          <AlarmSummaryCard label="정상" value={`${summary.counts.normal}장`} />
          <AlarmSummaryCard
            label="판정불가 (미계측)"
            value={`${summary.counts.unmeasured}장`}
            aux={pct(summary.counts.unmeasured, totalWafers)}
            tone="muted"
            title="선정 인자가 하나도 계측되지 않아 판정할 수 없는 wafer"
          />
          <AlarmSummaryCard
            label="알람군 평균수율"
            value={summary.alarm_group_yield_avg != null ? summary.alarm_group_yield_avg.toFixed(2) : "-"}
          />
          <AlarmSummaryCard
            label="무알람군 평균수율"
            value={summary.no_alarm_group_yield_avg != null ? summary.no_alarm_group_yield_avg.toFixed(2) : "-"}
          />
          <AlarmSummaryCard
            label="격차"
            value={yieldGap != null ? `${yieldGap > 0 ? "+" : ""}${yieldGap.toFixed(2)}%p` : "-"}
            tone={yieldGap != null && yieldGap < 0 ? "highlight" : "default"}
          />
        </section>
      ) : (
        <section className="alarmSummaryGrid">
          {Array.from({ length: 6 }).map((_, index) => (
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
            <h2>알람 목록 ({alarms?.total ?? 0}건)</h2>
          </div>
          {alarms && alarms.items.length > 0 && (
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
        {!loading && alarms && alarms.items.length === 0 && (
          <p className="emptyMessage">조건에 맞는 알람이 없습니다.</p>
        )}
        {!loading && alarms && alarms.items.length > 0 && alarmTable.sorted.length === 0 && (
          <NoSearchResults onClear={() => alarmTable.setSearch("")} />
        )}
        {!loading && alarms && alarmTable.sorted.length > 0 && (
          <>
            <ScrollTableBody>
              <table>
                <thead>
                  <tr>
                    <th style={{ width: "12%" }}>Wafer</th>
                    <th style={{ width: "8%" }}>LOT</th>
                    <th style={{ width: "6%" }}>Step</th>
                    <th style={{ width: "13%" }}>인자</th>
                    <th style={{ width: "6%" }}>타깃</th>
                    <th className="numCol" style={{ width: "8%" }}>값</th>
                    <th style={{ width: "14%" }}>정상범위</th>
                    <th className="numCol" style={{ width: "8%" }}>이탈량</th>
                    <th style={{ width: "7%" }}>방향</th>
                    <th style={{ width: "9%" }}>심각성</th>
                    <th className="numCol" style={{ width: "9%" }}>실측값</th>
                  </tr>
                </thead>
                <tbody>
                  {alarmTable.sorted.map((item, index) => (
                    <AlarmRow key={`${item.lot_wafer_id}-${item.feature}-${index}`} item={item} />
                  ))}
                </tbody>
              </table>
            </ScrollTableBody>
            <TableCaption total={alarmTable.sorted.length} shown={Math.min(10, alarmTable.sorted.length)} />
          </>
        )}
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">LOT</span>
            <h2>LOT별 알람 집계 (상위)</h2>
          </div>
        </div>
        {summary && summary.top_lots.length > 0 ? (
          <div className="tableWrap">
            <table>
              <thead><tr><th style={{ width: "70%" }}>LOT</th><th className="numCol" style={{ width: "30%" }}>알람 건수</th></tr></thead>
              <tbody>
                {summary.top_lots.slice(0, 10).map((lot) => (
                  <tr key={lot.lot_id}><td>{lot.lot_id}</td><td className="numCol">{lot.alarm_count}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
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
        <p className="alarmSummaryNote" style={{ margin: "0 0 4px" }}>
          알람은 관리한계(LCL/UCL) 이탈을 나타내는 이상 탐지이고, 개선 권장은 권장 구간 이탈을 나타내는 개선 제안입니다. 이미 알람으로 잡힌 wafer는 같은 인자에 대해 중복 집계하지 않습니다.
        </p>
        {loading && <p className="emptyMessage">불러오는 중…</p>}
        {!loading && recommendations && recommendations.items.length === 0 && (
          <p className="emptyMessage">권장할 항목이 없습니다.</p>
        )}
        {!loading && recommendations && recommendations.items.length > 0 && recommendationTable.sorted.length === 0 && (
          <NoSearchResults onClear={() => recommendationTable.setSearch("")} />
        )}
        {!loading && recommendations && recommendationTable.sorted.length > 0 && (
          <>
            <ScrollTableBody>
              <table>
                <thead>
                  <tr>
                    <th style={{ width: "12%" }}>Wafer</th>
                    <th style={{ width: "8%" }}>LOT</th>
                    <th style={{ width: "6%" }}>Step</th>
                    <th style={{ width: "13%" }}>인자</th>
                    <th style={{ width: "6%" }}>타깃</th>
                    <th className="numCol" style={{ width: "8%" }}>현재값</th>
                    <th style={{ width: "15%" }}>권장 구간</th>
                    <th style={{ width: "9%" }}>이동 방향</th>
                    <th className="numCol" style={{ width: "9%" }}>기대 개선</th>
                    <th style={{ width: "9%" }}>태그</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendationTable.sorted.map((item, index) => (
                    <RecommendationRow key={`${item.lot_wafer_id}-${item.feature}-${index}`} item={item} />
                  ))}
                </tbody>
              </table>
            </ScrollTableBody>
            <TableCaption total={recommendationTable.sorted.length} shown={Math.min(10, recommendationTable.sorted.length)} />
          </>
        )}
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
  tone?: "default" | "highlight" | "muted";
  title?: string;
}) {
  return (
    <div className={`alarmSummaryCard ${tone !== "default" ? `tone-${tone}` : ""}`} title={title}>
      <span className="alarmSummaryLabel">{label}</span>
      <div className="alarmSummaryValueRow">
        <strong className="alarmSummaryValue">{value}</strong>
        {aux && <span className="alarmSummaryAux">{aux}</span>}
      </div>
    </div>
  );
}

function AlarmRow({ item }: { item: AlarmItem }) {
  const [lo, hi] = item.normal_range;
  const rangeText = `${lo != null ? lo.toFixed(1) : "-∞"} ~ ${hi != null ? hi.toFixed(1) : "+∞"}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  return (
    <tr>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.lot_wafer_id}</Link>
      </td>
      <td>{item.lot_id ?? "-"}</td>
      <td>{item.step}</td>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.feature}</Link>
      </td>
      <td>{item.target}</td>
      <td className="numCol">{item.value.toFixed(2)}</td>
      <td title={`train에서 ${item.feature} 자체 분포의 IQR×1.5 관리한계 (Y와 무관)`}>{rangeText}</td>
      <td className="numCol">{item.deviation.toFixed(2)}</td>
      <td>{item.direction === "above" ? "높음" : "낮음"}</td>
      <td><SeverityBadge severity={item.severity} /></td>
      <td className="numCol">{item.actual_y != null ? item.actual_y.toFixed(2) : "-"}</td>
    </tr>
  );
}

function RecommendationRow({ item }: { item: RecommendationItem }) {
  const [lo, hi] = item.recommended_range;
  const rangeText = `${lo.toFixed(1)} ~ ${hi.toFixed(1)}`;
  const rootCauseHref = `/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`;
  const improvement = item.expected_improvement_pct != null ? `−${item.expected_improvement_pct.toFixed(0)}%` : "-";
  return (
    <tr>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.lot_wafer_id}</Link>
      </td>
      <td>{item.lot_id ?? "-"}</td>
      <td>{item.step}</td>
      <td>
        <Link href={rootCauseHref} title="원인 분석 산점도에서 열기">{item.feature}</Link>
      </td>
      <td>{item.target}</td>
      <td className="numCol">{item.value.toFixed(2)}</td>
      <td>{rangeText}</td>
      <td>
        <span className={`recommendationDirection dir-${item.direction}`}>{DIRECTION_LABEL[item.direction] ?? item.direction}</span>
      </td>
      <td className="numCol">{improvement}</td>
      <td><RecommendationTagBadge tag={item.tag} /></td>
    </tr>
  );
}
