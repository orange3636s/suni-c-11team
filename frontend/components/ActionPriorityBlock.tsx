"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { HScrollTableBody } from "@/components/DataTablePanel";
import { formatNumber, formatPct, formatSignedPp, isOutOfRange, rangeText } from "@/lib/fmeaFormat";
import type { ActionPriorityPayload, ActionPriorityRow } from "@/types/data";

// MB-5: 기대 회수가 이 값 미만이면 흐리게(실익 낮음) -- 백엔드
// (src/analysis/action_priority.py의 MIN_MEANINGFUL_EXPECTED_RECOVERY_PP)
// 와 값을 맞춘다. 판정 자체는 `row.dimmed`로 서버가 이미 내려주므로,
// 이 상수는 화면에는 쓰지 않는다(문서화 목적으로만 남긴다).
const SORT_OPTIONS = [
  { value: "expected", label: "기대 회수 순" },
  { value: "width", label: "회수 폭 순" },
] as const;
type SortValue = (typeof SORT_OPTIONS)[number]["value"];

// MB-2: "결함 수가 늘수록 Y3 손실이 늘어난다"류 한 줄 설명 -- 백엔드가
// 이미 계산해 보낸 relation_shape(구조화된 값)을 문장으로 바꾸는
// 순수 포맷팅이다(수치 계산이 아니다, fmeaFormat.ts의 RELATION_LABEL과
// 같은 성격).
function describeRelation(row: ActionPriorityRow): string {
  switch (row.relation_shape) {
    case "monotonic_increasing":
      return "값이 클수록 손실이 늘어납니다";
    case "monotonic_decreasing":
      return "값이 작을수록 손실이 늘어납니다";
    case "u_shape":
      return "권장 구간을 벗어날수록 손실이 늘어납니다";
    default:
      return "관계가 뚜렷하지 않습니다";
  }
}

function sortRows(rows: ActionPriorityRow[], sort: SortValue): ActionPriorityRow[] {
  const key = sort === "expected" ? "expected_recovery_pp" : "recovery_width_pp";
  return [...rows].sort((a, b) => (b[key] ?? -Infinity) - (a[key] ?? -Infinity));
}

function ActionPriorityRowView({
  row,
  rank,
  showTarget,
}: {
  row: ActionPriorityRow;
  rank: number;
  showTarget: boolean;
}) {
  const outOfRange = isOutOfRange(row);
  const measurementRatePct = row.total_wafers > 0 ? (row.measured_count / row.total_wafers) * 100 : 0;
  return (
    <tr className={row.dimmed ? "actionPriorityRow dimmed" : "actionPriorityRow"}>
      <td className={`data numCol${rank === 1 ? " actionPriorityRankTop" : ""}`}>{rank}</td>
      <td className={`data${showTarget ? "" : " actionPriorityGrouped"}`}>
        <Link
          className="fmeaFeatureLink"
          href={`/root-cause?target=${encodeURIComponent(row.target)}&feature=${encodeURIComponent(row.feature)}`}
        >
          {row.feature}
        </Link>
        {showTarget && <span className="actionPriorityTargetTag">→ {row.target}</span>}
      </td>
      {row.dimmed ? (
        <td colSpan={3} className="actionPriorityDimReason">
          {row.dim_reason}
        </td>
      ) : (
        <>
          <td className="actionPriorityDescription colNoTruncate">{describeRelation(row)}</td>
          <td>
            <span className={outOfRange ? "actionPriorityCurrentOut" : undefined}>{formatNumber(row.factor_value)}</span>
            {" → "}
            {rangeText(row)}
            <div className="actionPrioritySubline">
              계측 {row.measured_count.toLocaleString()}장 중 {row.out_of_range_count.toLocaleString()}장이 구간 밖
            </div>
          </td>
          <td className="actionPriorityExpected numCol">
            <span className={row.recovery_width_pp != null && row.recovery_width_pp > 0 ? "actionPriorityPositive" : undefined}>
              {formatSignedPp(row.expected_recovery_pp)}
            </span>
            <div className="actionPrioritySubline">
              회수 {formatNumber(row.recovery_width_pp, 2)} × 비중 {formatPct(row.share_pct, 1)}
            </div>
          </td>
        </>
      )}
      <td className="numCol">
        {row.measured_count.toLocaleString()}/{row.total_wafers.toLocaleString()}
        <div className="actionPrioritySubline">계측률 {formatPct(measurementRatePct, 1)}</div>
      </td>
    </tr>
  );
}

/** MB: 모니터링 홈 블록① 조치 우선순위 -- 타깃별 파레토 기여율 10%
 * 이상인 인자를 기대 회수(회수 폭 × 손실 비중) 순으로 보여준다. 계산은
 * 전부 백엔드(src/analysis/action_priority.py, train.CSV 기준)에서
 * 끝났으므로 이 컴포넌트는 정렬 선택과 표시만 한다. */
export default function ActionPriorityBlock({
  data,
  error,
}: {
  data: ActionPriorityPayload | null;
  error?: string | null;
}) {
  const [sort, setSort] = useState<SortValue>("expected");
  const sorted = useMemo(() => (data ? sortRows(data.rows, sort) : []), [data, sort]);

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">블록①</span>
          <h2>조치 우선순위</h2>
        </div>
        {data && sorted.length > 0 && (
          <select
            className="actionPrioritySortSelect"
            value={sort}
            onChange={(event) => setSort(event.target.value as SortValue)}
            aria-label="정렬 기준"
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {error ? (
        <div className="analysisErrorBox" role="alert">
          <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
          <div className="analysisErrorBody">
            <p className="analysisErrorMessage">조치 우선순위를 계산하지 못했습니다 — {error}</p>
          </div>
          <Link href="/root-cause" className="button sm secondary">
            원인 분석에서 다시 실행
          </Link>
        </div>
      ) : !data ? (
        <p className="emptyMessage">분석을 실행하면 조치 우선순위가 계산됩니다.</p>
      ) : sorted.length === 0 ? (
        <p className="emptyMessage">기여율 10% 이상 인자가 있는 타깃이 없습니다.</p>
      ) : (
        <>
          <p className="sectionCaption">
            train.CSV 기준 · 타깃별 기여율 10% 이상 인자 전체 · {sorted.length}행
          </p>
          <HScrollTableBody rows={5} minWidth={900}>
            <table className="dataTable actionPriorityTable">
              <thead>
                <tr>
                  <th>순위</th>
                  <th>인자</th>
                  <th colSpan={3}>내용</th>
                  <th>계측 수</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, index) => {
                  const previous = sorted[index - 1];
                  const showTarget = !previous || previous.target !== row.target;
                  return (
                    <ActionPriorityRowView
                      key={`${row.target}-${row.feature}`}
                      row={row}
                      rank={index + 1}
                      showTarget={showTarget}
                    />
                  );
                })}
              </tbody>
            </table>
          </HScrollTableBody>
          {data.no_qualifying_factor.length > 0 && (
            <p className="tableCaption">
              {data.no_qualifying_factor
                .map((n) => `${n.target}는 기여율 10% 이상 인자가 없습니다 (최대 ${n.max_contribution_pct.toFixed(1)}%)`)
                .join(" · ")}
            </p>
          )}
        </>
      )}
    </section>
  );
}
