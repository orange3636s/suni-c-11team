"use client";

import Link from "next/link";
import type { ActionPriorityPayload, ActionPriorityRow } from "@/types/data";

function CoverageTrack({ row }: { row: ActionPriorityRow }) {
  const total = row.total_wafers || 1;
  const outPct = (row.out_of_range_count / total) * 100;
  const inCount = Math.max(0, row.measured_count - row.out_of_range_count);
  const inPct = (inCount / total) * 100;
  const unmeasuredCount = Math.max(0, row.total_wafers - row.measured_count);
  const unmeasuredPct = 100 - outPct - inPct;
  return (
    <div className="coverageRow">
      <div className="coverageRowHead">
        <Link
          className="fmeaFeatureLink"
          href={`/root-cause?target=${encodeURIComponent(row.target)}&feature=${encodeURIComponent(row.feature)}`}
        >
          {row.feature}
        </Link>
        <span className="actionPriorityTargetTag">→ {row.target}</span>
      </div>
      <div className="coverageTrack" title={`구간 밖 ${row.out_of_range_count} · 구간 안 ${inCount} · 미계측 ${unmeasuredCount}`}>
        <span className="coverageSeg coverageSeg-out" style={{ width: `${outPct}%` }} />
        <span className="coverageSeg coverageSeg-in" style={{ width: `${inPct}%` }} />
        <span className="coverageSeg coverageSeg-unmeasured" style={{ width: `${unmeasuredPct}%` }} />
      </div>
      <div className="coverageRowCaption">
        구간 밖 {row.out_of_range_count.toLocaleString()} · 구간 안 {inCount.toLocaleString()} · 미계측{" "}
        {unmeasuredCount.toLocaleString()} / 전체 {row.total_wafers.toLocaleString()}
      </div>
    </div>
  );
}

/** 모니터링 홈 블록② 조치 가능 범위 -- "구간 밖 N장"이 계측된 것
 * 중에서만 센 값임을, 전체 1,000장을 분모로 한 스택 막대로 드러낸다.
 * 블록①과 같은 원천(action_priority.py)의 행을 재사용한다 -- 별도
 * 조회 없음. */
export default function CoverageBlock({ data, error }: { data: ActionPriorityPayload | null; error?: string | null }) {
  const rows = data?.rows ?? [];
  const avgMeasurementRatePct =
    rows.length > 0
      ? (rows.reduce((sum, r) => sum + (r.total_wafers > 0 ? r.measured_count / r.total_wafers : 0), 0) / rows.length) * 100
      : 0;

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">블록②</span>
          <h2>조치 가능 범위</h2>
        </div>
      </div>

      {error ? (
        <div className="analysisErrorBox" role="alert">
          <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
          <div className="analysisErrorBody">
            <p className="analysisErrorMessage">조치 가능 범위를 계산하지 못했습니다 — {error}</p>
          </div>
        </div>
      ) : !data || rows.length === 0 ? (
        <p className="emptyMessage">분석을 실행하면 조치 가능 범위가 계산됩니다.</p>
      ) : (
        <>
          <div className="coverageLegend">
            <span>
              <i className="coverageLegendSwatch coverageSeg-out" /> 구간 밖 — 조치 대상
            </span>
            <span>
              <i className="coverageLegendSwatch coverageSeg-in" /> 구간 안
            </span>
            <span>
              <i className="coverageLegendSwatch coverageSeg-unmeasured" /> 미계측 — 판정 불가
            </span>
          </div>
          {rows.map((row) => (
            <CoverageTrack key={`${row.target}-${row.feature}`} row={row} />
          ))}
          <p className="tableCaption">
            {rows.length}개 인자 모두 계측률이 {avgMeasurementRatePct.toFixed(0)}% 이하입니다. 구간 밖 wafer가 많아 보이지만
            계측된 것 중에서만 센 값입니다. 계측을 10%p 늘리면 조치 대상이 약{" "}
            {data.estimated_additional_action_wafers.toLocaleString()}장 늘어날 것으로 추정됩니다.
          </p>
        </>
      )}
    </section>
  );
}
