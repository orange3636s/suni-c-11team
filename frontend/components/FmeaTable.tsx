"use client";

import Link from "next/link";
import { BADGE_TOOLTIP, badgeClass, inlineFactorAction } from "@/lib/fmeaActions";
import { formatNumber, formatPct, formatSignedPp, isOutOfRange, rangeText } from "@/lib/fmeaFormat";
import type { FmeaFactorItem, FmeaTablePayload } from "@/types/data";

const LOW_MEASUREMENT_RATE_PCT = 10;
// 지시서 KC-2: 목업(Step1_D1 행)과 같은 처리 -- MNAR 경고 임계는
// fmeaActions.ts의 MNAR_WARNING_THRESHOLD_PP(3.0)와 같은 값이다.
const MNAR_ROW_THRESHOLD_PP = 3.0;

// 지시서 KC-2: MNAR 경고에 해당하는 행이 여럿이어도 "한 행만" 강조한다
// (여러 행이 깔리면 강조가 아니다) -- |mnar_gap_pp|가 가장 큰 행 하나만.
function topMnarFeature(items: FmeaFactorItem[]): string | null {
  let best: FmeaFactorItem | null = null;
  for (const item of items) {
    if (item.mnar_gap_pp == null || Math.abs(item.mnar_gap_pp) < MNAR_ROW_THRESHOLD_PP) continue;
    if (best == null || Math.abs(item.mnar_gap_pp) > Math.abs(best.mnar_gap_pp ?? 0)) best = item;
  }
  return best?.feature ?? null;
}

function FmeaRow({
  item,
  fmea,
  mnarHighlighted,
}: {
  item: FmeaFactorItem;
  fmea: FmeaTablePayload;
  mnarHighlighted: boolean;
}) {
  const outOfRange = isOutOfRange(item);
  const action = inlineFactorAction(item, fmea);
  return (
    <tr className={mnarHighlighted ? "fmeaMnarRow" : undefined}>
      <td className="data fmeaSticky fmeaStickyTarget">{item.target}</td>
      <td className="data fmeaSticky fmeaStickyFeature">
        <Link
          className="fmeaFeatureLink"
          href={`/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`}
        >
          {item.feature}
        </Link>
      </td>
      <td className="numCol">{formatPct(item.contribution_pct, 1)}</td>
      <td className={outOfRange ? "fmeaOutOfRange" : undefined}>{formatNumber(item.factor_value)}</td>
      <td>{rangeText(item)}</td>
      <td className="numCol">{formatPct(item.deviation_rate_pct, 0)}</td>
      <td className={item.defect_rate_deviation_pct != null && item.defect_rate_deviation_pct > 0 ? "fmeaDeviationPositive" : "fmeaDeviationMuted"}>
        {formatSignedPp(item.defect_rate_deviation_pct)}
      </td>
      <td className={`numCol${item.measurement_rate < LOW_MEASUREMENT_RATE_PCT ? " fmeaRateLow" : ""}`}>
        {formatPct(item.measurement_rate, 1)} · 검출 {formatPct(item.worst_decile_measurement_rate_pct, 1)}
      </td>
      <td className="fmeaColAction">
        {action.text}{" "}
        <span className={badgeClass(action.strength)} title={BADGE_TOOLTIP[action.strength]}>
          {action.strength}
        </span>
      </td>
    </tr>
  );
}

// 지시서 JA-3: fmea가 비었을 때 사유별로 다른 문구를 보여준다 --
// "없다"만 말하면 계산이 아예 안 된 것인지, 계산은 됐는데 실패한
// 것인지, 계산은 성공했는데 유의미한 인자가 없는 것인지 구분되지 않는다.
function EmptyState({ data, error }: { data: FmeaTablePayload | null; error?: string | null }) {
  if (error) {
    return (
      <div className="analysisErrorBox" role="alert">
        <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
        <div className="analysisErrorBody">
          <p className="analysisErrorMessage">FMEA 분석표를 생성하지 못했습니다 — {error}</p>
        </div>
        <Link href="/root-cause" className="button sm secondary">
          원인 분석에서 다시 실행
        </Link>
      </div>
    );
  }
  if (!data) {
    return <p className="emptyMessage">분석을 실행하면 FMEA 분석표가 생성됩니다.</p>;
  }
  if (data.no_qualifying_factor.length > 0) {
    return (
      <p className="emptyMessage">
        기여율 20% 이상 인자가 있는 타깃이 없습니다 ({data.no_qualifying_factor
          .map((n) => `${n.target}는 최대 ${n.max_contribution_pct.toFixed(1)}%`)
          .join(", ")}).
      </p>
    );
  }
  return <p className="emptyMessage">계측된 인자가 없어 계산할 수 없습니다.</p>;
}

/** FMEA 분석표 (모니터링 홈, 작업 지시서 WE) -- 유의 인자 표를 대체한다.
 * 계산은 전부 백엔드에서 끝났으므로(자동 갱신·수동 "다시 분석" 저장
 * 둘 다 같은 함수를 공유한다, JA-1) 이 컴포넌트는 표시만 한다. */
export default function FmeaTable({ data, error }: { data: FmeaTablePayload | null; error?: string | null }) {
  const mnarFeature = data ? topMnarFeature(data.items) : null;

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">FMEA 분석표</span>
          <h2>불량 원인별 잠재 원인</h2>
        </div>
      </div>

      {!data || data.items.length === 0 ? (
        <EmptyState data={data} error={error} />
      ) : (
        <>
          <p className="sectionCaption">
            불량률 편차 내림차순 · 타깃별 기여율 20% 이상 · {data.items.length}행 · {(data.dataset_id || "-").toUpperCase()} · {data.total_wafers.toLocaleString()} wf
          </p>
          <p className="fmeaDescription">
            각 타깃에서 파레토 기여율 20% 이상인 인자를 전부 표시합니다.
          </p>

          <div className="tableWrap fmeaScrollTable">
            <table className="fmeaTable">
              <thead>
                <tr>
                  <th className="fmeaSticky fmeaStickyTarget">타깃</th>
                  <th className="fmeaSticky fmeaStickyFeature">잠재 원인</th>
                  <th className="numCol">기여율</th>
                  <th>인자값</th>
                  <th>권장 구간</th>
                  <th className="numCol">이탈률</th>
                  <th>불량률 편차</th>
                  <th className="numCol">계측률 · 검출률</th>
                  <th className="fmeaColAction">권고 조치</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <FmeaRow
                    key={`${item.target}-${item.feature}`}
                    item={item}
                    fmea={data}
                    mnarHighlighted={item.feature === mnarFeature}
                  />
                ))}
              </tbody>
            </table>
          </div>
          <p className="meScrollHint" aria-hidden="true">← 좌우 스크롤</p>

          {data.no_qualifying_factor.length > 0 && (
            <div className="fmeaNotice fmeaNoticeInfo">
              <p>
                {data.no_qualifying_factor
                  .map((n) => `${n.target}는 기여율 20% 이상 인자가 없습니다 (최대 ${n.max_contribution_pct.toFixed(1)}%)`)
                  .join(" · ")}
              </p>
            </div>
          )}
        </>
      )}
    </section>
  );
}
