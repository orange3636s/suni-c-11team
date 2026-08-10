"use client";

import { BADGE_TOOLTIP, NOT_PROPOSED_FOOTNOTE, badgeClass, buildRecommendedActions } from "@/lib/fmeaActions";
import type { FmeaTablePayload, MeasurementExpansionResponse } from "@/types/data";

function formatCount(value: number): string {
  return `${value.toLocaleString()}장`;
}

// 작업 지시서 WF-4: 인자별 "수율 기여"가 2자리에서는 거의 같은 값으로
// 뭉친다(실제로는 인자마다 다른 부트스트랩 시뮬레이션 결과인데, 공통
// 배율(action_target_ratio × yield_gap_pp / total_wafers)이 작아 2자리
// 반올림에서 차이가 사라진다) -- 3자리로 늘려 차이를 드러낸다.
function formatYieldPpPrecise(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}%p`;
}

function formatYieldPp(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%p`;
}

function formatRate(value: number): string {
  return `${value.toFixed(1)}%`;
}

function expectedEffectText(reason: string, additionalJudged: number): string {
  if (reason === "계측률이 가장 낮아 판정 공백이 큽니다") {
    return `판정 공백이 ${additionalJudged.toLocaleString()}장 줄어듭니다`;
  }
  if (reason === "추정이 흔들려 권장구간 신뢰도가 낮습니다") return "권장구간 추정이 안정됩니다";
  if (reason === "추정 안정화로 권장구간이 정밀해집니다") return "권장구간이 정밀해집니다";
  return "추가 계측이 필요하지 않습니다";
}

function RecommendedActionsTable({ data }: { data: FmeaTablePayload | null }) {
  const rows = buildRecommendedActions(data);

  return (
    <div className="actionBlockSection">
      <h3 className="actionBlockSubheading">권고 조치</h3>
      <p className="sectionCaption">실익(불량률 편차) 순 · 기여율 20% 이상 핵심인자가 계측된 wafer 대상</p>

      {rows.length === 0 ? (
        <p className="emptyMessage">권고할 조치가 없습니다.</p>
      ) : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th className="numCol">#</th>
                <th>조치</th>
                <th>대상</th>
                <th>기대 효과</th>
                <th>근거 강도</th>
                <th>비고</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <td className="numCol">{row.order}</td>
                  <td className="data">{row.action}</td>
                  <td>{row.target}</td>
                  <td>{row.expectedEffect}</td>
                  <td>
                    <span className={badgeClass(row.strength)} title={BADGE_TOOLTIP[row.strength]}>
                      {row.strength}
                    </span>
                  </td>
                  <td className="fmeaActionNote">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="fmeaFootnote">{NOT_PROPOSED_FOOTNOTE}</p>
    </div>
  );
}

function MeasurementExpansionSection({ data }: { data: MeasurementExpansionResponse | null }) {
  if (!data) return null;

  if (!data.show_full_card) {
    const judgeable = data.total_wafers - data.action_blocked_wafers;
    return (
      <div className="actionBlockSection">
        <h3 className="actionBlockSubheading">계측 확대</h3>
        <p className="emptyMessage">
          계측률이 충분해 계측 확대 제안이 없습니다 (판정 가능 {judgeable.toLocaleString()}장 / {data.total_wafers.toLocaleString()}장)
        </p>
      </div>
    );
  }

  return (
    <div className="actionBlockSection">
      <h3 className="actionBlockSubheading">계측 확대</h3>
      <p className="sectionCaption">핵심인자가 하나도 계측되지 않은 wafer 대상</p>

      <div className="meMetricRow">
        <div className="meMetricTile">
          <span className="meMetricLabel">판정 불가</span>
          <strong className="meMetricValue">{formatCount(data.action_blocked_wafers)}</strong>
          <span className="meMetricSub">
            전체의 {data.total_wafers > 0 ? ((data.action_blocked_wafers / data.total_wafers) * 100).toFixed(1) : "0.0"}%
          </span>
        </div>
        <div className="meMetricTile meMetricTileEmphasis">
          <span className="meMetricLabel">계측 +10%p 시 추가 판정</span>
          <strong className="meMetricValue">{formatCount(data.additional_judged)}</strong>
          <span className="meMetricSub">그중 조치 대상 {formatCount(data.action_target)} 권장구간 밖으로</span>
        </div>
        <div className="meMetricTile">
          <span className="meMetricLabel">기대 개선</span>
          <strong className="meMetricValue">{formatYieldPp(data.expected_yield_gain_pp)}</strong>
          <span className="meMetricSub">현재 판정 가능한 wafer에 추가로 얻는 몫</span>
        </div>
      </div>

      <div className="tableWrap meScrollTable">
        <table className="meTable">
          <thead>
            <tr>
              <th>인자</th>
              <th className="numCol">계측률</th>
              <th>권고</th>
              <th className="meReasonColHeader">기대 효과</th>
              <th className="numCol">추가 판정</th>
              <th className="numCol">수율 기여</th>
            </tr>
          </thead>
          <tbody>
            {data.priorities.map((priority, index) => {
              const isLowest = index === 0;
              const isMaintain = priority.recommendation === "유지";
              return (
                <tr key={`${priority.feature}-${priority.target}`} className={isMaintain ? "meRowMuted" : "meRowFlagged"}>
                  <td className="data">{priority.feature} → {priority.target}</td>
                  <td className={`numCol${isLowest ? " meRateLowest" : ""}`}>{formatRate(priority.measurement_rate)}</td>
                  <td>
                    <span
                      className={`meRecommendationBadge${isMaintain ? " meRecommendationMaintain" : ""}`}
                      title={priority.reason}
                    >
                      {priority.recommendation}
                    </span>
                  </td>
                  <td className="meReasonCell">{expectedEffectText(priority.reason, priority.additional_judged)}</td>
                  <td className="numCol">+{priority.additional_judged.toLocaleString()}장</td>
                  <td className="numCol">{formatYieldPpPrecise(priority.yield_contribution_pp)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="meScrollHint" aria-hidden="true">← 좌우 스크롤</p>
      <p className="fmeaFootnote">
        판정 불가 {data.action_blocked_wafers.toLocaleString()}장 · 계측 +10%p 시 추가 판정 {data.additional_judged.toLocaleString()}장 ·
        기대 개선 {formatYieldPp(data.expected_yield_gain_pp)}
      </p>
      <p className="fmeaFootnote">계측을 늘리면 새 인자 발견·신뢰도 향상·오탐 감소가 기대됩니다.</p>
    </div>
  );
}

/** 조치 블록 (모니터링 홈, 작업 지시서 WF) -- 권고 조치 표와 계측 확대
 * 표를 한 카드 두 표로 합친다(세로가 길어지지 않게 카드는 나누지
 * 않는다). "제안하지 않음" 행은 표에 넣지 않고 하단 한 줄로 대신한다
 * (WF-2). 계측 확대의 하단 부가 효과 카드 3개도 한 줄로 대신한다(WF-3).
 */
export default function ActionBlock({
  fmea,
  measurementExpansion,
}: {
  fmea: FmeaTablePayload | null;
  measurementExpansion: MeasurementExpansionResponse | null;
}) {
  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">조치</span>
          <h2>권고 조치 · 계측 확대</h2>
        </div>
      </div>
      <RecommendedActionsTable data={fmea} />
      <MeasurementExpansionSection data={measurementExpansion} />
    </section>
  );
}
