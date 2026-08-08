import { kindLabel } from "@/lib/kindLabels";
import type { MeasurementExpansionResponse } from "@/types/data";

function formatCount(value: number): string {
  return `${value.toLocaleString()}장`;
}

function formatYieldPp(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%p`;
}

function formatRate(value: number): string {
  return `${value.toFixed(1)}%`;
}

const DISCOVERY_CARD_COLOR = "#9333EA";
const RELIABILITY_CARD_COLOR = "#0891B2";
const FALSE_POSITIVE_CARD_COLOR = "#0D9668";

// `_recommend()`(src/analysis/measurement_expansion.py)가 내려주는 고정된
// 4개 reason 문구 중 하나를 그대로 매칭해 "기대 효과"를 미래형으로 다시
// 쓴다 (spec §B-1/§B-2) -- reason 자체("사유")는 툴팁으로 옮기고, 이
// 열에는 결과만 미래형으로 보여준다. 백엔드 문구가 바뀌면 이 매칭도 함께
// 갱신해야 한다.
function expectedEffectText(reason: string, additionalJudged: number): string {
  if (reason === "계측률이 가장 낮아 판정 공백이 큽니다") {
    return `판정 공백이 ${additionalJudged.toLocaleString()}장 줄어듭니다`;
  }
  if (reason === "추정이 흔들려 권장구간 신뢰도가 낮습니다") return "권장구간 추정이 안정됩니다";
  if (reason === "추정 안정화로 권장구간이 정밀해집니다") return "권장구간이 정밀해집니다";
  return "추가 계측이 필요하지 않습니다";
}

/** '계측 확대 제안' 카드 (spec 문구 전수 검토 PART B, 배치 순서는 §A-0
 * 참고 -- 산점도·Box Plot 아래, 가장 마지막에 렌더된다). 값은 분석 실행
 * 시 한 번 계산되어 `data`로 그대로 전달되며, 이 컴포넌트는 어떤 통계도
 * 재계산하지 않는다 (spec §B-7). "권고"라는 단어가 사전 알람 로그의
 * "개선 권고"(wafer 대상)와 겹쳐 혼동을 주므로 화면 표시는 "제안"으로
 * 바꿨다 -- 내부 API 필드명(MeasurementExpansionResponse 등)은 그대로다.
 */
export default function MeasurementExpansionCard({ data }: { data: MeasurementExpansionResponse | null }) {
  if (!data) return null;

  if (!data.show_full_card) {
    const judgeable = data.total_wafers - data.action_blocked_wafers;
    return (
      <section className="resultCard measurementExpansionCollapsed">
        계측률이 충분해 계측 확대 제안이 없습니다 (판정 가능 {judgeable.toLocaleString()}장 / {data.total_wafers.toLocaleString()}장)
      </section>
    );
  }

  const discoveryCard = data.new_factor_discoveries.length > 0 ? data.new_factor_discoveries : null;

  return (
    <section className="resultCard measurementExpansionCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">계측 우선순위</span>
          <h2>계측 확대 제안</h2>
        </div>
      </div>

      <div className="meMetricRow">
        <div className="meMetricTile">
          <span className="meMetricLabel">지금 조치할 수 없는 wafer</span>
          <strong className="meMetricValue">{formatCount(data.action_blocked_wafers)}</strong>
          <span className="meMetricSub">
            전체의 {data.total_wafers > 0 ? ((data.action_blocked_wafers / data.total_wafers) * 100).toFixed(1) : "0.0"}% 계측이 없어 판정 불가
          </span>
        </div>
        <div className="meMetricTile meMetricTileEmphasis">
          <span className="meMetricLabel">계측 +10%p 시 추가 판정</span>
          <strong className="meMetricValue">{formatCount(data.additional_judged)}</strong>
          <span className="meMetricSub">그중 조치 대상 {formatCount(data.action_target)} 권장구간 밖으로</span>
        </div>
        <div className="meMetricTile">
          <span className="meMetricLabel">기대 수율 개선</span>
          <strong className="meMetricValue">{formatYieldPp(data.expected_yield_gain_pp)}</strong>
          <span className="meMetricSub">현재 판정 가능한 wafer에 추가로 얻는 몫</span>
        </div>
      </div>

      <p className="meFootnote">
        계측되지 않은 wafer는 이상 여부를 판정할 수 없어 조치 기회 자체가 없습니다. 계측을 늘리면 그만큼 개선 대상이 드러납니다.
      </p>

      {/* 모바일 반응형 패치 S-3: ≤767px에서는 컬럼을 카드로 접지 않고
          가로 스크롤을 기본 전략으로 쓴다(하지 말 것: 표 -> 카드 전환).
          .meScrollTable이 첫 컬럼 sticky·셀 패딩 축소·스크롤 힌트를
          globals.css에서 켠다. ≤479px에서는 저우선순위 "기대 효과"
          열(구 "사유")을 숨기되, 아래 캡션으로 숨겼다는 사실 자체는
          남긴다(정보를 조용히 지우지 않는다). */}
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
                    {/* 사유(reason)는 배지 툴팁으로 옮긴다 (spec §B-2) -- 열이
                        많아 좁으므로, 문제 서술은 호버로만 보여주고 표에는
                        결과(기대 효과)만 남긴다. */}
                    <span
                      className={`meRecommendationBadge${isMaintain ? " meRecommendationMaintain" : ""}`}
                      title={priority.reason}
                    >
                      {priority.recommendation}
                    </span>
                  </td>
                  <td className="meReasonCell">{expectedEffectText(priority.reason, priority.additional_judged)}</td>
                  <td className="numCol">+{priority.additional_judged.toLocaleString()}장</td>
                  <td className="numCol">
                    {formatYieldPp(priority.yield_contribution_pp)}
                    {/* 보정 §I-1: 이건 근거 밴드가 아니라 진행 바다 --
                        "어디부터 어디까지 불확실한가"(구간)가 아니라
                        "얼마나 찼는가"(0부터 채움)를 말하므로, 근거
                        밴드(.evidence-band)와 다른 클래스(.contribution-bar)를
                        쓴다 -- 두 개념을 같은 클래스로 섞지 않는다. 실제
                        lo/hi 구간은 API가 점추정만 주므로 만들어내지
                        않는다. 스케일은 고정(0~3%p)한다. tick은 쓰지 않는다. */}
                    {priority.yield_contribution_pp != null && (
                      <div className="contribution-bar">
                        <div className="track">
                          <div
                            className="fill"
                            style={{ width: `${Math.min(100, Math.max(0, (priority.yield_contribution_pp / 3) * 100))}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="meScrollHint" aria-hidden="true">← 좌우 스크롤</p>
      <p className="meHiddenColumnNote">기대 효과 열 숨김 — 767px 이상에서 표시</p>
      <p className="meFootnote">
        미계측 wafer 중 무작위로 추가 계측한다고 가정한 추정치입니다.
      </p>

      <div className="meSideEffectRow">
        {discoveryCard && (
          <div className="meSideEffectCard" style={{ borderColor: DISCOVERY_CARD_COLOR }}>
            <strong style={{ color: DISCOVERY_CARD_COLOR }}>새 원인 인자 발견</strong>
            <span className="meSideEffectSubtitle">
              {discoveryCard.map((d) => `${d.feature} (${kindLabel(d.kind)})`).join(" · ")}
            </span>
            <p>
              계측 2배 시 {[...new Set(discoveryCard.map((d) => d.target))].join(", ")}의 새 인자 {discoveryCard.length}개가 통계적으로 확인될 것으로 추정됩니다.
            </p>
          </div>
        )}
        <div className="meSideEffectCard" style={{ borderColor: RELIABILITY_CARD_COLOR }}>
          <strong style={{ color: RELIABILITY_CARD_COLOR }}>분석 신뢰도 향상</strong>
          <span className="meSideEffectSubtitle">전 인자 대상</span>
          <p>현재는 계측된 wafer에서만 관계가 확인되어 전체 적용에 한계가 있습니다.</p>
        </div>
        <div className="meSideEffectCard" style={{ borderColor: FALSE_POSITIVE_CARD_COLOR }}>
          <strong style={{ color: FALSE_POSITIVE_CARD_COLOR }}>불필요한 조치 감소</strong>
          <span className="meSideEffectSubtitle">오탐 비용 절감</span>
          <p>판정 근거가 늘어 정상 wafer를 조치 대상으로 잘못 분류하는 일이 줍니다.</p>
        </div>
      </div>

      <p className="meFootnote meFootnoteFinal">계측 비용과 위 개선 효과를 비교해 확대 여부를 판단하시기 바랍니다.</p>
    </section>
  );
}
