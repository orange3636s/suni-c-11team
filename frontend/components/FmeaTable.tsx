"use client";

import Link from "next/link";
import { inlineFactorAction } from "@/lib/fmeaActions";
import { formatNumber, formatPct, formatSignedPp, isOutOfRange, rangeText } from "@/lib/fmeaFormat";
import type { FmeaFactorItem, FmeaTablePayload } from "@/types/data";

const RPN_RED = 300;
const RPN_AMBER = 100;
const LOW_MEASUREMENT_RATE_PCT = 10;
const MIN_YIELD_DEVIATION_PP = 0.3;

function rpnClass(rpn: number): string {
  if (rpn >= RPN_RED) return "fmeaRpn-red";
  if (rpn >= RPN_AMBER) return "fmeaRpn-amber";
  return "";
}

function badgeClass(strength: string): string {
  if (strength === "확정") return "badge badge-green";
  if (strength === "실험 후보") return "badge badge-amber";
  return "badge badge-neutral";
}

function FmeaRow({ item, fmea }: { item: FmeaFactorItem; fmea: FmeaTablePayload }) {
  const outOfRange = isOutOfRange(item);
  const action = inlineFactorAction(item, fmea);
  return (
    <tr>
      <td className="data fmeaSticky fmeaStickyTarget">{item.target}</td>
      <td className="fmeaColScore">{item.severity_score}</td>
      <td className="data fmeaSticky fmeaStickyFeature">
        <Link href={`/root-cause?target=${encodeURIComponent(item.target)}&feature=${encodeURIComponent(item.feature)}`}>
          {item.feature}
        </Link>
      </td>
      <td className={outOfRange ? "fmeaOutOfRange" : undefined}>{formatNumber(item.factor_value)}</td>
      <td>{rangeText(item)}</td>
      <td className="numCol">{formatPct(item.deviation_rate_pct, 0)}</td>
      <td className="fmeaColScore">{item.occurrence_score}</td>
      <td>
        <span className="fmeaDetectionKind">{item.detection_kind}</span> {item.detection_method}
      </td>
      <td className={`numCol${item.measurement_rate < LOW_MEASUREMENT_RATE_PCT ? " fmeaRateLow" : ""}`}>
        {formatPct(item.measurement_rate, 1)}
      </td>
      <td className="fmeaColScore">{item.detection_score}</td>
      <td className={`numCol fmeaColRpn ${rpnClass(item.rpn)}`}>{item.rpn}</td>
      <td className={item.yield_deviation != null && item.yield_deviation >= MIN_YIELD_DEVIATION_PP ? "fmeaDeviationPositive" : "fmeaDeviationMuted"}>
        {formatSignedPp(item.yield_deviation)}
      </td>
      <td>{formatPct(item.expected_yield, 2)}</td>
      <td className="fmeaColAction">
        {action.text} <span className={badgeClass(action.strength)}>{action.strength}</span>
      </td>
    </tr>
  );
}

/** FMEA 분석표 (모니터링 홈, 지시서 IB) -- 유의 인자 표를 대체한다. 계산은
 * 전부 백엔드(자동 갱신 스냅샷)에서 끝났으므로 이 컴포넌트는 표시만
 * 한다. `data`가 null이면(스냅샷이 아직 한 번도 안 돌았거나, 계산에
 * 실패했거나) 대기 안내를 보여준다 -- "결과 없음"과 "실패"를 굳이
 * 구분하지 않는다(다른 자동 갱신 계산과 같은 원칙). */
export default function FmeaTable({ data }: { data: FmeaTablePayload | null }) {
  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">FMEA 분석표</span>
          <h2>불량 원인별 잠재 원인</h2>
        </div>
      </div>

      {!data ? (
        <p className="emptyMessage">다음 자동 갱신을 기다리는 중입니다 — FMEA 분석표는 자동 갱신 스냅샷에서만 계산됩니다.</p>
      ) : (
        <>
          <p className="sectionCaption">
            RPN 내림차순 · 상위 {data.items.length}개 · {(data.dataset_id || "-").toUpperCase()} · {data.total_wafers.toLocaleString()} wf
          </p>
          <p className="fmeaDescription">
            타깃별 인자를 효과크기(ε²)로 선정하고, 검출 능력까지 반영한 위험 우선순위를 산출합니다.
          </p>

          {data.items.length === 0 ? (
            <p className="emptyMessage">실익 기준(편차 ≥ 0.3%p)을 통과한 인자가 없습니다.</p>
          ) : (
            <>
              <div className="tableWrap fmeaScrollTable">
                <table className="fmeaTable">
                  <thead>
                    <tr>
                      <th className="fmeaSticky fmeaStickyTarget">타깃</th>
                      <th className="fmeaColScore">S</th>
                      <th className="fmeaSticky fmeaStickyFeature">잠재 원인</th>
                      <th>인자값</th>
                      <th>권장 구간</th>
                      <th className="numCol">이탈률</th>
                      <th className="fmeaColScore">O</th>
                      <th>검출 기법</th>
                      <th className="numCol">계측률</th>
                      <th className="fmeaColScore">D</th>
                      <th className="numCol fmeaColRpn">RPN</th>
                      <th>수율 편차</th>
                      <th>예상 수율</th>
                      <th className="fmeaColAction">권고 조치</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((item) => (
                      <FmeaRow key={`${item.target}-${item.feature}`} item={item} fmea={data} />
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="meScrollHint" aria-hidden="true">← 좌우 스크롤</p>

              <div className="fmeaRuleGrid">
                <div className="fmeaRuleCard">
                  <span className="fmeaRuleLabel">S · 심각도</span>
                  <span className="fmeaRuleWhat">불량 모드 손실 기여율</span>
                  <span className="fmeaRuleFormula">ceil(기여율% / 5)</span>
                </div>
                <div className="fmeaRuleCard">
                  <span className="fmeaRuleLabel">O · 발생도</span>
                  <span className="fmeaRuleWhat">권장 구간 이탈 비율</span>
                  <span className="fmeaRuleFormula">ceil(이탈률% / 10)</span>
                </div>
                <div className="fmeaRuleCard">
                  <span className="fmeaRuleLabel">D · 검출도</span>
                  <span className="fmeaRuleWhat">계측률의 역수</span>
                  <span className="fmeaRuleFormula">ceil((100−계측률%) / 10)</span>
                </div>
                <div className="fmeaRuleCard">
                  <span className="fmeaRuleLabel">RPN</span>
                  <span className="fmeaRuleWhat">위험 우선순위</span>
                  <span className="fmeaRuleFormula">S × O × D</span>
                </div>
              </div>

              <div className="fmeaNotice fmeaNoticeCaution">
                <p>
                  RPN만 보고 판단하지 마세요 — 수율 편차를 함께 봐야 합니다. RPN이 높은 이유는 발생도가 아니라 계측률이
                  낮아 D가 크기 때문일 수 있습니다.
                </p>
                <p>
                  실익 기준(편차 ≥ 0.3%p) 미달 {data.excluded_count}개는 표에서 제외했습니다
                  {data.excluded_negative_count > 0 ? ` (그중 편차가 음수인 인자 ${data.excluded_negative_count}개 포함)` : ""}.
                </p>
              </div>
              <div className="fmeaNotice fmeaNoticeInfo">
                <p>R In-line 샘플 계측(약 15%) · D Defect 검사(약 5%, 사후 선별) · Config 장비/챔버 기록(전수, 결측 0%)</p>
                <p>Config는 D=1로 유리하나 600건 검정 FDR 통과 0건이라 잠재 원인으로 등재하지 않았습니다.</p>
              </div>
              <div className="fmeaNotice fmeaNoticeInfo">
                <p>
                  자동 생성 초안입니다. S·O·D는 관례상 엔지니어가 판단하는 값이나 여기서는 분석 결과로부터 규칙
                  기반으로 산출했습니다. 공정 지식에 따른 검토 없이 확정하지 마세요.
                </p>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
