"use client";

import type { ActionPriorityPayload, FmeaTablePayload } from "@/types/data";

/** 데이터 한계 진단 (모니터링 홈) -- 조치 블록 바로
 * 다음에 둔다("계측을 늘려야 한다"의 근거가 여기 있다). 계측 편향(MNAR)
 * 과 분산 분해를 한 블록에 나란히, 둘 다 상시 표시한다(접지 않는다).
 * 분산 분해 하단의 불량모드별 변동 기여는 fmea(eval 기준)가 아니라
 * actionPriority(항상 train.CSV 실측 기준)에서 온다 -- eval 내장본은
 * Y가 전량 결측이라 값이 100% 모델 예측값이라 이 지표를 왜곡한다. */
export default function DataLimitationDiagnostics({
  fmea,
  actionPriority,
}: {
  fmea: FmeaTablePayload | null;
  actionPriority?: ActionPriorityPayload | null;
}) {
  if (!fmea) return null;
  const mnarRows = [...fmea.mnar_rate_report].sort((a, b) => b.ratio - a.ratio);
  const vd = fmea.variance_decomposition;
  const modeShare = actionPriority?.mode_variance_share ?? null;

  if (mnarRows.length === 0 && !vd) return null;

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">LIMITS</span>
          <h2>데이터 한계 진단</h2>
        </div>
      </div>

      <div className="limitsGrid">
        <div className="limitsPanel">
          <h3 className="actionBlockSubheading">계측 편향 진단 (MNAR)</h3>
          {mnarRows.length === 0 ? (
            <p className="emptyMessage">표본이 부족해 계측 편향을 계산할 수 없습니다.</p>
          ) : (
            <>
              <div className="mnarList">
                {mnarRows.map((row) => {
                  const maxRate = Math.max(row.overall_rate_pct, row.worst_decile_rate_pct, 1);
                  const severe = row.ratio >= 3;
                  return (
                    <div className="mnarRow" key={`${row.target}-${row.feature}`}>
                      <div className="mnarLabel">
                        <span className="mnarFeature">{row.feature}</span>
                        <span className="modeLossTarget">{row.target}</span>
                      </div>
                      <div className="mnarBars">
                        <div className="mnarBarLine">
                          <div className="mnarBarTrack">
                            <div className="mnarBarFill mnarBarFill-overall" style={{ width: `${(row.overall_rate_pct / maxRate) * 100}%` }} />
                          </div>
                          <span className="mnarBarValue">전체 {row.overall_rate_pct.toFixed(1)}%</span>
                        </div>
                        <div className="mnarBarLine">
                          <div className="mnarBarTrack">
                            <div
                              className={`mnarBarFill ${severe ? "mnarBarFill-worst-red" : "mnarBarFill-worst-amber"}`}
                              style={{ width: `${(row.worst_decile_rate_pct / maxRate) * 100}%` }}
                            />
                          </div>
                          <span className="mnarBarValue">최악 10% {row.worst_decile_rate_pct.toFixed(1)}%</span>
                        </div>
                      </div>
                      <span className={`mnarRatio${severe ? " mnarRatio-red" : ""}`}>{row.ratio.toFixed(2)}x</span>
                    </div>
                  );
                })}
              </div>
              <p className="fmeaFootnote">
                계측이 무작위였다면 두 막대의 높이가 같아야 합니다. 최악 10% wafer에서 계측률이 높다는 것은 불량
                징후를 보고 측정을 결정했다는 뜻이며, 이 경우 계측된 wafer만으로 추정한 관계는 전체 wafer로
                일반화되지 않습니다.{" "}
                {mnarRows[0] && `${mnarRows[0].feature}은(는) ${mnarRows[0].ratio.toFixed(2)}배로 `}
                {mnarRows[0] && mnarRows[0].ratio >= 3 ? "사실상 사후 확인 계측에 가깝습니다." : ""}
              </p>
            </>
          )}
        </div>

        <div className="limitsPanel">
          <h3 className="actionBlockSubheading">분산 분해</h3>
          {!vd ? (
            <p className="emptyMessage">랏 정보가 부족해 분산 분해를 계산할 수 없습니다.</p>
          ) : (
            <>
              <div className="varianceDecompList">
                <div className="varianceDecompRow">
                  <span className="varianceDecompLabel">랏 간</span>
                  <div className="varianceDecompTrack">
                    <div className="varianceDecompFill varianceDecompFill-between" style={{ width: `${vd.between_lot_pct}%` }} />
                    <div
                      className="varianceDecompExpectedLine"
                      style={{ left: `${Math.min(100, vd.no_effect_expected_pct)}%` }}
                      title={`무효과 기대값 ${vd.no_effect_expected_pct.toFixed(1)}% (=1/${vd.wafers_per_lot.toFixed(0)})`}
                    />
                  </div>
                  <span className="varianceDecompValue">{vd.between_lot_pct.toFixed(1)}%</span>
                </div>
                <div className="varianceDecompRow">
                  <span className="varianceDecompLabel">랏 내</span>
                  <div className="varianceDecompTrack">
                    <div className="varianceDecompFill varianceDecompFill-within" style={{ width: `${vd.within_lot_pct}%` }} />
                  </div>
                  <span className="varianceDecompValue">{vd.within_lot_pct.toFixed(1)}%</span>
                </div>
              </div>
              <p className="varianceDecompExpectedLabel">
                무효과 기대값 {vd.no_effect_expected_pct.toFixed(1)}% (=1/{vd.wafers_per_lot.toFixed(0)}) — 점선 위치
              </p>
              <p className="fmeaFootnote">
                랏당 {vd.wafers_per_lot.toFixed(0)}장에서는 랏 효과가 없어도 var(랏평균)/var(Y)의 기대값이{" "}
                {vd.no_effect_expected_pct.toFixed(1)}%입니다. 관측값 {vd.between_lot_pct.toFixed(1)}%
                {Math.abs(vd.between_lot_pct - vd.no_effect_expected_pct) <= 2
                  ? "은 그 기대값과 사실상 같으므로 랏 효과의 증거가 아닙니다."
                  : "은 그 기대값과 차이가 있어 랏 효과가 있을 수 있습니다."}{" "}
                ICC(1,1) = {vd.icc.toFixed(3)}. 수율 변동의 {vd.within_lot_pct.toFixed(1)}%가 같은 랏 안에서 wafer마다
                발생합니다. 랏 단위 순위표를 제공하지 않는 이유입니다.
              </p>

              {modeShare && modeShare.length > 0 && (
                <div className="modeShareSection">
                  <div className="modeShareHeading">
                    <h4 className="modeShareTitle">불량모드별 변동 기여</h4>
                    <span className="modeShareBasis">train.CSV 실측 기준</span>
                  </div>

                  <div className="modeShareLegend">
                    {modeShare.map((r, i) => (
                      <span key={r.target} className="modeShareLegendItem">
                        <i className="modeShareSwatch" style={{ background: `var(--mode-share-${i + 1})` }} />
                        {r.target}
                      </span>
                    ))}
                  </div>

                  <div className="modeShareBar">
                    {modeShare.map((r, i) => (
                      <div
                        key={r.target}
                        className="modeShareSeg"
                        style={{ flexGrow: r.variance_share_pct, background: `var(--mode-share-${i + 1})` }}
                        title={`${r.target} · 변동 기여 ${r.variance_share_pct.toFixed(1)}% · 평균 손실 ${r.mean_loss_pp.toFixed(2)}%p (비중 ${r.mean_share_pct.toFixed(1)}%)`}
                      />
                    ))}
                  </div>

                  <div className="modeShareLabels">
                    {modeShare.map((r) => (
                      <div key={r.target} className="modeShareLabel" style={{ flexGrow: r.variance_share_pct }}>
                        {r.variance_share_pct >= 10 ? (
                          <>
                            {r.target} <b>{r.variance_share_pct.toFixed(1)}%</b>
                          </>
                        ) : null}
                      </div>
                    ))}
                  </div>

                  <p className="fmeaFootnote">
                    {modeShare[0].target} 한 모드가 수율 변동의 {modeShare[0].variance_share_pct.toFixed(1)}%를
                    만듭니다. 랏 내 변동 {vd ? `${vd.within_lot_pct.toFixed(1)}%` : ""}가 어느 불량모드에서 오는지를
                    나눈 값이며, 평균 손실 비중이 아니라 <b>웨이퍼별 편차를 만드는 몫</b>입니다.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
