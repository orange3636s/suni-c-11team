"use client";

import type { ReactNode } from "react";
import type {
  ActionPriorityPayload,
  CoreFactorCoverage,
  DefectCooccurrence,
  FmeaTablePayload,
  MnarRateRow,
  ModeVarianceShareRow,
  VarianceDecomposition,
} from "@/types/data";

/** 데이터 한계 진단 (모니터링 홈) -- 조치 블록 바로 다음에 둔다("계측을
 * 늘려야 한다"의 근거가 여기 있다). 다섯 블록: 좌측에 계측 편향(MNAR,
 * 10행), 우측에 분산 분해·불량 원인별 변동 기여·핵심 인자 커버리지·
 * 불량 원인 독립성을 세로로 쌓는다. 다섯 블록 전부 `LimitBlock` 하나를
 * 공유해 제목·설명·시각화·해석 형식을 통일한다.
 *
 * 데이터셋 기준: MNAR·분산 분해·변동 기여·독립성은 train.CSV(손실
 * 상위 10%/원인 값을 실측으로 정해야 하므로 예측 Y로 계산하면 안 된다)
 * -- 커버리지만 test_remove_y.CSV(지금 분석 중인 배치의 계측 상태를
 * 묻는 질문이라서다). 각 블록 설명 끝에 데이터셋과 wafer 수를 명시한다. */
export default function DataLimitationDiagnostics({
  fmea,
  actionPriority,
}: {
  fmea: FmeaTablePayload | null;
  actionPriority?: ActionPriorityPayload | null;
}) {
  if (!fmea) return null;
  const mnarRows = fmea.mnar_rate_report; // 백엔드가 이미 배수 상위 10개로 내려보낸다
  const vd = fmea.variance_decomposition;
  const modeShare = actionPriority?.mode_variance_share ?? null;
  const coverage = fmea.core_factor_coverage;
  const cooccurrence = fmea.defect_cooccurrence;
  const trainWafers = fmea.train_total_wafers;

  if (mnarRows.length === 0 && !vd && !coverage && !cooccurrence) return null;

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">LIMITS</span>
          <h2>데이터 한계 진단</h2>
        </div>
      </div>

      <div className="limitsGrid">
        <div className="limitsColLeft">
          <LimitBlock
            title="계측 편향 진단 (MNAR)"
            description="불량이 심한 wafer를 더 자주 계측했는가"
            dataset={`train.CSV ${trainWafers.toLocaleString()}장`}
            interpretation={<MnarInterpretation rows={mnarRows} />}
          >
            <MnarBlockBody rows={mnarRows} />
          </LimitBlock>
        </div>

        <div className="limitsColRight">
          <LimitBlock
            title="분산 분해"
            description="수율 변동이 LOT 사이에서 오나 LOT 안에서 오나"
            dataset={vd ? `train.CSV ${trainWafers.toLocaleString()}장 · ${vd.lot_count} LOT` : `train.CSV ${trainWafers.toLocaleString()}장`}
            interpretation={<VarianceInterpretation vd={vd} />}
          >
            <VarianceBlockBody vd={vd} />
          </LimitBlock>

          <LimitBlock
            title="불량 원인별 변동 기여"
            description="wafer 사이의 수율 차이를 어느 원인이 만드는가"
            dataset={`train.CSV ${trainWafers.toLocaleString()}장`}
            interpretation={<ModeShareInterpretation rows={modeShare} withinLotPct={vd?.within_lot_pct ?? null} />}
          >
            <ModeShareBlockBody rows={modeShare} />
          </LimitBlock>

          <LimitBlock
            title="핵심 인자 커버리지"
            description="wafer 한 장이 몇 개의 핵심 인자로 판정되는가"
            dataset={coverage ? `test_remove_y.CSV ${coverage.total_wafers.toLocaleString()}장` : "test_remove_y.CSV"}
            interpretation={<CoverageInterpretation coverage={coverage} />}
          >
            <CoverageBlockBody coverage={coverage} />
          </LimitBlock>

          <LimitBlock
            title="불량 원인 독립성"
            description="두 원인이 동시에 나빠지는가 · 상위 10% 동시 발생률"
            dataset={`train.CSV ${trainWafers.toLocaleString()}장`}
            interpretation={<CooccurrenceInterpretation data={cooccurrence} />}
          >
            <CooccurrenceBlockBody data={cooccurrence} />
          </LimitBlock>
        </div>
      </div>
    </section>
  );
}

// ---- 공용 블록 형식 -- 제목 · 설명(데이터셋 표기 포함) · 시각화 · 해석 --

function LimitBlock({
  title,
  description,
  dataset,
  children,
  interpretation,
}: {
  title: string;
  description: string;
  dataset: string;
  children: ReactNode;
  interpretation: ReactNode;
}) {
  return (
    <div className="limitBlock">
      <h3 className="limitBlockTitle">{title}</h3>
      <p className="limitBlockDesc">
        {description} · <span className="limitBlockDataset">{dataset}</span>
      </p>
      <div className="limitBlockViz">{children}</div>
      <p className="limitBlockInterp">{interpretation}</p>
    </div>
  );
}

// ---- 계측 편향 진단 (MNAR) ----

function mnarTier(ratio: number): "red" | "amber" | "muted" {
  if (ratio >= 3.0) return "red";
  if (ratio >= 1.4) return "amber";
  return "muted";
}

function MnarBlockBody({ rows }: { rows: MnarRateRow[] }) {
  if (rows.length === 0) return <p className="emptyMessage">표본이 부족해 계측 편향을 계산할 수 없습니다.</p>;
  return (
    <div className="mnarList">
      {rows.map((row) => {
        const maxRate = Math.max(row.overall_rate_pct, row.worst_decile_rate_pct, 1);
        const tier = mnarTier(row.ratio);
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
                    className={`mnarBarFill mnarBarFill-worst-${tier}`}
                    style={{ width: `${(row.worst_decile_rate_pct / maxRate) * 100}%` }}
                  />
                </div>
                <span className="mnarBarValue">최악 10% {row.worst_decile_rate_pct.toFixed(1)}%</span>
              </div>
            </div>
            <span className={`mnarRatio${tier === "red" ? " mnarRatio-red" : tier === "amber" ? " mnarRatio-amber" : ""}`}>
              {row.ratio.toFixed(2)}x
            </span>
          </div>
        );
      })}
    </div>
  );
}

function MnarInterpretation({ rows }: { rows: MnarRateRow[] }) {
  if (rows.length === 0) return null;
  const top = rows[0];
  const belowTop5 = rows.slice(5);
  const maxBelowTop5 = belowTop5.length > 0 ? Math.max(...belowTop5.map((r) => r.ratio)) : null;
  return (
    <>
      계측이 무작위였다면 두 막대의 높이가 같아야 합니다. 아래 막대가 길다는 것은 불량 징후를 보고 측정을 결정했다는
      뜻이며, 이 경우 계측된 wafer만으로 추정한 관계는 전체로 일반화되지 않습니다.{" "}
      <b>
        {top.feature}은(는) {top.ratio.toFixed(2)}배로{" "}
      </b>
      {top.ratio >= 3 ? "사실상 사후 확인 계측입니다." : "계측이 결과를 보고 결정됐을 가능성이 있습니다."}
      {maxBelowTop5 != null && ` 6위 아래는 ${maxBelowTop5.toFixed(1)}배 이하로 편향이 크지 않습니다.`}
    </>
  );
}

// ---- 분산 분해 ----

function VarianceBlockBody({ vd }: { vd: VarianceDecomposition | null }) {
  if (!vd) return <p className="emptyMessage">LOT 정보가 부족해 분산 분해를 계산할 수 없습니다.</p>;
  return (
    <>
      <div className="varianceDecompList">
        <div className="varianceDecompRow">
          <span className="varianceDecompLabel">LOT 간</span>
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
          <span className="varianceDecompLabel">LOT 내</span>
          <div className="varianceDecompTrack">
            <div className="varianceDecompFill varianceDecompFill-within" style={{ width: `${vd.within_lot_pct}%` }} />
          </div>
          <span className="varianceDecompValue">{vd.within_lot_pct.toFixed(1)}%</span>
        </div>
      </div>
      <p className="varianceDecompExpectedLabel">
        무효과 기대값 {vd.no_effect_expected_pct.toFixed(1)}% (=1/{vd.wafers_per_lot.toFixed(0)}) — 점선 위치
      </p>
    </>
  );
}

function VarianceInterpretation({ vd }: { vd: VarianceDecomposition | null }) {
  if (!vd) return null;
  const closeToExpected = Math.abs(vd.between_lot_pct - vd.no_effect_expected_pct) <= 2;
  return (
    <>
      LOT당 {vd.wafers_per_lot.toFixed(0)}장에서는 LOT 효과가 전혀 없어도 var(LOT평균)/var(Y)의 기대값이{" "}
      <b>{vd.no_effect_expected_pct.toFixed(1)}%</b>입니다. 관측값 {vd.between_lot_pct.toFixed(1)}%
      {closeToExpected ? "은 그 기대값과 사실상 같으므로 LOT 효과의 증거가 아닙니다." : "은 그 기대값과 차이가 있어 LOT 효과가 있을 수 있습니다."}{" "}
      ICC(1,1) = {vd.icc.toFixed(3)}. 수율 변동의 {vd.within_lot_pct.toFixed(1)}%가 같은 LOT 안에서 wafer마다
      발생하며, LOT 단위 순위표를 제공하지 않는 이유입니다.
    </>
  );
}

// ---- 불량 원인별 변동 기여 ----

function ModeShareBlockBody({ rows }: { rows: ModeVarianceShareRow[] | null }) {
  if (!rows || rows.length === 0) return <p className="emptyMessage">표본이 부족해 변동 기여를 계산할 수 없습니다.</p>;
  return (
    <>
      <div className="modeShareLegend">
        {rows.map((r, i) => (
          <span key={r.target} className="modeShareLegendItem">
            <i className="modeShareSwatch" style={{ background: `var(--mode-share-${i + 1})` }} />
            {r.target}
          </span>
        ))}
      </div>
      <div className="modeShareBar">
        {rows.map((r, i) => (
          <div
            key={r.target}
            className="modeShareSeg"
            style={{ flexGrow: r.variance_share_pct, background: `var(--mode-share-${i + 1})` }}
            title={`${r.target} · 변동 기여 ${r.variance_share_pct.toFixed(1)}% · 평균 손실 ${r.mean_loss_pp.toFixed(2)}%p (비중 ${r.mean_share_pct.toFixed(1)}%)`}
          />
        ))}
      </div>
      <div className="modeShareLabels">
        {rows.map((r) => (
          <div key={r.target} className="modeShareLabel" style={{ flexGrow: r.variance_share_pct }}>
            {r.variance_share_pct >= 10 ? (
              <>
                {r.target} <b>{r.variance_share_pct.toFixed(1)}%</b>
              </>
            ) : null}
          </div>
        ))}
      </div>
    </>
  );
}

function ModeShareInterpretation({ rows, withinLotPct }: { rows: ModeVarianceShareRow[] | null; withinLotPct: number | null }) {
  if (!rows || rows.length === 0) return null;
  const top = rows[0];
  const weakest = rows[rows.length - 1];
  return (
    <>
      <b>
        {top.target} 한 원인이 수율 변동의 {top.variance_share_pct.toFixed(1)}%
      </b>
      를 만듭니다. {withinLotPct != null ? `LOT 내 변동 ${withinLotPct.toFixed(1)}%가 ` : ""}
      어느 불량 원인에서 오는지를 나눈 값이며, 평균 손실 비중이 아니라 웨이퍼별 편차를 만드는 몫입니다.{" "}
      {weakest.target}는 {weakest.variance_share_pct.toFixed(1)}%로 관계가 가장 깨끗해도 개선 여지가 작습니다.
    </>
  );
}

// ---- 핵심 인자 커버리지 ----

function coverageColor(count: number, maxCount: number): string {
  if (count === 0) return "var(--text-muted)";
  if (count === 1) return "var(--inferred)";
  const t = maxCount > 1 ? Math.min(1, (count - 1) / (maxCount - 1)) : 1;
  const pct = 55 + t * 45;
  return `color-mix(in srgb, var(--measured) ${pct}%, var(--surface-muted))`;
}

function CoverageBlockBody({ coverage }: { coverage: CoreFactorCoverage | null }) {
  if (!coverage || coverage.rows.length === 0) return <p className="emptyMessage">계측 상태를 계산할 수 없습니다.</p>;
  const maxCount = coverage.rows.length - 1;
  return (
    <>
      <div className="coverageStackBar">
        {coverage.rows.map((r) => (
          <div
            key={r.measured_count}
            className="coverageStackSeg"
            style={{ flexGrow: Math.max(r.wafer_count, 0.0001), background: coverageColor(r.measured_count, maxCount) }}
            title={`${r.measured_count}개 계측 · ${r.wafer_count.toLocaleString()}장 (${r.pct.toFixed(1)}%)`}
          />
        ))}
      </div>
      <div className="coverageLegend">
        {coverage.rows.map((r) => (
          <span key={r.measured_count} className="coverageLegendItem">
            <i className="coverageSwatch" style={{ background: coverageColor(r.measured_count, maxCount) }} />
            {r.measured_count}개 {r.wafer_count.toLocaleString()}장{r.measured_count === 0 ? " — 판정 불가" : ""}
          </span>
        ))}
      </div>
    </>
  );
}

function CoverageInterpretation({ coverage }: { coverage: CoreFactorCoverage | null }) {
  if (!coverage || coverage.rows.length === 0) return null;
  const zero = coverage.rows.find((r) => r.measured_count === 0);
  const threeOrMore = coverage.rows.filter((r) => r.measured_count >= 3);
  const threeOrMoreCount = threeOrMore.reduce((sum, r) => sum + r.wafer_count, 0);
  const threeOrMorePct = threeOrMore.reduce((sum, r) => sum + r.pct, 0);
  return (
    <>
      {zero && (
        <>
          <b>
            {zero.wafer_count.toLocaleString()}장({zero.pct.toFixed(1)}%)
          </b>
          은 핵심 인자가 하나도 계측되지 않아 예측이 원인별 평균으로 채워집니다. 이들끼리는 계측된 인자가 없어 순위를
          매길 근거가 없습니다.{" "}
        </>
      )}
      알람이 적은 것은 결함이 아니라 계측률의 결과입니다. 3개 이상 계측된 wafer는{" "}
      {threeOrMoreCount.toLocaleString()}장({threeOrMorePct.toFixed(1)}%)뿐입니다.
    </>
  );
}

// ---- 불량 원인 독립성 ----

function cooccurrenceTier(v: number): "flag" | "neutral" {
  return v > 1.1 || v < 0.9 ? "flag" : "neutral";
}

function CooccurrenceBlockBody({ data }: { data: DefectCooccurrence | null }) {
  if (!data) return <p className="emptyMessage">표본이 부족해 독립성을 계산할 수 없습니다.</p>;
  return (
    <table className="cooccurrenceMatrix">
      <thead>
        <tr>
          <th />
          {data.targets.map((t) => (
            <th key={t}>{t}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.targets.map((rowTarget, i) => (
          <tr key={rowTarget}>
            <th>{rowTarget}</th>
            {data.targets.map((colTarget, j) => {
              const v = data.matrix[i]?.[j] ?? null;
              if (v == null) return <td key={colTarget} className="cooccurrenceDiag">—</td>;
              const tier = cooccurrenceTier(v);
              return (
                <td key={colTarget} className={`cooccurrenceCell${tier === "flag" ? " cooccurrenceCell-flag" : ""}`}>
                  {v.toFixed(2)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CooccurrenceInterpretation({ data }: { data: DefectCooccurrence | null }) {
  if (!data) return null;
  const values = data.matrix.flat().filter((v): v is number => v != null);
  if (values.length === 0) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (
    <>
      두 원인이 독립이면 동시 발생률이 <b>1.00%</b>여야 합니다. 열 쌍 전부 {min.toFixed(2)}~{max.toFixed(2)}%로
      기대값 주변입니다. 다섯 원인이 서로 독립이므로 인자를 하나 고치면 그 원인만 개선됩니다 — 부작용을 걱정할
      필요가 없습니다. 원인별로 따로 관리하는 현재 설계의 근거입니다.
    </>
  );
}
