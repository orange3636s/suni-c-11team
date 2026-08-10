"use client";

import type { YieldSummary } from "@/types/data";

/** 수율 분포 히스토그램 (모니터링 홈, 작업 지시서 WD) -- 판정 가능/미계측
 * 두 켜로 쌓는다(필수). 미계측을 함께 쌓지 않으면 이 그림은 왜곡이다 --
 * 핵심 인자가 없어 값이 한 구간에 몰려도 공정 분포로 오독된다. */
export default function YieldHistogram({ summary }: { summary: YieldSummary | null }) {
  if (!summary || summary.histogram.length === 0) return null;

  const maxCount = Math.max(...summary.histogram.map((b) => b.judgeable_count + b.not_judgeable_count), 1);
  const totalNotJudgeable = summary.histogram.reduce((sum, b) => sum + b.not_judgeable_count, 0);
  const dominantBin = summary.histogram.reduce(
    (best, bin) => (bin.not_judgeable_count > (best?.not_judgeable_count ?? -1) ? bin : best),
    null as YieldSummary["histogram"][number] | null,
  );

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">수율 분포</span>
          <h2>판정 가능 vs 미계측</h2>
        </div>
      </div>
      <p className="sectionCaption">판정 가능(실색) · 미계측(빗금) 스택 — 등간격이 아니라 관심 구간을 좁게 잡은 구간입니다</p>

      <div className="yieldHistogram">
        {summary.histogram.map((bin) => {
          const total = bin.judgeable_count + bin.not_judgeable_count;
          const judgeableHeight = (bin.judgeable_count / maxCount) * 100;
          const notJudgeableHeight = (bin.not_judgeable_count / maxCount) * 100;
          return (
            <div className="yieldHistogramCol" key={bin.label}>
              <div className="yieldHistogramBarStack" title={`${bin.label}: 판정 가능 ${bin.judgeable_count} · 미계측 ${bin.not_judgeable_count}`}>
                {bin.not_judgeable_count > 0 && (
                  <div className="yieldHistogramSeg yieldHistogramSeg-notJudgeable" style={{ height: `${notJudgeableHeight}%` }} />
                )}
                {bin.judgeable_count > 0 && (
                  <div className="yieldHistogramSeg yieldHistogramSeg-judgeable" style={{ height: `${judgeableHeight}%` }} />
                )}
                {total === 0 && <div className="yieldHistogramSeg yieldHistogramSeg-empty" />}
              </div>
              <span className="yieldHistogramCount">{total.toLocaleString()}</span>
              <span className="yieldHistogramLabel">{bin.label}</span>
            </div>
          );
        })}
      </div>

      <p className="sectionCaption">
        {dominantBin && totalNotJudgeable > 0
          ? `${dominantBin.label} 구간의 ${(dominantBin.judgeable_count + dominantBin.not_judgeable_count).toLocaleString()}장 중 ` +
            `${dominantBin.not_judgeable_count.toLocaleString()}장은 핵심 인자 미계측으로 평균값이 대입된 wafer입니다. ` +
            `실제 분포가 아니라 결측 구조를 반영합니다.`
          : `현재 판정 불가(핵심 인자 미계측) wafer가 없어 이 분포는 결측 구조 없이 실측 그대로입니다.`}
      </p>
    </section>
  );
}
