"use client";

import type { YieldSummary } from "@/types/data";

/** 상단 요약 카드 4개 (모니터링 홈, 작업 지시서 WB) -- 목표 수율 개념을
 * 전면 폐기하고 대신 "지금 상황이 어떤가"를 네 개의 서로 다른 질문으로
 * 답한다: 평균은 현황, 하위 10장은 조치 대상의 심각도, 판정 가능·불가는
 * 이 화면을 얼마나 믿을 수 있는지. 목표 대비 갭·상태 배지는 넣지 않는다.
 */
export default function SummaryCards({ summary }: { summary: YieldSummary | null }) {
  if (!summary) {
    return (
      <section className="resultCard panel-primary">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">판단 요약</span>
            <h2>공정 현황 요약</h2>
          </div>
        </div>
        <p className="sectionCaption">예측 없음 — 원인 분석을 실행하면 요약 카드가 표시됩니다.</p>
      </section>
    );
  }

  const notJudgeable = summary.total_wafers - summary.judgeable_count;

  return (
    <section className="resultCard panel-primary">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">판단 요약</span>
          <h2>공정 현황 요약</h2>
        </div>
      </div>
      <div className="summaryCardsGrid">
        <div className="summaryCard">
          <span className="summaryCardLabel">예측 수율 평균</span>
          <strong className="summaryCardValue">{summary.predicted_mean.toFixed(2)}%</strong>
          <span className="summaryCardSub">
            최저 {summary.predicted_min.toFixed(2)} · 최고 {summary.predicted_max.toFixed(2)}
          </span>
        </div>
        <div className="summaryCard">
          <span className="summaryCardLabel">하위 {summary.bottom_n}장 평균</span>
          <strong className="summaryCardValue summaryCardValue-amber">
            {summary.bottom_mean != null ? `${summary.bottom_mean.toFixed(2)}%` : "-"}
          </strong>
          <span className="summaryCardSub">
            {summary.bottom_mean != null
              ? `전체 평균 대비 ${(summary.bottom_mean - summary.predicted_mean).toFixed(1)}%p`
              : `표본이 ${summary.bottom_n}장 미만입니다`}
          </span>
        </div>
        <div className="summaryCard">
          <span className="summaryCardLabel">판정 가능</span>
          <strong className="summaryCardValue">
            {summary.judgeable_count.toLocaleString()} / {summary.total_wafers.toLocaleString()}
          </strong>
          <span className="summaryCardSub">핵심 인자 1개 이상 계측</span>
        </div>
        <div className="summaryCard">
          <span className="summaryCardLabel">판정 불가</span>
          <strong className="summaryCardValue summaryCardValue-muted">{notJudgeable.toLocaleString()}</strong>
          <span className="summaryCardSub">계측 확대로 해소 가능</span>
        </div>
      </div>
    </section>
  );
}
