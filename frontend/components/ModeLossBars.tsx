"use client";

import type { YieldSummary } from "@/types/data";

/** 모드별 손실 막대 (모니터링 홈, 작업 지시서 WC) -- 어느 불량(타깃)이
 * 손실을 주도하는지 손실 큰 순으로 다섯 줄 보여준다. 1위만 강조색,
 * 나머지는 무채색 -- "1위가 무엇인지"가 핵심이지 다섯 개를 똑같이 강조하면
 * 무엇도 강조되지 않는다. */
export default function ModeLossBars({ summary }: { summary: YieldSummary | null }) {
  if (!summary || summary.mode_loss.length === 0) return null;

  const maxLoss = Math.max(...summary.mode_loss.map((m) => m.avg_loss_pct), 0.0001);
  const top = summary.mode_loss[0];
  const gapText =
    top.train_avg_loss_pct != null
      ? Math.abs(top.avg_loss_pct - top.train_avg_loss_pct) <= 0.5
        ? "모델이 분포를 재현하고 있습니다"
        : "학습 데이터와 분석 데이터의 분포가 다를 수 있습니다"
      : null;

  return (
    <section className="resultCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">손실 원인</span>
          <h2>모드별 손실</h2>
        </div>
      </div>
      <p className="sectionCaption">어느 불량이 손실을 주도하는지 손실 큰 순</p>

      <div className="modeLossList">
        {summary.mode_loss.map((mode, index) => {
          const isTop = index === 0;
          const widthPct = (mode.avg_loss_pct / maxLoss) * 100;
          return (
            <div className="modeLossRow" key={mode.target}>
              <div className="modeLossLabel">
                <span className="modeLossTarget">{mode.target}</span>
                {mode.feature && <span className="modeLossFeature">{mode.feature}</span>}
              </div>
              <div className="modeLossBarTrack">
                <div
                  className={`modeLossBarFill${isTop ? " modeLossBarFill-top" : ""}`}
                  style={{ width: `${Math.max(2, widthPct)}%` }}
                />
              </div>
              <div className="modeLossFigures">
                <span className="modeLossPp">{mode.avg_loss_pct.toFixed(2)}%p</span>
                <span className="modeLossContribution">{mode.contribution_pct.toFixed(1)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {gapText && (
        <p className="sectionCaption">
          {top.target} 예측 평균 {top.avg_loss_pct.toFixed(2)}%p
          {top.train_avg_loss_pct != null && ` · 학습 실측 ${top.train_avg_loss_pct.toFixed(2)}%p`} — {gapText}
        </p>
      )}
    </section>
  );
}
