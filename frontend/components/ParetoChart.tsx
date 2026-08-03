"use client";

import { useState } from "react";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ParetoRankingItem } from "@/types/data";

export type ParetoCountMode = "10" | "20" | "all";

type TooltipState = { x: number; y: number; item: ParetoRankingItem } | null;

function formatQ(q: number): string {
  return q < 0.001 ? q.toExponential(2) : q.toFixed(4);
}

export default function ParetoChart({
  target,
  items,
  countMode,
  onCountModeChange,
  onBarClick,
  activeFeature,
}: {
  target: string;
  items: ParetoRankingItem[];
  countMode: ParetoCountMode;
  onCountModeChange: (mode: ParetoCountMode) => void;
  onBarClick: (item: ParetoRankingItem) => void;
  activeFeature?: string | null;
}) {
  const theme = useResolvedTheme();
  const [tooltip, setTooltip] = useState<TooltipState>(null);

  if (items.length === 0) return null;

  const limit = countMode === "10" ? 10 : countMode === "20" ? 20 : items.length;
  const visible = items.slice(0, Math.min(limit, items.length));
  const n = visible.length;
  // Where cumulative contribution first reaches 80%, across the FULL pool
  // (not just what's currently shown) -- Y4 never reaches this within its
  // top 10/20, and that must show as a caption, not a fabricated divider.
  const crossIndex = items.findIndex((item) => item.cumulative_pct >= 80);
  const crossVisible = crossIndex >= 0 && crossIndex < n;

  const lineColor = theme === "dark" ? "#FF8B3D" : "#F76808";
  const thresholdColor = theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)";

  const xCenter = (i: number) => ((i + 0.5) / n) * 100;
  const yFor = (pct: number) => 100 - pct;
  const linePoints = visible.map((item, i) => `${xCenter(i)},${yFor(item.cumulative_pct)}`).join(" ");
  const thresholdY = yFor(80);
  const dividerX = crossVisible ? ((crossIndex + 1) / n) * 100 : null;

  return (
    <section className="resultCard paretoChartCard">
      <div className="paretoChartHeader">
        <div>
          <span className="sectionLabel">PARETO</span>
          <h2>{target} 상관 인자 기여도</h2>
        </div>
        <div className="paretoCountToggle" role="tablist" aria-label="표시 개수">
          {(["10", "20", "all"] as ParetoCountMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              role="tab"
              aria-selected={countMode === mode}
              className={countMode === mode ? "active" : ""}
              onClick={() => onCountModeChange(mode)}
            >
              {mode === "all" ? "전체" : `상위 ${mode}개`}
            </button>
          ))}
        </div>
      </div>

      <div className="paretoChartBody">
        <span className="paretoAxisLabel paretoAxisLabel-left">기여율 (%)</span>
        <div className="paretoPlotArea">
          {visible.map((item, i) => {
            const emphasized = crossVisible ? i <= crossIndex : true;
            return (
              <button
                key={item.feature}
                type="button"
                className={[
                  "paretoBar",
                  emphasized ? "emphasized" : "muted",
                  item.significant ? "significant" : "insignificant",
                  activeFeature === item.feature ? "active" : "",
                ].join(" ").trim()}
                style={{ left: `${(i / n) * 100}%`, width: `${(1 / n) * 100}%`, height: `${Math.max(item.contribution_pct, 0.5)}%` }}
                onMouseEnter={(event) => setTooltip({ x: event.clientX, y: event.clientY, item })}
                onMouseMove={(event) => setTooltip({ x: event.clientX, y: event.clientY, item })}
                onMouseLeave={() => setTooltip(null)}
                onClick={() => onBarClick(item)}
                aria-label={`${item.feature}, 기여율 ${item.contribution_pct.toFixed(1)}%${item.significant ? "" : " (통계적으로 유의하지 않음)"}`}
              />
            );
          })}
          <svg className="paretoOverlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <line x1="0" y1={thresholdY} x2="100" y2={thresholdY} className="paretoThresholdLine" stroke={thresholdColor} />
            {dividerX != null && (
              <line x1={dividerX} y1="0" x2={dividerX} y2="100" className="paretoDividerLine" />
            )}
            <polyline points={linePoints} className="paretoCumulativeLine" stroke={lineColor} fill="none" />
            {visible.map((item, i) => (
              <circle key={item.feature} cx={xCenter(i)} cy={yFor(item.cumulative_pct)} r="1.3" fill={lineColor} />
            ))}
          </svg>
        </div>
        <span className="paretoAxisLabel paretoAxisLabel-right">누적 기여율 (%)</span>
      </div>

      <div className="paretoFeatureLabels">
        {visible.map((item) => (
          <span key={item.feature} className="paretoFeatureLabel" style={{ width: `${(1 / n) * 100}%` }} title={item.feature}>
            {item.feature}
          </span>
        ))}
      </div>
      <p className="paretoAxisTitle">상관 인자</p>

      {crossVisible ? (
        <p className="paretoCaption">상위 {crossIndex + 1}개 인자가 전체 기여도의 80%</p>
      ) : (
        <p className="paretoCaption">
          상위 {n}개 누적 {visible[n - 1].cumulative_pct.toFixed(1)}% — 80% 도달에 {crossIndex + 1}개 필요
        </p>
      )}

      <div className="paretoLegend">
        <span><i className="paretoLegendSwatch significant" aria-hidden="true" /> FDR 통과</span>
        <span><i className="paretoLegendSwatch insignificant" aria-hidden="true" /> 통계적 유의성 미달</span>
      </div>

      {tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.item.feature}</strong>
          <div className="heatmapTooltipRow"><span>ε²</span><b>{tooltip.item.eps2.toFixed(3)}</b></div>
          <div className="heatmapTooltipRow"><span>기여율</span><b>{tooltip.item.contribution_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>누적</span><b>{tooltip.item.cumulative_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltip.item.n_observed.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>q</span><b>{formatQ(tooltip.item.q_value)}</b></div>
          <div className="heatmapTooltipRow"><span>FDR 통과</span><b>{tooltip.item.significant ? "예" : "아니오"}</b></div>
        </div>
      )}
    </section>
  );
}
