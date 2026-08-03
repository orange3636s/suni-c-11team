"use client";

import { useRef, useState } from "react";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ParetoRankingItem } from "@/types/data";

export type ParetoCountMode = "10" | "20" | "all";

const BAR_WIDTH = 48;
const BAR_GAP = 12;
const BAR_SLOT = BAR_WIDTH + BAR_GAP;
const LEFT_TICKS = [0, 25, 50, 75, 100];
const RIGHT_TICKS = [0, 20, 40, 60, 80, 100];

type TooltipState = { x: number; y: number; index: number } | null;

function formatP(p: number): string {
  return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
}

const TIER_LABEL: Record<string, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };

function shortenFeatureName(feature: string, maxLength = 10): string {
  return feature.length > maxLength ? `${feature.slice(0, maxLength - 1)}…` : feature;
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
  const plotRef = useRef<HTMLDivElement>(null);

  if (items.length === 0) return null;

  const limit = countMode === "10" ? 10 : countMode === "20" ? 20 : items.length;
  const visible = items.slice(0, Math.min(limit, items.length));
  const n = visible.length;
  const crossIndex = items.findIndex((item) => item.cumulative_pct >= 80);
  const crossVisible = crossIndex >= 0 && crossIndex < n;

  const lineColor = theme === "dark" ? "#F87171" : "#DC2626";
  const thresholdColor = theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)";
  const markerFill = theme === "dark" ? "#2C2C2E" : "#FFFFFF";

  const plotWidth = n * BAR_SLOT;
  const xCenter = (i: number) => i * BAR_SLOT + BAR_WIDTH / 2;
  const yFor = (pct: number) => 100 - pct;
  const linePoints = visible.map((item, i) => `${xCenter(i)},${yFor(item.cumulative_pct)}`).join(" ");
  const thresholdY = yFor(80);
  const dividerX = crossVisible ? crossIndex * BAR_SLOT + BAR_SLOT / 2 - BAR_GAP / 2 : null;

  function handlePlotMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relativeX = event.clientX - rect.left;
    const index = Math.min(n - 1, Math.max(0, Math.round(relativeX / BAR_SLOT - 0.5)));
    setTooltip({ x: event.clientX, y: event.clientY, index });
  }

  const tooltipItem = tooltip ? visible[tooltip.index] : null;

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
        <div className="paretoAxisCol paretoAxisCol-left">
          <span className="paretoAxisLabel">기여율 (%)</span>
          <div className="paretoTickCol">
            {LEFT_TICKS.map((tick) => (
              <span key={tick} className="paretoTick" style={{ bottom: `${tick}%` }}>{tick}</span>
            ))}
          </div>
        </div>

        <div className="paretoPlotScroll">
          <div
            ref={plotRef}
            className="paretoPlotArea"
            style={{ width: plotWidth }}
            onMouseMove={handlePlotMouseMove}
            onMouseLeave={() => setTooltip(null)}
          >
            {visible.map((item, i) => {
              const emphasized = crossVisible ? i <= crossIndex : true;
              return (
                <button
                  key={item.feature}
                  type="button"
                  className={[
                    "paretoBar",
                    emphasized ? "emphasized" : "muted",
                    `tier-${item.confidence_tier}`,
                    activeFeature === item.feature ? "active" : "",
                  ].join(" ").trim()}
                  style={{ left: i * BAR_SLOT, width: BAR_WIDTH, height: `${Math.max(item.contribution_pct, 0.5)}%` }}
                  onClick={() => onBarClick(item)}
                  aria-label={`${item.feature}, 기여율 ${item.contribution_pct.toFixed(1)}%, 신뢰도 ${TIER_LABEL[item.confidence_tier]}`}
                />
              );
            })}
            <svg
              className="paretoOverlay"
              viewBox={`0 0 ${plotWidth} 100`}
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <line x1="0" y1={thresholdY} x2={plotWidth} y2={thresholdY} className="paretoThresholdLine" stroke={thresholdColor} vectorEffect="non-scaling-stroke" />
              {dividerX != null && (
                <line x1={dividerX} y1="0" x2={dividerX} y2="100" className="paretoDividerLine" vectorEffect="non-scaling-stroke" />
              )}
              <polyline points={linePoints} className="paretoCumulativeLine" stroke={lineColor} fill="none" vectorEffect="non-scaling-stroke" />
              {visible.map((item, i) => (
                <circle
                  key={item.feature}
                  cx={xCenter(i)}
                  cy={yFor(item.cumulative_pct)}
                  r="3"
                  fill={markerFill}
                  stroke={lineColor}
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {tooltip && (
                <line
                  x1={xCenter(tooltip.index)} y1="0" x2={xCenter(tooltip.index)} y2="100"
                  className="paretoGuideLine" vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
            <span className="paretoThresholdLabel" style={{ top: `${thresholdY}%` }}>80%</span>
          </div>

          <div className="paretoFeatureLabels" style={{ width: plotWidth }}>
            {visible.map((item) => (
              <span
                key={item.feature}
                className="paretoFeatureLabel"
                style={{ width: BAR_SLOT }}
                title={item.feature}
              >
                {shortenFeatureName(item.feature)}
              </span>
            ))}
          </div>
          <p className="paretoAxisTitle" style={{ width: plotWidth }}>상관 인자</p>
        </div>

        <div className="paretoAxisCol paretoAxisCol-right">
          <div className="paretoTickCol">
            {RIGHT_TICKS.map((tick) => (
              <span key={tick} className="paretoTick" style={{ bottom: `${tick}%` }}>{tick}</span>
            ))}
          </div>
          <span className="paretoAxisLabel paretoAxisLabel-right">누적 기여율 (%)</span>
        </div>
      </div>

      {crossVisible ? (
        <p className="paretoCaption">상위 {crossIndex + 1}개 인자가 전체 기여도의 80%</p>
      ) : (
        <p className="paretoCaption">
          상위 {n}개 누적 {visible[n - 1].cumulative_pct.toFixed(1)}% — 80% 도달에 {crossIndex + 1}개 필요
        </p>
      )}

      <div className="paretoLegend">
        <span><i className="paretoLegendSwatch tier-strong" /> 강함 (p&lt;0.01)</span>
        <span><i className="paretoLegendSwatch tier-moderate" /> 보통 (p&lt;0.05)</span>
        <span><i className="paretoLegendSwatch tier-weak" /> 약함 (p&lt;0.20)</span>
        <span><i className="paretoLegendSwatch tier-reference" /> 참고 (p≥0.20)</span>
      </div>

      {tooltipItem && tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltipItem.feature}</strong>
          <div className="heatmapTooltipRow"><span>기여율</span><b>{tooltipItem.contribution_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>누적 기여율</span><b>{tooltipItem.cumulative_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>ε²</span><b>{tooltipItem.eps2.toFixed(3)}</b></div>
          <div className="heatmapTooltipRow"><span>p값</span><b>{formatP(tooltipItem.p_value)}</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltipItem.n_observed.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>신뢰도</span><b>{TIER_LABEL[tooltipItem.confidence_tier]}</b></div>
        </div>
      )}
    </section>
  );
}
