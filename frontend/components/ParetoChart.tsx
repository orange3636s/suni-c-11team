"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ParetoRankingItem } from "@/types/data";

const MAX_BAR = 48;
const MIN_BAR = 28;
const PADDING = 56;
const PLOT_HEIGHT = 360;
const LABEL_HEIGHT = 84;
const LABEL_MIN_GAP = 4; // px of breathing room required between adjacent horizontal labels
const LEFT_TICKS = [0, 25, 50, 75, 100];
const RIGHT_TICKS = [0, 20, 40, 60, 80, 100];
const LABEL_FONT = "10px var(--font-ui, system-ui, sans-serif)";

const TIER_LABEL: Record<string, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };

type TooltipState = { x: number; y: number; index: number } | null;

function formatP(p: number): string {
  return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
}

function shortenFeatureName(feature: string, maxLength = 11): string {
  return feature.length > maxLength ? `${feature.slice(0, maxLength - 1)}…` : feature;
}

let measureCanvas: HTMLCanvasElement | null = null;
function measureTextWidth(text: string): number {
  if (typeof document === "undefined") return text.length * 6;
  if (!measureCanvas) measureCanvas = document.createElement("canvas");
  const ctx = measureCanvas.getContext("2d");
  if (!ctx) return text.length * 6;
  ctx.font = LABEL_FONT;
  return ctx.measureText(text).width;
}

function computeBarLayout(n: number, containerWidth: number) {
  for (let barWidth = MAX_BAR; barWidth >= MIN_BAR; barWidth -= 1) {
    const gap = Math.max(16, barWidth * 0.4);
    const plotWidth = PADDING * 2 + n * barWidth + (n - 1) * gap;
    if (plotWidth <= containerWidth || barWidth === MIN_BAR) {
      return { barWidth, gap, plotWidth, slot: barWidth + gap };
    }
  }
  const gap = Math.max(16, MIN_BAR * 0.4);
  const plotWidth = PADDING * 2 + n * MIN_BAR + (n - 1) * gap;
  return { barWidth: MIN_BAR, gap, plotWidth, slot: MIN_BAR + gap };
}

function useContainerWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    observer.observe(el);
    setWidth(el.getBoundingClientRect().width || 600);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

export default function ParetoChart({
  target,
  items,
  n80,
  onBarClick,
  activeFeature,
}: {
  target: string;
  items: ParetoRankingItem[];
  n80: number | null;
  onBarClick: (item: ParetoRankingItem) => void;
  activeFeature?: string | null;
}) {
  const theme = useResolvedTheme();
  const [containerRef, containerWidth] = useContainerWidth();
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  const n = items.length;
  const reached80 = items.length > 0 && items[items.length - 1].cumulative_pct >= 80;
  const lastPct = items.length > 0 ? items[items.length - 1].cumulative_pct : 0;

  const layout = useMemo(() => computeBarLayout(Math.max(n, 1), containerWidth), [n, containerWidth]);

  const rotateLabels = useMemo(() => {
    if (n < 2) return false;
    const widths = items.map((item) => measureTextWidth(shortenFeatureName(item.feature)));
    return widths.some((w) => w + LABEL_MIN_GAP > layout.slot);
  }, [items, layout.slot, n]);

  const lineColor = theme === "dark" ? "#F87171" : "#DC2626";
  const thresholdColor = theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)";
  const markerFill = theme === "dark" ? "#2C2C2E" : "#FFFFFF";
  const rightAxisColor = lineColor;

  if (n === 0) return null;

  const xCenter = (i: number) => PADDING + i * layout.slot + layout.barWidth / 2;
  const xLeft = (i: number) => PADDING + i * layout.slot;
  const yFor = (pct: number) => PLOT_HEIGHT * (1 - Math.min(pct, 100) / 100);

  function handlePlotMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relativeX = event.clientX - rect.left - PADDING;
    const index = Math.min(n - 1, Math.max(0, Math.round(relativeX / layout.slot - 0.5)));
    setTooltip({ x: event.clientX, y: event.clientY, index });
  }

  const tooltipItem = tooltip ? items[tooltip.index] : null;

  return (
    <section className="resultCard paretoChartCard">
      <div className="paretoChartHeader">
        <div>
          <span className="sectionLabel">PARETO</span>
          <h2>{target} 상관 인자 기여도</h2>
        </div>
        <div className="paretoLegend paretoLegendInline">
          <span><i className="paretoLegendSwatch tier-strong" /> 강함</span>
          <span><i className="paretoLegendSwatch tier-moderate" /> 보통</span>
          <span><i className="paretoLegendSwatch tier-weak" /> 약함</span>
          <span><i className="paretoLegendSwatch tier-reference" /> 참고</span>
        </div>
      </div>

      <div className="paretoChartBody" ref={containerRef}>
        <div className="paretoAxisCol paretoAxisCol-left" style={{ height: PLOT_HEIGHT }}>
          <span className="paretoAxisLabel">기여율 (%)</span>
          <div className="paretoTickCol" style={{ height: PLOT_HEIGHT }}>
            {LEFT_TICKS.map((tick) => (
              <span key={tick} className="paretoTick" style={{ bottom: `${tick}%` }}>{tick}</span>
            ))}
          </div>
        </div>

        <div className="paretoPlotScroll">
          <div className="paretoPlotCenterer" style={{ minWidth: layout.plotWidth }}>
            <div
              ref={plotRef}
              className="paretoPlotArea"
              style={{ width: layout.plotWidth, height: PLOT_HEIGHT }}
              onMouseMove={n > 1 ? handlePlotMouseMove : undefined}
              onMouseLeave={() => setTooltip(null)}
            >
              <svg
                className="paretoOverlay"
                width={layout.plotWidth}
                height={PLOT_HEIGHT}
                aria-hidden="true"
              >
                <line
                  x1={0} y1={yFor(80)} x2={layout.plotWidth} y2={yFor(80)}
                  className="paretoThresholdLine" stroke={thresholdColor}
                />
                <text x={layout.plotWidth - 4} y={yFor(80) - 4} textAnchor="end" className="paretoThresholdLabel">80%</text>

                {items.map((item, i) => {
                  const barHeight = Math.max((item.contribution_pct / 100) * PLOT_HEIGHT, 2);
                  return (
                    <rect
                      key={item.feature}
                      className={["paretoBar", `tier-${item.confidence_tier}`, activeFeature === item.feature ? "active" : ""].join(" ")}
                      x={xLeft(i)}
                      y={PLOT_HEIGHT - barHeight}
                      width={layout.barWidth}
                      height={barHeight}
                      rx={4}
                      ry={4}
                      onClick={() => onBarClick(item)}
                      style={{ cursor: "pointer" }}
                    >
                      <title>{`${item.feature}, 기여율 ${item.contribution_pct.toFixed(1)}%, 신뢰도 ${TIER_LABEL[item.confidence_tier]}`}</title>
                    </rect>
                  );
                })}

                {n > 1 && (
                  <polyline
                    points={items.map((item, i) => `${xCenter(i)},${yFor(item.cumulative_pct)}`).join(" ")}
                    className="paretoCumulativeLine"
                    stroke={lineColor}
                    fill="none"
                  />
                )}
                {n > 1 &&
                  items.map((item, i) => (
                    <circle
                      key={item.feature}
                      cx={xCenter(i)}
                      cy={yFor(item.cumulative_pct)}
                      r={3}
                      fill={markerFill}
                      stroke={lineColor}
                      strokeWidth={1.5}
                    />
                  ))}

                {n === 1 && (
                  <SingleBarMarker
                    x={xCenter(0)}
                    y={yFor(items[0].cumulative_pct)}
                    plotWidth={layout.plotWidth}
                    pct={items[0].cumulative_pct}
                    lineColor={lineColor}
                    markerFill={markerFill}
                  />
                )}

                {tooltip && n > 1 && (
                  <line
                    x1={xCenter(tooltip.index)} y1={0} x2={xCenter(tooltip.index)} y2={PLOT_HEIGHT}
                    className="paretoGuideLine"
                  />
                )}
              </svg>
            </div>

            <div
              className={`paretoFeatureLabels ${rotateLabels ? "rotated" : ""}`}
              style={{ width: layout.plotWidth, height: LABEL_HEIGHT }}
            >
              {items.map((item, i) => (
                <span
                  key={item.feature}
                  className="paretoFeatureLabel"
                  style={{ left: xLeft(i), width: layout.barWidth }}
                  title={item.feature}
                >
                  <span className="paretoFeatureLabelText">{shortenFeatureName(item.feature)}</span>
                </span>
              ))}
            </div>
            <p className="paretoAxisTitle" style={{ width: layout.plotWidth }}>상관 인자</p>
          </div>
        </div>

        <div className="paretoAxisCol paretoAxisCol-right" style={{ height: PLOT_HEIGHT }}>
          <div className="paretoTickCol" style={{ height: PLOT_HEIGHT }}>
            {RIGHT_TICKS.map((tick) => (
              <span key={tick} className="paretoTick" style={{ bottom: `${tick}%`, color: rightAxisColor }}>{tick}</span>
            ))}
          </div>
          <span className="paretoAxisLabel paretoAxisLabel-right" style={{ color: rightAxisColor }}>누적 기여율 (%)</span>
        </div>
      </div>

      {n === 1 ? (
        reached80 ? (
          <p className="paretoCaption">1개 인자가 전체 기여도의 {lastPct.toFixed(1)}%</p>
        ) : (
          <p className="paretoCaption">
            1개 인자가 전체 기여도의 {lastPct.toFixed(1)}% — 80% 도달에 {n80 ?? "?"}개 필요
          </p>
        )
      ) : reached80 ? (
        <p className="paretoCaption">상위 {n}개 인자가 전체 기여도의 {lastPct.toFixed(1)}%</p>
      ) : (
        <p className="paretoCaption">
          상위 {n}개 인자가 전체 기여도의 {lastPct.toFixed(1)}% — 80%에 도달하지 못했습니다
        </p>
      )}

      {tooltipItem && tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltipItem.feature}</strong>
          <div className="heatmapTooltipRow"><span>기여율</span><b>{tooltipItem.contribution_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>누적 기여율</span><b>{tooltipItem.cumulative_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>ε²</span><b>{tooltipItem.eps2.toFixed(3)}</b></div>
          <div className="heatmapTooltipRow"><span>p값</span><b>{formatP(tooltipItem.p_value)}</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltipItem.n_observed.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>등급</span><b>{TIER_LABEL[tooltipItem.confidence_tier]}</b></div>
        </div>
      )}
    </section>
  );
}

function SingleBarMarker({
  x,
  y,
  plotWidth,
  pct,
  lineColor,
  markerFill,
}: {
  x: number;
  y: number;
  plotWidth: number;
  pct: number;
  lineColor: string;
  markerFill: string;
}) {
  return (
    <g>
      <line x1={x} y1={y} x2={plotWidth} y2={y} stroke={lineColor} strokeWidth={1} strokeDasharray="2 2" />
      <circle cx={x} cy={y} r={3} fill="#FFFFFF" stroke={lineColor} strokeWidth={1.6} />
      <text x={plotWidth - 4} y={y - 6} textAnchor="end" className="paretoSingleMarkerValue" fill={lineColor}>
        {pct.toFixed(1)}%
      </text>
      <circle cx={x} cy={y} r={3} fill={markerFill} opacity={0} />
    </g>
  );
}
