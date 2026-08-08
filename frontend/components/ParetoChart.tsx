"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { TIER_LABEL } from "@/lib/confidenceTier";
import { formatPValue } from "@/lib/numberFormat";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ParetoRankingItem } from "@/types/data";

const MAX_BAR = 48;
// PARETO_TOP_N=10으로 늘면서 28px 하한으로는 10개가 가로 스크롤 없이
// 들어가는 폭을 확보하기 어려운 카드 크기가 있었다 -- 20까지 더 좁힐 수
// 있게 낮춘다. computeBarLayout은 이 하한에서도 해를 못 찾으면 그대로
// MIN_BAR로 폴백하고 .paretoPlotScroll이 가로 스크롤을 넘겨받는다.
const MIN_BAR = 20;
const PADDING = 56;
const PLOT_HEIGHT = 360;
const FLAT_LABEL_HEIGHT = 18; // one horizontal line of label text, tightly boxed
const ROTATED_LABEL_HEIGHT = 84; // -90deg labels need room to run vertically
const LABEL_MIN_GAP = 4; // px of breathing room required between adjacent horizontal labels
// Fixed widths from .paretoAxisCol (label 14 + gap 8 + tickCol 22 = 44) on
// each side, plus the two 6px gaps in .paretoChartBody -- the space the
// axis columns always take up and that computeBarLayout must not offer to
// the plot, or the plot claims width the flex row can't actually give it.
const AXIS_RESERVED_WIDTH = 44 * 2 + 6 * 2;
const LEFT_TICKS = [0, 25, 50, 75, 100];
const RIGHT_TICKS = [0, 20, 40, 60, 80, 100];
const LABEL_FONT = "10px var(--font-ui, system-ui, sans-serif)";

type TooltipState = { x: number; y: number; index: number } | null;

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
  embedded = false,
  height,
}: {
  target: string;
  items: ParetoRankingItem[];
  n80: number | null;
  onBarClick: (item: ParetoRankingItem) => void;
  activeFeature?: string | null;
  // 인자 카드의 보기 토글 안에서 렌더될 때 true (spec: Pareto를 산점도
  // 카드로 병합) -- 바깥 resultCard 래퍼와 자체 제목/범례 헤더를 생략하고
  // 차트 본문만 반환한다. 기본값 false는 기존 호출부(HeatmapParetoSection
  // 등 독립 카드로 쓰는 곳)를 그대로 유지하기 위함이다.
  embedded?: boolean;
  // embedded일 때만 사용 -- 카드의 chartHeight(데스크톱/모바일)에 맞춘다.
  // 생략 시 기존 PLOT_HEIGHT 상수로 대체한다.
  height?: number;
}) {
  const theme = useResolvedTheme();
  const [containerRef, containerWidth] = useContainerWidth();
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const plotHeight = embedded && height ? height : PLOT_HEIGHT;

  const n = items.length;
  const reached80 = items.length > 0 && items[items.length - 1].cumulative_pct >= 80;
  const lastPct = items.length > 0 ? items[items.length - 1].cumulative_pct : 0;

  const plotAvailableWidth = Math.max(containerWidth - AXIS_RESERVED_WIDTH, MIN_BAR);
  const layout = useMemo(() => computeBarLayout(Math.max(n, 1), plotAvailableWidth), [n, plotAvailableWidth]);

  const rotateLabels = useMemo(() => {
    if (n < 2) return false;
    const widths = items.map((item) => measureTextWidth(shortenFeatureName(item.feature)));
    return widths.some((w) => w + LABEL_MIN_GAP > layout.slot);
  }, [items, layout.slot, n]);

  // 누적 곡선/마커는 실측이 아니라 계산된 값이라 --inferred, 80% 임계선은
  // 신호가 아니므로 중립 --line 을 쓴다 (지시서 N-2). ScatterChart의
  // TREND_COLOR와 동일한 상수쌍.
  const lineColor = theme === "dark" ? "#97A3B8" : "#7C8AA5";
  const thresholdColor = theme === "dark" ? "rgba(255, 255, 255, 0.14)" : "#D9DEE6";
  const markerFill = theme === "dark" ? "#2C2C2E" : "#FFFFFF";
  const rightAxisColor = lineColor;

  if (n === 0) return null;

  const xCenter = (i: number) => PADDING + i * layout.slot + layout.barWidth / 2;
  const xLeft = (i: number) => PADDING + i * layout.slot;
  // plotHeight를 닫는다 -- embedded 모드에서는 이 값이 PLOT_HEIGHT(360)와
  // 다르다(카드 높이 480/240을 그대로 받음). 막대·눈금 컬럼은 이미
  // plotHeight를 쓰는데 이 함수만 PLOT_HEIGHT를 참조하면, 누적선/마커/80%
  // 임계선이 막대·눈금과 다른 좌표계로 그려진다(버그 원인).
  const yFor = (pct: number) => plotHeight * (1 - Math.min(pct, 100) / 100);

  function handlePlotMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relativeX = event.clientX - rect.left - PADDING;
    const index = Math.min(n - 1, Math.max(0, Math.round(relativeX / layout.slot - 0.5)));
    setTooltip({ x: event.clientX, y: event.clientY, index });
  }

  const tooltipItem = tooltip ? items[tooltip.index] : null;

  // Chart body -- shared as-is between the standalone card (non-embedded)
  // and the embedded-in-a-factor-card mode; only the wrapper around it
  // (resultCard section + title/legend header) differs (see `embedded`
  // below).
  const body = (
    <>
      <div className="paretoChartBody" ref={containerRef}>
        <div className="paretoAxisCol paretoAxisCol-left" style={{ height: plotHeight }}>
          <span className="paretoAxisLabel"><span className="paretoAxisLabelText">기여율 (%)</span></span>
          <div className="paretoTickCol" style={{ height: plotHeight }}>
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
              style={{ width: layout.plotWidth, height: plotHeight }}
              onMouseMove={n > 1 ? handlePlotMouseMove : undefined}
              onMouseLeave={() => setTooltip(null)}
            >
              <svg
                className="paretoOverlay"
                width={layout.plotWidth}
                height={plotHeight}
                aria-hidden="true"
              >
                <line
                  x1={0} y1={yFor(80)} x2={layout.plotWidth} y2={yFor(80)}
                  className="paretoThresholdLine" stroke={thresholdColor}
                />
                <text x={layout.plotWidth - 4} y={yFor(80) - 4} textAnchor="end" className="paretoThresholdLabel">80%</text>

                {items.map((item, i) => {
                  // 막대 상단도 yFor()로 계산한다 (누적 마커/임계선과 같은
                  // 식) -- 두 계산이 갈라져 있으면 한쪽만 고쳤을 때 다시
                  // 어긋난다. 최소 높이 보정은 height에만 적용하고 yTop은
                  // 건드리지 않는다 -- 그래야 기여율이 0에 가까운 막대에서
                  // 상단이 실제 값보다 위로 튀지 않는다(하단으로만 살짝
                  // 넘치는 쪽을 택한다).
                  const yTop = yFor(item.contribution_pct);
                  const barHeight = Math.max(plotHeight - yTop, 2);
                  return (
                    <rect
                      key={item.feature}
                      className={["paretoBar", `tier-${item.confidence_tier}`, activeFeature === item.feature ? "active" : ""].join(" ")}
                      x={xLeft(i)}
                      y={yTop}
                      width={layout.barWidth}
                      height={barHeight}
                      rx={2}
                      ry={2}
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
                    x1={xCenter(tooltip.index)} y1={0} x2={xCenter(tooltip.index)} y2={plotHeight}
                    className="paretoGuideLine"
                  />
                )}
              </svg>
            </div>

            <div
              className={`paretoFeatureLabels ${rotateLabels ? "rotated" : ""}`}
              style={{ width: layout.plotWidth, height: rotateLabels ? ROTATED_LABEL_HEIGHT : FLAT_LABEL_HEIGHT }}
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

        <div className="paretoAxisCol paretoAxisCol-right" style={{ height: plotHeight }}>
          <div className="paretoTickCol" style={{ height: plotHeight }}>
            {RIGHT_TICKS.map((tick) => (
              <span key={tick} className="paretoTick" style={{ bottom: `${tick}%`, color: rightAxisColor }}>{tick}</span>
            ))}
          </div>
          <span className="paretoAxisLabel paretoAxisLabel-right" style={{ color: rightAxisColor }}><span className="paretoAxisLabelText">누적 기여율 (%)</span></span>
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
          <div className="heatmapTooltipRow"><span>p값</span><b>{formatPValue(tooltipItem.p_value)}</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltipItem.n_observed.toLocaleString()}</b></div>
          {/* 지시서 CD: 알람 등급(심각/위험/주의)과 겹치지 않게 -- 이 값은
              인자-타깃 연관의 세기(강함/보통/근거 부족/관계 없음)다. */}
          <div className="heatmapTooltipRow"><span>상관성</span><b>{TIER_LABEL[tooltipItem.confidence_tier]}</b></div>
        </div>
      )}
    </>
  );

  if (embedded) return body;

  return (
    <section className="resultCard paretoChartCard">
      <div className="paretoChartHeader">
        <div>
          <span className="sectionLabel">PARETO</span>
          <h2>{target} 상관 인자 기여도</h2>
        </div>
        <div className="paretoLegend paretoLegendInline">
          <span><i className="paretoLegendSwatch tier-strong" /> {TIER_LABEL.strong}</span>
          <span><i className="paretoLegendSwatch tier-moderate" /> {TIER_LABEL.moderate}</span>
          <span><i className="paretoLegendSwatch tier-weak" /> {TIER_LABEL.weak}</span>
          <span><i className="paretoLegendSwatch tier-reference" /> {TIER_LABEL.reference}</span>
        </div>
      </div>
      {body}
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
