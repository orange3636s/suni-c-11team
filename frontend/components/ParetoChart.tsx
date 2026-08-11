"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import InterpretationCard, { type InterpretationRow } from "@/components/InterpretationCard";
import { TIER_LABEL } from "@/lib/confidenceTier";
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
// 좌우 y축의 최상단 눈금 라벨(100)이 .paretoTick의
// `bottom:100%` + `transform:translateY(50%)` 배치 때문에 tick column의
// y=0(플롯 최상단) 경계 위로 반쯤 넘쳐 그려진다 -- 해석 카드 쪽 여백만
// 늘려서는(margin) 못 고친다, 이 넘침 자체가 차트 내부 좌표계 문제이기
// 때문이다. 플롯 영역(및 그 안의 눈금 라벨) 시작 위치를 이 값만큼 아래로
// 밀어 넘침이 카드가 아니라 이 여백 안에서만 일어나게 한다. plotHeight
// (막대·선 비례 계산)는 그대로 두고 전체 렌더 높이만 늘린다.
const CHART_PADDING_TOP = 16;
const FLAT_LABEL_HEIGHT = 18; // one horizontal line of label text, tightly boxed
const ROTATED_LABEL_HEIGHT = 84; // -90deg labels need room to run vertically
const LABEL_MIN_GAP = 4; // px of breathing room required between adjacent horizontal labels
// Fixed widths from .paretoAxisCol (label 14 + gap 8 + tickCol 22 = 44) on
// each side, plus the two 6px gaps in .paretoChartBody -- the space the
// axis columns always take up and that computeBarLayout must not offer to
// the plot, or the plot claims width the flex row can't actually give it.
const AXIS_RESERVED_WIDTH = 44 * 2 + 6 * 2;
// Pareto 썸네일은 막대가 위에서 시작하고 하단 x축 라벨이 생략돼,
// 본 차트와 같은 여백 비율을 쓰면 상단이 붙어 보인다. 썸네일 전용
// 상수로 상단에 여유를 확보한다(본 차트와 공유하지 않는다).
const THUMBNAIL_TOP_PADDING_RATIO = 0.12;
const LEFT_TICKS = [0, 25, 50, 75, 100];
const RIGHT_TICKS = [0, 20, 40, 60, 80, 100];
const LABEL_FONT = "10px var(--font-ui, system-ui, sans-serif)";

type TooltipState = { x: number; y: number; index: number } | null;

function shortenFeatureName(feature: string, maxLength = 11): string {
  return feature.length > maxLength ? `${feature.slice(0, maxLength - 1)}…` : feature;
}

/** Pareto 종합 문구의 순수 텍스트 버전 -- 차트 본문
 * (summaryCaption)과 즐겨찾기 스냅샷 저장(root-cause/page.tsx)이 같은
 * 문구를 쓴다. 로직이 두 곳에 따로 있으면 나중에 서로 어긋난다. */
export function buildParetoSummaryText(items: ParetoRankingItem[], n80: number | null): string {
  const n = items.length;
  if (n === 0) return "";
  const lastPct = items[n - 1].cumulative_pct;
  const reached80 = lastPct >= 80;
  if (n === 1) {
    return reached80
      ? `1개 인자가 전체 기여도의 ${lastPct.toFixed(1)}%`
      : `1개 인자가 전체 기여도의 ${lastPct.toFixed(1)}% — 80% 도달에 ${n80 ?? "?"}개 필요`;
  }
  return reached80
    ? `상위 ${n}개 인자가 전체 기여도의 ${lastPct.toFixed(1)}%`
    : `상위 ${n}개 인자가 전체 기여도의 ${lastPct.toFixed(1)}% — 80%에 도달하지 못했습니다`;
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
  thumbnail = false,
  reliabilityText,
  cardRef,
  headerActions,
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
  // 즐겨찾기 카드의 미리보기 -- 해석 문구(종합 캡션)를 그리지
  // 않고(그 문구는 카드 본문이 저장 시점 스냅샷으로 따로 보여준다),
  // 상단에 여유 패딩을 둬 Scatter/Box 썸네일과 같은 높이로 보이게 한다.
  thumbnail?: boolean;
  // "보통" 등급 인자의 설명력이 낮다는 안내(caller가 계산해 넘김,
  // root-cause/page.tsx의 buildModerateInterpretation과 동일 로직) --
  // Pareto 종합 문구("해석")와 카드 하나로 합친다. 빈 문자열/undefined면
  // "신뢰도" 행을 만들지 않는다.
  reliabilityText?: string;
  // 이미지 저장 버튼이 이 카드 루트 노드를
  // DOM 캡처해 PNG로 굽는다 -- non-embedded(독립 카드) 모드에서만 의미가
  // 있다.
  cardRef?: RefObject<HTMLElement | null>;
  // 타깃당 1개로 고정된 카드의 헤더에 얹을 액션(이미지
  // 저장 버튼 등) -- root-cause/page.tsx가 소유한 상태(export 진행 여부
  // 등)를 이 컴포넌트에 넣지 않기 위해 완성된 노드를 그대로 받는다.
  // non-embedded 모드에서만 렌더된다.
  headerActions?: ReactNode;
}) {
  const theme = useResolvedTheme();
  const [containerRef, containerWidth] = useContainerWidth();
  const [tooltip, setTooltip] = useState<TooltipState>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const baseHeight = embedded && height ? height : PLOT_HEIGHT;
  const topPadding = thumbnail ? baseHeight * THUMBNAIL_TOP_PADDING_RATIO : 0;
  const plotHeight = baseHeight - topPadding;

  const n = items.length;

  const plotAvailableWidth = Math.max(containerWidth - AXIS_RESERVED_WIDTH, MIN_BAR);
  const layout = useMemo(() => computeBarLayout(Math.max(n, 1), plotAvailableWidth), [n, plotAvailableWidth]);

  const rotateLabels = useMemo(() => {
    if (n < 2) return false;
    const widths = items.map((item) => measureTextWidth(shortenFeatureName(item.feature)));
    return widths.some((w) => w + LABEL_MIN_GAP > layout.slot);
  }, [items, layout.slot, n]);

  // 누적 곡선/마커는 실측이 아니라 계산된 값이라 --inferred, 80% 임계선은
  // 신호가 아니므로 중립 --line 을 쓴다. ScatterChart의
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

  // Pareto의 종합 문구(해석) -- 메타 줄 바로 아래·차트 위에 두고,
  // "해석" 라벨을 붙여 Scatter/Box와 같은 InterpretationCard로
  // 그린다(신뢰도 행은 caller가 넘겨줄 때만).
  // 썸네일(즐겨찾기 카드 미리보기)에서는 그리지 않는다 -- 그 문구는
  // 저장 시점 스냅샷으로 카드 본문이 따로 보여준다(DE-1).
  const summaryRows: InterpretationRow[] = [
    { label: "해석", text: buildParetoSummaryText(items, n80) },
    ...(reliabilityText ? [{ label: "신뢰도" as const, text: reliabilityText }] : []),
  ];
  const summaryCaption = <InterpretationCard rows={summaryRows} />;

  // Chart body -- shared as-is between the standalone card (non-embedded)
  // and the embedded-in-a-factor-card mode; only the wrapper around it
  // (resultCard section + title/legend header) differs (see `embedded`
  // below).
  const body = (
    <>
      {!thumbnail && summaryCaption}
      {/* 상단 여백 -- 최상단 눈금 라벨(100)이 tick column의 y=0
          경계 위로 넘쳐 그려지는 것 자체는 막지 않되(그 라벨의 상대
          위치는 그대로), 그 넘침이 이 빈 공간 안에서 끝나도록 플롯 전체를
          아래로 민다. plotHeight(막대·선 비례 계산)는 건드리지 않으므로
          막대가 작아지지 않고, 전체 렌더 높이만 CHART_PADDING_TOP만큼
          늘어난다. */}
      <div className="paretoChartBody" ref={containerRef} style={{ paddingTop: CHART_PADDING_TOP }}>
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
                      <title>{`${item.feature}, 기여율 ${item.contribution_pct.toFixed(1)}%, 신뢰도 ${TIER_LABEL[item.confidence_tier]}${item.under_sampled ? " (표본 부족)" : ""}`}</title>
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

      {tooltipItem && tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltipItem.feature}</strong>
          <div className="heatmapTooltipRow"><span>기여율</span><b>{tooltipItem.contribution_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>누적 기여율</span><b>{tooltipItem.cumulative_pct.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow">
            <span>Adj R²</span>
            <b>
              {tooltipItem.adj_r2.toFixed(3)}
              {tooltipItem.degree != null && ` (${tooltipItem.degree}차)`}
            </b>
          </div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltipItem.n_observed.toLocaleString()}</b></div>
          {/* 알람 등급(심각/위험/주의)과 겹치지 않게 -- 이 값은
              인자-타깃 연관의 세기(강함/보통/근거 부족/관계 없음)다. */}
          <div className="heatmapTooltipRow"><span>상관성</span><b>{TIER_LABEL[tooltipItem.confidence_tier]}</b></div>
          {tooltipItem.under_sampled && (
            <div className="heatmapTooltipRow"><span>표본</span><b className="paretoUnderSampledLabel">부족 (등급 하향 반영됨)</b></div>
          )}
        </div>
      )}
    </>
  );

  if (embedded) {
    // 썸네일 컨테이너에 상단 여유 패딩을 줘 Scatter/Box 썸네일과
    // 나란히 놓아도 위쪽이 붙어 보이지 않게 한다.
    if (thumbnail) return <div style={{ paddingTop: topPadding }}>{body}</div>;
    return body;
  }

  return (
    <section className="resultCard paretoChartCard" ref={cardRef}>
      <div className="paretoChartHeader">
        <div>
          <span className="sectionLabel">PARETO</span>
          {/* 파레토는 타깃당 1개로 고정되며, 인자명이 아니라
              "R/D/Config vs {타깃}"이 제목이다 -- 특정 인자 하나가 아니라
              이 타깃의 인자 순위 전체를 보여주는 차트이기 때문이다. */}
          <h2>R/D/Config vs {target}</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div className="paretoLegend paretoLegendInline">
            <span><i className="paretoLegendSwatch tier-strong" /> {TIER_LABEL.strong}</span>
            <span><i className="paretoLegendSwatch tier-moderate" /> {TIER_LABEL.moderate}</span>
            <span><i className="paretoLegendSwatch tier-weak" /> {TIER_LABEL.weak}</span>
            <span><i className="paretoLegendSwatch tier-reference" /> {TIER_LABEL.reference}</span>
          </div>
          {headerActions}
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
