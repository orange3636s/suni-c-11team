"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ReferenceLine, ReferenceLineKey, ScatterPoint, ScreeningScatterResponse } from "@/types/data";

export type ScatterColorMode = "default" | "config_model" | "lot" | "alarm";

const MARGIN = { top: 36, right: 28, bottom: 46, left: 58 };
const HEIGHT = 420;
const LABEL_MIN_GAP = 34;

type LineMeta = {
  legendId: string;
  shortLabel: string;
  legendLabel: string;
  legendDesc: string;
  defaultVisible: boolean;
  priority: number; // higher survives label-collision pruning first
  dash: string;
  strokeWidth: number;
};

const LINE_META: Record<ReferenceLineKey, LineMeta> = {
  mean: { legendId: "mean", shortLabel: "평균", legendLabel: "평균", legendDesc: "중심", defaultVisible: true, priority: 50, dash: "6 4", strokeWidth: 1.5 },
  iqr_lo: { legendId: "iqr", shortLabel: "LCL", legendLabel: "LCL/UCL (IQR 1.5배)", legendDesc: "관리한계, 알람 판정 기준", defaultVisible: true, priority: 40, dash: "10 5", strokeWidth: 2 },
  iqr_hi: { legendId: "iqr", shortLabel: "UCL", legendLabel: "LCL/UCL (IQR 1.5배)", legendDesc: "관리한계, 알람 판정 기준", defaultVisible: true, priority: 40, dash: "10 5", strokeWidth: 2 },
  q1: { legendId: "q1q3", shortLabel: "Q1", legendLabel: "Q1 / Q3", legendDesc: "사분위", defaultVisible: false, priority: 20, dash: "none", strokeWidth: 1.2 },
  q3: { legendId: "q1q3", shortLabel: "Q3", legendLabel: "Q1 / Q3", legendDesc: "사분위", defaultVisible: false, priority: 20, dash: "none", strokeWidth: 1.2 },
  s3_lo: { legendId: "s3", shortLabel: "−3σ", legendLabel: "±3σ", legendDesc: "참조선", defaultVisible: false, priority: 30, dash: "8 3 2 3", strokeWidth: 1.3 },
  s3_hi: { legendId: "s3", shortLabel: "+3σ", legendLabel: "±3σ", legendDesc: "참조선", defaultVisible: false, priority: 30, dash: "8 3 2 3", strokeWidth: 1.3 },
  s6_lo: { legendId: "s6", shortLabel: "−6σ", legendLabel: "±6σ", legendDesc: "참조선", defaultVisible: false, priority: 10, dash: "6 3 1 3 1 3", strokeWidth: 1.1 },
  s6_hi: { legendId: "s6", shortLabel: "+6σ", legendLabel: "±6σ", legendDesc: "참조선", defaultVisible: false, priority: 10, dash: "6 3 1 3 1 3", strokeWidth: 1.1 },
};

// Button order follows the spec's worked example: LCL/UCL, ±3σ, ±6σ, 평균, Q1/Q3.
const LEGEND_ORDER = ["iqr", "s3", "s6", "mean", "q1q3"];
const QUARTILE_SIGMA_KEYS = new Set<ReferenceLineKey>(["q1", "q3", "s3_lo", "s3_hi", "s6_lo", "s6_hi"]);

function formatNum1(value: number): string {
  return value.toFixed(1);
}

/** Legend button text with the actual computed values baked in (spec §4-6) --
 * separate from LINE_META.legendLabel, which stays a static category name used
 * by the per-line hover tooltip. */
function buildLegendLabel(groupId: string, lines: ReferenceLine[]): string {
  const find = (key: ReferenceLineKey) => lines.find((l) => l.key === key);
  if (groupId === "mean") {
    const mean = find("mean");
    return mean ? `평균 (${formatNum1(mean.value)})` : "평균";
  }
  if (groupId === "iqr") {
    const lo = find("iqr_lo");
    const hi = find("iqr_hi");
    if (lo?.drawable && hi?.drawable) return `LCL/UCL (IQR 1.5배 · ${formatNum1(lo.value)} / ${formatNum1(hi.value)})`;
    if (hi?.drawable) return `UCL (${formatNum1(hi.value)})`;
    if (lo?.drawable) return `LCL (${formatNum1(lo.value)})`;
    return "LCL/UCL";
  }
  if (groupId === "q1q3") {
    const q1 = find("q1");
    const q3 = find("q3");
    return q1 && q3 ? `Q1 / Q3 (${formatNum1(q1.value)} / ${formatNum1(q3.value)})` : "Q1 / Q3";
  }
  if (groupId === "s3" || groupId === "s6") {
    const lo = find(groupId === "s3" ? "s3_lo" : "s6_lo");
    const hi = find(groupId === "s3" ? "s3_hi" : "s6_hi");
    const label = groupId === "s3" ? "±3σ" : "±6σ";
    if (lo && hi) {
      const deviation = (hi.value - lo.value) / 2;
      return `${label} (${formatNum1(lo.value)} / ${formatNum1(hi.value)}, ±${formatNum1(deviation)})`;
    }
    return label;
  }
  return "";
}

const LINE_COLOR: Record<string, { light: string; dark: string }> = {
  mean: { light: "#059669", dark: "#34D399" },
  iqr: { light: "#0E306D", dark: "#7BA3E8" },
  s3: { light: "#9CA3AF", dark: "#6B7280" },
  s6: { light: "#D1D5DB", dark: "#4B5563" },
  q1q3: { light: "#3B82F6", dark: "#93C5FD" },
};
const TREND_COLOR = { light: "#DC2626", dark: "#F87171" };

function modelOf(config: string | null): string {
  if (!config) return "미계측";
  const match = /Model(\d+)/.exec(config);
  return match ? `Model${match[1]}` : config;
}
const MODEL_COLORS = ["#1D4ED8", "#059669", "#B45309"];
const LOT_PALETTE = ["#1D4ED8", "#059669", "#B45309", "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#DC2626"];

function colorForPoint(point: ScatterPoint, mode: ScatterColorMode, lotIndex: Map<string, number>): { color: string; size: number; opacity: number } {
  if (mode === "alarm") {
    return point.in_range
      ? { color: "#1D4ED8", size: 4.5, opacity: 0.7 }
      : { color: "#F59E0B", size: 5.5, opacity: 0.9 };
  }
  if (mode === "config_model") {
    const key = modelOf(point.config);
    let idx = lotIndex.get(key);
    if (idx == null) {
      idx = lotIndex.size;
      lotIndex.set(key, idx);
    }
    return { color: MODEL_COLORS[idx % MODEL_COLORS.length], size: 5, opacity: 0.85 };
  }
  if (mode === "lot") {
    const key = point.lot_id ?? "미상";
    let idx = lotIndex.get(key);
    if (idx == null) {
      idx = lotIndex.size;
      lotIndex.set(key, idx);
    }
    return { color: LOT_PALETTE[idx % LOT_PALETTE.length], size: 5, opacity: 0.85 };
  }
  // default
  return point.in_range
    ? { color: "#93C5FD", size: 4.5, opacity: 0.5 }
    : { color: "#1D4ED8", size: 5.5, opacity: 0.85 };
}

type LineLayout = { line: ReferenceLine; xPixel: number; row: number; hidden: boolean };

function layoutLabels(entries: Array<{ line: ReferenceLine; xPixel: number }>): LineLayout[] {
  const byPriority = [...entries].sort((a, b) => LINE_META[b.line.key].priority - LINE_META[a.line.key].priority);
  const placedPerRow: number[][] = [[], []];
  const result: LineLayout[] = [];
  for (const entry of byPriority) {
    let row = -1;
    for (let candidate = 0; candidate < 2; candidate += 1) {
      const conflict = placedPerRow[candidate].some((x) => Math.abs(x - entry.xPixel) < LABEL_MIN_GAP);
      if (!conflict) {
        row = candidate;
        placedPerRow[candidate].push(entry.xPixel);
        break;
      }
    }
    result.push({ line: entry.line, xPixel: entry.xPixel, row: row < 0 ? 0 : row, hidden: row < 0 });
  }
  return result;
}

type TrendHover = {
  screenX: number;
  screenY: number;
  dataX: number;
  dataY: number;
  ciLo: number;
  ciHi: number;
  n: number;
};

type LineHover = { key: ReferenceLineKey; x: number; y: number };

type PointHover = { screenX: number; screenY: number; clientX: number; clientY: number; point: ScatterPoint };

const POINT_HOVER_RADIUS_PX = 16;

export default function ScatterChart({
  data,
  colorMode,
  onSelectWafer,
  height = HEIGHT,
}: {
  data: ScreeningScatterResponse;
  colorMode: ScatterColorMode;
  onSelectWafer: (point: ScatterPoint) => void;
  height?: number;
}) {
  const theme = useResolvedTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerWidth, setContainerWidth] = useState(680);
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(["mean", "iqr"]));
  const [trendVisible, setTrendVisible] = useState(true);
  const [lineHover, setLineHover] = useState<LineHover | null>(null);
  const [trendHover, setTrendHover] = useState<TrendHover | null>(null);
  const [pointHover, setPointHover] = useState<PointHover | null>(null);
  const [disabledHint, setDisabledHint] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setContainerWidth(width);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const plotWidth = Math.max(containerWidth - MARGIN.left - MARGIN.right, 120);
  const plotHeight = height - MARGIN.top - MARGIN.bottom;

  const { xDomain, yDomain, xScale, yScale } = useMemo(() => {
    const xValues = data.points.map((p) => p.x);
    const yValues = data.points.map((p) => p.y);
    const xMin = xValues.length ? Math.min(...xValues) : 0;
    const xMax = xValues.length ? Math.max(...xValues) : 1;
    const yMin = yValues.length ? Math.min(...yValues) : 0;
    const yMax = yValues.length ? Math.max(...yValues) : 1;
    const xPad = (xMax - xMin) * 0.04 || 1;
    const yPad = (yMax - yMin) * 0.1 || 1;
    const xDom: [number, number] = [xMin - xPad, xMax + xPad];
    const yDom: [number, number] = [yMin - yPad, yMax + yPad];
    return {
      xDomain: xDom,
      yDomain: yDom,
      xScale: (v: number) => ((v - xDom[0]) / (xDom[1] - xDom[0])) * plotWidth,
      yScale: (v: number) => plotHeight - ((v - yDom[0]) / (yDom[1] - yDom[0])) * plotHeight,
    };
  }, [data.points, plotWidth, plotHeight]);

  // A plain object recreated each render -- cheap (just an empty Map),
  // and colorForPoint mutates it while walking data.points below to
  // assign each distinct config/lot a stable color index for this render.
  const lotIndex = new Map<string, number>();

  const drawableLines = data.reference_lines.filter((l) => l.drawable);
  const labelLayout = useMemo(
    () => layoutLabels(drawableLines.map((line) => ({ line, xPixel: xScale(line.value) }))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data.reference_lines, plotWidth],
  );

  const trendPath = useMemo(() => {
    if (!data.bins.length) return { line: "", band: "" };
    const points = data.bins.map((b) => [xScale(b.x_mean), yScale(b.y_mean)] as const);
    const line = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
    const upper = data.bins.map((b) => [xScale(b.x_mean), yScale(b.y_hi)] as const);
    const lower = [...data.bins].reverse().map((b) => [xScale(b.x_mean), yScale(b.y_lo)] as const);
    const band = [...upper, ...lower].map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";
    return { line, band };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.bins, plotWidth, plotHeight]);

  function isGroupFullyNonDrawable(groupId: string): boolean {
    return data.reference_lines.filter((l) => LINE_META[l.key].legendId === groupId).every((l) => !l.drawable);
  }

  function toggleGroup(groupId: string, event: React.MouseEvent) {
    if (isGroupFullyNonDrawable(groupId)) {
      setDisabledHint({ x: event.clientX, y: event.clientY, text: "데이터 범위 밖" });
      window.setTimeout(() => setDisabledHint(null), 1500);
      return;
    }
    setVisibleGroups((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }

  function findNearestPoint(relativeX: number, relativeY: number): ScatterPoint | null {
    // Linear scan over up to ~1,470 points is cheap per mousemove (no
    // per-point DOM listeners needed) -- nearest-point-within-radius,
    // not a list of every point under the cursor.
    let best: ScatterPoint | null = null;
    let bestDistance = POINT_HOVER_RADIUS_PX;
    for (const point of data.points) {
      const dx = xScale(point.x) - relativeX;
      const dy = yScale(point.y) - relativeY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = point;
      }
    }
    return best;
  }

  function handlePlotOverlayMouseMove(event: React.MouseEvent<SVGRectElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relativeX = event.clientX - rect.left - MARGIN.left;
    const relativeY = event.clientY - rect.top - MARGIN.top;
    const nearest = findNearestPoint(relativeX, relativeY);
    if (nearest) {
      setPointHover({ screenX: xScale(nearest.x), screenY: yScale(nearest.y), clientX: event.clientX, clientY: event.clientY, point: nearest });
    } else {
      setPointHover(null);
    }
    handleTrendMouseMove(event);
  }

  function handleTrendMouseMove(event: React.MouseEvent<SVGRectElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !data.bins.length) return;
    const relativeX = event.clientX - rect.left - MARGIN.left;
    const dataX = xDomain[0] + (relativeX / plotWidth) * (xDomain[1] - xDomain[0]);
    const bins = data.bins;
    let ciLo: number, ciHi: number, dataY: number, n: number;
    if (dataX <= bins[0].x_mean) {
      ({ y_mean: dataY, y_lo: ciLo, y_hi: ciHi, n } = bins[0]);
    } else if (dataX >= bins[bins.length - 1].x_mean) {
      ({ y_mean: dataY, y_lo: ciLo, y_hi: ciHi, n } = bins[bins.length - 1]);
    } else {
      let left = bins[0];
      let right = bins[bins.length - 1];
      for (let i = 0; i < bins.length - 1; i += 1) {
        if (dataX >= bins[i].x_mean && dataX <= bins[i + 1].x_mean) {
          left = bins[i];
          right = bins[i + 1];
          break;
        }
      }
      const t = (dataX - left.x_mean) / (right.x_mean - left.x_mean || 1);
      dataY = left.y_mean + t * (right.y_mean - left.y_mean);
      ciLo = left.y_lo + t * (right.y_lo - left.y_lo);
      ciHi = left.y_hi + t * (right.y_hi - left.y_hi);
      n = t < 0.5 ? left.n : right.n;
    }
    setTrendHover({ screenX: xScale(dataX), screenY: yScale(dataY), dataX, dataY, ciLo, ciHi, n });
  }

  const withinControl = trendHover
    ? (data.normal_range.lo == null || trendHover.dataX >= data.normal_range.lo) &&
      (data.normal_range.hi == null || trendHover.dataX <= data.normal_range.hi)
    : null;

  function renderLineBody(line: ReferenceLine) {
    const meta = LINE_META[line.key];
    if (!visibleGroups.has(meta.legendId)) return null;
    const x = xScale(line.value);
    const greyedOut = (line.key === "iqr_lo" || line.key === "iqr_hi") && !line.alarm_relevant;
    const color = greyedOut ? (theme === "dark" ? "#6B7280" : "#9CA3AF") : (theme === "dark" ? LINE_COLOR[meta.legendId].dark : LINE_COLOR[meta.legendId].light);
    return (
      <g key={line.key}>
        <line
          x1={x} x2={x} y1={0} y2={plotHeight}
          stroke={color}
          strokeWidth={meta.strokeWidth}
          strokeDasharray={meta.dash === "none" ? undefined : meta.dash}
          opacity={greyedOut ? 0.55 : 0.9}
        />
        {/* widened invisible hit area */}
        <rect
          x={x - 6} y={0} width={12} height={plotHeight}
          fill="transparent" style={{ cursor: "help" }}
          onMouseEnter={(event) => setLineHover({ key: line.key, x: event.clientX, y: event.clientY })}
          onMouseMove={(event) => setLineHover({ key: line.key, x: event.clientX, y: event.clientY })}
          onMouseLeave={() => setLineHover(null)}
        />
      </g>
    );
  }

  function renderLineLabel(line: ReferenceLine) {
    const meta = LINE_META[line.key];
    if (!visibleGroups.has(meta.legendId)) return null;
    const layout = labelLayout.find((l) => l.line.key === line.key);
    if (!layout || layout.hidden) return null;
    const greyedOut = (line.key === "iqr_lo" || line.key === "iqr_hi") && !line.alarm_relevant;
    const color = greyedOut ? (theme === "dark" ? "#6B7280" : "#9CA3AF") : (theme === "dark" ? LINE_COLOR[meta.legendId].dark : LINE_COLOR[meta.legendId].light);
    const x = xScale(line.value);
    return (
      <foreignObject key={line.key} x={x - 20} y={layout.row * 16} width={40} height={14} style={{ overflow: "visible", pointerEvents: "none" }}>
        <div className="scatterLineLabel" style={{ color }}>{meta.shortLabel}</div>
      </foreignObject>
    );
  }

  return (
    <div className="scatterChart" ref={containerRef}>
      <div className="scatterChartMeta">
        <span>n={data.n.toLocaleString()}</span>
        <span>ε²={data.eps2.toFixed(3)}</span>
        <span>p-value {data.p_value.toFixed(3)}</span>
        <span>등급 {{ strong: "강함", moderate: "보통", weak: "약함", reference: "참고" }[data.confidence_tier]}</span>
      </div>

      <svg ref={svgRef} width="100%" height={height} className="scatterChartSvg" role="img" aria-label={`${data.axis.x_label} vs ${data.axis.y_label} 산점도`}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* axis ticks */}
          {niceTicks(yDomain, 5).map((tick) => (
            <g key={`y-${tick}`}>
              <line x1={0} x2={plotWidth} y1={yScale(tick)} y2={yScale(tick)} className="scatterGridLine" />
              <text x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="scatterTickLabel">{formatTick(tick)}</text>
            </g>
          ))}
          {niceTicks(xDomain, 6).map((tick) => (
            <text key={`x-${tick}`} x={xScale(tick)} y={plotHeight + 20} textAnchor="middle" className="scatterTickLabel">{formatTick(tick)}</text>
          ))}

          {/* shading: amber outside IQR control limits, faint blue between Q1/Q3 */}
          {(() => {
            const iqrLo = data.reference_lines.find((l) => l.key === "iqr_lo");
            const iqrHi = data.reference_lines.find((l) => l.key === "iqr_hi");
            const loX = iqrLo?.drawable ? xScale(iqrLo.value) : 0;
            const hiX = iqrHi?.drawable ? xScale(iqrHi.value) : plotWidth;
            return visibleGroups.has("iqr") ? (
              <>
                {iqrLo?.drawable && <rect x={0} y={0} width={Math.max(loX, 0)} height={plotHeight} className="scatterOutsideShade" />}
                {iqrHi?.drawable && <rect x={hiX} y={0} width={Math.max(plotWidth - hiX, 0)} height={plotHeight} className="scatterOutsideShade" />}
              </>
            ) : null;
          })()}
          {(() => {
            if (!visibleGroups.has("q1q3")) return null;
            const q1 = data.reference_lines.find((l) => l.key === "q1");
            const q3 = data.reference_lines.find((l) => l.key === "q3");
            if (!q1?.drawable || !q3?.drawable) return null;
            const x1 = xScale(q1.value);
            const x2 = xScale(q3.value);
            return <rect x={Math.min(x1, x2)} y={0} width={Math.abs(x2 - x1)} height={plotHeight} className="scatterQuartileShade" />;
          })()}

          {/* data points -- painted before every reference line/curve (spec
              §4-1: lines and the trend curve must sit above the points,
              never hidden behind them). */}
          {data.points.map((point, index) => {
            const style = colorForPoint(point, colorMode, lotIndex);
            const isHovered = pointHover?.point === point;
            return (
              <circle
                key={point.lot_wafer_id ?? index}
                cx={xScale(point.x)}
                cy={yScale(point.y)}
                r={isHovered ? style.size * 1.5 : style.size}
                fill={style.color}
                opacity={isHovered ? 1 : style.opacity}
                stroke={isHovered ? (theme === "dark" ? "#FFFFFF" : "#0E306D") : "none"}
                strokeWidth={isHovered ? 1.5 : 0}
                style={{ cursor: "pointer" }}
                onClick={() => onSelectWafer(point)}
              />
            );
          })}

          {/* reference line bodies, stacked bottom-to-top per spec §4-1:
              Q1/Q3 + ±3σ/±6σ, then 평균, then LCL/UCL last (closest to the
              trend curve/labels above them). Labels are a separate,
              always-topmost pass below so they never sit under a
              later-drawn line. */}
          {drawableLines.filter((line) => QUARTILE_SIGMA_KEYS.has(line.key)).map((line) => renderLineBody(line))}
          {drawableLines.filter((line) => line.key === "mean").map((line) => renderLineBody(line))}
          {drawableLines.filter((line) => line.key === "iqr_lo" || line.key === "iqr_hi").map((line) => renderLineBody(line))}

          {/* confidence band + trend curve -- above every reference line */}
          {trendVisible && data.bins.length > 0 && (
            <>
              <path d={trendPath.band} className="scatterTrendBand" style={{ fill: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} />
              <path d={trendPath.line} className="scatterTrendLine" style={{ stroke: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} fill="none" />
            </>
          )}

          {/* line name labels -- topmost of all */}
          {drawableLines.map((line) => renderLineLabel(line))}

          {/* continuous point+trend hover overlay -- nearest-point search,
              not per-point listeners (see findNearestPoint's comment). */}
          <rect
            x={0} y={0} width={plotWidth} height={plotHeight} fill="transparent"
            onMouseMove={handlePlotOverlayMouseMove}
            onMouseLeave={() => {
              setPointHover(null);
              setTrendHover(null);
            }}
          />
          {trendHover && !pointHover && (
            <>
              <line x1={trendHover.screenX} x2={trendHover.screenX} y1={0} y2={plotHeight} className="scatterGuideLine" />
              <circle
                cx={trendHover.screenX} cy={trendHover.screenY} r={4}
                fill={theme === "dark" ? "#2C2C2E" : "#FFFFFF"}
                stroke={theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light}
                strokeWidth={2}
              />
            </>
          )}
        </g>
      </svg>

      <p className="scatterAxisTitle">{data.axis.x_label}</p>

      <div className="scatterLegend">
        {LEGEND_ORDER.map((groupId) => {
          const sampleLine = data.reference_lines.find((l) => LINE_META[l.key].legendId === groupId);
          if (!sampleLine) return null;
          const meta = LINE_META[sampleLine.key];
          const disabled = isGroupFullyNonDrawable(groupId);
          const active = visibleGroups.has(groupId) && !disabled;
          return (
            <button
              key={groupId}
              type="button"
              className={`scatterLegendItem ${active ? "active" : ""} ${disabled ? "disabled" : ""}`}
              onClick={(event) => toggleGroup(groupId, event)}
              title={disabled ? "데이터 범위 밖" : meta.legendDesc}
            >
              <i className="scatterLegendSwatch" style={{ background: theme === "dark" ? LINE_COLOR[groupId].dark : LINE_COLOR[groupId].light }} />
              {buildLegendLabel(groupId, data.reference_lines)}
            </button>
          );
        })}
        <button
          type="button"
          className={`scatterLegendItem ${trendVisible ? "active" : ""}`}
          onClick={() => setTrendVisible((v) => !v)}
          title="구간 평균 불량률"
        >
          <i className="scatterLegendSwatch" style={{ background: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} />
          구간 평균 불량률
        </button>
      </div>

      {lineHover && (() => {
        const line = data.reference_lines.find((l) => l.key === lineHover.key);
        if (!line) return null;
        const meta = LINE_META[line.key];
        return (
          <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
            <strong>{meta.legendLabel} {meta.shortLabel && `(${meta.shortLabel})`}</strong>
            <div className="heatmapTooltipRow"><span>값</span><b>{line.value.toFixed(1)}</b></div>
            <div className="heatmapTooltipRow"><span>계산</span><b>{line.formula}</b></div>
            {line.key !== "mean" && (
              <div className="heatmapTooltipRow"><span>이 선 밖</span><b>{line.outside_count.toLocaleString()}장</b></div>
            )}
          </div>
        );
      })()}

      {trendHover && !pointHover && (
        <div className="heatmapTooltip" style={{ left: MARGIN.left + trendHover.screenX + 14, top: MARGIN.top + trendHover.screenY + 14 }}>
          <strong>{data.axis.x_label.split(" ")[0]} = {trendHover.dataX.toFixed(1)}</strong>
          <div className="heatmapTooltipRow"><span>구간 평균 불량률</span><b>{trendHover.dataY.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>95% 신뢰구간</span><b>{trendHover.ciLo.toFixed(2)} ~ {trendHover.ciHi.toFixed(2)}</b></div>
          <div className="heatmapTooltipRow"><span>이 구간 wafer 수</span><b>{trendHover.n.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>관리한계 내</span><b>{withinControl ? "예" : "아니오"}</b></div>
        </div>
      )}

      {pointHover && (
        <div className="heatmapTooltip" style={{ left: pointHover.clientX + 14, top: pointHover.clientY + 14 }}>
          {pointHover.point.lot_wafer_id && <strong>{pointHover.point.lot_wafer_id}</strong>}
          <div className="heatmapTooltipRow"><span>{data.axis.x_label.split(" ")[0]}</span><b>{pointHover.point.x.toFixed(1)}</b></div>
          <div className="heatmapTooltipRow"><span>{data.axis.y_label.split(" ")[0]}</span><b>{pointHover.point.y.toFixed(1)}</b></div>
          <div className="heatmapTooltipRow"><span>관리한계</span><b>{pointHover.point.in_range ? "내" : "밖"}</b></div>
          {/* 현재 Color By 기준값 -- 기존 항목은 그대로 두고 한 줄만 덧붙인다
              (spec §5-3). 항상 최신 colorMode를 읽으므로 전환 시 즉시 반영된다. */}
          {colorMode === "config_model" && pointHover.point.config && (() => {
            const parts = parseConfigParts(pointHover.point.config);
            return (
              <div className="scatterColorByRow">
                <div className="heatmapTooltipRow"><span>설비</span><b>{pointHover.point.config}</b></div>
                {parts && (
                  <div className="heatmapTooltipRow"><span /><b>모델 {parts.model} · 장비 {parts.equipment} · 챔버 {parts.chamber}</b></div>
                )}
              </div>
            );
          })()}
          {colorMode === "lot" && pointHover.point.lot_id && (
            <div className="scatterColorByRow">
              <div className="heatmapTooltipRow"><span>LOT</span><b>{pointHover.point.lot_id}</b></div>
            </div>
          )}
          {colorMode === "alarm" && (
            <div className="scatterColorByRow">
              <div className="heatmapTooltipRow"><span>판정</span><b>{alarmVerdict(pointHover.point, data.reference_lines)}</b></div>
            </div>
          )}
        </div>
      )}

      {disabledHint && (
        <div className="heatmapTooltip" style={{ left: disabledHint.x + 14, top: disabledHint.y + 14 }}>
          <strong>{disabledHint.text}</strong>
        </div>
      )}
    </div>
  );
}

function niceTicks(domain: [number, number], count: number): number[] {
  const [min, max] = domain;
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const step = (max - min) / count;
  const ticks: number[] = [];
  for (let i = 0; i <= count; i += 1) ticks.push(min + step * i);
  return ticks;
}

function formatTick(value: number): string {
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1);
}

/** `Step16_Model2_EQB_CH3` -> model/equipment/chamber (display-only split,
 * mirrors what colorForPoint's modelOf() already does for the "Config
 * 모델별" coloring itself -- src/config_parser.py deliberately never
 * decomposes Config server-side, so this stays purely a tooltip label). */
function parseConfigParts(config: string): { model: string; equipment: string; chamber: string } | null {
  const withoutStepPrefix = config.replace(/^Step\d+_/, "");
  const parts = withoutStepPrefix.split("_");
  if (parts.length !== 3) return null;
  const [model, equipment, chamber] = parts;
  return { model, equipment, chamber };
}

function alarmVerdict(point: ScatterPoint, referenceLines: ReferenceLine[]): string {
  if (point.in_range) return "관리한계 내";
  const lo = referenceLines.find((l) => l.key === "iqr_lo");
  const hi = referenceLines.find((l) => l.key === "iqr_hi");
  if (hi?.drawable && point.x > hi.value) return `관리한계 밖 (상한 +${(point.x - hi.value).toFixed(1)})`;
  if (lo?.drawable && point.x < lo.value) return `관리한계 밖 (하한 -${(lo.value - point.x).toFixed(1)})`;
  return "관리한계 밖";
}
