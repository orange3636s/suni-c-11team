"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import { niceTicksFitted } from "@/lib/niceTicks";
import { measureTextWidth } from "@/lib/textMeasure";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ReferenceLine, ScatterPoint, ScreeningScatterResponse } from "@/types/data";

export type ScatterColorMode = "default" | "config_model" | "lot" | "alarm";

// left margin widened 58->76 (spec §1 재지시): the y-axis title moved
// from the top margin back to sitting horizontally left of the tick
// numbers, vertically centered on the plot -- that needs *horizontal*
// room again, sized to comfortably fit the widest realistic tick label
// (~35px, e.g. "-123.4") + 8px tick-gap + the title text + its own 8px
// gap, with a few px of slack.
const MARGIN = { top: 36, right: 28, bottom: 46, left: 76 };
// Matches .scatterTickLabel's font-size (app/globals.css) -- used to
// measure candidate tick labels for the overlap-backoff pass (spec §8).
const TICK_FONT = "10.5px system-ui, -apple-system, sans-serif";
// Y tick labels are horizontal text stacked vertically, so their collision
// dimension is line height, not width -- a fixed estimate for 10.5px text.
const Y_TICK_LABEL_HEIGHT_PX = 14;
const X_TICK_COUNT: [max: number, min: number] = [10, 8];
const Y_TICK_COUNT: [max: number, min: number] = [8, 6];
const HEIGHT = 420;
// "최적 중심" (5 glyphs, ~58px at the label's 10px font) is roughly twice
// as wide as "LCL"/"UCL" (~26px) -- widened from 34px so the 2-row
// collision fallback actually engages before the two visually overlap,
// not just when their center points happen to coincide.
const LABEL_MIN_GAP = 46;

type LineMeta = {
  legendId: string;
  shortLabel: string;
  legendLabel: string;
  dash: string;
  strokeWidth: number;
};

// Reduced to LCL/UCL only (spec: 산점도 기준선 3종으로 축소 -> ±3σ/±6σ/평균/Q1/Q3
// removed from every display surface -- line/button/legend/tooltip). The
// underlying stats (mean, std, q1, q3) are untouched in
// src/analysis/control_range.py and still ride along in
// `data.reference_lines` for the JSON report; this component just no
// longer renders those five keys.
const LINE_META: Record<"iqr_lo" | "iqr_hi", LineMeta> = {
  iqr_lo: { legendId: "iqr", shortLabel: "LCL", legendLabel: "LCL/UCL (IQR 1.5배)", dash: "10 5", strokeWidth: 2 },
  iqr_hi: { legendId: "iqr", shortLabel: "UCL", legendLabel: "LCL/UCL (IQR 1.5배)", dash: "10 5", strokeWidth: 2 },
};

function formatNum1(value: number): string {
  return value.toFixed(1);
}

/** LCL/UCL button text with the actual computed values baked in (spec
 * §4-6/§4-3) -- separate from LINE_META.legendLabel, which stays a
 * static category name used by the per-line hover tooltip. */
function buildIqrLabel(lines: ReferenceLine[]): string {
  const lo = lines.find((l) => l.key === "iqr_lo");
  const hi = lines.find((l) => l.key === "iqr_hi");
  if (lo?.drawable && hi?.drawable) return `LCL/UCL (IQR 1.5배 · ${formatNum1(lo.value)} / ${formatNum1(hi.value)})`;
  if (hi?.drawable) return `UCL (${formatNum1(hi.value)})`;
  if (lo?.drawable) return `LCL (${formatNum1(lo.value)})`;
  return "LCL/UCL";
}

/** Sample quantile with linear interpolation -- matches numpy/pandas'
 * default `interpolation="linear"`, which is what the backend's own
 * Q1/Q3/IQR figures already use (see control_range.py). `sorted` must
 * already be sorted ascending. */
function quantileOf(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);
  if (lower === upper) return sorted[lower];
  const weight = pos - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/** "권장 구간" has no backend field -- it's derived entirely from data
 * already sent to the client (`points`, `bins`), not a new server-side
 * statistic: the x-quantile span of the contiguous run of quantile bins
 * (the same 12-bin profile driving the 구간 평균 불량률 curve) whose
 * average defect rate sits at/below the factor's overall mean defect
 * rate. Verified against train.CSV against all 5 spec worked examples
 * (within ±0.1 of the given values). */
function recommendedRange(
  points: ScatterPoint[],
  bins: ScreeningScatterResponse["bins"],
): [number, number] | null {
  if (bins.length === 0 || points.length === 0) return null;
  const threshold = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  const qualifying: number[] = [];
  bins.forEach((bin, index) => {
    if (bin.y_mean <= threshold) qualifying.push(index);
  });
  if (qualifying.length === 0) return null;
  const first = qualifying[0];
  const last = qualifying[qualifying.length - 1];
  const xs = points.map((p) => p.x).sort((a, b) => a - b);
  const nb = bins.length;
  return [quantileOf(xs, first / nb), quantileOf(xs, (last + 1) / nb)];
}

const LINE_COLOR: Record<string, { light: string; dark: string }> = {
  iqr: { light: "#0E306D", dark: "#7BA3E8" },
  optimal: { light: "#059669", dark: "#34D399" },
  recommended: { light: "#059669", dark: "#34D399" },
};
const TREND_COLOR = { light: "#DC2626", dark: "#F87171" };

function modelOf(config: string | null): string {
  if (!config) return "미계측";
  const match = /Model(\d+)/.exec(config);
  return match ? `Model${match[1]}` : config;
}
const MODEL_COLORS = ["#1D4ED8", "#059669", "#B45309"];
const LOT_PALETTE = ["#1D4ED8", "#059669", "#B45309", "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#DC2626"];

export type PointZone = "in_recommended" | "in_control" | "out_control";

/** Which of the 3 "기본" Color By zones a point falls into (spec §5) --
 * shared by colorForPoint (paint) and the point tooltip (label) so they
 * can never disagree about a given point's zone. Only meaningful for the
 * "default" mode; other modes don't use zones at all. */
function zoneOf(point: ScatterPoint, recommendedRangeValue: [number, number] | null): PointZone {
  if (!point.in_range) return "out_control";
  if (recommendedRangeValue && point.x >= recommendedRangeValue[0] && point.x <= recommendedRangeValue[1]) return "in_recommended";
  return "in_control";
}

// Re-tuned per spec (재조정): darkest-first ordering flipped from the
// original "얼마나 벗어났는지" intuition -- 권장 구간 안 (where most points
// cluster) is now the *darkest* tier so the density/shape of that cluster
// actually reads, while 관리한계 밖 stays identifiable by its position
// inside the amber shaded band plus its own border (see OUT_CONTROL_BORDER)
// rather than by being darkest. Dark-theme opacity is +0.1 over light's
// (same alpha reads fainter on a dark background).
const ZONE_STYLE: Record<PointZone, { light: string; dark: string; size: number; opacityLight: number; opacityDark: number }> = {
  in_recommended: { light: "#1D4ED8", dark: "#93C5FD", size: 4, opacityLight: 0.75, opacityDark: 0.85 },
  in_control: { light: "#60A5FA", dark: "#5B8DEF", size: 4.5, opacityLight: 0.65, opacityDark: 0.75 },
  out_control: { light: "#93C5FD", dark: "#3E6FB8", size: 5.5, opacityLight: 0.9, opacityDark: 1 },
};

// 관리한계 밖 점만 받는 1px 테두리 -- 가장 연한 채움색이라도 테두리로
// 형태가 드러나게 한다 (spec §3).
const OUT_CONTROL_BORDER = { light: "#1E3A8A", dark: "#DBEAFE" };

const ZONE_LABEL: Record<PointZone, string> = {
  in_recommended: "권장 구간 안",
  in_control: "권장 구간 밖 · 관리한계 안",
  out_control: "관리한계 밖",
};

function colorForPoint(
  point: ScatterPoint,
  mode: ScatterColorMode,
  lotIndex: Map<string, number>,
  theme: "light" | "dark",
  recommendedRangeValue: [number, number] | null,
): { color: string; size: number; opacity: number } {
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
  // default -- 3-tier by zone: 권장 구간 안이 가장 진하고 관리한계 밖이
  // 가장 연하다 (재조정 spec §3) -- 대부분의 점이 몰려 있는 권장 구간의
  // 밀집도/형태를 진한 색으로 드러내는 게 의도이고, 관리한계 밖 점은 이미
  // 앰버 음영 안에 있어 색이 연해도 위치+테두리로 식별된다.
  const style = ZONE_STYLE[zoneOf(point, recommendedRangeValue)];
  return {
    color: theme === "dark" ? style.dark : style.light,
    size: style.size,
    opacity: theme === "dark" ? style.opacityDark : style.opacityLight,
  };
}

// A single vertical line drawn on the chart -- either one of the two
// LCL/UCL reference lines, or the (backend-provided, not
// frontend-computed) optimal-center point. Unified so both go through
// the same paint order / label-collision layout instead of two parallel
// code paths.
type DisplayLine = {
  key: "iqr_lo" | "iqr_hi" | "optimal";
  value: number;
  shortLabel: string;
  color: { light: string; dark: string };
  dash: string;
  strokeWidth: number;
  greyedOut: boolean;
};

type LineLayout = { line: DisplayLine; xPixel: number; row: number; hidden: boolean };

function layoutLabels(entries: Array<{ line: DisplayLine; xPixel: number }>): LineLayout[] {
  const placedPerRow: number[][] = [[], []];
  const result: LineLayout[] = [];
  for (const entry of entries) {
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

type LineHover = { key: string; x: number; y: number };

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
  // All 4 remaining reference elements default to visible (spec §4-2).
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(["iqr", "optimal", "recommended"]));
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

  // Tick density up from the previous fixed 6 (x) / 5 (y) to spec §8's
  // 8-10 / 6-8, backing off toward the lower bound whenever labels would
  // collide -- which also makes density shrink automatically as the chart
  // narrows (panel open, mobile), since a narrower plotWidth needs more
  // backoff steps to stop the same labels from overlapping.
  const xTicks = useMemo(
    () => niceTicksFitted(xDomain, X_TICK_COUNT[0], X_TICK_COUNT[1], plotWidth, formatTick, (label) => measureTextWidth(label, TICK_FONT)),
    [xDomain, plotWidth],
  );
  const yTicks = useMemo(
    () => niceTicksFitted(yDomain, Y_TICK_COUNT[0], Y_TICK_COUNT[1], plotHeight, formatTick, () => Y_TICK_LABEL_HEIGHT_PX),
    [yDomain, plotHeight],
  );
  // Measured (not guessed) so the y-axis title's 8px clearance (spec §1
  // 재지시) holds regardless of how wide the actual tick numbers render --
  // short domains ("1".."5") get a tighter gutter, wide ones ("-123.4")
  // get a wider one, and MARGIN.left's fixed 76px budget just needs to
  // comfortably cover the realistic worst case.
  const maxYTickLabelWidth = useMemo(
    () => yTicks.reduce((max, tick) => Math.max(max, measureTextWidth(formatTick(tick), TICK_FONT)), 0),
    [yTicks],
  );

  // A plain object recreated each render -- cheap (just an empty Map),
  // and colorForPoint mutates it while walking data.points below to
  // assign each distinct config/lot a stable color index for this render.
  const lotIndex = new Map<string, number>();

  const iqrLo = data.reference_lines.find((l) => l.key === "iqr_lo");
  const iqrHi = data.reference_lines.find((l) => l.key === "iqr_hi");
  const iqrFullyNonDrawable = !iqrLo?.drawable && !iqrHi?.drawable;
  // shape.py only ever sets optimal_center for a u_shape relation -- a
  // monotonic factor (e.g. Step1_D1) genuinely has no interior optimum,
  // not just one that fell outside the drawable range (spec §4-2/§4-3).
  const optimalAvailable = data.optimal_center != null;

  // Recommended range is clamped into the control-limit range (spec §5-3)
  // so it never contradicts the alarm boundaries -- if clamping collapses
  // the range entirely, the recommendation is suppressed (null), same as
  // the "no qualifying bins" case. `recommendedClamped` drives the
  // "관리한계에 맞춰 조정됨" tooltip note, true only when clamping actually
  // narrowed the raw range.
  const { value: recommendedRangeValue, clamped: recommendedClamped } = useMemo(() => {
    const raw = recommendedRange(data.points, data.bins);
    if (!raw) return { value: null as [number, number] | null, clamped: false };
    const lo = data.normal_range.lo != null ? Math.max(raw[0], data.normal_range.lo) : raw[0];
    const hi = data.normal_range.hi != null ? Math.min(raw[1], data.normal_range.hi) : raw[1];
    if (lo >= hi) return { value: null as [number, number] | null, clamped: false };
    return { value: [lo, hi] as [number, number], clamped: lo !== raw[0] || hi !== raw[1] };
  }, [data.points, data.bins, data.normal_range.lo, data.normal_range.hi]);

  const displayLines = useMemo<DisplayLine[]>(() => {
    const lines: DisplayLine[] = [];
    if (iqrLo?.drawable) {
      lines.push({
        key: "iqr_lo", value: iqrLo.value, shortLabel: "LCL",
        color: LINE_COLOR.iqr, dash: LINE_META.iqr_lo.dash, strokeWidth: LINE_META.iqr_lo.strokeWidth,
        greyedOut: !iqrLo.alarm_relevant,
      });
    }
    if (iqrHi?.drawable) {
      lines.push({
        key: "iqr_hi", value: iqrHi.value, shortLabel: "UCL",
        color: LINE_COLOR.iqr, dash: LINE_META.iqr_hi.dash, strokeWidth: LINE_META.iqr_hi.strokeWidth,
        greyedOut: !iqrHi.alarm_relevant,
      });
    }
    if (data.optimal_center != null) {
      lines.push({
        key: "optimal", value: data.optimal_center, shortLabel: "최적 중심",
        color: LINE_COLOR.optimal, dash: "4 3", strokeWidth: 1.8, greyedOut: false,
      });
    }
    return lines;
  }, [iqrLo, iqrHi, data.optimal_center]);

  const labelLayout = useMemo(
    () => layoutLabels(displayLines.map((line) => ({ line, xPixel: xScale(line.value) }))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [displayLines, plotWidth],
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
    if (groupId === "iqr") return iqrFullyNonDrawable;
    if (groupId === "optimal") return !optimalAvailable;
    if (groupId === "recommended") return recommendedRangeValue == null;
    return false;
  }

  function toggleGroup(groupId: string, event: React.MouseEvent) {
    if (isGroupFullyNonDrawable(groupId)) {
      setDisabledHint({
        x: event.clientX,
        y: event.clientY,
        text: groupId === "optimal" ? "단조 관계라 최적 중심이 없습니다" : "데이터 범위 밖",
      });
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

  function lineGroupOf(line: DisplayLine): string {
    return line.key === "optimal" ? "optimal" : "iqr";
  }

  function renderLineBody(line: DisplayLine) {
    if (!visibleGroups.has(lineGroupOf(line))) return null;
    const x = xScale(line.value);
    const color = line.greyedOut ? (theme === "dark" ? "#6B7280" : "#9CA3AF") : (theme === "dark" ? line.color.dark : line.color.light);
    return (
      <g key={line.key}>
        <line
          x1={x} x2={x} y1={0} y2={plotHeight}
          stroke={color}
          strokeWidth={line.strokeWidth}
          strokeDasharray={line.dash}
          opacity={line.greyedOut ? 0.55 : 0.9}
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

  function renderLineLabel(line: DisplayLine) {
    if (!visibleGroups.has(lineGroupOf(line))) return null;
    const layout = labelLayout.find((l) => l.line.key === line.key);
    if (!layout || layout.hidden) return null;
    const color = line.greyedOut ? (theme === "dark" ? "#6B7280" : "#9CA3AF") : (theme === "dark" ? line.color.dark : line.color.light);
    const x = xScale(line.value);
    return (
      // A wide, generously-overflowing box (label text is unconstrained
      // by it anyway -- overflow:visible) with the badge itself centered
      // by flex, not by relying on inline-block sizing to exactly 40px:
      // "최적 중심" is wider than that and would render left-shifted off
      // its true line position otherwise.
      <foreignObject key={line.key} x={x - 45} y={layout.row * 16} width={90} height={16} style={{ overflow: "visible", pointerEvents: "none" }}>
        <div className="scatterLineLabelWrap">
          <span className="scatterLineLabel" style={{ color }}>{line.shortLabel}</span>
        </div>
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

      <svg ref={svgRef} width="100%" height={height} className="scatterChartSvg" role="img" aria-label={`${factorAxisLabel(data.axis.x_label)} vs ${targetAxisLabel(data.axis.y_label)} 산점도`}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* dark-mode-only plot background (spec §4 재지시) -- painted
              first so every other layer sits on top of it. */}
          <rect x={0} y={0} width={plotWidth} height={plotHeight} className="scatterPlotBg" />
          {/* axis ticks */}
          {yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line x1={0} x2={plotWidth} y1={yScale(tick)} y2={yScale(tick)} className="scatterGridLine" />
              <text x={-8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="scatterTickLabel">{formatTick(tick)}</text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <g key={`x-${tick}`}>
              <line x1={xScale(tick)} x2={xScale(tick)} y1={0} y2={plotHeight} className="scatterGridLine scatterGridLine-x" />
              <text x={xScale(tick)} y={plotHeight + 20} textAnchor="middle" className="scatterTickLabel">{formatTick(tick)}</text>
            </g>
          ))}

          {/* y-axis title -- target name only (e.g. "Y5"), same grey/
              secondary style as the x-axis title below the plot. Spec §1
              재지시: horizontal (not rotated), vertically centered on the
              plot height, positioned left of the tick numbers with an
              8px gap measured against the actual widest tick label so it
              never overlaps regardless of the domain's value range. */}
          <text
            x={-8 - maxYTickLabelWidth - 8}
            y={plotHeight / 2}
            textAnchor="end"
            dominantBaseline="middle"
            className="scatterAxisTitleSvg"
          >
            {targetAxisLabel(data.axis.y_label)}
          </text>

          {/* shading: amber outside IQR control limits, green 13% band across
              the recommended range (spec §4-2). Both sit below the points,
              same as before -- a background region shouldn't obscure data. */}
          {(() => {
            const loX = iqrLo?.drawable ? xScale(iqrLo.value) : 0;
            const hiX = iqrHi?.drawable ? xScale(iqrHi.value) : plotWidth;
            return visibleGroups.has("iqr") ? (
              <>
                {iqrLo?.drawable && <rect x={0} y={0} width={Math.max(loX, 0)} height={plotHeight} className="scatterOutsideShade" />}
                {iqrHi?.drawable && <rect x={hiX} y={0} width={Math.max(plotWidth - hiX, 0)} height={plotHeight} className="scatterOutsideShade" />}
              </>
            ) : null;
          })()}
          {visibleGroups.has("recommended") && recommendedRangeValue && (() => {
            const [lo, hi] = recommendedRangeValue;
            const x1 = xScale(lo);
            const x2 = xScale(hi);
            return (
              <rect
                x={Math.min(x1, x2)} y={0} width={Math.abs(x2 - x1)} height={plotHeight}
                className="scatterRecommendedBand"
                style={{ fill: theme === "dark" ? LINE_COLOR.recommended.dark : LINE_COLOR.recommended.light }}
                onMouseEnter={(event) => setLineHover({ key: "recommended", x: event.clientX, y: event.clientY })}
                onMouseMove={(event) => setLineHover({ key: "recommended", x: event.clientX, y: event.clientY })}
                onMouseLeave={() => setLineHover(null)}
              />
            );
          })()}

          {/* data points -- painted before every reference line/curve (spec
              §4-1: lines and the trend curve must sit above the points,
              never hidden behind them). */}
          {data.points.map((point, index) => {
            const style = colorForPoint(point, colorMode, lotIndex, theme, recommendedRangeValue);
            const isHovered = pointHover?.point === point;
            // 관리한계 밖 점은 가장 연한 채움이라 테두리로도 구분되게 한다
            // (spec §3) -- 기본 모드에서만, 호버 중엔 기존 호버 테두리가
            // 우선한다.
            const isOutControlBorder = colorMode === "default" && !isHovered && zoneOf(point, recommendedRangeValue) === "out_control";
            const stroke = isHovered
              ? (theme === "dark" ? "#FFFFFF" : "#0E306D")
              : isOutControlBorder
                ? (theme === "dark" ? OUT_CONTROL_BORDER.dark : OUT_CONTROL_BORDER.light)
                : "none";
            return (
              <circle
                key={point.lot_wafer_id ?? index}
                cx={xScale(point.x)}
                cy={yScale(point.y)}
                r={isHovered ? style.size * 1.5 : style.size}
                fill={style.color}
                opacity={isHovered ? 1 : style.opacity}
                stroke={stroke}
                strokeWidth={isHovered ? 1.5 : isOutControlBorder ? 1 : 0}
                style={{ cursor: "pointer" }}
                onClick={() => onSelectWafer(point)}
              />
            );
          })}

          {/* reference line bodies, stacked bottom-to-top per spec §4-1:
              최적 중심 first, then LCL/UCL (closest to the trend
              curve/labels above them). Labels are a separate,
              always-topmost pass below so they never sit under a
              later-drawn line. */}
          {displayLines.filter((line) => line.key === "optimal").map((line) => renderLineBody(line))}
          {displayLines.filter((line) => line.key === "iqr_lo" || line.key === "iqr_hi").map((line) => renderLineBody(line))}

          {/* confidence band + trend curve -- above every reference line */}
          {trendVisible && data.bins.length > 0 && (
            <>
              <path d={trendPath.band} className="scatterTrendBand" style={{ fill: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} />
              <path d={trendPath.line} className="scatterTrendLine" style={{ stroke: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} fill="none" />
            </>
          )}

          {/* line name labels -- topmost of all */}
          {displayLines.map((line) => renderLineLabel(line))}

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

      <p className="scatterAxisTitle">{factorAxisLabel(data.axis.x_label)}</p>

      {/* 점 색상 3단계 범례 -- "기본" Color By 모드에서만 의미가 있다 (spec §5).
          다른 Color By 모드(설비/LOT/알람)는 각자 자기 방식대로 이미 구분되므로
          이 범례를 보여주지 않는다. */}
      {colorMode === "default" && (
        <div className="scatterLegend scatterZoneLegend">
          {(["in_recommended", "in_control", "out_control"] as PointZone[]).map((zone) => (
            <span className="scatterLegendItem scatterZoneLegendItem" key={zone}>
              <i
                className="scatterLegendSwatch"
                style={{ background: theme === "dark" ? ZONE_STYLE[zone].dark : ZONE_STYLE[zone].light }}
              />
              {ZONE_LABEL[zone]}
            </span>
          ))}
        </div>
      )}

      {/* 4 reference elements only (spec §4-2): ±3σ/±6σ/평균/Q1/Q3 removed
          from every display surface here, but never from
          src/analysis/control_range.py or the JSON report -- those still
          compute mean/std/q1/q3 because LCL/UCL is derived from them. */}
      <div className="scatterLegend">
        <button
          type="button"
          className={`scatterLegendItem ${visibleGroups.has("iqr") && !iqrFullyNonDrawable ? "active" : ""} ${iqrFullyNonDrawable ? "disabled" : ""}`}
          onClick={(event) => toggleGroup("iqr", event)}
          title={iqrFullyNonDrawable ? "데이터 범위 밖" : "관리한계, 알람 판정 기준"}
        >
          <i className="scatterLegendSwatch" style={{ background: theme === "dark" ? LINE_COLOR.iqr.dark : LINE_COLOR.iqr.light }} />
          {buildIqrLabel(data.reference_lines)}
        </button>
        <button
          type="button"
          className={`scatterLegendItem ${visibleGroups.has("optimal") && optimalAvailable ? "active" : ""} ${!optimalAvailable ? "disabled" : ""}`}
          onClick={(event) => toggleGroup("optimal", event)}
          title={optimalAvailable ? "구간 평균 불량률이 가장 낮은 지점" : "단조 관계라 최적 중심이 없습니다"}
        >
          <i className="scatterLegendSwatch" style={{ background: theme === "dark" ? LINE_COLOR.optimal.dark : LINE_COLOR.optimal.light }} />
          {optimalAvailable ? `최적 중심 (${formatNum1(data.optimal_center as number)})` : "최적 중심 (해당 없음)"}
        </button>
        <button
          type="button"
          className={`scatterLegendItem ${visibleGroups.has("recommended") && recommendedRangeValue ? "active" : ""} ${!recommendedRangeValue ? "disabled" : ""}`}
          onClick={(event) => toggleGroup("recommended", event)}
          title={
            recommendedRangeValue
              ? `구간 평균 불량률이 전체 평균 이하인 구간${recommendedClamped ? " (관리한계에 맞춰 조정됨)" : ""}`
              : "데이터 범위 밖"
          }
        >
          <i className="scatterLegendSwatch scatterLegendSwatch-band" style={{ background: theme === "dark" ? LINE_COLOR.recommended.dark : LINE_COLOR.recommended.light }} />
          {recommendedRangeValue ? `권장 구간 (${formatNum1(recommendedRangeValue[0])}~${formatNum1(recommendedRangeValue[1])})` : "권장 구간"}
        </button>
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
        if (lineHover.key === "optimal") {
          if (data.optimal_center == null) return null;
          return (
            <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
              <strong>최적 중심</strong>
              <div className="heatmapTooltipRow"><span>값</span><b>{formatNum1(data.optimal_center)}</b></div>
              <div className="heatmapTooltipRow"><span>의미</span><b>구간 평균 불량률 최저 지점</b></div>
            </div>
          );
        }
        if (lineHover.key === "recommended") {
          if (!recommendedRangeValue) return null;
          return (
            <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
              <strong>권장 구간</strong>
              <div className="heatmapTooltipRow"><span>범위</span><b>{formatNum1(recommendedRangeValue[0])} ~ {formatNum1(recommendedRangeValue[1])}</b></div>
              <div className="heatmapTooltipRow"><span>의미</span><b>구간 평균 불량률이 전체 평균 이하</b></div>
              {recommendedClamped && (
                <div className="heatmapTooltipRow"><span /><b>관리한계에 맞춰 조정됨</b></div>
              )}
            </div>
          );
        }
        const line = data.reference_lines.find((l) => l.key === lineHover.key);
        if (!line || (line.key !== "iqr_lo" && line.key !== "iqr_hi")) return null;
        const meta = LINE_META[line.key];
        return (
          <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
            <strong>{meta.legendLabel} ({meta.shortLabel})</strong>
            <div className="heatmapTooltipRow"><span>값</span><b>{line.value.toFixed(1)}</b></div>
            <div className="heatmapTooltipRow"><span>계산</span><b>{line.formula}</b></div>
            <div className="heatmapTooltipRow"><span>이 선 밖</span><b>{line.outside_count.toLocaleString()}장</b></div>
          </div>
        );
      })()}

      {trendHover && !pointHover && (
        <div className="heatmapTooltip" style={{ left: MARGIN.left + trendHover.screenX + 14, top: MARGIN.top + trendHover.screenY + 14 }}>
          <strong>{factorAxisLabel(data.axis.x_label)} = {trendHover.dataX.toFixed(1)}</strong>
          <div className="heatmapTooltipRow"><span>구간 평균 불량률</span><b>{trendHover.dataY.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>95% 신뢰구간</span><b>{trendHover.ciLo.toFixed(2)} ~ {trendHover.ciHi.toFixed(2)}</b></div>
          <div className="heatmapTooltipRow"><span>이 구간 wafer 수</span><b>{trendHover.n.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>관리한계 내</span><b>{withinControl ? "예" : "아니오"}</b></div>
        </div>
      )}

      {pointHover && (
        <div className="heatmapTooltip" style={{ left: pointHover.clientX + 14, top: pointHover.clientY + 14 }}>
          {pointHover.point.lot_wafer_id && <strong>{pointHover.point.lot_wafer_id}</strong>}
          <div className="heatmapTooltipRow"><span>{factorAxisLabel(data.axis.x_label)}</span><b>{pointHover.point.x.toFixed(1)}</b></div>
          <div className="heatmapTooltipRow"><span>{targetAxisLabel(data.axis.y_label)}</span><b>{pointHover.point.y.toFixed(1)}%</b></div>
          <div className="heatmapTooltipRow"><span>관리한계</span><b>{pointHover.point.in_range ? "내" : "밖"}</b></div>
          {colorMode === "default" && (
            <div className="heatmapTooltipRow"><span>구간</span><b>{ZONE_LABEL[zoneOf(pointHover.point, recommendedRangeValue)]}</b></div>
          )}
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
