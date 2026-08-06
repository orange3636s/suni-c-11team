"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import { niceTicksFitted } from "@/lib/niceTicks";
import { measureTextWidth } from "@/lib/textMeasure";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { AlarmGrade, ReferenceLine, RelationShape, ScatterPoint, ScreeningScatterResponse, WindowMethod } from "@/types/data";

// "개선 권고"는 알람 삼각형 대상이 아니다 (spec §B: 마커는 심각/위험/주의 3
// 등급만 -- 알람이 아닌 참고용 등급까지 표시하면 무엇이 진짜 알람인지
// 흐려진다).
type AlarmMarkerGrade = "심각" | "위험" | "주의";
function isAlarmMarkerGrade(grade: AlarmGrade): grade is AlarmMarkerGrade {
  return grade === "심각" || grade === "위험" || grade === "주의";
}

export type ScatterColorMode = "default" | "config_model" | "lot" | "alarm";
export type ScatterView = "scatter" | "box";
// Unified with the trend curve's 12-quantile profile (spec §2: "Box Plot
// 도 12구간을 쓴다") -- was 10 (qcut decile) before, which is why the same
// factor's outlier count reads differently between the two profiles even
// though both are computed independently from the same points.
const BOX_BIN_COUNT = 12;
// 62%/50% of the per-category pixel spacing (spec §3-2/§3-4).
const BOX_WIDTH_RATIO = 0.62;
const BOX_CAP_WIDTH_RATIO = 0.32;
// Gaussian std-dev for point jitter, in category-axis units (spec §3-3).
const BOX_JITTER_STD = 0.1;

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

// 알람 판정 GBDT 전환 (spec §C-1/§C-3) -- 관리한계(IQR 1.5배, LCL/UCL)를
// 부분 의존도 기반 경고선으로 교체했다. ±3σ/±6σ/평균/Q1/Q3와 마찬가지로
// 기존 IQR 통계 자체는 src/analysis/control_range.py에 남아 있지만 화면에는
// 더 이상 그리지 않는다 (백엔드가 iqr_lo/iqr_hi를 reference_lines에서 이미
// 제외하고 warning_lo/warning_hi로 보낸다).
const LINE_META: Record<"warning_lo" | "warning_hi", LineMeta> = {
  warning_lo: { legendId: "warning", shortLabel: "경고선", legendLabel: "경고선 (예측 수율 기준)", dash: "10 5", strokeWidth: 2 },
  warning_hi: { legendId: "warning", shortLabel: "경고선", legendLabel: "경고선 (예측 수율 기준)", dash: "10 5", strokeWidth: 2 },
};

function formatNum1(value: number): string {
  return value.toFixed(1);
}

// Below this ε², the curve is close enough to flat that naming a shape
// (U자/단조) would read as a confident pattern the data doesn't actually
// support -- so the flatness message wins regardless of `relation_shape`
// (spec §2-2, priority 1).
const EPS2_FLAT_THRESHOLD = 0.02;

type ShapeCategory = "flat" | "u_shape" | "monotonic_increasing" | "monotonic_decreasing" | "unclear";

// ε² gates first (spec §7-1, priority 1): below threshold, the flatness
// message wins regardless of `relation_shape`, since naming a shape on a
// near-flat curve would read as a confident pattern the data doesn't
// actually support.
function classifyShape(eps2: number, shape: RelationShape): ShapeCategory {
  if (eps2 < EPS2_FLAT_THRESHOLD) return "flat";
  if (shape === "u_shape" || shape === "monotonic_increasing" || shape === "monotonic_decreasing") return shape;
  return "unclear";
}

/** Jargon-free reading of the chart above it (spec §7) -- no ρ/ε²/p-value,
 * factor/target names and the optimal-center value swapped in as real
 * numbers. The first sentence is identical regardless of `view` (spec
 * §7: "첫 줄은 완전히 같다"); only the second sentence -- how to read that
 * shape in *this* view -- differs, and is omitted entirely for the flat
 * category, which has nothing view-specific to add. `optimalCenter` only
 * ever backs the u_shape branch (monotonic relations have none by
 * construction -- see src/analysis/screening/shape.py); a u_shape whose
 * center got dropped (sparse min-y bin) still gets the U-shape message,
 * just without a number to anchor it to. */
function buildInterpretationTip(
  eps2: number,
  shape: RelationShape,
  optimalCenter: number | null,
  targetLabel: string,
  view: ScatterView,
): string {
  const category = classifyShape(eps2, shape);
  let first: string;
  switch (category) {
    case "flat":
      first = `곡선이 거의 평평합니다. 이 인자만으로는 ${targetLabel} 불량률을 설명하기 어렵습니다.`;
      break;
    case "u_shape":
      first = optimalCenter != null
        ? `값이 ${formatNum1(optimalCenter)}에서 멀어질수록 ${targetLabel} 불량률이 오르는 U자 형태입니다.`
        : `값이 낮아도 높아도 ${targetLabel} 불량률이 오르는 U자 형태입니다.`;
      break;
    case "monotonic_increasing":
      first = `값이 커질수록 ${targetLabel} 불량률이 오릅니다.`;
      break;
    case "monotonic_decreasing":
      first = `값이 커질수록 ${targetLabel} 불량률이 내려갑니다.`;
      break;
    default:
      first = `뚜렷한 방향성 없이 흩어져 있습니다. 이 인자만으로 ${targetLabel} 불량률의 방향을 판단하기 어렵습니다.`;
  }
  if (category === "flat") return first;
  if (view === "box") {
    return `${first} 상자가 낮고 짧을수록 평균이 낮고 흩어짐도 작다는 뜻입니다.`;
  }
  // 인과/조작 표현 금지 (spec 문구 전수 검토 §A-7, prompts/report_system.md
  // "절대 규칙 2"와 동일 기준) -- "~하면 유리합니다"/"조절하면 안 됩니다"는
  // 관찰된 상관관계를 개입 처방으로 읽히게 하므로, LLM 프롬프트가 이미 쓰는
  // "~구간에서 낮게/높게 관측된다" 식 서술로 통일한다.
  if (category === "u_shape") return `${first} 관측된 범위 양쪽 끝 모두에서 ${targetLabel} 불량률이 높게 나타납니다.`;
  if (category === "monotonic_increasing") return `${first} 낮은 구간에서 ${targetLabel} 불량률이 낮게 관측됩니다.`;
  if (category === "monotonic_decreasing") return `${first} 높은 구간에서 ${targetLabel} 불량률이 낮게 관측됩니다.`;
  return first;
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

/** Deterministic hash -> seeded PRNG -> gaussian sample, so a given
 * wafer's box-plot jitter offset never changes across re-renders (spec
 * §3-3: "지터 난수는 시드를 고정"), without needing to persist any extra
 * per-point state. */
function hashSeed(key: string): number {
  let h = 2166136261;
  for (let i = 0; i < key.length; i += 1) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function seededGaussian(key: string, stdDev: number): number {
  let seed = hashSeed(key);
  const next = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const u1 = Math.max(next(), 1e-9);
  const u2 = next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * stdDev;
}

type BoxJitteredPoint = { point: ScatterPoint; jitter: number; isOutlier: boolean };

type BoxBin = {
  index: number;
  xMean: number;
  xLo: number;
  xHi: number;
  n: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  whiskerLo: number;
  whiskerHi: number;
  members: BoxJitteredPoint[];
  outlierCount: number;
};

/** Equal-frequency 12-bin split of `points` by x -- matches pandas'
 * `qcut(x, 12)`, the same bin count as the trend curve/권장 구간 profile
 * (spec §2: 두 보기의 구간 수를 통일해 토글 시 빨간 곡선이 그대로 유지되게
 * 한다), though independently computed here since this needs full
 * per-bin y-quantile stats (Q1/median/Q3/whiskers), not just the mean the
 * trend profile carries: bin edges are the x-quantiles at each
 * one-twelfth, and membership is by value within (edge, edge] half-open
 * intervals (first bin's left edge inclusive, matching qcut's own
 * left-edge adjustment so the minimum value lands in bin 0) -- not a
 * plain positional slice of x-sorted points, which silently disagrees
 * with pandas whenever a quantile edge falls between two points with
 * different y-values (verified against train.CSV: positional slicing
 * undercounts Step16_R1/Y2's outliers at 30 vs qcut's 31). Box/whisker
 * stats per bin follow the standard Tukey convention: whiskers clamped to
 * the furthest in-fence data point, not the raw fence value itself, so a
 * whisker end always sits on a real wafer. */
function buildBoxBins(points: ScatterPoint[]): BoxBin[] {
  const n = points.length;
  if (n === 0) return [];
  const sortedX = points.map((p) => p.x).sort((a, b) => a - b);
  const edges = Array.from({ length: BOX_BIN_COUNT + 1 }, (_, i) => quantileOf(sortedX, i / BOX_BIN_COUNT));
  const chunks: ScatterPoint[][] = Array.from({ length: BOX_BIN_COUNT }, () => []);
  for (const point of points) {
    let idx = -1;
    for (let i = 0; i < BOX_BIN_COUNT; i += 1) {
      const lo = edges[i];
      const hi = edges[i + 1];
      if (i === 0 ? point.x >= lo && point.x <= hi : point.x > lo && point.x <= hi) {
        idx = i;
        break;
      }
    }
    if (idx === -1) idx = point.x <= edges[0] ? 0 : BOX_BIN_COUNT - 1;
    chunks[idx].push(point);
  }
  const bins: BoxBin[] = [];
  chunks.forEach((chunk, i) => {
    if (chunk.length === 0) return;
    const ys = chunk.map((p) => p.y).sort((a, b) => a - b);
    const q1 = quantileOf(ys, 0.25);
    const q3 = quantileOf(ys, 0.75);
    const median = quantileOf(ys, 0.5);
    const iqr = q3 - q1;
    const loFence = q1 - 1.5 * iqr;
    const hiFence = q3 + 1.5 * iqr;
    const inFenceYs = ys.filter((y) => y >= loFence && y <= hiFence);
    const whiskerLo = inFenceYs.length ? inFenceYs[0] : ys[0];
    const whiskerHi = inFenceYs.length ? inFenceYs[inFenceYs.length - 1] : ys[ys.length - 1];
    const mean = ys.reduce((sum, y) => sum + y, 0) / ys.length;
    let outlierCount = 0;
    const members: BoxJitteredPoint[] = chunk.map((point, memberIndex) => {
      const isOutlier = point.y < whiskerLo || point.y > whiskerHi;
      if (isOutlier) outlierCount += 1;
      const seedKey = point.lot_wafer_id ?? `bin${i}-idx${memberIndex}-x${point.x}-y${point.y}`;
      return { point, jitter: seededGaussian(seedKey, BOX_JITTER_STD), isOutlier };
    });
    const xs = chunk.map((p) => p.x);
    bins.push({
      index: bins.length,
      xMean: xs.reduce((sum, x) => sum + x, 0) / xs.length,
      xLo: Math.min(...xs),
      xHi: Math.max(...xs),
      n: chunk.length,
      mean, median, q1, q3, whiskerLo, whiskerHi,
      members, outlierCount,
    });
  });
  return bins;
}

/** Interpolates a real data-axis value into the box chart's 1..N
 * category-position space, using the bins' x-means as anchor points
 * (spec §3-5) -- values outside the first/last bin center clamp to the
 * nearest end instead of extrapolating. */
function categoryPositionForValue(value: number, bins: BoxBin[]): number {
  if (bins.length === 0) return 1;
  if (value <= bins[0].xMean) return 1;
  const last = bins[bins.length - 1];
  if (value >= last.xMean) return bins.length;
  for (let i = 0; i < bins.length - 1; i += 1) {
    const a = bins[i].xMean;
    const b = bins[i + 1].xMean;
    if (value >= a && value <= b) {
      const t = (value - a) / (b - a || 1);
      return i + 1 + t;
    }
  }
  return bins.length;
}

const LINE_COLOR: Record<string, { light: string; dark: string }> = {
  warning: { light: "#0E306D", dark: "#7BA3E8" },
};
// SPC/ML 방식 전환 (spec §3-4) -- 권장 구간 밴드/경계선/최적 중심/Box Plot
// 구간-내 박스 테두리가 전부 이 한 쌍을 공유한다. SPC는 사전 알람 로그의
// 기존 초록 토큰(#0D9668/#34D399, globals.css --band-inrec-*)과 동일해 두
// 화면이 시각적으로 일치하고, ML은 산점도의 파란 점(#1D4ED8/#60A5FA)·빨간
// 트렌드선(#DC2626)과 충분히 구분되도록 고른 보라(#9333EA/#C084FC)다.
const METHOD_COLOR: Record<WindowMethod, { light: string; dark: string }> = {
  spc: { light: "#0D9668", dark: "#34D399" },
  ml: { light: "#9333EA", dark: "#C084FC" },
};
const METHOD_LABEL: Record<WindowMethod, string> = { spc: "SPC", ml: "ML" };
const TREND_COLOR = { light: "#DC2626", dark: "#F87171" };

// 알람 판정 GBDT 전환 (spec §B-1) -- "주의"는 #EAB308 대신 #CA8A04를 쓴다:
// 빈 삼각형이라 테두리만 보이므로 밝은 노랑은 흰 배경에서 식별이 어렵다.
// 회색 4번째 등급은 만들지 않는다 (3단계만).
const ALARM_GRADE_COLOR: Record<"심각" | "위험" | "주의", { light: string; dark: string }> = {
  심각: { light: "#DC2626", dark: "#F87171" },
  위험: { light: "#EA580C", dark: "#FB923C" },
  주의: { light: "#CA8A04", dark: "#FACC15" },
};
// 점 반지름(style.size)의 diameter 기준 1.6배 (spec §B-1: "점이 4px이면
// 삼각형 한 변 6.5px" -- 점보다 조금 큰 정도로, 주변 점을 가리지 않는다).
const ALARM_TRIANGLE_RATIO = 1.6;
// z-index: 일반 점 < 박스 < 이상치 < 삼각형(항상 최상단), 여러 등급이
// 겹치면 심각이 가장 위 (spec §B-2).
const ALARM_GRADE_Z: Record<"심각" | "위험" | "주의", number> = { 심각: 3, 위험: 2, 주의: 1 };

// Box-plot-only palette (spec §3-2/§3-3) -- box border color depends on
// whether the bin's x-mean sits inside the recommended range (in which
// case it takes the active method's color, see METHOD_COLOR), everything
// else is fixed per element.
const BOX_COLOR = {
  boxOut: { light: "#0E306D", dark: "#7BA3E8" },
  median: { light: "#DC2626", dark: "#F87171" },
  whisker: { light: "#0E306D", dark: "#7BA3E8" },
  inlier: { light: "#2563EB", dark: "#60A5FA" },
  outlier: { light: "#DC2626", dark: "#F87171" },
};

// Box Plot legend (spec §5) -- small presentational swatches drawn as
// mini inline SVGs so each icon actually matches its written description
// (a bordered box + median tick, an I-beam whisker, jittered dots, hollow
// outlier rings, a dashed/dotted reference line, a translucent band).
// Rendered through the same `LegendCard` as every other legend entry
// (spec §4: box-plot legend items are click-toggleable too, apart from
// the box body itself), not a separate read-only component.
function IconBoxWhisker({ boxColor, medianColor }: { boxColor: string; medianColor: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <rect x="4" y="3" width="14" height="10" fill="none" stroke={boxColor} strokeWidth="1.6" />
      <line x1="4" y1="8" x2="18" y2="8" stroke={medianColor} strokeWidth="2" />
    </svg>
  );
}
function IconMedian({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <line x1="2" y1="8" x2="20" y2="8" stroke={color} strokeWidth="2.4" />
    </svg>
  );
}
function IconWhisker({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <line x1="11" y1="2" x2="11" y2="14" stroke={color} strokeWidth="1.2" />
      <line x1="7" y1="2" x2="15" y2="2" stroke={color} strokeWidth="1.2" />
      <line x1="7" y1="14" x2="15" y2="14" stroke={color} strokeWidth="1.2" />
    </svg>
  );
}
function IconDots({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <circle cx="6" cy="10" r="2" fill={color} opacity="0.6" />
      <circle cx="11" cy="5" r="2" fill={color} opacity="0.6" />
      <circle cx="16" cy="11" r="2" fill={color} opacity="0.6" />
    </svg>
  );
}
function IconOutlierDots({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <circle cx="6" cy="8" r="3" fill="none" stroke={color} strokeWidth="1.2" />
      <circle cx="12" cy="6" r="3" fill="none" stroke={color} strokeWidth="1.2" />
      <circle cx="18" cy="9" r="3" fill="none" stroke={color} strokeWidth="1.2" />
    </svg>
  );
}

/** 삼각형 꼭짓점 3개 -- 위쪽 꼭짓점이 (cx, cy - r)에 오는 정삼각형 근사.
 * 알람 판정 GBDT 전환 spec §B-1: "속이 빈 삼각형. 채우지 마라." */
function trianglePoints(cx: number, cy: number, r: number): string {
  const a = (cx - r * 0.87).toFixed(2);
  const b = (cx + r * 0.87).toFixed(2);
  const top = (cy - r).toFixed(2);
  const bottom = (cy + r * 0.5).toFixed(2);
  return `${cx.toFixed(2)},${top} ${a},${bottom} ${b},${bottom}`;
}

function IconTriangle({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <polygon points={trianglePoints(11, 9, 6)} fill="none" stroke={color} strokeWidth="1.4" />
    </svg>
  );
}
function IconDashedLine({ color, dotted }: { color: string; dotted?: boolean }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <line x1="2" y1="8" x2="20" y2="8" stroke={color} strokeWidth="1.8" strokeDasharray={dotted ? "2 3" : "5 3"} />
    </svg>
  );
}
function IconTrendLine({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <line x1="2" y1="8" x2="20" y2="8" stroke={color} strokeWidth="1.5" strokeDasharray="4 3" />
      <circle cx="11" cy="8" r="2.4" fill={color} />
    </svg>
  );
}
function IconBand({ color }: { color: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <rect x="2" y="1" width="18" height="14" fill={color} opacity="0.35" />
    </svg>
  );
}
/** Read-only-or-toggle legend card: swatch + bold label + small
 * description, stacked (spec §3). Renders as a `<button>` (and picks up
 * `.active`/`.disabled`) when `onClick` is given -- reused for both the
 * static Color By row and the still-toggleable reference-line row so
 * clicking LCL/UCL, 최적 중심, 권장 구간, 구간 평균 불량률 keeps working
 * exactly as before (spec §3-3: 기존 토글 기능 유지). */
function LegendCard({
  icon,
  label,
  desc,
  onClick,
  active,
  disabled,
  title,
}: {
  icon: React.ReactNode;
  label: string;
  desc?: string;
  onClick?: (event: React.MouseEvent) => void;
  active?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  const className = [
    "scatterLegendCard",
    onClick ? "interactive" : "",
    active ? "active" : "",
    disabled ? "disabled" : "",
  ].filter(Boolean).join(" ");
  const inner = (
    <>
      <span className="scatterLegendCardSwatch">{icon}</span>
      <span className="scatterLegendCardText">
        <strong>{label}</strong>
        {/* Only lines that name a computed threshold/value or a concrete
            action earn a description (spec §5-1) -- everything else is
            swatch + label only, so an empty/undefined desc renders no
            <small> at all rather than a blank line. */}
        {desc ? <small>{desc}</small> : null}
      </span>
    </>
  );
  if (onClick) {
    return (
      <button type="button" className={className} onClick={onClick} title={title}>
        {inner}
      </button>
    );
  }
  return (
    <div className={className} title={title}>
      {inner}
    </div>
  );
}

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

// Point-tooltip zone label (spec §5) -- the legend itself no longer names
// these zones (row 1's color-by breakdown was removed per spec §5-4: the
// legend describes reference lines/box elements only, never how points
// are colored), but the tooltip still needs a plain-language "which zone
// is this point in" line for "기본" Color By.
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
  key: "warning_lo" | "warning_hi" | "optimal";
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

type PointHover = { screenX: number; screenY: number; clientX: number; clientY: number; point: ScatterPoint; isOutlier?: boolean };

type BoxHover = { clientX: number; clientY: number; bin: BoxBin };

const POINT_HOVER_RADIUS_PX = 16;

export default function ScatterChart({
  data,
  colorMode,
  view,
  method,
  onSelectWafer,
  height = HEIGHT,
  alarmGradeByWaferId,
}: {
  data: ScreeningScatterResponse;
  colorMode: ScatterColorMode;
  // Owned by the caller now (spec §1-3: 보기 토글 lives in the card header,
  // same row as the title) -- this component only reads it to decide what
  // to render, it never renders the toggle buttons itself.
  view: ScatterView;
  // Which of data.methods.{spc,ml} drives the recommended-range band and
  // optimal-center line (spec §3) -- same "caller owns the toggle state"
  // pattern as `view`. Defaults to "spc" when the caller hasn't resolved
  // data.methods.adopted yet (first render before the toggle mounts).
  method?: WindowMethod;
  onSelectWafer: (point: ScatterPoint) => void;
  height?: number;
  // 알람 판정 GBDT 전환 (spec §B) -- wafer_id -> 등급. 카드 하나가 한 번의
  // "원인 분석 실행"에서 나온 알람 목록 전체를 공유하므로 caller가 한 번만
  // 가져와 모든 ScatterChart 인스턴스에 그대로 넘긴다 (여기서 재요청하지
  // 않는다).
  alarmGradeByWaferId?: Record<string, AlarmGrade>;
}) {
  const activeMethod: WindowMethod = method ?? "spc";
  const methodColor = METHOD_COLOR[activeMethod];
  const theme = useResolvedTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerWidth, setContainerWidth] = useState(680);
  // All 4 remaining reference elements default to visible (spec §4-2).
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(["warning", "optimal", "recommended"]));
  // 알람 심각도 3등급 -- 등급별로 독립적으로 켜고 끌 수 있다 (spec §B-3).
  // 기본은 전부 켜짐.
  const [visibleAlarmGrades, setVisibleAlarmGrades] = useState<Set<AlarmMarkerGrade>>(
    () => new Set(["심각", "위험", "주의"]),
  );
  const [trendVisible, setTrendVisible] = useState(true);
  // Box Plot's own togglable elements (spec §4) -- 개별 wafer/이상치 wafer
  // can be switched off from the legend same as every other reference
  // element; 상자/중앙값/수염 (the box body itself) intentionally have no
  // matching state, since those legend cards render with no onClick.
  const [boxPointsVisible, setBoxPointsVisible] = useState({ inlier: true, outlier: true });
  const [lineHover, setLineHover] = useState<LineHover | null>(null);
  const [trendHover, setTrendHover] = useState<TrendHover | null>(null);
  const [pointHover, setPointHover] = useState<PointHover | null>(null);
  const [disabledHint, setDisabledHint] = useState<{ x: number; y: number; text: string } | null>(null);
  const [boxHover, setBoxHover] = useState<BoxHover | null>(null);

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

  // 10-bin box/whisker profile computed client-side from the same points
  // the scatter view already has -- no extra API call (spec §7).
  const boxBins = useMemo(() => buildBoxBins(data.points), [data.points]);

  // Box mode's x-axis is categorical (1..N bin slots, evenly spaced with
  // half a slot of padding on each side) rather than continuous -- this
  // scale converts a category position to a pixel offset.
  const catDomain = useMemo<[number, number]>(() => [0.5, Math.max(boxBins.length, 1) + 0.5], [boxBins.length]);
  const catScale = (cat: number) => ((cat - catDomain[0]) / (catDomain[1] - catDomain[0])) * plotWidth;
  const catUnitPx = plotWidth / (catDomain[1] - catDomain[0]);
  const boxWidthPx = catUnitPx * BOX_WIDTH_RATIO;
  const boxCapWidthPx = catUnitPx * BOX_CAP_WIDTH_RATIO;

  // Shared x-position resolver for every reference element (LCL/UCL,
  // 최적 중심, 권장 구간, 구간 평균 불량률) -- scatter mode places them by
  // real value on the continuous xScale same as always; box mode
  // re-projects that same real value onto the categorical box axis by
  // interpolating against the box bins' x-means (spec §3-5), so "LCL"
  // still lands between the two box columns whose data actually spans it.
  const plotX = (value: number): number =>
    view === "box" ? catScale(categoryPositionForValue(value, boxBins)) : xScale(value);

  // Tick density up from the previous fixed 6 (x) / 5 (y) to spec §8's
  // 8-10 / 6-8, backing off toward the lower bound whenever labels would
  // collide -- which also makes density shrink automatically as the chart
  // narrows (panel open, mobile), since a narrower plotWidth needs more
  // backoff steps to stop the same labels from overlapping. Below a
  // ~240px-tall mobile chart (spec: JSON 보고서 버튼 제거 · 모바일 레이아웃
  // 전환 §B-6, "x축 4~5개 · y축 4개"), the desktop band itself drops too --
  // niceTicksFitted's own overlap backoff alone doesn't reliably reach
  // that few on a chart this compact.
  const isCompactChart = plotHeight < 200;
  const xTicks = useMemo(() => {
    const [max, min] = isCompactChart ? [5, 4] : X_TICK_COUNT;
    return niceTicksFitted(xDomain, max, min, plotWidth, formatTick, (label) => measureTextWidth(label, TICK_FONT));
  }, [xDomain, plotWidth, isCompactChart]);
  // Box mode: one tick per bin, at its column center, labeled with the
  // bin's x-mean rounded to an integer (spec §3-1) -- not the shared
  // niceTicksFitted continuous-axis logic, since these positions are
  // fixed category slots, not values to round to "nice" numbers. Falls
  // back to one decimal place for every label, uniformly, whenever
  // integer rounding would collide two adjacent bins onto the same
  // number (spec §2: "60이 두 번" -- 12 narrower bins make this far more
  // likely than the old 10).
  const boxXTicks = useMemo(() => {
    const rounded = boxBins.map((bin) => Math.round(bin.xMean));
    const hasDuplicate = rounded.some((value, i) => rounded.indexOf(value) !== i);
    return boxBins.map((bin, i) => ({
      pixel: catScale(bin.index + 1),
      label: hasDuplicate ? formatNum1(bin.xMean) : String(rounded[i]),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boxBins, plotWidth]);
  const yTicks = useMemo(() => {
    const [max, min] = isCompactChart ? [4, 4] : Y_TICK_COUNT;
    return niceTicksFitted(yDomain, max, min, plotHeight, formatTick, () => Y_TICK_LABEL_HEIGHT_PX);
  }, [yDomain, plotHeight, isCompactChart]);
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

  const warningLo = data.reference_lines.find((l) => l.key === "warning_lo");
  const warningHi = data.reference_lines.find((l) => l.key === "warning_hi");
  const warningFullyNonDrawable = !warningLo?.drawable && !warningHi?.drawable;

  // 알람 판정 GBDT 전환 (spec §B-4) -- 삼각형은 관리한계/권장구간과 무관하게
  // 어디에나 나타날 수 있다: 다변량 판정이므로 정상이다.
  function alarmGradeOf(waferId: string | null): AlarmMarkerGrade | null {
    if (!waferId || !alarmGradeByWaferId) return null;
    const grade = alarmGradeByWaferId[waferId];
    return grade && isAlarmMarkerGrade(grade) ? grade : null;
  }
  const alarmGradeCounts: Record<AlarmMarkerGrade, number> = { 심각: 0, 위험: 0, 주의: 0 };
  if (alarmGradeByWaferId) {
    for (const point of data.points) {
      const grade = alarmGradeOf(point.lot_wafer_id);
      if (grade) alarmGradeCounts[grade] += 1;
    }
  }
  const hasAnyAlarmMarker = alarmGradeCounts.심각 + alarmGradeCounts.위험 + alarmGradeCounts.주의 > 0;
  // The active method's own window/center (spec §3) -- null when
  // data.methods is null (Config factors, never routed through this
  // component in practice) or that method couldn't fit a window at all.
  const activeMethodWindow = data.methods?.[activeMethod] ?? null;

  // shape.py only ever sets optimal_center for a u_shape relation -- a
  // monotonic factor (e.g. Step1_D1) genuinely has no interior optimum,
  // not just one that fell outside the drawable range (spec §4-2/§4-3).
  // This gate is intentionally method-independent: whether a factor has
  // a genuine U-shaped optimum at all doesn't change when the SPC/ML
  // toggle flips, only *where* that optimum sits does.
  const optimalAvailable = data.optimal_center != null;
  // A classified center that got dropped downstream (fell outside its
  // own recommended window after control-range clamping, or was picked
  // from a sparse/outlier-widened bin) gets its own specific reason
  // instead of the generic "단조 관계라..." message (spec §3-3/§3-4).
  const optimalUnavailableReason = data.optimal_center_dropped_reason ?? "단조 관계라 최적 중심이 없습니다";
  // The displayed center value follows the active method once one is
  // available server-side; falls back to the legacy client-only value
  // (identical to methods.spc.optimal_center in practice, see
  // window_methods.py) if `data.methods` is missing for some reason.
  const displayedOptimalCenter = activeMethodWindow?.optimal_center ?? data.optimal_center;

  // Recommended range/최적중심 come straight from the backend's SPC/ML
  // comparison (spec §2-1/§3-3) -- already clamped into the control
  // range there, so no client-side re-derivation or re-clamping needed.
  // `recommendedClamped` drives the "관리한계에 맞춰 조정됨" tooltip note.
  const recommendedRangeValue: [number, number] | null = activeMethodWindow ? activeMethodWindow.window : null;
  const recommendedClamped = activeMethodWindow?.clamped ?? false;

  const displayLines = useMemo<DisplayLine[]>(() => {
    const lines: DisplayLine[] = [];
    if (warningLo?.drawable) {
      lines.push({
        key: "warning_lo", value: warningLo.value, shortLabel: "경고선",
        color: LINE_COLOR.warning, dash: LINE_META.warning_lo.dash, strokeWidth: LINE_META.warning_lo.strokeWidth,
        greyedOut: !warningLo.alarm_relevant,
      });
    }
    if (warningHi?.drawable) {
      lines.push({
        key: "warning_hi", value: warningHi.value, shortLabel: "경고선",
        color: LINE_COLOR.warning, dash: LINE_META.warning_hi.dash, strokeWidth: LINE_META.warning_hi.strokeWidth,
        greyedOut: !warningHi.alarm_relevant,
      });
    }
    if (displayedOptimalCenter != null) {
      lines.push({
        key: "optimal", value: displayedOptimalCenter, shortLabel: "최적 중심",
        color: methodColor, dash: "4 3", strokeWidth: 1.9, greyedOut: false,
      });
    }
    return lines;
  }, [warningLo, warningHi, displayedOptimalCenter, methodColor]);

  const labelLayout = useMemo(
    () => layoutLabels(displayLines.map((line) => ({ line, xPixel: plotX(line.value) }))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [displayLines, plotWidth, view, boxBins],
  );

  const trendPath = useMemo(() => {
    if (!data.bins.length) return { segments: [] as { d: string; dashed: boolean }[], band: "", markers: [] as { x: number; y: number }[] };
    const points = data.bins.map((b) => [plotX(b.x_mean), yScale(b.y_mean)] as const);
    // One segment per adjacent bin pair, dashed whenever either endpoint
    // bin is `sparse` (outlier-widened -- spec §3-4: "표본이 넓게 흩어진
    // 구간" gets a visibly different line style, not silently blended
    // into the solid curve). In box mode every segment is dashed
    // regardless of sparseness (spec §3-4: 실선->점선), plus a circle
    // marker is drawn at each bin so it reads distinctly from the box
    // plot's own solid median line.
    const segments = points.slice(1).map(([x2, y2], i) => {
      const [x1, y1] = points[i];
      return { d: `M${x1},${y1} L${x2},${y2}`, dashed: view === "box" || data.bins[i].sparse || data.bins[i + 1].sparse };
    });
    const upper = data.bins.map((b) => [plotX(b.x_mean), yScale(b.y_hi)] as const);
    const lower = [...data.bins].reverse().map((b) => [plotX(b.x_mean), yScale(b.y_lo)] as const);
    const band = [...upper, ...lower].map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";
    const markers = points.map(([x, y]) => ({ x, y }));
    return { segments, band, markers };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.bins, plotWidth, plotHeight, view, boxBins]);

  function isGroupFullyNonDrawable(groupId: string): boolean {
    if (groupId === "warning") return warningFullyNonDrawable;
    if (groupId === "optimal") return !optimalAvailable;
    if (groupId === "recommended") return recommendedRangeValue == null;
    return false;
  }

  function toggleGroup(groupId: string, event: React.MouseEvent) {
    if (isGroupFullyNonDrawable(groupId)) {
      setDisabledHint({
        x: event.clientX,
        y: event.clientY,
        text: groupId === "optimal" ? optimalUnavailableReason : "데이터 범위 밖",
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

  // Box mode's points sit at their bin's jittered category position, not
  // at their real x value -- so nearest-point search needs its own pass
  // over the already-jittered screen coordinates rather than xScale.
  function findNearestBoxPoint(relativeX: number, relativeY: number): { point: ScatterPoint; screenX: number; screenY: number; isOutlier: boolean } | null {
    let best: { point: ScatterPoint; screenX: number; screenY: number; isOutlier: boolean } | null = null;
    let bestDistance = POINT_HOVER_RADIUS_PX;
    for (const bin of boxBins) {
      for (const member of bin.members) {
        const screenX = catScale(bin.index + 1 + member.jitter);
        const screenY = yScale(member.point.y);
        const dx = screenX - relativeX;
        const dy = screenY - relativeY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = { point: member.point, screenX, screenY, isOutlier: member.isOutlier };
        }
      }
    }
    return best;
  }

  // Which box column (if any) the cursor sits over -- used for the box
  // summary tooltip (spec §5). Column hit area is the box's own pixel
  // width, not the full category slot, so hovering the empty gap between
  // two boxes doesn't spuriously show a tooltip. Note: this can't be a
  // plain onMouseEnter/Leave on the box <rect> itself, since the shared
  // plot overlay rect below paints on top of every other plot element and
  // would swallow those events -- same reason point/trend hover already
  // route through this one handler instead of per-shape listeners.
  function findBoxColumnAt(relativeX: number): BoxBin | null {
    for (const bin of boxBins) {
      const centerX = catScale(bin.index + 1);
      if (Math.abs(relativeX - centerX) <= boxWidthPx / 2) return bin;
    }
    return null;
  }

  function handlePlotOverlayMouseMove(event: React.MouseEvent<SVGRectElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const relativeX = event.clientX - rect.left - MARGIN.left;
    const relativeY = event.clientY - rect.top - MARGIN.top;
    if (view === "box") {
      const nearest = findNearestBoxPoint(relativeX, relativeY);
      if (nearest) {
        setPointHover({ screenX: nearest.screenX, screenY: nearest.screenY, clientX: event.clientX, clientY: event.clientY, point: nearest.point, isOutlier: nearest.isOutlier });
        setBoxHover(null);
        return;
      }
      setPointHover(null);
      const column = findBoxColumnAt(relativeX);
      setBoxHover(column ? { clientX: event.clientX, clientY: event.clientY, bin: column } : null);
      return;
    }
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

  const interpretationText = buildInterpretationTip(data.eps2, data.relation_shape, data.optimal_center, targetAxisLabel(data.axis.y_label), view);

  // Reference-line legend labels/descriptions (spec §5-1/§5-2) -- the
  // numeric value now rides in the *label* itself ("관리한계 LCL/UCL  49.9
  // / 69.9"), not the description, so the description is free to hold
  // only the one thing a swatch alone can't convey: what it means to be
  // outside it. A one-sided factor (e.g. Step1_D1, LCL-less) collapses to
  // a single value instead of a slash pair (spec §5-6).
  // 알람 판정 GBDT 전환 (spec §C-3/§C-4-1) -- 값은 라벨에, 실측 수율 차이는
  // 설명에 싣는다. 예측값이 아니라 관측값이다 (§C-4-1 "예측 기반 수치를
  // 범례에 쓰지 마라"). 표본이 30장 미만이면 백엔드가 gap을 null로 보내고,
  // 그 방향은 수치 없이 "표본이 적어..."로 대체한다.
  function warningLegendLabel(): string {
    if (warningHi?.drawable && !warningLo?.drawable) return `경고선  ${formatNum1(warningHi.value)}`;
    if (warningLo?.drawable && !warningHi?.drawable) return `경고선  ${formatNum1(warningLo.value)}`;
    if (warningLo?.drawable && warningHi?.drawable) return `경고선  ${formatNum1(warningLo.value)} / ${formatNum1(warningHi.value)}`;
    return "경고선";
  }
  function warningLegendDesc(): string {
    const parts: string[] = [];
    if (warningHi?.drawable) {
      parts.push(
        warningHi.observed_yield_gap_pp != null
          ? `이 값을 넘은 wafer의 실제 수율이 ${Math.abs(warningHi.observed_yield_gap_pp).toFixed(1)}%p 낮게 관측되었습니다`
          : "이 값을 넘은 표본이 적어 수율 차이를 계산할 수 없습니다",
      );
    }
    if (warningLo?.drawable) {
      parts.push(
        warningLo.observed_yield_gap_pp != null
          ? `이 값보다 낮은 wafer의 실제 수율이 ${Math.abs(warningLo.observed_yield_gap_pp).toFixed(1)}%p 낮게 관측되었습니다`
          : "이 값보다 낮은 표본이 적어 수율 차이를 계산할 수 없습니다",
      );
    }
    return parts.join(" · ") || "예측 수율이 낮게 관측되는 구간의 경계";
  }
  // 단조 인자는 최적 중심 자체가 개념적으로 없으므로 항목을 아예 숨긴다
  // (spec §3-4) -- u_shape/unclear에서 값이 빠진 경우는 기존처럼 "해당
  // 없음" 비활성 카드로 남겨 이유를 알 수 있게 한다.
  const showOptimalLegendCard = data.relation_shape !== "monotonic_increasing" && data.relation_shape !== "monotonic_decreasing";

  function lineGroupOf(line: DisplayLine): string {
    return line.key === "optimal" ? "optimal" : "warning";
  }

  function renderLineBody(line: DisplayLine) {
    if (!visibleGroups.has(lineGroupOf(line))) return null;
    const x = plotX(line.value);
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
    const x = plotX(line.value);
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
      {/* 해석 팁 (spec §2) -- 차트 바로 위, 제목/메타 정보와는 별도 줄로
          쌓인다. 통계 용어 없이 이 차트가 무엇을 보여주는지 한 줄로
          요약한다. */}
      <div className="scatterInterpretationTip">
        <span className="scatterInterpretationTipLabel">해석</span>
        <p className="scatterInterpretationTipBody">{interpretationText}</p>
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
          {view === "scatter"
            ? xTicks.map((tick) => (
                <g key={`x-${tick}`}>
                  <line x1={xScale(tick)} x2={xScale(tick)} y1={0} y2={plotHeight} className="scatterGridLine scatterGridLine-x" />
                  <text x={xScale(tick)} y={plotHeight + 20} textAnchor="middle" className="scatterTickLabel">{formatTick(tick)}</text>
                </g>
              ))
            : boxXTicks.map((tick) => (
                <g key={`xbox-${tick.pixel}`}>
                  <line x1={tick.pixel} x2={tick.pixel} y1={0} y2={plotHeight} className="scatterGridLine scatterGridLine-x" />
                  <text x={tick.pixel} y={plotHeight + 20} textAnchor="middle" className="scatterTickLabel">{tick.label}</text>
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

          {/* shading: amber outside the 경고선 (알람 판정 GBDT 전환 §C-3,
              IQR 관리한계 대신 PDP 기반 경고선), green 13% band across
              the recommended range (spec §4-2). Both sit below the points,
              same as before -- a background region shouldn't obscure data. */}
          {(() => {
            const loX = warningLo?.drawable ? plotX(warningLo.value) : 0;
            const hiX = warningHi?.drawable ? plotX(warningHi.value) : plotWidth;
            return visibleGroups.has("warning") ? (
              <>
                {warningLo?.drawable && <rect x={0} y={0} width={Math.max(loX, 0)} height={plotHeight} className="scatterOutsideShade" />}
                {warningHi?.drawable && <rect x={hiX} y={0} width={Math.max(plotWidth - hiX, 0)} height={plotHeight} className="scatterOutsideShade" />}
              </>
            ) : null;
          })()}
          {visibleGroups.has("recommended") && recommendedRangeValue && (() => {
            const [lo, hi] = recommendedRangeValue;
            const x1 = Math.min(plotX(lo), plotX(hi));
            const x2 = Math.max(plotX(lo), plotX(hi));
            const fill = theme === "dark" ? methodColor.dark : methodColor.light;
            return (
              <>
                <rect
                  x={x1} y={0} width={x2 - x1} height={plotHeight}
                  className="scatterRecommendedBand"
                  style={{ fill }}
                  onMouseEnter={(event) => setLineHover({ key: "recommended", x: event.clientX, y: event.clientY })}
                  onMouseMove={(event) => setLineHover({ key: "recommended", x: event.clientX, y: event.clientY })}
                  onMouseLeave={() => setLineHover(null)}
                />
                {/* 2px 경계선 -- alpha만으로는 그 안의 점이 묻히므로 두께로
                    보완한다 (spec §3-4: "밴드 알파를 0.15보다 올리지 마라"). */}
                <rect
                  x={x1} y={0} width={x2 - x1} height={plotHeight}
                  fill="none" stroke={fill} strokeWidth={2} pointerEvents="none"
                />
                {x2 - x1 > 60 && (
                  <text
                    x={(x1 + x2) / 2} y={16} textAnchor="middle"
                    className="scatterBandLabel"
                    style={{ fill }}
                    pointerEvents="none"
                  >
                    {`권장 구간 (${METHOD_LABEL[activeMethod]})`}
                  </text>
                )}
              </>
            );
          })()}

          {/* data points -- painted before every reference line/curve (spec
              §4-1: lines and the trend curve must sit above the points,
              never hidden behind them). */}
          {view === "scatter"
            ? data.points.map((point, index) => {
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
              })
            : boxBins.map((bin) => {
                const inRecommended = recommendedRangeValue != null && bin.xMean >= recommendedRangeValue[0] && bin.xMean <= recommendedRangeValue[1];
                const boxColor = theme === "dark"
                  ? (inRecommended ? methodColor.dark : BOX_COLOR.boxOut.dark)
                  : (inRecommended ? methodColor.light : BOX_COLOR.boxOut.light);
                const whiskerColor = theme === "dark" ? BOX_COLOR.whisker.dark : BOX_COLOR.whisker.light;
                const medianColor = theme === "dark" ? BOX_COLOR.median.dark : BOX_COLOR.median.light;
                // Outliers are always this fixed red, regardless of Color By
                // (spec §3-3): a hollow ring means "outside the whisker",
                // a statistical position independent of whichever coloring
                // scheme is currently selected.
                const outlierColor = theme === "dark" ? BOX_COLOR.outlier.dark : BOX_COLOR.outlier.light;
                const centerX = catScale(bin.index + 1);
                const q1Y = yScale(bin.q1);
                const q3Y = yScale(bin.q3);
                const medianY = yScale(bin.median);
                const whiskerLoY = yScale(bin.whiskerLo);
                const whiskerHiY = yScale(bin.whiskerHi);
                const boxTop = Math.min(q1Y, q3Y);
                const boxBottom = Math.max(q1Y, q3Y);
                return (
                  <g key={`box-${bin.index}`}>
                    {/* whisker: Q1-1.5*IQR ~ Q3+1.5*IQR, clamped to the box */}
                    <line x1={centerX} x2={centerX} y1={whiskerHiY} y2={boxTop} stroke={whiskerColor} strokeWidth={1.1} />
                    <line x1={centerX} x2={centerX} y1={boxBottom} y2={whiskerLoY} stroke={whiskerColor} strokeWidth={1.1} />
                    <line x1={centerX - boxCapWidthPx / 2} x2={centerX + boxCapWidthPx / 2} y1={whiskerHiY} y2={whiskerHiY} stroke={whiskerColor} strokeWidth={1.1} />
                    <line x1={centerX - boxCapWidthPx / 2} x2={centerX + boxCapWidthPx / 2} y1={whiskerLoY} y2={whiskerLoY} stroke={whiskerColor} strokeWidth={1.1} />
                    {/* jittered individual wafers -- inliers filled (painted per
                        the active Color By, spec §3), outliers hollow red
                        (spec §3-3: "이상치는 속이 빈 원. 채우지 마라"). Lower
                        radius/opacity than the scatter view (spec §3-2) since
                        these sit on top of the box itself. */}
                    {bin.members.map((member, mi) => {
                      const cx = catScale(bin.index + 1 + member.jitter);
                      const cy = yScale(member.point.y);
                      const isHovered = pointHover?.point === member.point;
                      if (member.isOutlier) {
                        if (!boxPointsVisible.outlier) return null;
                        return (
                          <circle
                            key={member.point.lot_wafer_id ?? `${bin.index}-out-${mi}`}
                            cx={cx} cy={cy} r={isHovered ? 6 : 4.5}
                            fill="none" stroke={outlierColor} strokeWidth={isHovered ? 1.8 : 1} opacity={0.85}
                            style={{ cursor: "pointer" }}
                            onClick={() => onSelectWafer(member.point)}
                          />
                        );
                      }
                      if (!boxPointsVisible.inlier) return null;
                      const style = colorForPoint(member.point, colorMode, lotIndex, theme, recommendedRangeValue);
                      return (
                        <circle
                          key={member.point.lot_wafer_id ?? `${bin.index}-in-${mi}`}
                          cx={cx} cy={cy} r={isHovered ? 4 : 2.2}
                          fill={style.color} opacity={isHovered ? 0.85 : 0.28}
                          style={{ cursor: "pointer" }}
                          onClick={() => onSelectWafer(member.point)}
                        />
                      );
                    })}
                    {/* box -- outline only, never filled, so jitter points under it
                        stay visible (spec §3-2). Hover is handled by the
                        shared plot overlay rect (see findBoxColumnAt), not
                        by listeners here -- the overlay paints on top of
                        every plot element and would swallow them anyway. */}
                    <rect
                      x={centerX - boxWidthPx / 2}
                      y={boxTop}
                      width={boxWidthPx}
                      height={Math.max(boxBottom - boxTop, 0.5)}
                      fill="none"
                      stroke={boxColor}
                      strokeWidth={1.6}
                    />
                    <line
                      x1={centerX - boxWidthPx / 2} x2={centerX + boxWidthPx / 2}
                      y1={medianY} y2={medianY}
                      stroke={medianColor} strokeWidth={2.2}
                      style={{ pointerEvents: "none" }}
                    />
                  </g>
                );
              })}

          {/* reference line bodies, stacked bottom-to-top per spec §4-1:
              최적 중심 first, then 경고선 (closest to the trend
              curve/labels above them). Labels are a separate,
              always-topmost pass below so they never sit under a
              later-drawn line. */}
          {displayLines.filter((line) => line.key === "optimal").map((line) => renderLineBody(line))}
          {displayLines.filter((line) => line.key === "warning_lo" || line.key === "warning_hi").map((line) => renderLineBody(line))}

          {/* confidence band + trend curve -- above every reference line.
              Segments touching a sparse (outlier-widened) bin render
              dashed instead of solid (spec §3-4). */}
          {trendVisible && data.bins.length > 0 && (
            <>
              {view === "scatter" && (
                <path d={trendPath.band} className="scatterTrendBand" style={{ fill: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }} />
              )}
              {trendPath.segments.map((segment, i) => (
                <path
                  key={i}
                  d={segment.d}
                  className="scatterTrendLine"
                  style={{ stroke: theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light }}
                  strokeDasharray={segment.dashed ? "5 4" : undefined}
                  fill="none"
                />
              ))}
              {/* circle markers -- box mode only (spec §3-4), so the dashed
                  구간 평균 curve reads distinctly from the box plot's own
                  solid median line. */}
              {view === "box" &&
                trendPath.markers.map((marker, i) => (
                  <circle
                    key={`trend-marker-${i}`}
                    cx={marker.x}
                    cy={marker.y}
                    r={3}
                    fill={theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light}
                    style={{ pointerEvents: "none" }}
                  />
                ))}
            </>
          )}

          {/* 알람 심각도 삼각형 -- 항상 최상단(일반 점 < 박스 < 이상치 <
              삼각형, spec §B-2). 관리한계/권장구간과 무관하게 어디에나
              나타날 수 있다(다변량 판정이므로 정상, spec §B-4). */}
          {hasAnyAlarmMarker && view === "scatter" && (() => {
            const markers = data.points
              .map((point) => ({ point, grade: alarmGradeOf(point.lot_wafer_id) }))
              .filter((m): m is { point: ScatterPoint; grade: AlarmMarkerGrade } => m.grade != null && visibleAlarmGrades.has(m.grade))
              .sort((a, b) => ALARM_GRADE_Z[a.grade] - ALARM_GRADE_Z[b.grade]);
            return markers.map(({ point, grade }, i) => {
              const style = colorForPoint(point, colorMode, lotIndex, theme, recommendedRangeValue);
              const color = theme === "dark" ? ALARM_GRADE_COLOR[grade].dark : ALARM_GRADE_COLOR[grade].light;
              return (
                <polygon
                  key={`alarm-${point.lot_wafer_id ?? i}`}
                  points={trianglePoints(xScale(point.x), yScale(point.y), style.size * ALARM_TRIANGLE_RATIO)}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.4}
                  style={{ pointerEvents: "none" }}
                />
              );
            });
          })()}
          {hasAnyAlarmMarker && view === "box" && (() => {
            const markers: { cx: number; cy: number; grade: AlarmMarkerGrade; id: string }[] = [];
            for (const bin of boxBins) {
              for (const member of bin.members) {
                if (member.isOutlier && !boxPointsVisible.outlier) continue;
                if (!member.isOutlier && !boxPointsVisible.inlier) continue;
                const grade = alarmGradeOf(member.point.lot_wafer_id);
                if (!grade || !visibleAlarmGrades.has(grade)) continue;
                markers.push({
                  cx: catScale(bin.index + 1 + member.jitter),
                  cy: yScale(member.point.y),
                  grade,
                  id: member.point.lot_wafer_id ?? `${bin.index}-${markers.length}`,
                });
              }
            }
            markers.sort((a, b) => ALARM_GRADE_Z[a.grade] - ALARM_GRADE_Z[b.grade]);
            return markers.map((m) => (
              <polygon
                key={`alarm-box-${m.id}`}
                points={trianglePoints(m.cx, m.cy, 4.5 * ALARM_TRIANGLE_RATIO)}
                fill="none"
                stroke={theme === "dark" ? ALARM_GRADE_COLOR[m.grade].dark : ALARM_GRADE_COLOR[m.grade].light}
                strokeWidth={1.4}
                style={{ pointerEvents: "none" }}
              />
            ));
          })()}

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
              setBoxHover(null);
            }}
          />
          {view === "scatter" && trendHover && !pointHover && (
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

      {/* 범례 (spec §5) -- 점의 색 구분(Color By)은 더 이상 범례에 나타나지
          않는다: 어떤 Color By를 고르든 이 아래 내용은 완전히 동일하다.
          Box Plot은 박스 자체를 읽는 법(1행)이 앞에 붙고, 기준선 4종 +
          구간 평균 불량률(row2)은 두 보기에서 문구까지 완전히 같다
          (spec §5-3: "Box Plot 2행이 Scatter Plot 1행과 글자까지
          동일"). */}
      <div className="scatterRefLegend">
        <div className="scatterRefLegendDivider" />

        {view === "box" && (
          <div className="scatterLegendRow">
            <LegendCard
              icon={<IconBoxWhisker boxColor={theme === "dark" ? methodColor.dark : methodColor.light} medianColor={theme === "dark" ? BOX_COLOR.median.dark : BOX_COLOR.median.light} />}
              label="상자 (Q1~Q3)"
            />
            <LegendCard
              icon={<IconMedian color={theme === "dark" ? BOX_COLOR.median.dark : BOX_COLOR.median.light} />}
              label="중앙값"
            />
            <LegendCard
              icon={<IconWhisker color={theme === "dark" ? BOX_COLOR.whisker.dark : BOX_COLOR.whisker.light} />}
              label="수염"
              desc="Q1−1.5×IQR ~ Q3+1.5×IQR"
            />
            <LegendCard
              icon={<IconDots color={theme === "dark" ? BOX_COLOR.inlier.dark : BOX_COLOR.inlier.light} />}
              label="개별 wafer"
              onClick={() => setBoxPointsVisible((cur) => ({ ...cur, inlier: !cur.inlier }))}
              active={boxPointsVisible.inlier}
            />
            <LegendCard
              icon={<IconOutlierDots color={theme === "dark" ? BOX_COLOR.outlier.dark : BOX_COLOR.outlier.light} />}
              label="이상치 wafer"
              onClick={() => setBoxPointsVisible((cur) => ({ ...cur, outlier: !cur.outlier }))}
              active={boxPointsVisible.outlier}
            />
          </div>
        )}

        {/* 기준선 4종 + 구간 평균 불량률 -- Color By와 무관하게 항상 동일.
            ±3σ/±6σ/평균/Q1/Q3는 여전히 어느 화면에도 노출하지 않는다
            (src/analysis/control_range.py에는 그대로 남아 있다). */}
        <div className="scatterLegendRow">
          <LegendCard
            icon={<IconDashedLine color={theme === "dark" ? LINE_COLOR.warning.dark : LINE_COLOR.warning.light} />}
            label={warningLegendLabel()}
            desc={warningLegendDesc()}
            onClick={(event) => toggleGroup("warning", event)}
            active={visibleGroups.has("warning") && !warningFullyNonDrawable}
            disabled={warningFullyNonDrawable}
            title={warningFullyNonDrawable ? "경고선 없음 (이 인자 단독으로는 예측 수율이 임계 아래로 내려가지 않음)" : "예측 수율이 낮아지는 구간의 경계"}
          />
          {showOptimalLegendCard && (
            <LegendCard
              icon={<IconDashedLine color={theme === "dark" ? methodColor.dark : methodColor.light} dotted />}
              label={optimalAvailable ? `최적 중심  ${formatNum1(displayedOptimalCenter as number)}` : "최적 중심"}
              desc={optimalAvailable ? "불량률이 가장 낮은 지점" : optimalUnavailableReason}
              onClick={(event) => toggleGroup("optimal", event)}
              active={visibleGroups.has("optimal") && optimalAvailable}
              disabled={!optimalAvailable}
              title={optimalAvailable ? "구간 평균 불량률이 가장 낮은 지점" : optimalUnavailableReason}
            />
          )}
          <LegendCard
            icon={<IconBand color={theme === "dark" ? methodColor.dark : methodColor.light} />}
            label={
              recommendedRangeValue
                ? `권장 구간 (${METHOD_LABEL[activeMethod]})  ${formatNum1(recommendedRangeValue[0])}~${formatNum1(recommendedRangeValue[1])}`
                : "권장 구간"
            }
            desc={
              recommendedRangeValue
                ? `이 범위로 관리 권장${recommendedClamped ? " (관리한계에 맞춰 조정됨)" : ""}`
                : "데이터 범위 밖"
            }
            onClick={(event) => toggleGroup("recommended", event)}
            active={visibleGroups.has("recommended") && !!recommendedRangeValue}
            disabled={!recommendedRangeValue}
            title={
              recommendedRangeValue
                ? `구간 평균 불량률이 전체 평균 이하인 구간${recommendedClamped ? " (관리한계에 맞춰 조정됨)" : ""}`
                : "데이터 범위 밖"
            }
          />
          <LegendCard icon={<IconBand color="#F59E0B" />} label="경고선 밖" desc="예측 수율이 낮게 관측되는 범위" />
          <LegendCard
            icon={<IconTrendLine color={theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light} />}
            label="구간 평균 불량률"
            onClick={() => setTrendVisible((v) => !v)}
            active={trendVisible}
            title="구간 평균 불량률"
          />
        </div>

        {/* 알람 심각도 삼각형 범례 -- 등급별로 독립적으로 on/off (spec §B-3).
            산점도/Box Plot 둘 다 이 범례를 공유한다. */}
        {alarmGradeByWaferId && (
          <div className="scatterLegendRow scatterAlarmLegendRow">
            <span className="scatterAlarmLegendLabel">알람 심각도</span>
            {(["심각", "위험", "주의"] as const).map((grade) => (
              <LegendCard
                key={grade}
                icon={<IconTriangle color={theme === "dark" ? ALARM_GRADE_COLOR[grade].dark : ALARM_GRADE_COLOR[grade].light} />}
                label={`${grade} ${alarmGradeCounts[grade]}`}
                onClick={() =>
                  setVisibleAlarmGrades((current) => {
                    const next = new Set(current);
                    if (next.has(grade)) next.delete(grade);
                    else next.add(grade);
                    return next;
                  })
                }
                active={visibleAlarmGrades.has(grade)}
                disabled={alarmGradeCounts[grade] === 0}
                title={`${grade} 등급 알람 (예측 수율 신뢰구간 상한 기준)`}
              />
            ))}
          </div>
        )}
        {hasAnyAlarmMarker && (
          <p className="scatterAlarmLegendNote">
            삼각형은 전체 인자를 종합한 판정이므로 권장 구간 안에도 나타날 수 있습니다.
          </p>
        )}
      </div>

      {lineHover && (() => {
        if (lineHover.key === "optimal") {
          if (displayedOptimalCenter == null) return null;
          return (
            <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
              <strong>최적 중심</strong>
              <div className="heatmapTooltipRow"><span>값</span><b>{formatNum1(displayedOptimalCenter)}</b></div>
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
        if (!line || (line.key !== "warning_lo" && line.key !== "warning_hi")) return null;
        const meta = LINE_META[line.key];
        return (
          <div className="heatmapTooltip" style={{ left: lineHover.x + 14, top: lineHover.y + 14 }}>
            <strong>{meta.legendLabel}</strong>
            <div className="heatmapTooltipRow"><span>값</span><b>{line.value.toFixed(1)}</b></div>
            <div className="heatmapTooltipRow"><span>이 선 밖</span><b>{line.outside_count.toLocaleString()}장</b></div>
            {line.observed_yield_gap_pp != null && (
              <div className="heatmapTooltipRow"><span>실측 수율 차이</span><b>{line.observed_yield_gap_pp.toFixed(1)}%p</b></div>
            )}
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
          {pointHover.isOutlier && (
            <div className="heatmapTooltipRow"><span /><b className="scatterOutlierTag">이상치</b></div>
          )}
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

      {boxHover && !pointHover && (() => {
        const bin = boxHover.bin;
        return (
          <div className="heatmapTooltip" style={{ left: boxHover.clientX + 14, top: boxHover.clientY + 14 }}>
            <strong>구간 {bin.index + 1} · {factorAxisLabel(data.axis.x_label)} {formatNum1(bin.xLo)} ~ {formatNum1(bin.xHi)}</strong>
            <div className="heatmapTooltipRow"><span>wafer</span><b>{bin.n.toLocaleString()}장</b></div>
            <div className="heatmapTooltipRow"><span>중앙값</span><b>{bin.median.toFixed(2)}</b></div>
            <div className="heatmapTooltipRow"><span>평균</span><b>{bin.mean.toFixed(2)}</b></div>
            <div className="heatmapTooltipRow"><span>Q1 / Q3</span><b>{bin.q1.toFixed(2)} / {bin.q3.toFixed(2)}</b></div>
            <div className="heatmapTooltipRow"><span>이상치</span><b>{bin.outlierCount.toLocaleString()}장</b></div>
          </div>
        );
      })()}

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
