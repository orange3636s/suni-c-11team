"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import InterpretationCard, { type InterpretationRow } from "@/components/InterpretationCard";
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import { parseConfig } from "@/lib/constants";
import { niceTicksFitted } from "@/lib/niceTicks";
import { measureTextWidth } from "@/lib/textMeasure";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { RelationShape, ScatterPoint, ScreeningScatterResponse, WindowMethod } from "@/types/data";

export type ScatterColorMode = "default" | "config_model" | "lot";
// 그룹 스크럽 슬라이더 정렬 기준 (spec CA) -- 기본은 평균순: Lot_ID는 처리
// 순서가 아니라 임의 부여이므로, 이름순으로 훑으면 점이 무작위로 튀어
// 아무 패턴도 안 보인다. 평균 y값 오름차순으로 훑어야 분포가 아래->위로
// 이동하는 것이 보인다.
type ScrubSort = "mean" | "name" | "count";
// 비선택 그룹의 투명도 (spec CA: "0.10 ~ 0.15", 지시서 X-3: 기본 불투명도가
// 낮아졌으니 하한 0.10을 쓴다) -- 색은 유지한 채 이만큼만 낮춘다. 회색으로
// 바꾸지 않는다.
const SCRUB_DIM_OPACITY = 0.1;
// 이보다 그룹 수가 많으면(LOT 390개 등) 눈금을 그리지 않는다 -- 스톱이
// 적은 Config 모드(3~4개)에서만 각 스톱이 어디인지 보이게 한다 (spec CA
// "그룹 수에 따른 차이").
const SCRUB_TICK_MAX_STOPS = 12;
// 인자 카드 "보기" 토글 전체 상태 (Pareto/Scatter/Box) -- Pareto는 이
// ScatterChart 컴포넌트가 아니라 카드가 직접 분기해 ParetoChart를 그리므로,
// 이 컴포넌트 자신은 QuickLookView(아래)만 받는다.
export type ScatterView = "pareto" | "scatter" | "box";
// ScatterChart가 실제로 렌더하는 뷰. Quick Look 패널에도 Pareto 옵션이 없어
// 그대로 재사용한다 -- ScatterView와 분리해 두면 quickLookView 같은 state에
// "pareto"가 흘러들 여지 자체가 타입 레벨에서 없어진다.
export type QuickLookView = "scatter" | "box";
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
// top/right trimmed 36/28 -> 28/16 (경고선 제거) -- 라벨 최고 행이
// warning_line(34)에서 window_bounds(20)로 낮아졌고, 오른쪽 끝에
// 붙던 경고선(warning_hi) 값 라벨도 사라져 두 여백 모두 그만큼
// 빈 공간으로 남았던 것을 걷어냈다.
const MARGIN = { top: 28, right: 16, bottom: 46, left: 76 };
// Matches .scatterTickLabel's font (app/globals.css) -- used to measure
// candidate tick labels for the overlap-backoff pass (spec §8). HP-HMI
// 전환으로 .scatterTickLabel이 등폭(var(--font-data))이 됐으므로, 폭
// 측정도 같은 폰트로 해야 충돌 판정이 실제 렌더 폭과 어긋나지 않는다.
const TICK_FONT = "10.5px ui-monospace, Menlo, monospace";
// Y tick labels are horizontal text stacked vertically, so their collision
// dimension is line height, not width -- a fixed estimate for 10.5px text.
const Y_TICK_LABEL_HEIGHT_PX = 14;
const X_TICK_COUNT: [max: number, min: number] = [10, 8];
const Y_TICK_COUNT: [max: number, min: number] = [8, 6];
const HEIGHT = 420;
// 기준선 라벨 겹침 수정 (spec §C-2) -- "최적 중심"/"권장 구간" 라벨이
// 서로의 x 위치가 가까울 때 겹치던 문제를, 2단 높이로 분산해 해결한다.
// offset은 plot 상단 가장자리(y=0) 기준 위쪽(margin 안)으로 얼마나
// 띄우는지이며, 값이 클수록 차트에서 더 멀다 -- 20이 위(권장 구간), 6이
// 차트에 가장 가깝다(최적 중심). 14px 간격은 라벨 한 줄의 대략적인
// 높이와 맞춘 값이라 두 줄이 서로 겹치지 않는다. (경고선 제거로 3단에서
// 2단으로 줄었다 -- 행 번호를 당겼다.)
type LabelRowKey = "window_bounds" | "optimal_center";
const LABEL_ROW_OFFSET: Record<LabelRowKey, number> = {
  optimal_center: 6,
  window_bounds: 20,
};
// 생략 순서 (spec §C-3): 권장 구간 경계(1) > 최적 중심(2, 가장 먼저
// 생략) -- 숫자가 작을수록 우선순위가 높다.
const LABEL_ROW_PRIORITY: Record<LabelRowKey, number> = {
  window_bounds: 1,
  optimal_center: 2,
};
// .scatterLineLabel의 font-weight/font-size와 맞춘 측정용 폰트 문자열
// (spec §C-4 폭 계산에 실제 렌더 폭을 써야 축약 여부 판단이 정확하다).
// HP-HMI 전환으로 .scatterLineLabel이 등폭이 됐으므로 측정 폰트도 맞춘다.
const LABEL_FONT = "700 10px ui-monospace, Menlo, monospace";
// 이 폭 미만이면 라벨을 축약형으로 바꾼다 (spec §C-4: "차트 폭 600px
// 미만이면 축약형을 쓴다").
const COMPACT_LABEL_CHART_WIDTH = 600;
// 권장 구간 밴드 폭이 이보다 좁으면 양끝 경계값 라벨을 생략한다 (spec
// §C-5).
const BAND_BOUNDARY_MIN_WIDTH = 60;

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
  view: QuickLookView,
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

// SPC/ML 방식 전환 (spec §3-4) -- 권장 구간 밴드/경계선/Box Plot 구간-내
// 박스 테두리가 이 한 쌍을 공유한다. HP-HMI 색 규율(§H-1) 적용 후:
// 권장 구간은 "정상 상태"(이상 신호가 아님)라 SPC/ML 모두 같은 무채색
// 하나로 통일했다 -- 이전에는 초록(SPC)/보라(ML)로 방식을 구분했지만,
// 신호색은 이상 상태에만 쓴다는 규율상 방식 구분에 색상을 쓸 이유가
// 없다. 방식 구분은 여전히 텍스트 라벨("SPC"/"ML")이 담당한다.
const METHOD_COLOR: Record<WindowMethod, { light: string; dark: string }> = {
  spc: { light: "#59636F", dark: "#9097A3" },
  ml: { light: "#59636F", dark: "#9097A3" },
};
const METHOD_LABEL: Record<WindowMethod, string> = { spc: "SPC", ml: "ML" };
// 최적 중심(optimal_center)은 이전엔 METHOD_COLOR를 함께 썼지만, 권장
// 구간(위)과 최적 중심은 이제 서로 다른 무채색 톤이어야 한다(§H-1: 최적
// 중심 = --text, 권장 구간 = 옅은 중립 배경) -- 그래서 별도 상수로
// 분리했다.
const OPTIMAL_CENTER_COLOR = { light: "#141A22", dark: "#F5F5F7" };
// 구간 평균 불량률 곡선은 "이 구간에서 불량이 얼마나 나는가"를 보여주는
// 차트의 핵심 판독 대상이다 -- 무채색/추정값 색(--inferred)으로 두면
// 배경의 점·격자와 명도가 비슷해져 정작 봐야 할 선이 가장 조용해진다.
// 색 규율의 목적은 색을 줄이는 게 아니라 봐야 할 것에 색이 가게 하는
// 것이므로, 이 선만은 신호색(--sig-red)을 쓴다.
const TREND_COLOR = { light: "#C0392B", dark: "#EE6B76" };

// Box-plot 색 규칙. 상자 테두리/수염은 "권장구간 안쪽인지"와 무관하게 항상
// 중립(var(--line))이다. 상자 자체는 옅게 채운다(8% var(--measured), 아래
// 렌더 지점의 fillOpacity). 중앙값은 무채색(var(--text)). 관리한계 개념
// 제거(GBDT 전환)에 따라 통계적 이상치(IQR 기준)는 항상 outlierNeutral
// 하나로 통일한다 -- 경고선은 화면에서 완전히 제거됐으므로 그 기준 채색도 없다.
// Y-5: --line의 다크 값을 rgba(255,255,255,.14)(1.57:1, WCAG UI 요소
// 3:1 기준 미달)에서 .34(3.05:1)로 올렸다 -- 여기 두 하드코딩 사본도
// 같이 맞춘다(둘 다 CSS 변수가 아니라 SVG stroke에 직접 넣는 JS 상수라
// 토큰을 바꿔도 저절로 따라오지 않는다).
const BOX_LINE_COLOR = { light: "#D9DEE6", dark: "rgba(255, 255, 255, 0.34)" }; // var(--line)
const BOX_FILL_COLOR = { light: "#0E306D", dark: "#7BA3E8" }; // var(--measured), 8% 채움으로만 씀
const BOX_COLOR = {
  median: { light: "#141A22", dark: "#F5F5F7" }, // var(--text)
  whisker: { light: "#D9DEE6", dark: "rgba(255, 255, 255, 0.34)" }, // var(--line)
  inlier: { light: "#0E306D", dark: "#7BA3E8" }, // var(--measured) -- DEFAULT_POINT_STYLE과 동일 상수
  outlierNeutral: { light: "#59636F", dark: "#9097A3" },
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

// dash 패턴은 실제 렌더 상수(최적 중심 라인의 "4 3")를 그대로 받는다 --
// 범례에서도 실선/점선 구분이 실제 차트와 같은 비율로 보이게 한다
// (지시서 N-1).
function IconDashedLine({ color, dash }: { color: string; dash: string }) {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <line x1="2" y1="8" x2="20" y2="8" stroke={color} strokeWidth="1.8" strokeDasharray={dash} />
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
      <button type="button" className={className} onClick={onClick} title={title} aria-disabled={disabled}>
        {inner}
      </button>
    );
  }
  return (
    <div className={className} title={title} aria-disabled={disabled}>
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

/** Config별/LOT별 Color By가 점 하나를 어느 그룹으로 보는지 -- colorForPoint
 * (칠하기)와 그룹 스크럽 슬라이더(spec CA, 강조)가 절대 다른 그룹 경계를
 * 갖지 않도록 단일 소스로 뺐다. "기본" 모드는 그룹 개념이 없어 null. */
function scrubGroupKeyOf(point: ScatterPoint, mode: ScatterColorMode): string | null {
  if (mode === "config_model") return modelOf(point.config);
  if (mode === "lot") return point.lot_id ?? "미상";
  return null;
}

// "기본" Color By 점 스타일 -- 관리한계(IQR/LCL/UCL) 개념 제거(GBDT 전환)에
// 이어, "기본"은 "구분하지 않음"이라는 뜻으로 통일했다(spec P-1): 권장
// 구간 안/밖을 포함한 어떤 zone 구분도 점 색·크기로 표현하지 않는다.
// 권장 구간은 참조선/밴드로 이미 표시되므로 점에서 중복 인코딩할
// 필요가 없다. --measured 토큰과 동일한 값.
//
// 점 농도 완화 (X-1) -- 이 0.28은 임의값이 아니라 Box Plot의 지터된 일반
// 점(POINT_OPACITY, 아래)이 이미 실제로 쓰고 있던 값을 그대로 가져온
// 것이다. 예전에는 Box Plot이 산점도보다 일부러 더 연했는데("점이
// 상자 위에 얹히므로"), 이제는 반대로 산점도가 Box Plot 무게에 맞춘다 --
// 두 뷰를 오갈 때 점의 무게가 달라지면 안 되기 때문이다(지시서 X그룹).
// 세 Color By 모드(기본/Config별/LOT별) 전부 이 값 하나를 공유한다 --
// 모드마다 다르면 전환할 때 밀도 인상이 바뀌어 비교가 안 된다.
const POINT_OPACITY = 0.28;
const DEFAULT_POINT_STYLE = { light: "#0E306D", dark: "#7BA3E8", size: 4 };

function colorForPoint(
  point: ScatterPoint,
  mode: ScatterColorMode,
  lotIndex: Map<string, number>,
  theme: "light" | "dark",
): { color: string; size: number; opacity: number } {
  if (mode === "config_model" || mode === "lot") {
    const key = scrubGroupKeyOf(point, mode) ?? "미상";
    let idx = lotIndex.get(key);
    if (idx == null) {
      idx = lotIndex.size;
      lotIndex.set(key, idx);
    }
    const palette = mode === "config_model" ? MODEL_COLORS : LOT_PALETTE;
    return { color: palette[idx % palette.length], size: 5, opacity: POINT_OPACITY };
  }
  return {
    color: theme === "dark" ? DEFAULT_POINT_STYLE.dark : DEFAULT_POINT_STYLE.light,
    size: DEFAULT_POINT_STYLE.size,
    opacity: POINT_OPACITY,
  };
}

// A single vertical line drawn on the chart -- the (backend-provided, not
// frontend-computed) optimal-center point. Kept as its own type so the
// paint order / label-collision layout stays generic over reference lines.
type DisplayLine = {
  key: "optimal";
  value: number;
  color: { light: string; dark: string };
  dash: string;
  strokeWidth: number;
  greyedOut: boolean;
};

// 기준선 라벨 하나 -- 최적 중심 vertical line뿐 아니라 권장 구간 밴드의
// 요약 라벨(window_bounds)도 같은 체계로 배치되므로, "라인이
// 있는가"와 무관하게 여기서는 x 위치 + 우선순위 행만 다룬다 (spec §C-6:
// Box Plot도 같은 로직을 쓴다 -- plotX가 이미 두 보기를 통일하므로
// 여기서부터는 view를 몰라도 된다).
type FloatingLabel = {
  key: string;
  rowKey: LabelRowKey;
  x: number;
  full: string;
  short: string;
  color: string;
};

type FloatingLabelPlacement = { key: string; x: number; y: number; hidden: boolean; text: string; color: string };

/** 폭 기반 충돌 판정으로 라벨을 3단 높이에 배치한다 (spec §C-2/§C-3).
 * 아무 라벨도 충돌하지 않으면 전부 가장 낮은 높이(차트에 가장 가까운
 * offset)에 그대로 둔다 -- 불필요한 계단을 만들지 않는다. 충돌이 하나라도
 * 있으면 각 라벨을 자기 우선순위의 지정 행으로 올리고, 그래도 같은 행
 * 안에서 겹치면(예: 최적 중심과 권장 구간 라벨이 서로 가까움) 우선순위가 낮은 쪽부터
 * 생략한다 -- 생략된 라벨은 그려지지 않지만, 그 라벨이 속한 선/밴드 자체는
 * 별도로 계속 그려지므로 호버 시 툴팁으로 값을 확인할 수 있다(spec §C-3
 * "생략된 라벨은 호버 시 툴팁으로 표시한다"). */
function layoutFloatingLabels(
  labels: FloatingLabel[],
  useShort: boolean,
  measure: (text: string) => number,
): FloatingLabelPlacement[] {
  if (labels.length === 0) return [];
  const textOf = (l: FloatingLabel) => (useShort ? l.short : l.full);
  const widthOf = new Map(labels.map((l) => [l.key, measure(textOf(l)) + 8]));
  const collide = (a: FloatingLabel, b: FloatingLabel) =>
    Math.abs(a.x - b.x) < (widthOf.get(a.key)! + widthOf.get(b.key)!) / 2 + 8;
  const anyCollision = labels.some((a, i) => labels.slice(i + 1).some((b) => collide(a, b)));
  if (!anyCollision) {
    const y = -LABEL_ROW_OFFSET.optimal_center;
    return labels.map((l) => ({ key: l.key, x: l.x, y, hidden: false, text: textOf(l), color: l.color }));
  }
  const sorted = [...labels].sort((a, b) => LABEL_ROW_PRIORITY[a.rowKey] - LABEL_ROW_PRIORITY[b.rowKey]);
  const placedByRow: FloatingLabel[] = [];
  const byKey = new Map<string, FloatingLabelPlacement>();
  for (const l of sorted) {
    const conflict = placedByRow.some((o) => o.rowKey === l.rowKey && collide(l, o));
    const y = -LABEL_ROW_OFFSET[l.rowKey];
    if (!conflict) placedByRow.push(l);
    byKey.set(l.key, { key: l.key, x: l.x, y, hidden: conflict, text: textOf(l), color: l.color });
  }
  return labels.map((l) => byKey.get(l.key)!);
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
// 드래그 선택 (spec B) -- 이동 거리가 이 미만이면 클릭(기존 포인트 상세),
// 이상이면 브러시 선택으로 본다.
const DRAG_THRESHOLD_PX = 5;
type DragPos = { x: number; y: number };
// 정보 박스 예상 크기 (CSS 값과 맞춰 둔다) -- 드래그 사각형이 우상단
// 코너의 이 영역을 침범하는지 판정하는 데만 쓰인다.
const STATS_BOX_WIDTH = 200;
const STATS_BOX_HEIGHT = 130;
// globals.css --accent (#0e306d) -- 라이트/다크 동일 값이라 이 파일의
// 다른 색처럼 테마별 쌍을 따로 두지 않는다.
const ACCENT_COLOR = "#0E306D";

// 즐겨찾기 썸네일 전용 점 반지름 (지시서 P-2) -- 본 차트의 DEFAULT_POINT_STYLE/
// MODEL_COLORS/LOT_PALETTE 크기와 절대 공유하지 않는 별도 상수다: 본 차트
// 값을 나중에 바꿔도 썸네일이 따라 커지지 않아야 한다. 썸네일은 식별용이지
// 값을 읽는 화면이 아니므로 3px대의 본 차트 점보다 한참 작게 잡는다.
const THUMBNAIL_POINT_RADIUS = 1.1;
const THUMBNAIL_POINT_OPACITY = 0.5;

export default function ScatterChart({
  data,
  colorMode,
  view,
  method,
  onSelectWafer,
  height = HEIGHT,
  reliabilityText,
  thumbnail = false,
}: {
  data: ScreeningScatterResponse;
  colorMode: ScatterColorMode;
  // Owned by the caller now (spec §1-3: 보기 토글 lives in the card header,
  // same row as the title) -- this component only reads it to decide what
  // to render, it never renders the toggle buttons itself. Never "pareto"
  // -- the card renders ParetoChart itself for that state instead of
  // reaching this component (see QuickLookView above).
  view: QuickLookView;
  // Which of data.methods.{spc,ml} drives the recommended-range band and
  // optimal-center line (spec §3) -- same "caller owns the toggle state"
  // pattern as `view`. Defaults to "spc" when the caller hasn't resolved
  // data.methods.adopted yet (first render before the toggle mounts).
  method?: WindowMethod;
  onSelectWafer: (point: ScatterPoint) => void;
  height?: number;
  // HA그룹: "보통" 등급 인자의 설명력이 낮다는 안내(caller가 계산해 넘김
  // -- root-cause/page.tsx의 buildModerateInterpretation과 동일 로직).
  // 이 컴포넌트가 자체 계산한 "해석" 행과 함께 카드 하나로 합쳐 그린다.
  // 빈 문자열/undefined면 "신뢰도" 행 자체를 만들지 않는다(조건부).
  reliabilityText?: string;
  // 즐겨찾기 썸네일 모드 (지시서 P-2) -- 점을 훨씬 작게, 참조선은 최적
  // 중심 하나만, 축은 눈금/라벨 없이 선만 남긴다. 식별용이라 값을 읽을
  // 필요가 없다.
  thumbnail?: boolean;
}) {
  const activeMethod: WindowMethod = method ?? "spc";
  const methodColor = METHOD_COLOR[activeMethod];
  const theme = useResolvedTheme();
  // 모바일 반응형 패치 S-4: ≤767px에서 점 반지름 4px -> 3px(투명도는
  // 그대로), 눈금은 양끝+중앙(3개)까지 줄인다.
  const isMobileLayout = useIsMobileLayout();
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerWidth, setContainerWidth] = useState(680);
  // 남은 2개 참조 요소(최적 중심/권장 구간)는 기본으로 켜져 있다 (spec §4-2,
  // 경고선 제거로 3개에서 2개로 줄었다).
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(["optimal", "recommended"]));
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
  // 드래그 선택 (spec B) -- 일시적 상태다. 뷰 전환·타깃 변경·카드 remount
  // 시 해제해야 하므로 아래 [data, view] 이펙트에서 지운다.
  const [dragStart, setDragStart] = useState<DragPos | null>(null);
  const [dragCurrent, setDragCurrent] = useState<DragPos | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<ScatterPoint[]>([]);
  // 정보 박스가 드래그 사각형이 우상단 코너를 침범할 때만 좌상단으로
  // flip한다 -- 선택이 끝난 뒤에는 박스가 사각형을 따라다니지 않으므로
  // (지시서 B) 이 값도 finishDrag 시점에 한 번만 정해지고 고정된다.
  const [statsBoxFlip, setStatsBoxFlip] = useState(false);

  // 그룹 스크럽 슬라이더 (spec CA) -- Config별/LOT별 Color By에서만 의미가
  // 있다. 위치 0 = 전체(강조 없음), 1..N = scrubGroups[index-1]. 카드별
  // 독립 상태이며, 서버·즐겨찾기에는 절대 저장하지 않는다(일시 상태).
  const [scrubSort, setScrubSort] = useState<ScrubSort>("mean");
  const [scrubIndex, setScrubIndex] = useState(0);
  const [scrubInputInvalid, setScrubInputInvalid] = useState(false);
  // 드래그 중 rAF로 스로틀링 (spec CA "성능") -- 네이티브 range input이
  // 매 스텝마다 onChange를 쏘아도, 실제 setState(리렌더)는 프레임당 최대
  // 1번만 반영한다.
  const pendingScrubIndexRef = useRef<number | null>(null);
  const scrubRafIdRef = useRef<number | null>(null);

  useEffect(() => {
    // setTimeout으로 감싼다 -- 이펙트 본문에서 곧장 setState를 부르면
    // cascading render 린트 규칙에 걸린다(이 파일의 다른 모달들과 같은
    // 우회 패턴).
    const timer = window.setTimeout(() => setSelectedPoints([]), 0);
    return () => window.clearTimeout(timer);
  }, [data, view]);

  // 그룹 스크럽 초기화 (spec CA "상태 범위") -- 색상 모드가 바뀌거나(Config
  // 인덱스를 LOT 모드로 끌고 가면 안 됨) 새 인자 데이터가 들어오면 0(전체)
  // 으로 되돌린다. `view`는 의도적으로 빠져 있다 -- 뷰 전환(산점도<->Box
  // Plot) 시에는 유지해야 한다(같은 데이터를 다른 형태로 보는 것뿐이므로).
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setScrubIndex(0);
      setScrubInputInvalid(false);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [data, colorMode]);

  useEffect(
    () => () => {
      if (scrubRafIdRef.current != null) cancelAnimationFrame(scrubRafIdRef.current);
    },
    [],
  );

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
  // 모바일 반응형 패치 S-4: ≤767px 뷰포트에서는 chart 픽셀 높이와 무관하게
  // 항상 3개(양끝+중앙)까지 더 줄인다 -- isCompactChart는 차트 자체의
  // 렌더 높이(썸네일 등에서도 발생 가능)를 기준으로 하는 기존 규칙이라
  // 그대로 남겨 두고, 뷰포트 폭 기준 규칙을 그 위에 얹는다.
  const xTicks = useMemo(() => {
    const [max, min] = isMobileLayout ? [3, 3] : isCompactChart ? [5, 4] : X_TICK_COUNT;
    return niceTicksFitted(xDomain, max, min, plotWidth, formatTick, (label) => measureTextWidth(label, TICK_FONT));
  }, [xDomain, plotWidth, isCompactChart, isMobileLayout]);
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
    const [max, min] = isMobileLayout ? [3, 3] : isCompactChart ? [4, 4] : Y_TICK_COUNT;
    return niceTicksFitted(yDomain, max, min, plotHeight, formatTick, () => Y_TICK_LABEL_HEIGHT_PX);
  }, [yDomain, plotHeight, isCompactChart, isMobileLayout]);
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

  // 그룹 스크럽 슬라이더 (spec CA) -- 점 -> 그룹키 매핑을 프레임마다 다시
  // 계산하지 않도록 미리 만들어 메모이즈한다("성능" 절: "그룹 -> 점 인덱스
  // 매핑을 미리 계산해 메모이즈한다. 매 프레임 filter를 돌리지 말 것").
  // colorMode가 "기본"이면 그룹 개념이 없으므로 null.
  const scrubKeyByPoint = useMemo(() => {
    if (colorMode === "default") return null;
    const map = new Map<ScatterPoint, string>();
    for (const point of data.points) {
      const key = scrubGroupKeyOf(point, colorMode);
      if (key != null) map.set(point, key);
    }
    return map;
  }, [data.points, colorMode]);

  // 그룹 목록 + 평균 y/개수 -- 정렬 기준이 바뀔 때만 다시 정렬한다(그룹
  // 집계 자체는 scrubKeyByPoint/data.points가 바뀔 때만). 표본이 1~2개뿐인
  // 그룹도 전부 포함한다 (spec CA "하지 말 것": "표본이 적다는 이유로 빼지
  // 말 것").
  const scrubGroups = useMemo(() => {
    if (!scrubKeyByPoint) return [] as { key: string; count: number; meanY: number }[];
    const stats = new Map<string, { count: number; sumY: number }>();
    for (const point of data.points) {
      const key = scrubKeyByPoint.get(point);
      if (key == null) continue;
      const entry = stats.get(key) ?? { count: 0, sumY: 0 };
      entry.count += 1;
      entry.sumY += point.y;
      stats.set(key, entry);
    }
    const list = Array.from(stats.entries()).map(([key, { count, sumY }]) => ({ key, count, meanY: sumY / count }));
    if (scrubSort === "mean") list.sort((a, b) => a.meanY - b.meanY);
    else if (scrubSort === "name") list.sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));
    else list.sort((a, b) => b.count - a.count);
    return list;
  }, [scrubKeyByPoint, data.points, scrubSort]);

  // 슬라이더 위치 0 = 전체(강조 없음) -- scrubIndex가 그룹 수를 넘으면(색상
  // 모드/데이터가 막 바뀌어 리셋 이펙트가 아직 반영되기 전의 한 프레임)
  // 안전하게 "전체"로 읽는다.
  const activeScrubGroup = scrubIndex >= 1 ? (scrubGroups[scrubIndex - 1]?.key ?? null) : null;
  const scrubGroupLabel = activeScrubGroup ?? "전체";

  /** 비선택 그룹의 점을 낮은 투명도로 흐리게 한다 (spec CA "시각 처리": 색은
   * 유지, 회색으로 바꾸지 않는다) -- 강조 그룹이 없거나(위치 0) 이 점이
   * 이미 호버/브러시 선택 중이면 원래 opacity를 그대로 쓴다.
   *
   * 지시서 X-3: 기본 불투명도를 POINT_OPACITY(0.28)로 낮추면서 대비가
   * 줄었다(이전 0.7~0.85 대 0.12 → 0.28 대 0.10). 선택 그룹은 원래
   * opacity를 그대로 넘기는 대신 1로 올려, 강조 대비가 오히려 커지게
   * 한다 -- 비선택 하한(SCRUB_DIM_OPACITY)만 낮추고 선택 쪽을 그대로
   * 두면 이전보다 덜 도드라져 보였을 것이다. */
  function applyScrubDim(point: ScatterPoint, opacity: number, isHovered: boolean, isSelected: boolean): number {
    if (isHovered || isSelected) return opacity;
    if (!activeScrubGroup) return opacity;
    return scrubKeyByPoint?.get(point) === activeScrubGroup ? 1 : SCRUB_DIM_OPACITY;
  }

  // 드래그 중 rAF 스로틀 (spec CA "성능") -- 슬라이더가 프레임당 여러 번
  // onChange를 쏘아도 setState(리렌더)는 프레임당 최대 1번만 반영한다.
  function scheduleScrubIndex(next: number) {
    pendingScrubIndexRef.current = next;
    if (scrubRafIdRef.current != null) return;
    scrubRafIdRef.current = requestAnimationFrame(() => {
      scrubRafIdRef.current = null;
      const value = pendingScrubIndexRef.current;
      pendingScrubIndexRef.current = null;
      if (value != null) {
        setScrubIndex(value);
        setScrubInputInvalid(false);
      }
    });
  }

  // 그룹명 직접 입력 -> 슬라이더 점프 (spec CA) -- 비우고 확정하면 0(전체)
  // 으로, 존재하지 않는 이름이면 경고 테두리만 표시하고 슬라이더는 그대로
  // 둔다.
  function commitScrubInput(raw: string) {
    const trimmed = raw.trim();
    if (!trimmed) {
      setScrubIndex(0);
      setScrubInputInvalid(false);
      return;
    }
    const index = scrubGroups.findIndex((group) => group.key.toLowerCase() === trimmed.toLowerCase());
    if (index === -1) {
      setScrubInputInvalid(true);
      return;
    }
    setScrubIndex(index + 1);
    setScrubInputInvalid(false);
  }

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
  const recommendedRangeValue: [number, number] | null = activeMethodWindow ? activeMethodWindow.window : null;

  const displayLines = useMemo<DisplayLine[]>(() => {
    const lines: DisplayLine[] = [];
    if (displayedOptimalCenter != null) {
      lines.push({
        key: "optimal", value: displayedOptimalCenter,
        color: OPTIMAL_CENTER_COLOR, dash: "4 3", strokeWidth: 1.9, greyedOut: false,
      });
    }
    return lines;
  }, [displayedOptimalCenter]);

  // 차트 폭이 좁으면 라벨을 축약형(값만)으로 바꾼다 (spec §C-4). 이 폭
  // 미만에서는 권장 구간 요약 라벨도 아예 빼고 밴드 안쪽 경계값 라벨에
  // 정보 전달을 맡긴다 (아래 BAND_BOUNDARY_MIN_WIDTH 조건부 렌더 참고).
  const isNarrowChart = containerWidth < COMPACT_LABEL_CHART_WIDTH;

  // 최적 중심 라인 라벨 + 권장 구간 요약 라벨을 한 배열로 모아 같은
  // 2단 충돌 배치 로직을 함께 통과시킨다 (spec §C-2) -- 꺼진 그룹은 애초에
  // 배치 대상에서 빼서, 숨겨진 라벨이 다른 라벨의 자리를 차지하지 않는다.
  const floatingLabels = useMemo<FloatingLabel[]>(() => {
    const items: FloatingLabel[] = [];
    for (const line of displayLines) {
      if (!visibleGroups.has("optimal")) continue;
      // 보정 §I-5: 최적 중심은 SPC/ML 방식에 따라 값 자체가 달라지므로
      // (권장 구간 라벨처럼) 방식을 라벨에 표기한다 -- 색으로는 더 이상
      // 구분하지 않으니(§H-1) 텍스트가 유일한 구분 수단이다.
      const methodSuffix = ` (${METHOD_LABEL[activeMethod]})`;
      const color = line.greyedOut ? (theme === "dark" ? "#6B7280" : "#9CA3AF") : (theme === "dark" ? line.color.dark : line.color.light);
      items.push({
        key: line.key,
        rowKey: "optimal_center",
        x: plotX(line.value),
        full: `최적 중심 ${formatNum1(line.value)}${methodSuffix}`,
        short: formatNum1(line.value),
        color,
      });
    }
    if (!isNarrowChart && visibleGroups.has("recommended") && recommendedRangeValue) {
      const [lo, hi] = recommendedRangeValue;
      const x1 = Math.min(plotX(lo), plotX(hi));
      const x2 = Math.max(plotX(lo), plotX(hi));
      items.push({
        key: "window_bounds",
        rowKey: "window_bounds",
        x: (x1 + x2) / 2,
        full: `권장 구간 (${METHOD_LABEL[activeMethod]}) ${formatNum1(Math.min(lo, hi))}~${formatNum1(Math.max(lo, hi))}`,
        short: "",
        color: theme === "dark" ? methodColor.dark : methodColor.light,
      });
    }
    return items;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayLines, visibleGroups, isNarrowChart, recommendedRangeValue, activeMethod, methodColor, theme, plotWidth, view, boxBins]);

  const labelLayout = useMemo(
    () => layoutFloatingLabels(floatingLabels, isNarrowChart, (text) => measureTextWidth(text, LABEL_FONT)),
    [floatingLabels, isNarrowChart],
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

  // 선택 표시 (spec B) -- 선택이 있을 때만 비선택 점을 흐리게 한다.
  const hasSelection = selectedPoints.length > 0;
  const selectedSet = new Set(selectedPoints);

  // Esc는 브러시 선택만 해제한다 (spec CA "드래그 브러시 선택과의 관계") --
  // 슬라이더(그룹 강조)는 건드리지 않는다. 선택이 없을 때는 리스너 자체를
  // 붙이지 않아 카드가 여러 개 떠 있어도 서로 방해하지 않는다.
  useEffect(() => {
    if (!hasSelection) return;
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") setSelectedPoints([]);
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [hasSelection]);

  // 드래그 브러시 히트테스트용 렌더 좌표 -- Box 모드는 지터된 화면 위치
  // 기준으로 판정한다(지시서 B: "지터된 점 좌표 기준"), Scatter 모드는
  // 실제 값 좌표. 숨겨진(inlier/outlier 토글 꺼짐) 점은 제외한다.
  // H-4: 이 배열(최대 1,470개 점)이 매 렌더(호버 포함)마다 새로 만들어지고
  // 있었다 -- 실제로 좌표가 바뀔 수 있는 입력이 바뀔 때만 다시 만든다.
  // catScale은 매 렌더 새로 만들어지는 함수라 그 자체를 deps에 넣으면
  // 메모가 무력화되므로, 대신 catScale의 실제 값을 좌우하는
  // plotWidth/catDomain을 deps로 쓴다.
  const dragHitPoints = useMemo<{ point: ScatterPoint; screenX: number; screenY: number }[]>(() => {
    return view === "box"
      ? boxBins.flatMap((bin) =>
          bin.members
            .filter((member) => (member.isOutlier ? boxPointsVisible.outlier : boxPointsVisible.inlier))
            .map((member) => ({
              point: member.point,
              screenX: catScale(bin.index + 1 + member.jitter),
              screenY: yScale(member.point.y),
            })),
        )
      : data.points.map((point) => ({ point, screenX: xScale(point.x), screenY: yScale(point.y) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, boxBins, data.points, boxPointsVisible, xScale, yScale, plotWidth, catDomain]);

  function toPlotRelative(clientX: number, clientY: number): DragPos | null {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return { x: clientX - rect.left - MARGIN.left, y: clientY - rect.top - MARGIN.top };
  }

  function handlePlotMouseDown(event: React.MouseEvent<SVGRectElement>) {
    const pos = toPlotRelative(event.clientX, event.clientY);
    if (!pos) return;
    // 새 드래그를 시작하면 이전 선택은 즉시 지운다 -- 그래야 드래그가
    // 결국 "빈 영역 클릭"으로 끝나도 선택 해제와 동일한 결과가 된다.
    setSelectedPoints([]);
    setDragStart(pos);
    setDragCurrent(pos);
  }

  function finishDrag(start: DragPos, end: DragPos) {
    const distance = Math.hypot(end.x - start.x, end.y - start.y);
    if (distance < DRAG_THRESHOLD_PX) {
      // 클릭으로 본다 -- 기존 포인트 클릭 상세를 그대로 재현한다. 이
      // 오버레이가 모든 하위 도형의 포인터 이벤트를 가로채므로(위
      // findBoxColumnAt 주석 참고) 개별 <circle>의 onClick은 실제로
      // 발생하지 않는다 -- 클릭 처리는 여기서 대신한다.
      if (view === "box") {
        const nearest = findNearestBoxPoint(end.x, end.y);
        if (nearest) onSelectWafer(nearest.point);
      } else {
        const nearest = findNearestPoint(end.x, end.y);
        if (nearest) onSelectWafer(nearest);
      }
      return;
    }
    const xLo = Math.min(start.x, end.x);
    const xHi = Math.max(start.x, end.x);
    const yLo = Math.min(start.y, end.y);
    const yHi = Math.max(start.y, end.y);
    const hits = dragHitPoints
      .filter((p) => p.screenX >= xLo && p.screenX <= xHi && p.screenY >= yLo && p.screenY <= yHi)
      .map((p) => p.point);
    // 우상단 코너에 정보 박스가 뜨는데 드래그 사각형이 그 자리를
    // 침범하면 박스가 선택 점을 가리므로 좌상단으로 flip (지시서 B).
    setStatsBoxFlip(xHi > plotWidth - STATS_BOX_WIDTH && yLo < STATS_BOX_HEIGHT);
    setSelectedPoints(hits);
  }

  // 드래그 중에는 커서가 오버레이 밖으로 나가도 계속 추적해야 하므로
  // window 리스너를 쓴다 (오버레이 자체의 onMouseMove/onMouseUp만으로는
  // 사각형 밖으로 나가는 순간 드래그가 끊긴다).
  useEffect(() => {
    if (!dragStart) return;
    function handleWindowMouseMove(event: MouseEvent) {
      const pos = toPlotRelative(event.clientX, event.clientY);
      if (pos) setDragCurrent(pos);
    }
    function handleWindowMouseUp(event: MouseEvent) {
      const end = toPlotRelative(event.clientX, event.clientY) ?? dragCurrent ?? dragStart;
      if (dragStart && end) finishDrag(dragStart, end);
      setDragStart(null);
      setDragCurrent(null);
    }
    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", handleWindowMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragStart]);

  function handlePlotOverlayMouseMove(event: React.MouseEvent<SVGRectElement>) {
    if (dragStart) return; // 드래그 중엔 브러시 사각형만 갱신한다 (window 리스너), 호버는 잠시 멈춘다.
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

  // C-5: data.optimal_center는 SPC 고정값이다 -- ML로 전환하면 라인·범례는
  // displayedOptimalCenter(ML 중심)를 따르는데 해석 문장만 SPC 값에
  // 머물러 있었다.
  const interpretationText = buildInterpretationTip(data.eps2, data.relation_shape, displayedOptimalCenter, targetAxisLabel(data.axis.y_label), view);
  // HA그룹: 해석(항상) + 신뢰도(조건부, caller가 넘겨줄 때만) 두 행을
  // 카드 하나로 합친다 -- 순서는 해석 -> 신뢰도(무엇인지 먼저, 얼마나
  // 믿을지 나중).
  const interpretationRows: InterpretationRow[] = [
    { label: "해석", text: interpretationText },
    ...(reliabilityText ? [{ label: "신뢰도" as const, text: reliabilityText }] : []),
  ];

  // 단조 인자는 최적 중심 자체가 개념적으로 없으므로 항목을 아예 숨긴다
  // (spec §3-4) -- u_shape/unclear에서 값이 빠진 경우는 기존처럼 "해당
  // 없음" 비활성 카드로 남겨 이유를 알 수 있게 한다.
  const showOptimalLegendCard = data.relation_shape !== "monotonic_increasing" && data.relation_shape !== "monotonic_decreasing";

  function renderLineBody(line: DisplayLine) {
    if (!visibleGroups.has("optimal")) return null;
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

  // line 라벨/권장 구간 라벨 공용 렌더러 -- labelLayout이 이미 x/y/숨김
  // 여부/축약 텍스트까지 계산해 뒀으므로 여기서는 그리기만 한다 (spec
  // §C-6: Box Plot도 plotX를 통해 같은 좌표계를 쓰므로 별도 분기가 없다).
  function renderFloatingLabel(key: string) {
    const layout = labelLayout.find((l) => l.key === key);
    if (!layout || layout.hidden || !layout.text) return null;
    return (
      // A wide, generously-overflowing box (label text is unconstrained
      // by it anyway -- overflow:visible) with the badge itself centered
      // by flex, not by relying on inline-block sizing to exactly 40px:
      // "권장 구간 (SPC) 54.7~61.5" is much wider than that and would
      // render left-shifted off its true position otherwise.
      <foreignObject key={key} x={layout.x - 70} y={layout.y} width={140} height={14} style={{ overflow: "visible", pointerEvents: "none" }}>
        <div className="scatterLineLabelWrap">
          <span className="scatterLineLabel" style={{ color: layout.color }}>{layout.text}</span>
        </div>
      </foreignObject>
    );
  }

  return (
    <div className="scatterChart" ref={containerRef}>
      {/* 해석+신뢰도 카드 (HA그룹) -- 차트 바로 위, 제목/메타 정보와는
          별도 줄로 쌓인다. 통계 용어 없이 이 차트가 무엇을 보여주는지
          한 줄로 요약하고, 설명력이 낮은 경우에는 그 사실을 같은 카드
          안 두 번째 행으로 덧붙인다(카드를 두 개 쌓지 않는다). */}
      <InterpretationCard rows={interpretationRows} />

      <div className="scatterPlotWrap">
      <svg ref={svgRef} width="100%" height={height} className="scatterChartSvg" role="img" aria-label={`${factorAxisLabel(data.axis.x_label)} vs ${targetAxisLabel(data.axis.y_label)} 산점도`}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* dark-mode-only plot background (spec §4 재지시) -- painted
              first so every other layer sits on top of it. */}
          <rect x={0} y={0} width={plotWidth} height={plotHeight} className="scatterPlotBg" />
          {/* axis ticks -- 썸네일은 눈금/라벨 없이 축 선만 (지시서 P-2).
              식별용이라 값을 읽을 필요가 없다. */}
          {thumbnail ? (
            <>
              <line x1={0} x2={0} y1={0} y2={plotHeight} className="scatterGridLine" />
              <line x1={0} x2={plotWidth} y1={plotHeight} y2={plotHeight} className="scatterGridLine" />
            </>
          ) : (
            <>
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
            </>
          )}

          {/* 썸네일에서는 최적 중심 참조선 하나만 남긴다 (지시서 P-2) --
              권장 구간 밴드는 3종 참조선 중 하나라 여기서 함께 생략한다. */}
          {!thumbnail && visibleGroups.has("recommended") && recommendedRangeValue && (() => {
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
                {/* 밴드 양끝 안쪽 경계값 (spec §C-5) -- "권장 구간 (SPC)
                    54.7~61.5" 이름표는 이제 위쪽 3단 라벨 체계
                    (window_bounds)로 옮겨졌으니, 밴드 안에는 그 라벨이
                    좁은 화면에서 생략되더라도 경계값만은 항상 읽히도록
                    작은 값 라벨만 남긴다. SPC/ML 모두 동일하게 적용된다. */}
                {x2 - x1 >= BAND_BOUNDARY_MIN_WIDTH && (
                  <>
                    <text x={x1 + 4} y={20} textAnchor="start" className="scatterBandBoundaryLabel" style={{ fill }} pointerEvents="none">
                      {formatNum1(Math.min(lo, hi))}
                    </text>
                    <text x={x2 - 4} y={20} textAnchor="end" className="scatterBandBoundaryLabel" style={{ fill }} pointerEvents="none">
                      {formatNum1(Math.max(lo, hi))}
                    </text>
                  </>
                )}
              </>
            );
          })()}

          {/* data points -- painted before every reference line/curve (spec
              §4-1: lines and the trend curve must sit above the points,
              never hidden behind them). */}
          {view === "scatter"
            ? data.points.map((point, index) => {
                const style = colorForPoint(point, colorMode, lotIndex, theme);
                const isHovered = pointHover?.point === point;
                const isSelected = selectedSet.has(point);
                const stroke = isHovered ? (theme === "dark" ? "#FFFFFF" : "#0E306D") : "none";
                // 모바일 반응형 패치 S-4: ≤767px에서 점 반지름 1px 축소
                // (기본 4px -> 3px, 투명도는 손대지 않는다).
                const size = thumbnail ? THUMBNAIL_POINT_RADIUS : isMobileLayout ? Math.max(style.size - 1, 2) : style.size;
                const opacity = thumbnail ? THUMBNAIL_POINT_OPACITY : style.opacity;
                const baseOpacity = isHovered || isSelected ? 1 : hasSelection ? opacity / 3 : opacity;
                return (
                  <circle
                    key={point.lot_wafer_id ?? index}
                    cx={xScale(point.x)}
                    cy={yScale(point.y)}
                    r={isHovered ? size * 1.5 : isSelected ? size + 1 : size}
                    fill={isSelected && !isHovered ? ACCENT_COLOR : style.color}
                    opacity={applyScrubDim(point, baseOpacity, isHovered, isSelected)}
                    stroke={isSelected && !isHovered ? ACCENT_COLOR : stroke}
                    strokeWidth={isHovered ? 1.5 : isSelected ? 1.5 : 0}
                    style={{ cursor: "pointer" }}
                    onClick={() => onSelectWafer(point)}
                  />
                );
              })
            : boxBins.map((bin) => {
                // 보정 §I-3: 상자 테두리·수염은 이제 권장구간 안/밖과
                // 무관하게 항상 중립(var(--line)) 하나다.
                const boxColor = theme === "dark" ? BOX_LINE_COLOR.dark : BOX_LINE_COLOR.light;
                const boxFill = theme === "dark" ? BOX_FILL_COLOR.dark : BOX_FILL_COLOR.light;
                const whiskerColor = theme === "dark" ? BOX_COLOR.whisker.dark : BOX_COLOR.whisker.light;
                const medianColor = theme === "dark" ? BOX_COLOR.median.dark : BOX_COLOR.median.light;
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
                        (spec §3-3: "이상치는 속이 빈 원. 채우지 마라"). Radius
                        stays smaller than the scatter view's (spec §3-2) since
                        these sit on top of the box itself, but opacity is now
                        POINT_OPACITY -- the same value the scatter view's
                        colorForPoint uses (지시서 X그룹: 두 뷰의 점 무게가
                        같아야 한다). */}
                    {bin.members.map((member, mi) => {
                      const cx = catScale(bin.index + 1 + member.jitter);
                      const cy = yScale(member.point.y);
                      const isHovered = pointHover?.point === member.point;
                      const isSelected = selectedSet.has(member.point);
                      if (member.isOutlier) {
                        if (!boxPointsVisible.outlier) return null;
                        const baseOpacity = 0.85;
                        const opacity = applyScrubDim(
                          member.point,
                          isHovered || isSelected ? 1 : hasSelection ? baseOpacity / 3 : baseOpacity,
                          isHovered,
                          isSelected,
                        );
                        // 관리한계 개념 제거(GBDT 전환) -- 통계적 이상치(IQR
                        // 기준)는 항상 무채색 하나로 통일한다.
                        const outlierColor = theme === "dark" ? BOX_COLOR.outlierNeutral.dark : BOX_COLOR.outlierNeutral.light;
                        return (
                          <circle
                            key={member.point.lot_wafer_id ?? `${bin.index}-out-${mi}`}
                            cx={cx} cy={cy} r={thumbnail ? THUMBNAIL_POINT_RADIUS : isHovered ? 6 : isSelected ? 5.5 : 4.5}
                            fill="none"
                            stroke={isSelected && !isHovered ? ACCENT_COLOR : outlierColor}
                            strokeWidth={isHovered ? 1.8 : isSelected ? 1.6 : 1}
                            opacity={opacity}
                            style={{ cursor: "pointer" }}
                            onClick={() => onSelectWafer(member.point)}
                          />
                        );
                      }
                      if (!boxPointsVisible.inlier) return null;
                      const style = colorForPoint(member.point, colorMode, lotIndex, theme);
                      const baseOpacity = POINT_OPACITY;
                      const opacity = applyScrubDim(
                        member.point,
                        isHovered ? 0.85 : isSelected ? 1 : hasSelection ? baseOpacity / 3 : baseOpacity,
                        isHovered,
                        isSelected,
                      );
                      return (
                        <circle
                          key={member.point.lot_wafer_id ?? `${bin.index}-in-${mi}`}
                          cx={cx} cy={cy} r={thumbnail ? THUMBNAIL_POINT_RADIUS : isHovered ? 4 : isSelected ? 3.2 : 2.2}
                          fill={isSelected && !isHovered ? ACCENT_COLOR : style.color}
                          opacity={opacity}
                          style={{ cursor: "pointer" }}
                          onClick={() => onSelectWafer(member.point)}
                        />
                      );
                    })}
                    {/* box -- outline only, never filled, so jitter points under it
                        stay visible (spec §3-2). Hover is handled by the
                        shared plot overlay rect (see findBoxColumnAt), not
                        by listeners here -- the overlay paints on top of
                        every plot element and would swallow them anyway.
                        보정 §I-3: 8% var(--measured) 옅은 채움을 추가했다 --
                        fillOpacity가 낮아 밑에 깔린 jitter 점은 계속 보인다. */}
                    <rect
                      x={centerX - boxWidthPx / 2}
                      y={boxTop}
                      width={boxWidthPx}
                      height={Math.max(boxBottom - boxTop, 0.5)}
                      fill={boxFill}
                      fillOpacity={0.08}
                      stroke={boxColor}
                      strokeWidth={1.6}
                    />
                    <line
                      x1={centerX - boxWidthPx / 2} x2={centerX + boxWidthPx / 2}
                      y1={medianY} y2={medianY}
                      stroke={medianColor} strokeWidth={1.5}
                      style={{ pointerEvents: "none" }}
                    />
                  </g>
                );
              })}

          {/* reference line body -- 최적 중심 (spec §4-1). Labels are a
              separate, always-topmost pass below so they never sit under a
              later-drawn line. */}
          {displayLines.map((line) => renderLineBody(line))}

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

          {/* line name labels -- topmost of all. 썸네일은 라벨을 전부 생략한다
              (지시서 P-2). */}
          {!thumbnail && labelLayout.map((layout) => renderFloatingLabel(layout.key))}

          {/* 드래그 브러시 사각형 (spec B) -- 오버레이보다 먼저 그려서
              오버레이가 계속 포인터 이벤트를 받게 둔다(사각형 자체는
              pointerEvents:none). */}
          {dragStart && dragCurrent && (
            <rect
              x={Math.min(dragStart.x, dragCurrent.x)}
              y={Math.min(dragStart.y, dragCurrent.y)}
              width={Math.abs(dragCurrent.x - dragStart.x)}
              height={Math.abs(dragCurrent.y - dragStart.y)}
              className="scatterBrushRect"
              pointerEvents="none"
            />
          )}

          {/* continuous point+trend hover overlay -- nearest-point search,
              not per-point listeners (see findNearestPoint's comment).
              Also owns click-vs-drag selection (spec B): every pointer
              event in the plot lands here first (see findBoxColumnAt's
              comment on why per-shape listeners don't fire), so point
              click and the drag brush both route through this one rect. */}
          <rect
            x={0} y={0} width={plotWidth} height={plotHeight} fill="transparent"
            onMouseDown={handlePlotMouseDown}
            onMouseMove={handlePlotOverlayMouseMove}
            onMouseLeave={() => {
              if (dragStart) return; // drag continues via the window listener
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

      {selectedPoints.length > 0 && (
        <SelectionStatsBox
          points={selectedPoints}
          xLabel={factorAxisLabel(data.axis.x_label)}
          yLabel={targetAxisLabel(data.axis.y_label)}
          flip={statsBoxFlip}
          onClose={() => setSelectedPoints([])}
          // 브러시는 강조 그룹과 무관하게 전체에서 선택한다 (spec CA "드래그
          // 브러시 선택과의 관계") -- 통계 박스에는 교집합만 병기한다.
          emphasizedCount={
            activeScrubGroup ? selectedPoints.filter((p) => scrubKeyByPoint?.get(p) === activeScrubGroup).length : null
          }
        />
      )}
      </div>

      {!thumbnail && <p className="scatterAxisTitle">{factorAxisLabel(data.axis.x_label)}</p>}

      {/* 범례 (spec §5) -- 점의 색 구분(Color By)은 더 이상 범례에 나타나지
          않는다: 어떤 Color By를 고르든 이 아래 내용은 완전히 동일하다.
          Box Plot은 박스 자체를 읽는 법(1행)이 앞에 붙고, 기준선 4종 +
          구간 평균 불량률(row2)은 두 보기에서 문구까지 완전히 같다
          (spec §5-3: "Box Plot 2행이 Scatter Plot 1행과 글자까지
          동일"). */}
      <div className="scatterRefLegend">
        <div className="scatterRefLegendDivider" />

        {/* 그룹 스크럽 슬라이더 (spec CA) -- Config별/LOT별 Color By에서만
            렌더한다("기본"이면 그룹 개념이 없다). 위치 0 = 전체, 오른쪽으로
            움직이면 scrubSort 기준으로 정렬된 그룹을 하나씩 강조한다.
            참조선·최적중심·권장구간·박스 자체는 이 상태와 무관하게 항상
            전체 데이터 기준으로 그려진다(§C 재계산 금지 -- 장비별 Trellis
            모달의 역할과 겹치지 않게 한다). */}
        {colorMode !== "default" && (
          <div className="scatterScrubRow">
            <span className="scatterScrubLabel">그룹 강조</span>
            <button
              type="button"
              className="alertsPresetButton scatterScrubResetButton"
              onClick={() => {
                setScrubIndex(0);
                setScrubInputInvalid(false);
              }}
              disabled={scrubIndex === 0}
              title="강조를 해제하고 전체를 봅니다"
            >
              전체로
            </button>
            <div className="scatterScrubTrack">
              <input
                type="range"
                className="alertsGaugeSlider scatterScrubSlider"
                min={0}
                max={scrubGroups.length}
                step={1}
                value={Math.min(scrubIndex, scrubGroups.length)}
                onChange={(event) => scheduleScrubIndex(Number(event.target.value))}
                aria-label="그룹 강조 위치 (왼쪽 끝 = 전체)"
              />
              {/* 눈금 -- Config처럼 스톱이 적을 때만 그린다 (LOT 390개는
                  눈금이 뭉개진 띠가 될 뿐이라 그리지 않는다). */}
              {scrubGroups.length > 0 && scrubGroups.length <= SCRUB_TICK_MAX_STOPS && (
                <div className="scatterScrubTicks" aria-hidden="true">
                  {Array.from({ length: scrubGroups.length + 1 }, (_, i) => (
                    <span key={i} className="scatterScrubTick" style={{ left: `${(i / scrubGroups.length) * 100}%` }} />
                  ))}
                </div>
              )}
            </div>
            <input
              key={scrubIndex}
              type="text"
              className={`scatterScrubInput ${scrubInputInvalid ? "invalid" : ""}`}
              defaultValue={scrubGroupLabel}
              onFocus={(event) => event.currentTarget.select()}
              onBlur={(event) => commitScrubInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") commitScrubInput((event.target as HTMLInputElement).value);
              }}
              title="그룹명을 입력하면 그 위치로 이동합니다 (비우면 전체로)"
              aria-label="강조할 그룹명"
            />
            <select
              className="tableSortSelect scatterScrubSortSelect"
              value={scrubSort}
              onChange={(event) => {
                setScrubSort(event.target.value as ScrubSort);
                setScrubIndex(0);
              }}
              aria-label="그룹 강조 정렬 기준"
            >
              <option value="mean">정렬: 평균순</option>
              <option value="name">정렬: 이름순</option>
              <option value="count">정렬: 점 개수순</option>
            </select>
          </div>
        )}

        {view === "box" && (
          <div className="scatterLegendRow">
            <LegendCard
              icon={<IconBoxWhisker boxColor={theme === "dark" ? BOX_LINE_COLOR.dark : BOX_LINE_COLOR.light} medianColor={theme === "dark" ? BOX_COLOR.median.dark : BOX_COLOR.median.light} />}
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
              icon={<IconOutlierDots color={theme === "dark" ? BOX_COLOR.outlierNeutral.dark : BOX_COLOR.outlierNeutral.light} />}
              label="이상치 wafer"
              onClick={() => setBoxPointsVisible((cur) => ({ ...cur, outlier: !cur.outlier }))}
              active={boxPointsVisible.outlier}
            />
          </div>
        )}

        {/* 최적 중심 · 권장 구간 · 구간 평균 불량률 3종 -- Color By와
            무관하게 항상 동일. 관리한계(IQR/LCL/UCL)와 경고선은 화면에
            더는 노출하지 않는 개념이다 (src/analysis/control_range.py,
            warning_line.py에는 그대로 남아 있다). */}
        <div className="scatterLegendRow">
          {showOptimalLegendCard && (
            <LegendCard
              icon={<IconDashedLine color={theme === "dark" ? OPTIMAL_CENTER_COLOR.dark : OPTIMAL_CENTER_COLOR.light} dash="4 3" />}
              label={optimalAvailable ? `최적 중심  ${formatNum1(displayedOptimalCenter as number)} (${METHOD_LABEL[activeMethod]})` : "최적 중심"}
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
            desc={recommendedRangeValue ? "이 범위로 관리 권장" : "데이터 범위 밖"}
            onClick={(event) => toggleGroup("recommended", event)}
            active={visibleGroups.has("recommended") && !!recommendedRangeValue}
            disabled={!recommendedRangeValue}
            title={recommendedRangeValue ? "구간 평균 불량률이 전체 평균 이하인 구간" : "데이터 범위 밖"}
          />
          <LegendCard
            icon={<IconTrendLine color={theme === "dark" ? TREND_COLOR.dark : TREND_COLOR.light} />}
            label="구간 평균 불량률"
            onClick={() => setTrendVisible((v) => !v)}
            active={trendVisible}
            title="구간 평균 불량률"
          />
        </div>
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
            </div>
          );
        }
        return null;
      })()}

      {trendHover && !pointHover && (
        <div className="heatmapTooltip" style={{ left: MARGIN.left + trendHover.screenX + 14, top: MARGIN.top + trendHover.screenY + 14 }}>
          <strong>{factorAxisLabel(data.axis.x_label)} = {trendHover.dataX.toFixed(1)}</strong>
          <div className="heatmapTooltipRow"><span>구간 평균 불량률</span><b>{trendHover.dataY.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>95% 신뢰구간</span><b>{trendHover.ciLo.toFixed(2)} ~ {trendHover.ciHi.toFixed(2)}</b></div>
          <div className="heatmapTooltipRow"><span>이 구간 wafer 수</span><b>{trendHover.n.toLocaleString()}</b></div>
        </div>
      )}

      {pointHover && (
        <div className="heatmapTooltip" style={{ left: pointHover.clientX + 14, top: pointHover.clientY + 14 }}>
          {pointHover.point.lot_wafer_id && <strong>{pointHover.point.lot_wafer_id}</strong>}
          <div className="heatmapTooltipRow"><span>{factorAxisLabel(data.axis.x_label)}</span><b>{pointHover.point.x.toFixed(1)}</b></div>
          <div className="heatmapTooltipRow"><span>{targetAxisLabel(data.axis.y_label)}</span><b>{pointHover.point.y.toFixed(1)}%</b></div>
          {pointHover.isOutlier && (
            <div className="heatmapTooltipRow"><span /><b className="scatterOutlierTag">이상치</b></div>
          )}
          {/* 현재 Color By 기준값 -- 기존 항목은 그대로 두고 한 줄만 덧붙인다
              (spec §5-3). 항상 최신 colorMode를 읽으므로 전환 시 즉시 반영된다. */}
          {colorMode === "config_model" && pointHover.point.config && (() => {
            const parts = parseConfig(pointHover.point.config);
            return (
              <div className="scatterColorByRow">
                <div className="heatmapTooltipRow"><span>설비</span><b>{pointHover.point.config}</b></div>
                {parts && (
                  <div className="heatmapTooltipRow"><span /><b>모델 {parts.model} · 장비 {parts.eq} · 챔버 {parts.chamber}</b></div>
                )}
              </div>
            );
          })()}
          {colorMode === "lot" && pointHover.point.lot_id && (
            <div className="scatterColorByRow">
              <div className="heatmapTooltipRow"><span>LOT</span><b>{pointHover.point.lot_id}</b></div>
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

function meanOf(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
}

/** 드래그 선택 정보 박스 (spec B) -- 플롯 우측 상단에 고정 위치로 뜬다
 * (드래그 사각형을 따라다니지 않는다). 축 이름은 하드코딩하지 않고
 * `data.axis`에서 그대로 받은 실제 컬럼명을 쓴다. 1개 선택이면 통계
 * 대신 그 wafer 하나의 값만 간략히 보여준다 -- 기존 포인트 클릭
 * 상세(WaferDetailPopover)와 중복되지 않게, 표 없이 한 줄로만. */
function SelectionStatsBox({
  points,
  xLabel,
  yLabel,
  flip,
  onClose,
  emphasizedCount,
}: {
  points: ScatterPoint[];
  xLabel: string;
  yLabel: string;
  flip: boolean;
  onClose: () => void;
  // 그룹 스크럽 슬라이더가 강조 중일 때만 값이 있다 (spec CA) -- 브러시
  // 선택은 강조 그룹과 무관하게 전체에서 뽑히므로, 그 교집합만 괄호로
  // 병기한다. null이면 슬라이더가 "전체" 위치라 병기할 것이 없다.
  emphasizedCount: number | null;
}) {
  return (
    <div className={`heatmapTooltip scatterSelectionBox ${flip ? "flip" : ""}`}>
      <div className="scatterSelectionBoxHeader">
        <strong>
          선택 {points.length}개 점
          {emphasizedCount != null && ` (강조 그룹 내 ${emphasizedCount}개)`}
        </strong>
        <button type="button" className="scatterSelectionBoxClose" onClick={onClose} aria-label="선택 해제">✕</button>
      </div>
      {points.length === 1 ? (
        <div className="heatmapTooltipRow">
          <span>{points[0].lot_wafer_id ?? "-"}</span>
          <b>{xLabel} {formatTick(points[0].x)} · {yLabel} {formatTick(points[0].y)}</b>
        </div>
      ) : (
        <table className="scatterSelectionBoxTable">
          <thead>
            <tr><th /><th>{xLabel}</th><th>{yLabel}</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>평균</td>
              <td>{formatTick(meanOf(points.map((p) => p.x)))}</td>
              <td>{formatTick(meanOf(points.map((p) => p.y)))}</td>
            </tr>
            <tr>
              <td>중앙값</td>
              <td>{formatTick(quantileOf([...points.map((p) => p.x)].sort((a, b) => a - b), 0.5))}</td>
              <td>{formatTick(quantileOf([...points.map((p) => p.y)].sort((a, b) => a - b), 0.5))}</td>
            </tr>
            <tr>
              <td>최솟값</td>
              <td>{formatTick(Math.min(...points.map((p) => p.x)))}</td>
              <td>{formatTick(Math.min(...points.map((p) => p.y)))}</td>
            </tr>
            <tr>
              <td>최댓값</td>
              <td>{formatTick(Math.max(...points.map((p) => p.x)))}</td>
              <td>{formatTick(Math.max(...points.map((p) => p.y)))}</td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}

