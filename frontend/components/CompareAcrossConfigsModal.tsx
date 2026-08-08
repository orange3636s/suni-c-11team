"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { getScreeningScatter } from "@/lib/api";
import { effectSizeTierFromRho, TIER_TOOLTIP } from "@/lib/confidenceTier";
import { parseConfig, type ConfigParts } from "@/lib/constants";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ScatterPoint, ScreeningScatterResponse } from "@/types/data";

// n 미만이면 그룹 내 최적중심/구간/경고선을 그리지 않는다 -- 근거 없는
// 선을 그리면 오탐을 유도하므로 (지시서 "n 부족 처리").
const MIN_GROUP_N = 30;
// Y1~Y5 비교 모달(CompareAcrossTargetsModal의 MiniChart)과 완전히 같은
// 값으로 고정한다 (지시서 C-2: "기준은 Y1~Y5 비교 모달이다. 두 모달을
// 나란히 열었을 때 패널 카드가 같은 크기·같은 폰트로 보여야 한다") --
// 분할 기준(Model/EQ/Chamber)이 바뀌어도, 패널 수가 몇 개든 이 값은
// 절대 재계산하지 않는다. 패널이 늘면 .compareModalScroll의 가로
// 스크롤로 처리한다.
const CHART_W = 320;
const CHART_H = 260;
const MARGIN = { top: 26, right: 10, bottom: 26, left: 36 };
// 구간 평균 불량률 계산은 산점도/Box Plot과 동일한 12분위를 쓴다
// (ScatterChart.tsx의 BOX_BIN_COUNT와 맞춤).
const BIN_COUNT = 12;
// 최적 중심이 x 범위의 이 비율을 넘게 흩어지면 "따로 분리" 문구를 얹는다.
const CENTER_SPREAD_RATIO = 0.15;

// 발산(초록/빨강) 팔레트 제거 (지시서 N-3) -- 그룹별 최적 중심은 --text,
// 구간 평균 곡선은 --inferred, 경고선은 --sig-amber. ScatterChart.tsx의
// OPTIMAL_CENTER_COLOR/TREND_COLOR/LINE_COLOR.warning과 동일한 상수쌍이다.
const TEXT_COLOR = { light: "#141A22", dark: "#F5F5F7" };
const INFERRED_COLOR = { light: "#7C8AA5", dark: "#97A3B8" };
const WARNING_COLOR = { light: "#B8791A", dark: "#E4A23E" };
const GRAY = { light: "#9CA3AF", dark: "#6B7280" };
// 산점도 "기본" 모드의 단일 점 색과 동일한 상수(--measured, 지시서 P-1).
const POINT_COLOR = { light: "#0E306D", dark: "#7BA3E8" };

type SplitMode = "model" | "eq" | "chamber";
// 계층 순서(Model이 상위)와 일치시킨다 (지시서 C-1) -- Object.keys 순회
// 순서로 토글 버튼이 렌더되므로 이 선언 순서가 곧 화면 순서다.
const SPLIT_LABEL: Record<SplitMode, string> = { model: "Model", eq: "EQ 호기", chamber: "Chamber" };

function formatNum(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
}

function groupKeyFor(parts: ConfigParts, mode: SplitMode): string {
  if (mode === "eq") return parts.eq;
  if (mode === "model") return parts.model;
  return parts.chamber;
}
const UNKNOWN_GROUP = "미상";

/** Spearman rho, computed client-side per group (그룹 내 재계산 -- 백엔드는
 * 전체 데이터의 상관만 계산했으므로 층화 후 값은 여기서 다시 구해야
 * 한다). Average-rank ties, same convention as the backend's scipy call. */
function rankOf(values: number[]): number[] {
  const n = values.length;
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1]] === values[order[i]]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k]] = avgRank;
    i = j + 1;
  }
  return ranks;
}
function spearmanRho(points: { x: number; y: number }[]): number | null {
  const n = points.length;
  if (n < 2) return null;
  const rx = rankOf(points.map((p) => p.x));
  const ry = rankOf(points.map((p) => p.y));
  const meanRx = rx.reduce((s, v) => s + v, 0) / n;
  const meanRy = ry.reduce((s, v) => s + v, 0) / n;
  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = rx[i] - meanRx;
    const dy = ry[i] - meanRy;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  if (varX === 0 || varY === 0) return null;
  return cov / Math.sqrt(varX * varY);
}

// Abramowitz-Stegun erf approximation -- feeds the Fisher z pairwise
// difference test below.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * y;
}
function normalCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

// C-2: "다릅니다"를 단정하려면 그룹 쌍을 직접 비교하는 검정이 필요하다 --
// 예전에는 max|rho|/min|rho| 비율이 임의 상수(1.5) 미만이면 "유사"로
// 봤는데, 이 앱 전체가 FDR 규율을 쓰면서 이 화면만 검정 없이 결론을
// 냈다. 두 독립 상관계수의 차이는 Fisher z 변환 후 표준정규로 검정한다
// (Fisher, 1921) -- 각 그룹의 상관은 서로 다른 wafer 표본에서 나온
// 독립 추정치이므로 이 검정이 성립한다.
function fisherZ(r: number): number {
  const clamped = Math.max(-0.999999, Math.min(r, 0.999999));
  return 0.5 * Math.log((1 + clamped) / (1 - clamped));
}
function pairwiseSlopeDiffPValue(r1: number, n1: number, r2: number, n2: number): number | null {
  if (n1 <= 3 || n2 <= 3) return null;
  const se = Math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3));
  if (se === 0) return null;
  const z = (fisherZ(r1) - fisherZ(r2)) / se;
  return 2 * (1 - normalCdf(Math.abs(z)));
}
function formatPValue(p: number): string {
  return p < 0.001 ? "p<.001" : `p=${p.toFixed(3)}`;
}

type BinStat = { x_mean: number; y_mean: number; n: number; x_lo: number; x_hi: number };

/** 그룹 점들을 x 기준 12분위로 나눠 각 bin의 x/y 평균을 낸다 (지시서: 산점도/
 * Box Plot과 같은 BOX_BIN_COUNT=12). 등분위(같은 개수)로 자르는 quantile
 * binning -- 값 범위 등분이 아니다. */
function computeBins(points: { x: number; y: number }[]): BinStat[] {
  const n = points.length;
  if (n === 0) return [];
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const binCount = Math.min(BIN_COUNT, n);
  const bins: BinStat[] = [];
  for (let i = 0; i < binCount; i++) {
    const start = Math.floor((i * n) / binCount);
    const end = Math.floor(((i + 1) * n) / binCount);
    const chunk = sorted.slice(start, end);
    if (chunk.length === 0) continue;
    const xMean = chunk.reduce((s, p) => s + p.x, 0) / chunk.length;
    const yMean = chunk.reduce((s, p) => s + p.y, 0) / chunk.length;
    bins.push({ x_mean: xMean, y_mean: yMean, n: chunk.length, x_lo: chunk[0].x, x_hi: chunk[chunk.length - 1].x });
  }
  return bins;
}

/** 그룹 내 최적중심/최적구간/경고선을 12-bin 프로파일에서 재계산한다
 * (지시서 "그룹별 재계산 규칙" -- 전체 데이터로 계산된 값을 복사하면 이
 * 기능의 목적, 즉 패널마다 달라지는지를 보는 것이 무너진다). SPC 결과와
 * 정확히 일치할 필요는 없다 -- 세 패널에 같은 규칙만 일관되게 적용되면
 * 패널 간 상대 비교는 유효하다. */
function computeGroupWindow(points: { x: number; y: number }[]): {
  bins: BinStat[];
  optimalCenter: number | null;
  rangeLo: number | null;
  rangeHi: number | null;
  warningX: number | null;
} {
  const bins = computeBins(points);
  if (bins.length === 0) return { bins, optimalCenter: null, rangeLo: null, rangeHi: null, warningX: null };

  let minIdx = 0;
  for (let i = 1; i < bins.length; i++) if (bins[i].y_mean < bins[minIdx].y_mean) minIdx = i;
  const overallMean = points.reduce((s, p) => s + p.y, 0) / points.length;
  const threshold = (bins[minIdx].y_mean + overallMean) / 2;
  let lo = minIdx;
  let hi = minIdx;
  while (lo - 1 >= 0 && bins[lo - 1].y_mean <= threshold) lo--;
  while (hi + 1 < bins.length && bins[hi + 1].y_mean <= threshold) hi++;

  const variance = points.reduce((s, p) => s + (p.y - overallMean) ** 2, 0) / points.length;
  const sigma = Math.sqrt(variance);
  const warnThreshold = overallMean + 0.5 * sigma;
  let warningX: number | null = null;
  for (const b of bins) {
    if (b.y_mean > warnThreshold) {
      warningX = b.x_lo;
      break;
    }
  }

  return { bins, optimalCenter: bins[minIdx].x_mean, rangeLo: bins[lo].x_lo, rangeHi: bins[hi].x_hi, warningX };
}

type GroupStat = {
  key: string;
  points: ScatterPoint[];
  n: number;
  rho: number | null;
  bins: BinStat[];
  optimalCenter: number | null;
  rangeLo: number | null;
  rangeHi: number | null;
  warningX: number | null;
  insufficientN: boolean;
};

function buildGroupStat(key: string, points: ScatterPoint[]): GroupStat {
  const n = points.length;
  const rho = spearmanRho(points);
  const insufficientN = n < MIN_GROUP_N;
  if (insufficientN) {
    return { key, points, n, rho, bins: [], optimalCenter: null, rangeLo: null, rangeHi: null, warningX: null, insufficientN };
  }
  const window = computeGroupWindow(points);
  return { key, points, n, rho, ...window, insufficientN };
}

export default function CompareAcrossConfigsModal({
  feature,
  step,
  target,
  datasetId,
  onClose,
}: {
  feature: string;
  step: number;
  target: string;
  datasetId: string;
  onClose: () => void;
}) {
  const theme = useResolvedTheme();
  const [data, setData] = useState<ScreeningScatterResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [splitMode, setSplitMode] = useState<SplitMode>("model");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    // setState는 setTimeout 콜백 안에서 호출한다 (CompareAcrossTargetsModal과
    // 같은 패턴) -- 이펙트 본문에서 직접 호출하면 cascading render 린트
    // 규칙에 걸린다.
    const timer = window.setTimeout(() => {
      setLoading(true);
      setData(null);
      // 백엔드 신규 API 없이: 기존 getScreeningScatter를 1회만 호출하고,
      // 응답에 이미 실려 있는 ScatterPoint.config로 프론트에서 그룹핑한다.
      getScreeningScatter(datasetId, target, feature)
        .then((d) => {
          if (!cancelled) setData(d);
        })
        .catch(() => {
          if (!cancelled) setData(null);
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [datasetId, target, feature]);

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const xDomain = useMemo<[number, number]>(() => {
    if (!data || data.points.length === 0) return [0, 1];
    let min = Infinity;
    let max = -Infinity;
    for (const p of data.points) {
      if (p.x < min) min = p.x;
      if (p.x > max) max = p.x;
    }
    if (!Number.isFinite(min)) return [0, 1];
    const pad = (max - min) * 0.04 || 1;
    return [min - pad, max + pad];
  }, [data]);

  // y 도메인은 0부터 -- 패널마다 자동 스케일하면 패널 간 비교가 무의미해
  //지므로 (지시서 "필수 규칙 — 축 공유"), 모든 패널이 이 도메인을 그대로
  // 공유한다.
  const yDomain = useMemo<[number, number]>(() => {
    if (!data || data.points.length === 0) return [0, 1];
    let max = -Infinity;
    for (const p of data.points) if (p.y > max) max = p.y;
    if (!Number.isFinite(max) || max <= 0) return [0, 1];
    return [0, max * 1.08];
  }, [data]);

  const groups = useMemo<GroupStat[]>(() => {
    if (!data) return [];
    const byKey = new Map<string, ScatterPoint[]>();
    const unknown: ScatterPoint[] = [];
    for (const p of data.points) {
      const parts = p.config ? parseConfig(p.config) : null;
      // 이 인자가 속한 스텝의 Config만 쓴다 -- 다른 스텝 Config가 섞여
      // 들어오면(또는 형식이 다른 미지 Config면) "미상"으로 모은다.
      if (!parts || parts.step !== step) {
        unknown.push(p);
        continue;
      }
      const key = groupKeyFor(parts, splitMode);
      const arr = byKey.get(key);
      if (arr) arr.push(p);
      else byKey.set(key, [p]);
    }
    const orderedKeys = Array.from(byKey.keys()).sort();
    const stats = orderedKeys.map((key) => buildGroupStat(key, byKey.get(key)!));
    if (unknown.length > 0) stats.push(buildGroupStat(UNKNOWN_GROUP, unknown));
    return stats;
  }, [data, splitMode, step]);

  const interpretation = useMemo<string[] | null>(() => {
    if (loading || groups.length < 2) return null;
    const known = groups.filter((g): g is GroupStat & { rho: number } => g.rho != null);
    if (known.length < 2) return null;

    const lines: string[] = [];
    const rhoList = known.map((g) => `${g.key} ${g.rho >= 0 ? "+" : ""}${g.rho.toFixed(2)}`).join(" / ");

    // C-2: 모든 쌍을 Fisher z 차이 검정으로 비교하고, 다중 비교이므로
    // Bonferroni로 유의수준을 쌍 수만큼 보정한다(α=0.05/쌍 수). 임의
    // 비율 상수로 "다르다"를 단정하던 이전 로직을 대체한다.
    const pairs: { a: GroupStat & { rho: number }; b: GroupStat & { rho: number }; p: number }[] = [];
    for (let i = 0; i < known.length; i++) {
      for (let j = i + 1; j < known.length; j++) {
        const p = pairwiseSlopeDiffPValue(known[i].rho, known[i].n, known[j].rho, known[j].n);
        if (p != null) pairs.push({ a: known[i], b: known[j], p });
      }
    }

    if (pairs.length === 0) {
      // 표본이 너무 작아 어떤 쌍도 검정할 수 없다 -- 근거 없이 단정하지 않는다.
    } else {
      const alpha = 0.05 / pairs.length;
      const significant = pairs.filter((pair) => pair.p < alpha);
      if (significant.length === 0) {
        lines.push(`장비별 기울기 차이가 통계적으로 확인되지 않습니다 (${rhoList}).`);
      } else if (significant.length === pairs.length) {
        const desc = significant.map((pair) => `${pair.a.key}-${pair.b.key} ${formatPValue(pair.p)}`).join(", ");
        lines.push(`장비별 기울기가 통계적으로 다릅니다 (${desc}, ${pairs.length}쌍 보정). 전체 상관을 단일 결론으로 쓰지 마세요.`);
      } else {
        const only = significant.length === 1;
        const names = significant.map((pair) => `${pair.a.key}과 ${pair.b.key}`).join(", ");
        const pDesc = only
          ? formatPValue(significant[0].p)
          : significant.map((pair) => `${pair.a.key}-${pair.b.key} ${formatPValue(pair.p)}`).join(", ");
        lines.push(
          `${names} 사이에${only ? "만" : ""} 차이 근거가 있습니다 (${pDesc}, ${pairs.length}쌍 보정). 나머지 쌍은 구별되지 않습니다.`,
        );
      }
    }

    const centersKnown = groups.filter((g): g is GroupStat & { optimalCenter: number } => g.optimalCenter != null);
    if (centersKnown.length >= 2) {
      const xs = centersKnown.map((g) => g.optimalCenter);
      const spread = Math.max(...xs) - Math.min(...xs);
      const xRange = xDomain[1] - xDomain[0];
      if (xRange > 0 && spread > CENTER_SPREAD_RATIO * xRange) {
        lines.push("장비별 최적 중심이 흩어져 있습니다 — 권장구간을 장비별로 분리해야 할 수 있습니다.");
      }
    }
    return lines;
  }, [loading, groups, xDomain]);

  const yAxisLabel = `${target} (%)`;

  return (
    <div className="compareModalBackdrop" onClick={onClose} role="presentation">
      <div className="compareModal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-label={`${feature} 장비별 Trellis 비교`}>
        <div className="compareModalHeader">
          <div>
            <h2>{feature} vs {target} — 장비별 Trellis</h2>
            {data && <p className="compareModalMeta">n={data.n.toLocaleString()} 계측 · Step{step}</p>}
            <div className="scatterViewToggleRow" style={{ marginTop: 6 }}>
              <span className="scatterViewToggleLabel">분할 기준</span>
              <div className="scatterViewToggle" role="group" aria-label="장비 분할 기준">
                {(Object.keys(SPLIT_LABEL) as SplitMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={`scatterViewToggleBtn ${splitMode === mode ? "active" : ""}`}
                    onClick={() => setSplitMode(mode)}
                  >
                    {SPLIT_LABEL[mode]}
                  </button>
                ))}
              </div>
            </div>
            {interpretation && interpretation.length > 0 && (
              <p className="compareModalInterpretation">
                {interpretation.map((line) => (
                  <span key={line}>{line}</span>
                ))}
              </p>
            )}
          </div>
          <button type="button" className="compareModalClose" onClick={onClose} aria-label="닫기">✕</button>
        </div>

        {loading ? (
          <p className="emptyMessage">불러오는 중…</p>
        ) : groups.length === 0 ? (
          <p className="emptyMessage">이 인자의 장비(Config) 정보를 찾을 수 없어 Trellis를 표시할 수 없습니다.</p>
        ) : (
          <div className="compareModalScroll" ref={scrollRef}>
            {groups.map((group, index) => (
              <TrellisPanel
                key={group.key}
                group={group}
                index={index}
                width={CHART_W}
                xDomain={xDomain}
                yDomain={yDomain}
                globalOptimalCenter={data?.optimal_center ?? null}
                theme={theme}
                yAxisLabel={yAxisLabel}
              />
            ))}
          </div>
        )}

        <div className="compareModalFooter">
          <div className="compareModalLegend">
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? GRAY.dark : GRAY.light }} /> 전체 최적 중심 (층화 전)</span>
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? TEXT_COLOR.dark : TEXT_COLOR.light }} /> 그룹별 최적 중심</span>
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? WARNING_COLOR.dark : WARNING_COLOR.light }} /> 그룹별 경고선</span>
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? INFERRED_COLOR.dark : INFERRED_COLOR.light }} /> 구간 평균 불량률</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrellisPanel({
  group,
  index,
  width,
  xDomain,
  yDomain,
  globalOptimalCenter,
  theme,
  yAxisLabel,
}: {
  group: GroupStat;
  index: number;
  width: number;
  xDomain: [number, number];
  yDomain: [number, number];
  globalOptimalCenter: number | null;
  theme: "light" | "dark";
  yAxisLabel: string;
}) {
  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = CHART_H - MARGIN.top - MARGIN.bottom;
  const xScale = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plotWidth;
  const yScale = (v: number) => plotHeight - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * plotHeight;

  // 미니 패널은 작으므로 눈금을 양끝 2개로 덜어낸다 (지시서 N-3) -- 값을
  // 읽는 화면이 아니라 형태를 보는 화면이다.
  const xTicks = useMemo(() => [xDomain[0], xDomain[1]], [xDomain]);
  const yTicks = useMemo(() => [yDomain[0], yDomain[1]], [yDomain]);

  const textColor = theme === "dark" ? TEXT_COLOR.dark : TEXT_COLOR.light;
  const inferredColor = theme === "dark" ? INFERRED_COLOR.dark : INFERRED_COLOR.light;
  const warningColor = theme === "dark" ? WARNING_COLOR.dark : WARNING_COLOR.light;
  const grayColor = theme === "dark" ? GRAY.dark : GRAY.light;
  const pointColor = theme === "dark" ? POINT_COLOR.dark : POINT_COLOR.light;

  // C-2: 패널별 무보정 p 배지는 제거했다 -- 장비 간 차이는 이제 모달
  // 상단의 Fisher z + Bonferroni 쌍별 검정 문구가 담당한다. 효과크기
  // 등급(배지)만 이 패널에 남는다: "관계의 크기"는 그룹 간 차이와
  // 무관한 별개 질문이다.
  const tier = !group.insufficientN && group.rho != null ? effectSizeTierFromRho(group.rho) : null;

  return (
    <div className={`compareMiniChart ${group.insufficientN ? "insufficientN" : ""}`} style={{ width }}>
      <div className="compareMiniChartHeader">
        <span className="compareMiniChartTitle">
          {group.key}
          <span className="compareMiniChartRho">
            {" "}n={group.n}
            {group.rho != null && <> · ρ={group.rho >= 0 ? "+" : ""}{group.rho.toFixed(2)}</>}
          </span>
        </span>
        {group.insufficientN ? (
          <span className="confidenceBadge tier-reference">표본 부족(n={group.n})</span>
        ) : tier ? (
          <ConfidenceBadge
            tier={tier}
            title={`이 그룹 내에서 인자-타깃 관계의 효과크기 등급입니다 (그룹 간 차이와는 무관). ${TIER_TOOLTIP[tier]}`}
          />
        ) : null}
      </div>

      <svg width={width} height={CHART_H} className="compareMiniChartSvg">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line x1={0} x2={plotWidth} y1={yScale(tick)} y2={yScale(tick)} className="compareMiniGridLine" />
              {index === 0 && (
                <text x={-4} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="compareMiniTick">{formatNum(tick)}</text>
              )}
            </g>
          ))}
          {xTicks.map((tick) => (
            <text key={`x-${tick}`} x={xScale(tick)} y={plotHeight + 14} textAnchor="middle" className="compareMiniTick">{formatNum(tick)}</text>
          ))}
          {index === 0 && (
            <text
              x={0}
              y={0}
              textAnchor="middle"
              dominantBaseline="middle"
              className="compareMiniTick"
              transform={`translate(${-MARGIN.left + 10}, ${plotHeight / 2}) rotate(-90)`}
            >
              {yAxisLabel}
            </text>
          )}

          {/* 권장 구간 밴드: 옅은 중립 배경만, 테두리 없음 -- 작은 패널에서는
              덜어낼수록 핵심 3선(최적 중심/구간 평균 곡선/경고선)이 산다
              (지시서 N-3). */}
          {!group.insufficientN && group.rangeLo != null && group.rangeHi != null && (
            <rect
              x={xScale(group.rangeLo)}
              y={0}
              width={Math.max(xScale(group.rangeHi) - xScale(group.rangeLo), 0)}
              height={plotHeight}
              fill={grayColor}
              opacity={0.1}
            />
          )}

          {/* points painted before reference lines/curves so lines stay visible on top --
              산점도와 같은 --measured 단일색, 패널이 작으므로 반지름은 더 줄인다. */}
          {group.points.map((p, i) => (
            <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={1.7} fill={pointColor} opacity={0.5} />
          ))}

          {globalOptimalCenter != null && (
            <line x1={xScale(globalOptimalCenter)} x2={xScale(globalOptimalCenter)} y1={0} y2={plotHeight} stroke={grayColor} strokeWidth={1.2} strokeDasharray="2 3" opacity={0.85} />
          )}
          {!group.insufficientN && group.optimalCenter != null && (
            <line x1={xScale(group.optimalCenter)} x2={xScale(group.optimalCenter)} y1={0} y2={plotHeight} stroke={textColor} strokeWidth={1.4} strokeDasharray="4 3" />
          )}
          {!group.insufficientN && group.warningX != null && (
            <line x1={xScale(group.warningX)} x2={xScale(group.warningX)} y1={0} y2={plotHeight} stroke={warningColor} strokeWidth={1.4} strokeDasharray="10 5" />
          )}

          {!group.insufficientN && group.bins.length > 0 && (
            <path
              d={group.bins.map((b, i) => `${i === 0 ? "M" : "L"}${xScale(b.x_mean)},${yScale(b.y_mean)}`).join(" ")}
              fill="none"
              stroke={inferredColor}
              strokeWidth={1.8}
              opacity={0.9}
            />
          )}
        </g>
      </svg>
    </div>
  );
}
