"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { getScreeningScatter } from "@/lib/api";
import { DIVERGING_GREEN, DIVERGING_RED } from "@/lib/constants";
import { niceTicks, niceTicksFitted } from "@/lib/niceTicks";
import { measureTextWidth } from "@/lib/textMeasure";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { RelationShape, ScreeningScatterResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
const CHART_W = 320;
const CHART_H = 260;
const CHART_GAP = 24;
const MARGIN = { top: 26, right: 10, bottom: 26, left: 36 };
const STRONG_RHO = 0.15;
const POINT_HOVER_RADIUS = 16;
// Tighter than the main chart's 8-10/6-8 (spec §8: mini-charts get their
// own defaults) -- these panels are ~320x260 vs. the main chart's much
// larger plot area, so the same density would just overlap.
const MINI_TICK_FONT = "9px system-ui, -apple-system, sans-serif";
const MINI_X_TICK_COUNT: [max: number, min: number] = [5, 4];
const MINI_Y_TICK_COUNT = 4;

const NAVY = { light: "#0E306D", dark: "#7BA3E8" };
const GREEN = DIVERGING_GREEN;
const RED = DIVERGING_RED;

function formatNum(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
}

// A target that "dominates" needs its |rho| to clear the runner-up by this
// ratio, otherwise the spread reads as broad-and-even rather than
// one-target-leads (spec §2-1 pattern B vs. C).
const DOMINANCE_RATIO = 1.5;

/** Korean 이/가 particle for a word ending in digit, hangul, or latin letter
 * (spec §2-4) -- picked by trailing-syllable batchim, not by hardcoding
 * "이(가)" for every factor name. */
function particle(word: string, withBatchim: string, without: string): string {
  const last = word[word.length - 1] ?? "";
  let hasBatchim: boolean;
  if (last >= "0" && last <= "9") {
    hasBatchim = "0136780".includes(last); // 영/일/삼/육/칠/팔 -> batchim
  } else if (last >= "가" && last <= "힣") {
    hasBatchim = (last.charCodeAt(0) - 0xac00) % 28 !== 0;
  } else {
    hasBatchim = "lmnr".includes(last.toLowerCase());
  }
  return hasBatchim ? withBatchim : without;
}

type TargetKey = (typeof TARGETS)[number];
type KnownEntry = { target: TargetKey; rho: number; shape: RelationShape };
type DirectionKind = "up" | "down" | "mixed" | "u";

/** U-shape takes priority over sign (spec §2-3): a factor that hurts both
 * tails reads as monotonic-up/down only by coincidence of which half of the
 * range has more points, so the shape check is applied before the sign
 * check rather than after. */
function classifyDirection(entries: KnownEntry[]): DirectionKind {
  if (entries.some((e) => e.shape === "u_shape")) return "u";
  const signs = entries.map((e) => Math.sign(e.rho));
  if (signs.every((s) => s >= 0)) return "up";
  if (signs.every((s) => s <= 0)) return "down";
  return "mixed";
}

const DIRECTION_PHRASE: Record<DirectionKind, string> = {
  up: "값이 커질수록 불량률이 함께 올라갑니다",
  down: "값이 커질수록 불량률이 함께 내려갑니다",
  mixed: "값에 따라 불량률이 함께 변합니다",
  u: "값이 양쪽 끝으로 갈수록 불량률이 함께 올라갑니다",
};

const CHANGE_VERB: Record<DirectionKind, string> = {
  up: "올라가지만",
  down: "내려가지만",
  mixed: "변하지만",
  u: "휘어지지만",
};

/** Plain-language read of the 5 mini-charts: which target(s) the factor
 * actually moves the needle on, in wording that doesn't require knowing
 * what rho/eps2/p-value mean (spec §5-2). Branches on how many targets
 * clear STRONG_RHO -- a single "strongest wins" reduce (the pre-existing
 * approach) reads as selective even when a factor moves every target
 * together, which is a real pattern in its own right (spec §2-1).
 */
function buildInterpretation(feature: string, entries: KnownEntry[]): string[] | null {
  if (entries.length === 0) return null;

  const featureParticle = particle(feature, "이", "가");
  const strong = entries.filter((e) => Math.abs(e.rho) >= STRONG_RHO);
  const sortedAbs = entries.map((e) => Math.abs(e.rho)).sort((a, b) => b - a);
  const top = entries.reduce((best, e) => (Math.abs(e.rho) > Math.abs(best.rho) ? e : best));
  const second = sortedAbs[1] ?? 0;

  if (strong.length === 0) {
    return [
      "다섯 유형 모두에서 곡선이 거의 평평합니다.",
      "이 인자만으로는 특정 불량 유형을 설명하기 어렵다는 뜻입니다.",
    ];
  }

  // 인과 표현 금지 (spec 문구 전수 검토 §A-7, prompts/report_system.md
  // "절대 규칙 2"와 동일 기준) -- "작용함을 뜻합니다"/"영향을 주면서"는 관측된
  // 상관관계를 인과로 읽히게 하므로, "함께 나타남" 식 서술로 통일한다.
  if (strong.length === 1) {
    const main = strong[0].target;
    const others = TARGETS.filter((t) => t !== main);
    return [
      `${others.join(", ")}에서는 값이 변해도 불량률이 거의 일정합니다.`,
      `${main}에서만 곡선이 뚜렷하게 휘어, ${feature}${featureParticle} ${main} 불량과만 뚜렷하게 함께 나타남을 뜻합니다.`,
    ];
  }

  if (Math.abs(top.rho) >= second * DOMINANCE_RATIO) {
    const others = strong.filter((e) => e.target !== top.target).map((e) => e.target);
    const direction = classifyDirection(strong);
    return [
      `${others.join(", ")}에서 불량률이 함께 ${CHANGE_VERB[direction]}, ${top.target}에서 가장 뚜렷합니다.`,
      `${feature}${featureParticle} 여러 불량 유형과 함께 나타나며 ${top.target}에서 특히 뚜렷하게 관측됨을 뜻합니다.`,
    ];
  }

  const direction = classifyDirection(strong);
  return [
    `${TARGETS.join(", ")} 모두에서 ${DIRECTION_PHRASE[direction]}.`,
    `${feature}${featureParticle} 특정 불량 유형이 아니라 전반에 걸쳐 함께 나타남을 뜻합니다.`,
  ];
}

type PointHover = { screenX: number; screenY: number; x: number; y: number } | null;

export default function CompareAcrossTargetsModal({
  feature,
  originTarget,
  datasetId,
  onClose,
  onSelectTarget,
}: {
  feature: string;
  originTarget: string;
  datasetId: string;
  onClose: () => void;
  onSelectTarget: (target: string) => void;
}) {
  const theme = useResolvedTheme();
  const [dataByTarget, setDataByTarget] = useState<Record<string, ScreeningScatterResponse | null>>({});
  const [loading, setLoading] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      void Promise.all(
        TARGETS.map((t) =>
          getScreeningScatter(datasetId, t, feature)
            .then((d) => [t, d] as const)
            .catch(() => [t, null] as const),
        ),
      ).then((results) => {
        if (cancelled) return;
        setDataByTarget(Object.fromEntries(results));
        setLoading(false);
      });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // Mounts once per (feature, origin target) open -- this reuses no
    // cached client state by design (see design note in the parent: the
    // heavy stats -- eps2/p-value/control limits -- were already computed
    // during "원인 분석 실행"; this call is the cheap per-target point
    // fetch, not a re-run of the screening pipeline).
  }, [datasetId, feature]);

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  useEffect(() => {
    if (loading || !scrollRef.current) return;
    const index = TARGETS.indexOf(originTarget as (typeof TARGETS)[number]);
    if (index >= 0) scrollRef.current.scrollLeft = index * (CHART_W + CHART_GAP);
  }, [loading, originTarget]);

  const xDomain = useMemo<[number, number]>(() => {
    let min = Infinity;
    let max = -Infinity;
    for (const t of TARGETS) {
      const d = dataByTarget[t];
      if (!d) continue;
      for (const p of d.points) {
        if (p.x < min) min = p.x;
        if (p.x > max) max = p.x;
      }
    }
    if (!Number.isFinite(min)) return [0, 1];
    const pad = (max - min) * 0.04 || 1;
    return [min - pad, max + pad];
  }, [dataByTarget]);

  const originData = dataByTarget[originTarget];
  const lcl = originData?.reference_lines.find((l) => l.key === "iqr_lo");
  const ucl = originData?.reference_lines.find((l) => l.key === "iqr_hi");
  const optimalCenter = originData?.optimal_center ?? null;

  const interpretation = useMemo(() => {
    if (loading) return null;
    const known = TARGETS.map((t) => {
      const d = dataByTarget[t];
      return d?.spearman_r != null ? { target: t, rho: d.spearman_r, shape: d.relation_shape } : null;
    }).filter((entry): entry is KnownEntry => entry != null);
    return buildInterpretation(feature, known);
  }, [loading, dataByTarget, feature]);

  return (
    <div className="compareModalBackdrop" onClick={onClose} role="presentation">
      <div className="compareModal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-label={`${feature} 불량 유형별 영향 비교`}>
        <div className="compareModalHeader">
          <div>
            <h2>{feature} vs Y1 ~ Y5 — 불량 유형별 영향 비교</h2>
            {originData && (
              <p className="compareModalMeta">
                n={originData.n.toLocaleString()} 계측
                {lcl?.drawable && ucl?.drawable && ` · 관리한계 ${formatNum(lcl.value)} ~ ${formatNum(ucl.value)}`}
              </p>
            )}
            {interpretation && (
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
        ) : (
          <div className="compareModalScroll" ref={scrollRef}>
            {TARGETS.map((t) => (
              <MiniChart
                key={t}
                target={t}
                data={dataByTarget[t]}
                xDomain={xDomain}
                isOrigin={t === originTarget}
                theme={theme}
                onSelectTarget={() => {
                  onSelectTarget(t);
                  onClose();
                }}
              />
            ))}
          </div>
        )}

        <div className="compareModalFooter">
          <div className="compareModalLegend">
            {lcl && ucl && (
              <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? NAVY.dark : NAVY.light }} /> 관리한계 LCL/UCL ({formatNum(lcl.value)} / {formatNum(ucl.value)})</span>
            )}
            {optimalCenter != null && <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? GREEN.dark : GREEN.light }} /> 최적 중심 ({formatNum(optimalCenter)})</span>}
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? RED.dark : RED.light }} /> 구간 평균 불량률</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MiniChart({
  target,
  data,
  xDomain,
  isOrigin,
  theme,
  onSelectTarget,
}: {
  target: string;
  data: ScreeningScatterResponse | null;
  xDomain: [number, number];
  isOrigin: boolean;
  theme: "light" | "dark";
  onSelectTarget: () => void;
}) {
  const [hover, setHover] = useState<PointHover>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const plotWidth = CHART_W - MARGIN.left - MARGIN.right;
  const plotHeight = CHART_H - MARGIN.top - MARGIN.bottom;

  const rho = data?.spearman_r ?? null;
  const isStrong = rho != null && Math.abs(rho) >= STRONG_RHO;

  const yDomain = useMemo<[number, number]>(() => {
    if (!data || data.points.length === 0) return [0, 1];
    const values = data.points.map((p) => p.y);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min) * 0.1 || 1;
    return [min - pad, max + pad];
  }, [data]);

  const xScale = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * plotWidth;
  const yScale = (v: number) => plotHeight - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * plotHeight;

  const xTicks = useMemo(
    () => niceTicksFitted(xDomain, MINI_X_TICK_COUNT[0], MINI_X_TICK_COUNT[1], plotWidth, formatNum, (label) => measureTextWidth(label, MINI_TICK_FONT)),
    [xDomain, plotWidth],
  );
  const yTicks = useMemo(() => niceTicks(yDomain, MINI_Y_TICK_COUNT), [yDomain]);

  const iqrLo = data?.reference_lines.find((l) => l.key === "iqr_lo");
  const iqrHi = data?.reference_lines.find((l) => l.key === "iqr_hi");
  const optimalCenter = data?.optimal_center ?? null;

  function handleMouseMove(event: React.MouseEvent<SVGRectElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !data) return;
    const relativeX = event.clientX - rect.left - MARGIN.left;
    const relativeY = event.clientY - rect.top - MARGIN.top;
    let best: { x: number; y: number } | null = null;
    let bestDistance = POINT_HOVER_RADIUS;
    for (const p of data.points) {
      const dx = xScale(p.x) - relativeX;
      const dy = yScale(p.y) - relativeY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = { x: p.x, y: p.y };
      }
    }
    if (best) setHover({ screenX: xScale(best.x), screenY: yScale(best.y), x: best.x, y: best.y });
    else setHover(null);
  }

  const pointOpacity = isStrong ? 0.55 : 0.18;
  const pointColor = isStrong ? "#1D4ED8" : "#93C5FD";
  const trendOpacity = isStrong ? 1 : 0.45;
  const lineOpacity = isStrong ? 1 : 0.5;
  const titleColor = isStrong ? (theme === "dark" ? "#7BA3E8" : "#0E306D") : "#9CA3AF";
  const navyColor = theme === "dark" ? NAVY.dark : NAVY.light;
  const greenColor = theme === "dark" ? GREEN.dark : GREEN.light;
  const redColor = theme === "dark" ? RED.dark : RED.light;

  return (
    <div className={`compareMiniChart ${isOrigin ? "origin" : ""}`} style={{ width: CHART_W }}>
      <div className="compareMiniChartHeader">
        <span className="compareMiniChartTitle" style={{ color: titleColor }}>
          {target}{isOrigin && " ★"} {rho != null && <span className="compareMiniChartRho">ρ={rho >= 0 ? "+" : ""}{rho.toFixed(2)}</span>}
        </span>
        {data && <ConfidenceBadge tier={data.confidence_tier} />}
      </div>

      {!data ? (
        <p className="emptyMessage">데이터 없음</p>
      ) : (
        <svg ref={svgRef} width={CHART_W} height={CHART_H} className="compareMiniChartSvg">
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {yTicks.map((tick) => (
              <g key={`y-${tick}`}>
                <line x1={0} x2={plotWidth} y1={yScale(tick)} y2={yScale(tick)} className="compareMiniGridLine" />
                <text x={-4} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" className="compareMiniTick">{formatNum(tick)}</text>
              </g>
            ))}
            {xTicks.map((tick) => (
              <text key={`x-${tick}`} x={xScale(tick)} y={plotHeight + 14} textAnchor="middle" className="compareMiniTick">{formatNum(tick)}</text>
            ))}

            {iqrLo?.drawable && (
              <rect x={0} y={0} width={Math.max(xScale(iqrLo.value), 0)} height={plotHeight} className="compareMiniOutsideShade" />
            )}
            {iqrHi?.drawable && (
              <rect x={xScale(iqrHi.value)} y={0} width={Math.max(plotWidth - xScale(iqrHi.value), 0)} height={plotHeight} className="compareMiniOutsideShade" />
            )}

            {/* points painted before every reference line/curve so lines
                stay visible on top (spec §4-1, applies to this mini chart too) */}
            {data.points.map((p, i) => (
              <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={2.4} fill={pointColor} opacity={pointOpacity} />
            ))}

            {optimalCenter != null && (
              <line x1={xScale(optimalCenter)} x2={xScale(optimalCenter)} y1={0} y2={plotHeight} stroke={greenColor} strokeWidth={1.3} strokeDasharray="4 3" opacity={lineOpacity} />
            )}
            {iqrLo?.drawable && (
              <line x1={xScale(iqrLo.value)} x2={xScale(iqrLo.value)} y1={0} y2={plotHeight} stroke={navyColor} strokeWidth={1.6} strokeDasharray="7 4" opacity={lineOpacity} />
            )}
            {iqrHi?.drawable && (
              <line x1={xScale(iqrHi.value)} x2={xScale(iqrHi.value)} y1={0} y2={plotHeight} stroke={navyColor} strokeWidth={1.6} strokeDasharray="7 4" opacity={lineOpacity} />
            )}

            {data.bins.length > 0 && (
              <path
                d={data.bins.map((b, i) => `${i === 0 ? "M" : "L"}${xScale(b.x_mean)},${yScale(b.y_mean)}`).join(" ")}
                fill="none"
                stroke={redColor}
                strokeWidth={2}
                opacity={trendOpacity}
              />
            )}

            {hover && (
              <>
                <line x1={hover.screenX} x2={hover.screenX} y1={0} y2={plotHeight} className="compareMiniGuideLine" />
                <circle cx={hover.screenX} cy={hover.screenY} r={3.5} fill={theme === "dark" ? "#2C2C2E" : "#fff"} stroke="#1D4ED8" strokeWidth={1.5} />
              </>
            )}

            {/* margin-click switches segment; point-radius excluded so it doesn't fight point hover */}
            <rect
              x={0} y={0} width={plotWidth} height={plotHeight} fill="transparent" style={{ cursor: "pointer" }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHover(null)}
              onClick={(event) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (!rect) return;
                const relativeX = event.clientX - rect.left - MARGIN.left;
                const relativeY = event.clientY - rect.top - MARGIN.top;
                const nearPoint = data.points.some((p) => Math.hypot(xScale(p.x) - relativeX, yScale(p.y) - relativeY) < POINT_HOVER_RADIUS);
                if (!nearPoint) onSelectTarget();
              }}
            />
          </g>
        </svg>
      )}

      {hover && (
        <div className="heatmapTooltip compareMiniTooltip">
          <div className="heatmapTooltipRow"><span>x</span><b>{hover.x.toFixed(1)}</b></div>
          <div className="heatmapTooltipRow"><span>y</span><b>{hover.y.toFixed(1)}</b></div>
        </div>
      )}
    </div>
  );
}
