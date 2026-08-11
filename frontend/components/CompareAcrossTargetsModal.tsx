"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import { getScreeningScatter } from "@/lib/api";
import { evaluateCurve, fitDefectRateCurve, type CurveFitResult } from "@/lib/defectRateCurve";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { RelationShape, ScreeningScatterResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
const CHART_W = 320;
const CHART_H = 260;
const CHART_GAP = 24;
const MARGIN = { top: 26, right: 10, bottom: 26, left: 36 };
// 부호 있는 순위상관(|rho| >= 0.15)으로 "뚜렷한 관계"를 가르던 기준을
// 설명력 기준으로 옮겼다 -- 이 인자들 상당수가 U자라 전체구간 rho는
// 표본 절반에 대해 반대 부호로 읽힌다. 0.02는 confidence_tier가
// "관계 없음(참고)"을 가르는 하한(grade_thresholds.yaml의
// min_eps2_reference)과 같은 값이라, 이 문단과 등급 배지가 어긋나지
// 않는다.
const STRONG_ADJ_R2 = 0.02;
const POINT_HOVER_RADIUS = 16;

// 그룹별 최적 중심은 무채색(--text)이다. 구간 평균 곡선은 이 차트의 핵심
// 판독 대상이라 신호색(--sig-red)을 쓴다 -- 발산(초록/빨강) 팔레트는
// 쓰지 않는다.
const TEXT_COLOR = { light: "#141A22", dark: "#F5F5F7" };
const INFERRED_COLOR = { light: "#C0392B", dark: "#EE6B76" };
// 산점도 "기본" 모드와 동일한 --measured 단일 점 색.
const POINT_COLOR = { light: "#0E306D", dark: "#7BA3E8" };

function formatNum(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
}

/** Samples the fitted curve across its observed x-domain into an SVG
 * polyline path -- shared math (lib/defectRateCurve), per-panel pixel
 * scale (this file). Each mini chart already fetches its own target's
 * points independently, so this is naturally "its own" fit already. */
function buildCurvePathD(fit: CurveFitResult, xScale: (v: number) => number, yScale: (v: number) => number): string {
  const [lo, hi] = fit.domain;
  if (!(hi > lo)) return "";
  const STEPS = 48;
  const parts: string[] = [];
  for (let i = 0; i <= STEPS; i += 1) {
    const x = lo + ((hi - lo) * i) / STEPS;
    const y = evaluateCurve(fit, x);
    parts.push(`${i === 0 ? "M" : "L"}${xScale(x)},${yScale(y)}`);
  }
  return parts.join(" ");
}

// A target that "dominates" needs its Adjusted R2 to clear the runner-up
// by this ratio, otherwise the spread reads as broad-and-even rather than
// one-target-leads.
const DOMINANCE_RATIO = 1.5;

/** Korean 이/가 particle for a word ending in digit, hangul, or latin letter
 * -- picked by trailing-syllable batchim, not by hardcoding
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
type KnownEntry = { target: TargetKey; adjR2: number; shape: RelationShape };
type DirectionKind = "up" | "down" | "mixed" | "u";

/** U-shape takes priority: a factor that hurts both tails
 * reads as monotonic-up/down only by coincidence of which half of the
 * range has more points, so the U check runs first. Direction now comes
 * from the classified relation shape rather than a correlation sign --
 * the sign was removed everywhere precisely because it misreads U-shaped
 * factors. */
function classifyDirection(entries: KnownEntry[]): DirectionKind {
  if (entries.some((e) => e.shape === "u_shape")) return "u";
  const shapes = entries.map((e) => e.shape);
  if (shapes.every((s) => s === "monotonic_increasing")) return "up";
  if (shapes.every((s) => s === "monotonic_decreasing")) return "down";
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
 * what Adjusted R2/p-value mean. Branches on how many targets
 * clear STRONG_ADJ_R2 -- a single "strongest wins" reduce (the pre-existing
 * approach) reads as selective even when a factor moves every target
 * together, which is a real pattern in its own right.
 */
function buildInterpretation(feature: string, entries: KnownEntry[]): string[] | null {
  if (entries.length === 0) return null;

  const featureParticle = particle(feature, "이", "가");
  const strong = entries.filter((e) => e.adjR2 >= STRONG_ADJ_R2);
  const sortedDesc = entries.map((e) => e.adjR2).sort((a, b) => b - a);
  const top = entries.reduce((best, e) => (e.adjR2 > best.adjR2 ? e : best));
  const second = sortedDesc[1] ?? 0;

  if (strong.length === 0) {
    return [
      "다섯 유형 모두에서 곡선이 거의 평평합니다.",
      "이 인자만으로는 특정 불량 유형을 설명하기 어렵다는 뜻입니다.",
    ];
  }

  // 인과 표현 금지 (prompts/report_system.md "절대 규칙 2"와 동일
  // 기준) -- "작용함을 뜻합니다"/"영향을 주면서"는 관측된
  // 상관관계를 인과로 읽히게 하므로, "함께 나타남" 식 서술로 통일한다.
  if (strong.length === 1) {
    const main = strong[0].target;
    const others = TARGETS.filter((t) => t !== main);
    return [
      `${others.join(", ")}에서는 값이 변해도 불량률이 거의 일정합니다.`,
      `${main}에서만 곡선이 뚜렷하게 휘어, ${feature}${featureParticle} ${main} 불량과만 뚜렷하게 함께 나타남을 뜻합니다.`,
    ];
  }

  if (top.adjR2 >= second * DOMINANCE_RATIO) {
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
  // 패널 폭 320px -> 200px (≤767px). 가로 스크롤은
  // 모든 폭에서 유지한다 -- 패널을 세로로 쌓지 않는다(하지 말 것 목록).
  const isMobileLayout = useIsMobileLayout();
  const chartW = isMobileLayout ? 200 : CHART_W;

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
    // heavy stats -- Adjusted R2/p-value/control limits -- were already computed
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
    if (index >= 0) scrollRef.current.scrollLeft = index * (chartW + CHART_GAP);
  }, [loading, originTarget, chartW]);

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
  const optimalCenter = originData?.optimal_center ?? null;

  const interpretation = useMemo(() => {
    if (loading) return null;
    const known = TARGETS.map((t) => {
      const d = dataByTarget[t];
      return d != null ? { target: t, adjR2: d.adj_r2, shape: d.relation_shape } : null;
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
                width={chartW}
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
            {optimalCenter != null && <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? TEXT_COLOR.dark : TEXT_COLOR.light }} /> 최적 중심 ({formatNum(optimalCenter)})</span>}
            <span><i className="compareLegendSwatch" style={{ background: theme === "dark" ? INFERRED_COLOR.dark : INFERRED_COLOR.light }} /> 불량률 곡선</span>
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
  width,
  onSelectTarget,
}: {
  target: string;
  data: ScreeningScatterResponse | null;
  xDomain: [number, number];
  isOrigin: boolean;
  theme: "light" | "dark";
  // 320px(데스크톱) / 200px(≤767px) -- 호출부가
  // CHART_W 상수 대신 이 값을 넘긴다.
  width: number;
  onSelectTarget: () => void;
}) {
  const [hover, setHover] = useState<PointHover>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = CHART_H - MARGIN.top - MARGIN.bottom;

  const adjR2 = data?.adj_r2 ?? null;
  const degree = data?.degree ?? null;
  const isStrong = adjR2 != null && adjR2 >= STRONG_ADJ_R2;

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

  // 미니 패널은 양끝 2개 눈금만 -- 형태를 보는 화면이지 값을
  // 읽는 화면이 아니다.
  const xTicks = useMemo(() => [xDomain[0], xDomain[1]], [xDomain]);
  const yTicks = useMemo(() => [yDomain[0], yDomain[1]], [yDomain]);

  const optimalCenter = data?.optimal_center ?? null;
  // 이 패널의 데이터(해당 타깃만)로 자체 재적합한다 -- 5개 패널이
  // 서로 다른 타깃의 점을 쓰므로 이미 "그룹별 재적합"에 해당한다.
  const curveFit = useMemo(() => (data ? fitDefectRateCurve(data.points) : null), [data]);

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
  const pointColor = theme === "dark" ? POINT_COLOR.dark : POINT_COLOR.light;
  const trendOpacity = isStrong ? 1 : 0.45;
  const lineOpacity = isStrong ? 1 : 0.5;
  const titleColor = isStrong ? (theme === "dark" ? "#7BA3E8" : "#0E306D") : "#9CA3AF";
  const textColor = theme === "dark" ? TEXT_COLOR.dark : TEXT_COLOR.light;
  const inferredColor = theme === "dark" ? INFERRED_COLOR.dark : INFERRED_COLOR.light;

  return (
    <div className={`compareMiniChart ${isOrigin ? "origin" : ""}`} style={{ width }}>
      <div className="compareMiniChartHeader">
        <span className="compareMiniChartTitle" style={{ color: titleColor }}>
          {target}{isOrigin && " ★"}{" "}
          {adjR2 != null && (
            <span className="compareMiniChartStat">
              R²={adjR2.toFixed(3)}{degree != null && ` (${degree}차)`}
            </span>
          )}
        </span>
        {data && <ConfidenceBadge tier={data.confidence_tier} />}
      </div>

      {!data ? (
        <p className="emptyMessage">데이터 없음</p>
      ) : (
        <svg ref={svgRef} width={width} height={CHART_H} className="compareMiniChartSvg">
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

            {/* points painted before every reference line/curve so lines
                stay visible on top (applies to this mini chart too).
                패널이 작으므로 반지름을 산점도보다 더 줄인다. */}
            {data.points.map((p, i) => (
              <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={1.7} fill={pointColor} opacity={pointOpacity} />
            ))}

            {optimalCenter != null && (
              <line x1={xScale(optimalCenter)} x2={xScale(optimalCenter)} y1={0} y2={plotHeight} stroke={textColor} strokeWidth={1.3} strokeDasharray="4 3" opacity={lineOpacity} />
            )}

            {/* 이 타깃 데이터로 재적합한 불량률 곡선 -- 폴백이면
                기존 구간 평균 꺾은선을 그대로 그린다(계산 불변). */}
            {curveFit && curveFit.fallbackReason == null && (
              <path d={buildCurvePathD(curveFit, xScale, yScale)} fill="none" stroke={inferredColor} strokeWidth={2} opacity={trendOpacity}>
                <title>{`R²=${curveFit.r2.toFixed(2)} (${curveFit.degree === 2 ? "2차" : "1차"})`}</title>
              </path>
            )}
            {(!curveFit || curveFit.fallbackReason != null) && data.bins.length > 0 && (
              <path
                d={data.bins.map((b, i) => `${i === 0 ? "M" : "L"}${xScale(b.x_mean)},${yScale(b.y_mean)}`).join(" ")}
                fill="none"
                stroke={inferredColor}
                strokeWidth={2}
                opacity={trendOpacity}
              >
                {curveFit?.fallbackReason && <title>{curveFit.fallbackReason}</title>}
              </path>
            )}

            {hover && (
              <>
                <line x1={hover.screenX} x2={hover.screenX} y1={0} y2={plotHeight} className="compareMiniGuideLine" />
                <circle cx={hover.screenX} cy={hover.screenY} r={3.5} fill={theme === "dark" ? "#2C2C2E" : "#fff"} stroke={pointColor} strokeWidth={1.5} />
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
