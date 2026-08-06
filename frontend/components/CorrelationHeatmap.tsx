"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getScreeningHeatmap } from "@/lib/api";
import { formatQValue } from "@/lib/numberFormat";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfidenceTier, HeatmapMetric, HeatmapResponse } from "@/types/data";

const DEFAULT_ROW_LIMIT = 20;
const METRIC_LABEL: Record<HeatmapMetric, string> = { spearman: "상관계수 (ρ)", eps2: "설명력 (ε²)" };
type SortMode = "max_rho" | "min_rho" | "target" | "step";

/** ε² has no sign, so "절댓값"/"부호 포함" framing doesn't apply to it --
 * spearman shows the qualifier, eps2 shows the plain magnitude-only label
 * (spec §5-4-1/§5-4-2). */
function sortOptionLabels(metric: HeatmapMetric): { max: string; min: string; target: string; step: string } {
  if (metric === "eps2") {
    return { max: "최대 ε²", min: "최소 ε²", target: "특정 타깃 기준", step: "Step 순서" };
  }
  return {
    max: "최대 |ρ| (절댓값)",
    min: "최소 |ρ| (절댓값)",
    target: "특정 타깃 기준 (부호 포함)",
    step: "Step 순서",
  };
}

function featureStep(feature: string): number {
  const match = /^Step(\d+)_/.exec(feature);
  return match ? Number(match[1]) : 0;
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
}

function mixHex(from: string, to: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(from);
  const [r2, g2, b2] = hexToRgb(to);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function relativeLuminance(rgb: [number, number, number]): number {
  const [r, g, b] = rgb.map((c) => c / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

const THEME_COLORS = {
  light: { pos: "#B42318", neg: "#1849A9", center: "#FFFFFF" },
  dark: { pos: "#F97066", neg: "#84CAFF", center: "#2C2C2E" },
};

function cellBackground(value: number, min: number, max: number, theme: "light" | "dark"): { bg: string; light: boolean } {
  const { pos, neg, center } = THEME_COLORS[theme];
  if (min < 0) {
    const clipped = Math.max(min, Math.min(max, value));
    const denom = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const t = Math.min(1, Math.abs(clipped) / denom);
    const target = clipped >= 0 ? pos : neg;
    const bg = mixHex(center, target, t);
    return { bg, light: relativeLuminance(hexToRgb(target)) * t + relativeLuminance(hexToRgb(center)) * (1 - t) < 0.45 };
  }
  const clipped = Math.max(min, Math.min(max, value));
  const t = Math.min(1, clipped / (max || 1));
  const bg = mixHex(center, pos, t);
  return { bg, light: relativeLuminance(hexToRgb(pos)) * t + relativeLuminance(hexToRgb(center)) * (1 - t) < 0.45 };
}

export type HeatmapCellSelection = {
  target: string;
  feature: string;
  significant: boolean;
  qValue: number | null;
};

type TooltipState = {
  x: number;
  y: number;
  feature: string;
  target: string;
  value: number | null;
  n: number;
  q: number | null;
  significant: boolean;
  tier: ConfidenceTier | null;
};

const TIER_LABEL: Record<ConfidenceTier, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };

export default function CorrelationHeatmap({
  datasetId,
  enabled,
  onSelectCell,
}: {
  datasetId: string;
  enabled: boolean;
  onSelectCell: (selection: HeatmapCellSelection) => void;
}) {
  const theme = useResolvedTheme();
  const [metric, setMetric] = useState<HeatmapMetric>("spearman");
  const [data, setData] = useState<HeatmapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [significantOnly, setSignificantOnly] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("max_rho");
  const [sortTarget, setSortTarget] = useState<string>("");
  const [expanded, setExpanded] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // Brief settle animation on the rows whenever the sort/filter controls
  // below actually change the order (spec §5-4-3) -- triggered from
  // those controls' own onChange, not derived reactively from `rows`,
  // so it never fires a setState synchronously inside an effect.
  const [sorting, setSorting] = useState(false);
  const cache = useRef<Map<string, HeatmapResponse>>(new Map());

  function triggerRowSettle() {
    setSorting(true);
    window.setTimeout(() => setSorting(false), 220);
  }

  useEffect(() => {
    cache.current = new Map();
  }, [datasetId]);

  useEffect(() => {
    if (!enabled) return;
    const cacheKey = metric;
    const cached = cache.current.get(cacheKey);
    if (cached) {
      setData(cached);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        setLoading(true);
        setError("");
        try {
          const response = await getScreeningHeatmap(datasetId, metric);
          if (cancelled) return;
          cache.current.set(cacheKey, response);
          setData(response);
          if (!sortTarget && response.targets.length > 0) setSortTarget(response.targets[0]);
        } catch (failure) {
          if (!cancelled) setError(failure instanceof Error ? failure.message : "히트맵을 불러오지 못했습니다.");
        } finally {
          if (!cancelled) setLoading(false);
        }
      })();
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, metric, enabled]);

  const rows = useMemo(() => {
    if (!data) return [];
    let indices = data.features.map((_, index) => index);
    if (significantOnly) {
      indices = indices.filter((index) => data.significant[index].some(Boolean));
    }
    if (sortMode === "step") {
      indices = [...indices].sort((a, b) => featureStep(data.features[a]) - featureStep(data.features[b]));
    } else if (sortMode === "target" && sortTarget) {
      // Signed, not absolute -- "Y1 기준" means "가장 많이 올리는 인자가
      // 위로", so a strong negative correlation sorts to the bottom, not
      // the top (spec §5-4-2; this differs from max/min |ρ| on purpose).
      const colIndex = data.targets.indexOf(sortTarget);
      if (colIndex >= 0) {
        indices = [...indices].sort((a, b) => {
          const va = data.values[a][colIndex] ?? -Infinity;
          const vb = data.values[b][colIndex] ?? -Infinity;
          return vb - va;
        });
      }
    } else if (sortMode === "min_rho") {
      // The server's default order is already max|value| desc (see the
      // "max_rho" case below) -- reversing that exact order is exactly
      // ascending order by the same per-row max|value|, without
      // recomputing it client-side.
      indices = [...indices].reverse();
    }
    // sortMode "max_rho" keeps the server's default order (already max|rho| desc)
    return indices;
  }, [data, significantOnly, sortMode, sortTarget]);

  const visibleRows = expanded ? rows : rows.slice(0, DEFAULT_ROW_LIMIT);

  if (!enabled) {
    return (
      <section className="resultCard heatmapCard">
        <div className="heatmapHeaderRow">
          <div>
            <span className="sectionLabel">CORRELATION OVERVIEW</span>
            <h2>전체 인자 조망</h2>
          </div>
        </div>
        <p className="emptyMessage">원인 분석을 실행하면 상관관계 히트맵이 생성됩니다.</p>
      </section>
    );
  }

  if (loading && !data) {
    return (
      <section className="resultCard heatmapCard">
        <p className="emptyMessage">상관관계 히트맵을 계산하는 중…</p>
      </section>
    );
  }
  if (error) {
    return (
      <section className="resultCard heatmapCard">
        <p className="errorMessage">{error}</p>
      </section>
    );
  }
  if (!data) return null;

  const [scaleMin, scaleMax] = [data.scale.min, data.scale.max];
  const gridTemplateColumns = `160px repeat(${data.targets.length}, minmax(64px, 1fr))`;

  return (
    <section className="resultCard heatmapCard">
      <div className="heatmapHeaderRow">
        <div className="heatmapHeaderRowText">
          <span className="sectionLabel">CORRELATION OVERVIEW</span>
          <h2>전체 상관관계 히트맵</h2>
          <p className="heatmapIntro">산점도가 &ldquo;왜 이 인자인가&rdquo;를 보여준다면, 이 히트맵은 &ldquo;다른 인자들은 왜 아닌가&rdquo;를 보여줍니다.</p>
        </div>
        <div className="heatmapMetricToggle" role="tablist" aria-label="지표 선택">
          {(Object.keys(METRIC_LABEL) as HeatmapMetric[]).map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={metric === key}
              className={metric === key ? "active" : ""}
              onClick={() => setMetric(key)}
            >
              {METRIC_LABEL[key]}
            </button>
          ))}
        </div>
      </div>

      <div className="heatmapControls">
        <div className="fieldGroup">
          <span>정렬 기준</span>
          <select
            value={sortMode}
            onChange={(event) => {
              setSortMode(event.target.value as SortMode);
              triggerRowSettle();
            }}
          >
            <option value="max_rho">{sortOptionLabels(metric).max}</option>
            <option value="min_rho">{sortOptionLabels(metric).min}</option>
            <option value="target">{sortOptionLabels(metric).target}</option>
            <option value="step">{sortOptionLabels(metric).step}</option>
          </select>
        </div>
        {sortMode === "target" && (
          <div className="fieldGroup">
            <span>기준 타깃</span>
            <select
              value={sortTarget}
              onChange={(event) => {
                setSortTarget(event.target.value);
                triggerRowSettle();
              }}
            >
              {data.targets.map((target) => (
                <option key={target} value={target}>{target}</option>
              ))}
            </select>
          </div>
        )}
        <label>
          <input
            type="checkbox"
            checked={significantOnly}
            onChange={(event) => {
              setSignificantOnly(event.target.checked);
              triggerRowSettle();
            }}
          />
          유의 인자만 보기
        </label>
      </div>

      <div className={`heatmapScrollArea ${expanded ? "" : "collapsed"}`}>
        <div className={`heatmapGrid ${sorting ? "sorting" : ""}`} style={{ gridTemplateColumns }}>
          <div className="heatmapCornerCell heatmapColHeader" />
          {data.targets.map((target) => (
            <div key={target} className="heatmapColHeader">{target}</div>
          ))}
          {visibleRows.map((rowIndex) => {
            const feature = data.features[rowIndex];
            return (
              <FragmentRow
                key={feature}
                feature={feature}
                rowIndex={rowIndex}
                data={data}
                metric={metric}
                theme={theme}
                scaleMin={scaleMin}
                scaleMax={scaleMax}
                onHover={setTooltip}
                onSelectCell={onSelectCell}
              />
            );
          })}
        </div>
      </div>

      <div className="heatmapExpandRow">
        <button type="button" className="referenceOnlyToggle" onClick={() => setExpanded((current) => !current)}>
          {expanded ? "접기" : `전체 ${rows.length}행 보기`}
        </button>
      </div>

      <div className="heatmapColorbar">
        <span>{scaleMin.toFixed(2)}</span>
        <div
          className="heatmapColorbarTrack"
          style={{
            background:
              scaleMin < 0
                ? `linear-gradient(to right, ${THEME_COLORS[theme].neg}, ${THEME_COLORS[theme].center}, ${THEME_COLORS[theme].pos})`
                : `linear-gradient(to right, ${THEME_COLORS[theme].center}, ${THEME_COLORS[theme].pos})`,
          }}
        />
        <span>{scaleMax.toFixed(2)}</span>
      </div>

      <p className="heatmapCaption">
        {`Eq. ${data.excluded_configs}개는 범주형이므로 제외됨 — 원인분석 Pareto/산점도에서는 박스플롯으로 확인하세요. `}
        표본이 30개 미만인 셀은 사선 패턴으로 표시됩니다.
        <br />
        인자 선정은 ε² + BH-FDR 기준이며, 이 히트맵은 전체 조망용입니다.
      </p>

      {tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.feature} × {tooltip.target}</strong>
          <div className="heatmapTooltipRow"><span>{metric === "spearman" ? "ρ" : "ε²"}</span><b>{tooltip.value != null ? tooltip.value.toFixed(3) : "표본 부족"}</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltip.n.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>q</span><b>{formatQValue(tooltip.q)}</b></div>
          <div className="heatmapTooltipRow"><span>신뢰도</span><b>{tooltip.tier ? TIER_LABEL[tooltip.tier] : "-"}</b></div>
          <div className="heatmapTooltipRow"><span>FDR 통과</span><b>{tooltip.significant ? "예" : "아니오"}</b></div>
        </div>
      )}
    </section>
  );
}

function FragmentRow({
  feature,
  rowIndex,
  data,
  metric,
  theme,
  scaleMin,
  scaleMax,
  onHover,
  onSelectCell,
}: {
  feature: string;
  rowIndex: number;
  data: HeatmapResponse;
  metric: HeatmapMetric;
  theme: "light" | "dark";
  scaleMin: number;
  scaleMax: number;
  onHover: (tooltip: TooltipState | null) => void;
  onSelectCell: (selection: HeatmapCellSelection) => void;
}) {
  return (
    <>
      <div className="heatmapRowLabel">{feature}</div>
      {data.targets.map((target, colIndex) => {
        const value = data.values[rowIndex][colIndex];
        const n = data.n[rowIndex][colIndex];
        const q = data.q[rowIndex][colIndex];
        const significant = data.significant[rowIndex][colIndex];
        const tier = data.tier[rowIndex][colIndex];
        const masked = value == null;
        const style: React.CSSProperties = {};
        if (!masked) {
          const { bg, light } = cellBackground(value, scaleMin, scaleMax, theme);
          style.background = bg;
          style.color = light ? "var(--heatmap-text-inverse)" : "var(--heatmap-text)";
        }
        return (
          <button
            key={target}
            type="button"
            className={`heatmapCell ${significant ? "significant" : ""} ${masked ? "masked" : ""}`}
            style={style}
            onMouseEnter={(event) =>
              onHover({
                x: event.clientX,
                y: event.clientY,
                feature,
                target,
                value,
                n,
                q,
                significant,
                tier,
              })
            }
            onMouseMove={(event) =>
              onHover({
                x: event.clientX,
                y: event.clientY,
                feature,
                target,
                value,
                n,
                q,
                significant,
                tier,
              })
            }
            onMouseLeave={() => onHover(null)}
            onTouchStart={(event) =>
              onHover({
                x: event.touches[0]?.clientX ?? 0,
                y: event.touches[0]?.clientY ?? 0,
                feature,
                target,
                value,
                n,
                q,
                significant,
                tier,
              })
            }
            onClick={() => onSelectCell({ target, feature, significant, qValue: q })}
            aria-label={`${feature}, ${target}, ${metric === "spearman" ? "rho" : "eps2"} ${value != null ? value.toFixed(2) : "표본 부족"}`}
          >
            {masked ? "" : value!.toFixed(2)}
          </button>
        );
      })}
    </>
  );
}
