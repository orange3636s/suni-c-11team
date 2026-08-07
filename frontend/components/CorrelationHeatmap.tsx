"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getScreeningHeatmap } from "@/lib/api";
import { TIER_LABEL } from "@/lib/confidenceTier";
import { formatEps2, formatQValue } from "@/lib/numberFormat";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfidenceTier, ConfigHeatmapLevel, HeatmapKind, HeatmapMetric, HeatmapResponse } from "@/types/data";

const DEFAULT_ROW_LIMIT = 20;
const METRIC_LABEL: Record<HeatmapMetric, string> = { spearman: "상관계수 (ρ)", eps2: "설명력 (ε²)" };
const VIEW_KIND_LABEL: Record<HeatmapKind, string> = { numeric: "수치형", categorical: "범주형" };
const CONFIG_LEVEL_LABEL: Record<ConfigHeatmapLevel, string> = { model: "Model", eq: "EQ", chamber: "Chamber" };
// Config는 순서 없는 범주형이라 상관계수(ρ)가 정의되지 않는다 (spec E) --
// 범주형 보기에서는 기준 토글을 이 지표로 고정·비활성화한다.
const CATEGORICAL_METRIC: HeatmapMetric = "eps2";
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
  const [kind, setKind] = useState<HeatmapKind>("numeric");
  const [configLevel, setConfigLevel] = useState<ConfigHeatmapLevel>("eq");
  // 범주형 보기에서는 지표가 늘 ε²다 -- 토글에 쓸 "현재 표시 중인 지표"는
  // 이 값을 쓰고, `metric` state 자체는 수치형으로 돌아왔을 때 사용자가
  // 마지막으로 고른 값(ρ/ε²)을 그대로 기억하도록 건드리지 않는다.
  const effectiveMetric: HeatmapMetric = kind === "categorical" ? CATEGORICAL_METRIC : metric;
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
    // kind/configLevel까지 캐시 키에 넣는다 -- 안 그러면 수치형 캐시가
    // 범주형 조회에 잘못 재사용될 수 있다 (지시서 E의 백엔드 캐시 키
    // 주의사항과 같은 이유).
    const cacheKey = kind === "categorical" ? `categorical:${configLevel}` : `numeric:${metric}`;
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
          const response = await getScreeningHeatmap(datasetId, metric, kind, kind === "categorical" ? configLevel : undefined);
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
  }, [datasetId, metric, kind, configLevel, enabled]);

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
  // 범주형 헤더 요약 "검정 N건 · FDR 통과 M건" (지시서 E) -- 그리드에
  // 이미 실려 있는 값에서 파생만 한다, 별도 필드를 새로 만들지 않는다.
  const testedCount = data.values.reduce((sum, row) => sum + row.filter((v) => v != null).length, 0);
  const passedCount = data.significant.reduce((sum, row) => sum + row.filter(Boolean).length, 0);

  return (
    <section className="resultCard heatmapCard">
      <div className="heatmapHeaderRow">
        <div className="heatmapHeaderRowText">
          <span className="sectionLabel">CORRELATION OVERVIEW</span>
          <h2>전체 상관관계 히트맵</h2>
          {kind === "categorical" && (
            <p className="heatmapIntro">검정 {testedCount}건 · FDR 통과 {passedCount}건</p>
          )}
        </div>
        <div className="heatmapToggleStack">
          {/* 기준(ρ/ε²) — 범주형 보기에서는 ε²로 고정·비활성화한다 (spec E:
              "범주형 인자는 상관계수를 정의할 수 없어 설명력만 사용합니다"). */}
          <div className="scatterViewToggleRow">
            <span className="scatterViewToggleLabel">기준</span>
            <div
              className="scatterViewToggle"
              role="group"
              aria-label="지표 선택"
              title={kind === "categorical" ? "범주형 인자는 상관계수를 정의할 수 없어 설명력만 사용합니다." : undefined}
            >
              {(Object.keys(METRIC_LABEL) as HeatmapMetric[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`scatterViewToggleBtn ${effectiveMetric === key ? "active" : ""}`}
                  disabled={kind === "categorical"}
                  onClick={() => setMetric(key)}
                >
                  {METRIC_LABEL[key]}
                </button>
              ))}
            </div>
          </div>
          <div className="scatterViewToggleRow">
            <span className="scatterViewToggleLabel">보기</span>
            <div className="scatterViewToggle" role="group" aria-label="수치형/범주형 보기">
              {(Object.keys(VIEW_KIND_LABEL) as HeatmapKind[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`scatterViewToggleBtn ${kind === key ? "active" : ""}`}
                  onClick={() => setKind(key)}
                >
                  {VIEW_KIND_LABEL[key]}
                </button>
              ))}
            </div>
          </div>
          {kind === "categorical" && (
            <div className="scatterViewToggleRow">
              <span className="scatterViewToggleLabel">계층</span>
              <div className="scatterViewToggle" role="group" aria-label="Config 계층 선택">
                {(Object.keys(CONFIG_LEVEL_LABEL) as ConfigHeatmapLevel[]).map((level) => (
                  <button
                    key={level}
                    type="button"
                    className={`scatterViewToggleBtn ${configLevel === level ? "active" : ""}`}
                    onClick={() => setConfigLevel(level)}
                  >
                    {CONFIG_LEVEL_LABEL[level]}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {kind === "categorical" && passedCount === 0 && (
        <p className="heatmapEmptyBanner">
          이 데이터에서는 FDR 보정 후 유의한 Config 인자가 없습니다 ({testedCount}건 검정, 통과 0건). 장비 효과가 현재 해상도에서 검출 한계 이하입니다.
        </p>
      )}

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
            <option value="max_rho">{sortOptionLabels(effectiveMetric).max}</option>
            <option value="min_rho">{sortOptionLabels(effectiveMetric).min}</option>
            <option value="target">{sortOptionLabels(effectiveMetric).target}</option>
            <option value="step">{sortOptionLabels(effectiveMetric).step}</option>
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
                metric={effectiveMetric}
                kind={kind}
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
        {kind === "numeric" ? (
          <>
            {data.excluded_configs > 0 && `Eq. ${data.excluded_configs}개 제외. `}
            표본이 30개 미만인 셀은 사선 패턴으로 표시됩니다.
          </>
        ) : (
          <>
            색은 FDR 통과(q&lt;0.05) 셀에만 칠합니다 — 나머지는 회색 고정(고정 스케일 ε² {scaleMin.toFixed(2)}~{scaleMax.toFixed(2)}).
            <br />
            R/D {data.excluded_configs}개 제외.
          </>
        )}
      </p>

      {tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.feature} × {tooltip.target}</strong>
          <div className="heatmapTooltipRow"><span>{effectiveMetric === "spearman" ? "ρ" : "ε²"}</span><b>{tooltip.value != null ? tooltip.value.toFixed(3) : "표본 부족"}</b></div>
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
  kind,
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
  kind: HeatmapKind;
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
        // 범주형 보기 전용: 색은 FDR 게이트를 통과한 셀에만 칠한다 (지시서
        // E "색 스케일 규칙" -- 자동 정규화 금지와 짝을 이루는 규칙. q가
        // 없거나(=미검정) 0.05 이상이면 값은 있어도 회색 고정).
        const gated = kind === "categorical" && !masked && !significant;
        const style: React.CSSProperties = {};
        if (!masked && !gated) {
          const { bg, light } = cellBackground(value, scaleMin, scaleMax, theme);
          style.background = bg;
          style.color = light ? "var(--heatmap-text-inverse)" : "var(--heatmap-text)";
        }
        return (
          <button
            key={target}
            type="button"
            className={`heatmapCell ${significant ? "significant" : ""} ${masked ? "masked" : ""} ${gated ? "gated" : ""}`}
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
            aria-label={`${feature}, ${target}, ${metric === "spearman" ? "rho" : "eps2"} ${value != null ? formatEps2(value) : "표본 부족"}`}
          >
            {masked ? "" : formatEps2(value)}
          </button>
        );
      })}
    </>
  );
}
