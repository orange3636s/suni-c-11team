"use client";

import { useMemo, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { DIVERGING_GREEN, DIVERGING_RED } from "@/lib/constants";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfigTreemapGroup, ConfigTreemapResponse } from "@/types/data";

// n이 이 미만인 타일은 "표본 부족"으로 회색 처리하고 색을 칠하지 않는다.
// 25는 실측으로 고른 값이다 -- test.CSV 기준으로 임계가 30이면 조합의
// 56.7%가 회색이 되어 트리맵이 거의 비어 보이고, 25면 43.2%로 절반
// 미만이다. src/analysis/screening/selector.py의
// DEFAULT_MIN_N_CATEGORICAL과 같은 값을 유지한다.
const MIN_TILE_N = 25;
// 고정 ±3%p 스케일 -- 관측 최대/최소로 자동 정규화하면 0.9%p 차이가
// 새빨갛게 렌더되어 없는 신호를 만든다.
const COLOR_SPAN_PP = 3;

type ParsedGroup = ConfigTreemapGroup & { model: string; eq: string; chamber: string };

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgbToHex([r, g, b]: [number, number, number]): string {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}
function lerpColor(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexToRgb(a);
  const [br, bg, bb] = hexToRgb(b);
  return rgbToHex([ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t]);
}

// 불량률은 낮을수록 좋으므로 초록(낮음)-빨강(높음), 중앙은 전체 평균.
// RED/GREEN 토큰과 같은 값이다 (ParetoChart.tsx 등).
const RED = DIVERGING_RED;
const GREEN = DIVERGING_GREEN;
const CENTER = { light: "#FFFFFF", dark: "#2C2C2E" };

function colorForMean(mean: number, overallMean: number, theme: "light" | "dark"): string {
  const lo = overallMean - COLOR_SPAN_PP;
  const hi = overallMean + COLOR_SPAN_PP;
  const clamped = Math.max(lo, Math.min(hi, mean));
  const t = (clamped - lo) / (hi - lo || 1); // 0..1, 0.5 = 전체 평균
  const red = theme === "dark" ? RED.dark : RED.light;
  const green = theme === "dark" ? GREEN.dark : GREEN.light;
  const center = theme === "dark" ? CENTER.dark : CENTER.light;
  return t < 0.5 ? lerpColor(green, center, t / 0.5) : lerpColor(center, red, (t - 0.5) / 0.5);
}

// 타일이 채색되면(FDR 통과 시에만) 배경이 빨강~중립~초록 그라디언트를
// 오간다 -- 고정 텍스트색으로는 중앙 근처에서 대비가 무너진다. 셀 배경의
// 실제 밝기를 보고 매 타일마다 어두운/밝은 글자를 고른다(YIQ luma).
function lumaOf(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}
function textColorForTile(backgroundHex: string): string {
  return lumaOf(backgroundHex) > 0.5 ? "var(--ink)" : "#fff";
}

type HoverState = { group: ParsedGroup; x: number; y: number } | null;

/** Config별 트리맵 탭의 트리맵 하나(타깃 하나 담당) --
 * 스텝·타깃 선택기는 페이지 상단에 하나만 두고 이 컴포넌트는 이미
 * 조회된 `data`를 그리기만 한다. `loading`이면 새로 그리지 않고 직전
 * 데이터를 불투명도만 낮춰 보여준다 -- 조회 중 트리맵을 지우면 화면이
 * 매 조회마다 비어 보인다. */
export default function ConfigTreemap({
  target,
  step,
  data,
  loading,
  headerRight,
}: {
  target: string;
  step: number;
  data: ConfigTreemapResponse | null;
  loading: boolean;
  // 스텝 선택 드롭다운은 다섯 트리맵이 함께 쓰는 컨트롤이라 카드
  // 자체는 모른다 -- 부모(config-treemap/page.tsx)가 첫 번째(Y1) 카드에만
  // 넘겨서 같은 컨트롤이 다섯 번 반복되지 않게 한다.
  headerRight?: ReactNode;
}) {
  const theme = useResolvedTheme();
  const router = useRouter();
  const isMobileLayout = useIsMobileLayout();
  const [hover, setHover] = useState<HoverState>(null);

  const models = useMemo(() => {
    if (!data) return [];
    const parsed: ParsedGroup[] = data.groups.map((group) => ({ ...group, eq: group.equipment }));
    const byModel = new Map<string, Map<string, ParsedGroup[]>>();
    for (const p of parsed) {
      if (!byModel.has(p.model)) byModel.set(p.model, new Map());
      const eqMap = byModel.get(p.model)!;
      if (!eqMap.has(p.eq)) eqMap.set(p.eq, []);
      eqMap.get(p.eq)!.push(p);
    }
    const modelRows = Array.from(byModel.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([model, eqMap]) => {
        const eqRows = Array.from(eqMap.entries())
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([eq, chambers]) => ({
            eq,
            totalN: chambers.reduce((sum, c) => sum + c.n, 0),
            chambers: [...chambers].sort((a, b) => a.chamber.localeCompare(b.chamber)),
          }));
        return { model, totalN: eqRows.reduce((sum, row) => sum + row.totalN, 0), eqRows };
      });
    return modelRows;
  }, [data]);

  function handleTileClick(group: ParsedGroup) {
    const params = new URLSearchParams({ feature: `Step${step}_Config`, config: group.config });
    router.push(`/root-cause?${params.toString()}`);
  }

  return (
    <section className="resultCard monitoringTreemapCard" style={{ opacity: loading && data ? 0.5 : 1 }}>
      <div className="sectionHeading compact">
        <div>
          <h2>Config vs {target} 트리맵</h2>
        </div>
        {headerRight}
      </div>

      {!data && loading ? (
        <p className="emptyMessage">불러오는 중…</p>
      ) : !data || models.length === 0 ? (
        <p className="emptyMessage">{data?.empty_reason ?? `Step${step}에는 Config 데이터가 없습니다.`}</p>
      ) : (
        <>
          <div className="monitoringTreemap">
            {models.map((modelRow) => (
              <div
                key={modelRow.model}
                className="monitoringTreemapModel"
                style={isMobileLayout ? { flex: "0 0 100px" } : { flexGrow: modelRow.totalN, flexBasis: 0 }}
              >
                <div className="monitoringTreemapModelLabel">{modelRow.model}</div>
                <div className="monitoringTreemapEqRows">
                  {modelRow.eqRows.map((eqRow) => (
                    <div key={eqRow.eq} className="monitoringTreemapEqRow" style={{ flexGrow: eqRow.totalN, flexBasis: 0 }}>
                      <div className="monitoringTreemapEqLabel">{eqRow.eq}</div>
                      <div className="monitoringTreemapTiles">
                        {eqRow.chambers.map((chamber) => {
                          const insufficientN = chamber.n < MIN_TILE_N;
                          const shouldColor = !insufficientN && data.significant;
                          const tileBackground = shouldColor ? colorForMean(chamber.mean, data.overall_mean, theme) : undefined;
                          return (
                            <button
                              type="button"
                              key={chamber.config}
                              className={`monitoringTreemapTile ${insufficientN ? "insufficientN" : ""}`}
                              style={{
                                flexGrow: chamber.n,
                                flexBasis: 0,
                                background: tileBackground,
                                color: tileBackground ? textColorForTile(tileBackground) : undefined,
                              }}
                              onClick={() => handleTileClick(chamber)}
                              onMouseEnter={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseMove={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseLeave={() => setHover(null)}
                            >
                              <span className="monitoringTreemapTileTitle">{chamber.chamber}</span>
                              <span>{insufficientN ? "표본 부족" : `${chamber.mean.toFixed(1)}%`}</span>
                              <span className="monitoringTreemapTileN">n={chamber.n}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="monitoringTreemapCaption">
            면적 = 웨이퍼 수 · 색 = {data.target} 평균 불량률(전체 평균 {data.overall_mean.toFixed(1)}% 기준 ±{COLOR_SPAN_PP}%p 고정 스케일, 낮을수록 초록) · n&lt;{MIN_TILE_N} 회색
          </p>
        </>
      )}

      {hover && (
        <div className="heatmapTooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <strong>{hover.group.config}</strong>
          <div className="heatmapTooltipRow"><span>경로</span><b>{hover.group.model} → {hover.group.equipment} → {hover.group.chamber}</b></div>
          <div className="heatmapTooltipRow"><span>n</span><b>{hover.group.n}</b></div>
          <div className="heatmapTooltipRow"><span>{data?.target ?? target} 평균 불량률</span><b>{hover.group.mean.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>중앙값</span><b>{hover.group.median.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>p5</span><b>{hover.group.p5.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>p95</span><b>{hover.group.p95.toFixed(2)}%</b></div>
        </div>
      )}
    </section>
  );
}
