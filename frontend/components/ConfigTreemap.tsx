"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getTreemapData } from "@/lib/monitoringSource";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfigTreemapGroup, ConfigTreemapResponse } from "@/types/data";

const STEP_OPTIONS = Array.from({ length: 30 }, (_, i) => i + 1);
// n이 이 미만인 타일은 회색 처리하고 색을 칠하지 않는다 (지시서 §4③
// "표본 부족").
const MIN_TILE_N = 30;
// 고정 ±3%p 스케일 -- 관측 최대/최소로 자동 정규화하면 0.9%p 차이가
// 새빨갛게 렌더되어 없는 신호를 만든다 (지시서 §4③ "색 스케일 규칙").
const COLOR_SPAN_PP = 3;

const CONFIG_RE = /^Step\d+_(Model\d+)_(EQ[A-Z])_(CH\d+)$/;

type ParsedGroup = ConfigTreemapGroup & { model: string; eq: string; chamber: string };

function parseConfig(group: ConfigTreemapGroup): ParsedGroup | null {
  const m = CONFIG_RE.exec(group.config);
  if (!m) return null;
  return { ...group, model: m[1], eq: m[2], chamber: m[3] };
}

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

// 발산형 빨강(낮음)-초록(높음), 중앙은 전체 평균. 이 앱 전역에서 이미 쓰는
// RED/GREEN 토큰과 같은 값이다 (ParetoChart.tsx 등). dataviz 스킬의
// validate_palette.js로 확인함 -- 라이트 모드는 전 항목 PASS(CVD ΔE
// 8.6), 다크 모드는 CVD ΔE 6.5로 "6–8 완충 구간, 보조 인코딩이 있으면
// 허용" 등급이라 타일에 평균값을 항상 텍스트로 같이 표시한다(색만으로
// 판단하지 않도록).
const RED = { light: "#DC2626", dark: "#F87171" };
const GREEN = { light: "#059669", dark: "#34D399" };
const CENTER = { light: "#FFFFFF", dark: "#2C2C2E" };

function colorForMean(mean: number, overallMean: number, theme: "light" | "dark"): string {
  const lo = overallMean - COLOR_SPAN_PP;
  const hi = overallMean + COLOR_SPAN_PP;
  const clamped = Math.max(lo, Math.min(hi, mean));
  const t = (clamped - lo) / (hi - lo || 1); // 0..1, 0.5 = 전체 평균
  const red = theme === "dark" ? RED.dark : RED.light;
  const green = theme === "dark" ? GREEN.dark : GREEN.light;
  const center = theme === "dark" ? CENTER.dark : CENTER.light;
  return t < 0.5 ? lerpColor(red, center, t / 0.5) : lerpColor(center, green, (t - 0.5) / 0.5);
}

type HoverState = { group: ParsedGroup; x: number; y: number } | null;

export default function ConfigTreemap({ datasetId }: { datasetId: string }) {
  const theme = useResolvedTheme();
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [data, setData] = useState<ConfigTreemapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [hover, setHover] = useState<HoverState>(null);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      void getTreemapData(datasetId, step).then((result) => {
        if (cancelled) return;
        setData(result);
        setLoading(false);
      });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [datasetId, step]);

  const { models, unknownCount } = useMemo(() => {
    if (!data) return { models: [], unknownCount: 0 };
    const parsed: ParsedGroup[] = [];
    let unknown = 0;
    for (const group of data.groups) {
      const p = parseConfig(group);
      if (p) parsed.push(p);
      else unknown += 1;
    }
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
    return { models: modelRows, unknownCount: unknown };
  }, [data]);

  function handleTileClick(group: ParsedGroup) {
    const params = new URLSearchParams({ feature: `Step${step}_Config`, config: group.config });
    router.push(`/root-cause?${params.toString()}`);
  }

  return (
    <section className="resultCard monitoringTreemapCard">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">TREEMAP</span>
          <h2>Model → EQ → Chamber 수율</h2>
        </div>
        <label className="monitoringStepSelect">
          스텝
          <select value={step} onChange={(event) => setStep(Number(event.target.value))}>
            {STEP_OPTIONS.map((s) => (
              <option key={s} value={s}>Step{s}</option>
            ))}
          </select>
        </label>
      </div>

      {loading ? (
        <p className="emptyMessage">불러오는 중…</p>
      ) : !data || models.length === 0 ? (
        <p className="emptyMessage">Step{step}에는 Config 데이터가 없습니다.</p>
      ) : (
        <>
          <div className="monitoringTreemap">
            {models.map((modelRow) => (
              <div key={modelRow.model} className="monitoringTreemapModel" style={{ flexGrow: modelRow.totalN, flexBasis: 0 }}>
                <div className="monitoringTreemapModelLabel">{modelRow.model}</div>
                <div className="monitoringTreemapEqRows">
                  {modelRow.eqRows.map((eqRow) => (
                    <div key={eqRow.eq} className="monitoringTreemapEqRow" style={{ flexGrow: eqRow.totalN, flexBasis: 0 }}>
                      <div className="monitoringTreemapEqLabel">{eqRow.eq}</div>
                      <div className="monitoringTreemapTiles">
                        {eqRow.chambers.map((chamber) => {
                          const insufficientN = chamber.n < MIN_TILE_N;
                          return (
                            <button
                              type="button"
                              key={chamber.config}
                              className={`monitoringTreemapTile ${insufficientN ? "insufficientN" : ""}`}
                              style={{
                                flexGrow: chamber.n,
                                flexBasis: 0,
                                background: insufficientN ? undefined : colorForMean(chamber.mean, data.overall_mean, theme),
                              }}
                              onClick={() => handleTileClick(chamber)}
                              onMouseEnter={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseMove={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseLeave={() => setHover(null)}
                            >
                              <span className="monitoringTreemapTileTitle">{chamber.chamber}</span>
                              <span>{insufficientN ? "표본 부족" : `${chamber.mean.toFixed(1)}%`}</span>
                              <span>n={chamber.n}</span>
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
          {unknownCount > 0 && (
            <p className="emptyMessage monitoringTreemapUnknown">형식을 알 수 없는 Config {unknownCount}건은 표시에서 제외했습니다.</p>
          )}
          <p className="monitoringTreemapCaption">
            면적 = 웨이퍼 수 · 색 = 평균 수율(전체 평균 {data.overall_mean.toFixed(1)}% 기준 ±{COLOR_SPAN_PP}%p 고정 스케일) · n&lt;{MIN_TILE_N} 회색
          </p>
        </>
      )}

      {hover && (
        <div className="heatmapTooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <strong>{hover.group.config}</strong>
          <div className="heatmapTooltipRow"><span>n</span><b>{hover.group.n}</b></div>
          <div className="heatmapTooltipRow"><span>평균</span><b>{hover.group.mean.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>중앙값</span><b>{hover.group.median.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>p5</span><b>{hover.group.p5.toFixed(2)}%</b></div>
          <div className="heatmapTooltipRow"><span>p95</span><b>{hover.group.p95.toFixed(2)}%</b></div>
        </div>
      )}
    </section>
  );
}
