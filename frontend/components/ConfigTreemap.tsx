"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { DIVERGING_GREEN, DIVERGING_RED } from "@/lib/constants";
import { getDatasetSchema } from "@/lib/api";
import { getTreemapData } from "@/lib/monitoringSource";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfigTreemapGroup, ConfigTreemapResponse } from "@/types/data";

const TARGET_OPTIONS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
// n이 이 미만인 타일은 회색 처리하고 색을 칠하지 않는다 (지시서 §4③
// "표본 부족").
const MIN_TILE_N = 30;
// 고정 ±3%p 스케일 -- 관측 최대/최소로 자동 정규화하면 0.9%p 차이가
// 새빨갛게 렌더되어 없는 신호를 만든다 (지시서 §4③ "색 스케일 규칙").
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
// RED/GREEN 토큰과 같은 값이다 (ParetoChart.tsx 등). dataviz 스킬의
// validate_palette.js로 확인함 -- 라이트 모드는 전 항목 PASS(CVD ΔE
// 8.6), 다크 모드는 CVD ΔE 6.5로 "6–8 완충 구간, 보조 인코딩이 있으면
// 허용" 등급이라 타일에 평균값을 항상 텍스트로 같이 표시한다(색만으로
// 판단하지 않도록).
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

// Z-3: 타일이 채색되면(FDR 통과 시에만, 현재 데이터에서는 없음) 배경이
// 빨강~중립~초록 그라디언트를 오간다 -- 고정 텍스트색으로는 중앙(다크
// 중립 #2C2C2E) 근처에서 대비가 무너진다. 셀 배경의 실제 밝기를 보고
// 매 타일마다 어두운/밝은 글자를 고른다(YIQ luma, 관용적 임계값 0.5).
function lumaOf(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}
function textColorForTile(backgroundHex: string): string {
  return lumaOf(backgroundHex) > 0.5 ? "var(--ink)" : "#fff";
}

type HoverState = { group: ParsedGroup; x: number; y: number } | null;

export default function ConfigTreemap({
  datasetId,
  initialStep,
  initialData = null,
  onDataChange,
}: {
  datasetId: string;
  initialStep?: number;
  // E-4: 부모(모니터링 홈)의 monitoringHome 캐시에서 넘어온, 마지막으로
  // 조회했던 스텝의 결과 -- 탭을 나갔다 돌아왔을 때(이 컴포넌트가
  // 통째로 언마운트/리마운트될 때) 이 값이 있으면 재조회하지 않는다.
  initialData?: { step: number; target?: string; data: ConfigTreemapResponse | null } | null;
  onDataChange?: (step: number, target: string, data: ConfigTreemapResponse | null) => void;
}) {
  const theme = useResolvedTheme();
  const router = useRouter();
  // 모바일 반응형 패치 S-4: ≤767px에서 3-Model 가로 분할이 읽기 힘들어
  // 세로로 쌓는다(각 Model 블록 ~100px). 데스크톱/태블릿의 면적비례
  // flexGrow는 가로 분할 전제라 세로로 바꾸면 그대로 못 쓴다 -- 폭 대신
  // 높이가 비례해 버려 Model마다 제각각 높이가 되므로, 이 폭에서는
  // 고정 높이로 바꾼다.
  const isMobileLayout = useIsMobileLayout();
  const [step, setStep] = useState(initialStep ?? initialData?.step ?? 1);
  const [target, setTarget] = useState(initialData?.target ?? initialData?.data?.target ?? "Y1");
  const [stepOptions, setStepOptions] = useState<number[]>([]);
  const [optionsDataset, setOptionsDataset] = useState("");
  const optionsReady = optionsDataset === datasetId;
  const hasCachedInitial = initialData != null && initialData.step === step && (initialData.target ?? "Y1") === target;
  const [data, setData] = useState<ConfigTreemapResponse | null>(hasCachedInitial ? initialData!.data : null);
  const [loading, setLoading] = useState(!hasCachedInitial);
  const [hover, setHover] = useState<HoverState>(null);
  // 캐시는 최초 마운트 시 한 번만 소비한다 -- 이후 사용자가 직접 스텝을
  // 바꾸면(활발히 보고 있는 중이므로) 항상 새로 조회해야 한다.
  const consumedInitialRef = useRef(hasCachedInitial);

  useEffect(() => {
    let cancelled = false;
    void getDatasetSchema(datasetId)
      .then((schema) => {
        if (cancelled) return;
        const available = schema.config_steps.length > 0 ? schema.config_steps : schema.steps_present;
        const storedStep = Number(window.localStorage.getItem(`monitoring-treemap-step:${datasetId}`));
        const nextStep = available.includes(storedStep) ? storedStep : available.includes(step) ? step : (available[0] ?? step);
        const storedTarget = window.localStorage.getItem(`monitoring-treemap-target:${datasetId}`);
        const nextTarget = TARGET_OPTIONS.includes(storedTarget as (typeof TARGET_OPTIONS)[number]) ? storedTarget! : target;
        setStepOptions(available);
        setStep(nextStep);
        setTarget(nextTarget);
        setOptionsDataset(datasetId);
      })
      .catch(() => {
        if (cancelled) return;
        setStepOptions([step]);
        setOptionsDataset(datasetId);
      });
    return () => { cancelled = true; };
    // Dataset change is the only reason to reload the schema/options.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  useEffect(() => {
    if (!optionsReady) return;
    if (consumedInitialRef.current && initialData?.step === step && (initialData.target ?? "Y1") === target) {
      consumedInitialRef.current = false;
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setLoading(true);
      window.localStorage.setItem(`monitoring-treemap-step:${datasetId}`, String(step));
      window.localStorage.setItem(`monitoring-treemap-target:${datasetId}`, target);
      void getTreemapData(datasetId, step, target).then((result) => {
        if (cancelled) return;
        setData(result);
        setLoading(false);
        onDataChange?.(step, target, result);
      });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, optionsReady, step, target]);

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
    <section className="resultCard monitoringTreemapCard">
      <div className="sectionHeading compact">
        <div>
          <h2>설비 구성 트리맵 Model → EQ → Chamber 수율</h2>
        </div>
        <div className="monitoringTreemapControls">
          <label className="monitoringStepSelect">
            스텝
            <select value={step} onChange={(event) => setStep(Number(event.target.value))}>
              {stepOptions.map((s) => (
                <option key={s} value={s}>Step{s}</option>
              ))}
            </select>
          </label>
          <label className="monitoringStepSelect">
            불량 원인
            <select value={target} onChange={(event) => setTarget(event.target.value)}>
              {TARGET_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
        </div>
      </div>

      {loading ? (
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
                          const shouldColor = !insufficientN;
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
                                // 중립 타일(항상 var(--text), 현재 데이터의
                                // 전량)은 CSS 기본값을 그대로 쓴다 -- 채색된
                                // 타일만 셀 밝기에 맞춰 글자색을 정한다.
                                color: tileBackground ? textColorForTile(tileBackground) : undefined,
                              }}
                              onClick={() => handleTileClick(chamber)}
                              onMouseEnter={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseMove={(event) => setHover({ group: chamber, x: event.clientX, y: event.clientY })}
                              onMouseLeave={() => setHover(null)}
                            >
                              <span className="monitoringTreemapTileTitle">{chamber.chamber}</span>
                              <span>{insufficientN ? "표본 부족" : `${chamber.mean.toFixed(1)}%`}</span>
                              {/* 모바일 반응형 패치 S-4: 타일 텍스트가 좁아지면(≤767px)
                                  n= 카운트부터 뺀다 -- 채널명 + 값은 항상 남긴다. */}
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
