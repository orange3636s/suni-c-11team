"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getScreeningHeatmap } from "@/lib/api";
import { TIER_LABEL } from "@/lib/confidenceTier";
import { formatEps2, formatQValue } from "@/lib/numberFormat";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfidenceTier, HeatmapKind, HeatmapResponse } from "@/types/data";

// 작업 지시서 WJ-1: 기본 7행만 보여주고 "전체 N행 보기"로 펼친다 --
// 스크롤 상자와 자르기를 함께 쓰지 않는다(펼치면 전체가 그대로 자라난다,
// WJ-1 "하지 말 것"). 정렬을 바꾸면 다시 7행으로 접힌다(아래 sortMode/
// significantOnly onChange가 setExpanded(false)를 함께 호출).
const DEFAULT_ROW_LIMIT = 7;
const VIEW_KIND_LABEL: Record<HeatmapKind, string> = { numeric: "수치형", categorical: "범주형" };
// TC-1: 뷰마다 다른 제목 -- 하나의 제목에 토글만 두지 않고, 지금 보고
// 있는 게 뭔지 제목이 직접 말하게 한다.
const VIEW_TITLE: Record<HeatmapKind, string> = {
  numeric: "R, D vs Y1~Y5 상관관계 히트맵",
  categorical: "Config vs Y1~Y5 상관관계 히트맵",
};
// TC-4: eps2 >= 이 값이고 |rho| < 이 값이면 U자(비단조) 관계로 판정한다 --
// 순위상관(rho)만 보면 약해 보이지만 실제 설명력(eps2)은 높은 경우.
const U_SHAPE_EPS2_MIN = 0.05;
const U_SHAPE_RHO_MAX = 0.15;
// TC-4: 정렬 드롭다운은 세 항목만 남긴다 -- 절댓값 정렬·특정 타깃 기준·
// Step 순서는 제거했다. 기본값은 "최대 ε²"(관계 강도 순 -- U자 인자도
// 상위에 온다).
type SortMode = "max_eps2" | "max_rho" | "min_rho";
const SORT_OPTION_LABEL: Record<SortMode, string> = {
  max_eps2: "최대 ε²",
  max_rho: "최대 ρ",
  min_rho: "최소 ρ",
};

function isUShape(eps2Value: number | null, rhoValue: number | null): boolean {
  return eps2Value != null && rhoValue != null && eps2Value >= U_SHAPE_EPS2_MIN && Math.abs(rhoValue) < U_SHAPE_RHO_MAX;
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

// TC-4: 농도(강도)는 eps2, 색상(방향)은 rho 부호 -- 둘을 분리해서 받는다.
// rho가 없으면(categorical, 방향 미정의) 항상 양의 방향(빨강 계열) 하나만
// 쓴다(기존 범주형 동작과 동일).
function cellBackground(eps2Value: number, rhoValue: number | null, epsMax: number, theme: "light" | "dark"): { bg: string; light: boolean } {
  const { pos, neg, center } = THEME_COLORS[theme];
  const target = rhoValue != null && rhoValue < 0 ? neg : pos;
  const t = Math.min(1, Math.max(0, eps2Value) / (epsMax || 1));
  const bg = mixHex(center, target, t);
  const light = relativeLuminance(hexToRgb(target)) * t + relativeLuminance(hexToRgb(center)) * (1 - t) < 0.45;
  return { bg, light };
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
  eps2: number | null;
  rho: number | null;
  n: number;
  q: number | null;
  significant: boolean;
  tier: ConfidenceTier | null;
  uShape: boolean;
  // QA-3: 상관계수는 그려지지만(n>=30) 종류별 표본 게이트 미달로 유의
  // 인자 목록에는 없는 셀.
  gateExcluded: boolean;
};

export default function CorrelationHeatmap({
  datasetId,
  enabled,
  onSelectCell,
  initialCache,
  onCacheUpdate,
}: {
  datasetId: string;
  enabled: boolean;
  onSelectCell: (selection: HeatmapCellSelection) => void;
  // TA그룹: 탭 왕복으로 이 컴포넌트가 언마운트/리마운트돼도 재요청하지
  // 않도록, 상위(AnalysisStateProvider의 analysis.heatmap)가 들고 있는
  // 캐시를 씨앗으로 받고, 새로 조회할 때마다 갱신분을 되돌려준다. 키는
  // 이제 kind 하나뿐이다(TC-4: metric 토글 제거, TC-3: config_level 제거).
  initialCache?: Record<string, HeatmapResponse>;
  onCacheUpdate?: (cache: Record<string, HeatmapResponse>) => void;
}) {
  const theme = useResolvedTheme();
  const [kind, setKind] = useState<HeatmapKind>("numeric");
  // TA그룹: 마운트 시점의 initialCache로 씨앗을 뿌린다 -- 탭을 나갔다
  // 돌아와 이 컴포넌트가 다시 마운트될 때, AnalysisStateProvider가 들고
  // 있던 이전 결과를 그대로 이어받아 재요청을 건너뛴다. `data`보다 먼저
  // 선언해야 아래 lazy initializer가 이 값을 바로 읽을 수 있다.
  const cache = useRef<Map<string, HeatmapResponse>>(new Map(Object.entries(initialCache ?? {})));
  // datasetId가 "바뀔 때만" 캐시를 비운다 -- 마운트 첫 렌더에서도 이
  // effect는 한 번 실행되므로, 매번 비우면 위에서 뿌린 씨앗이 즉시
  // 날아간다(datasetId 자체는 그대로인데도).
  const previousDatasetId = useRef(datasetId);
  // 씨앗 캐시에 현재 보기(kind 초기값)가 이미 있으면 첫 페인트부터 그걸로
  // 채운다 -- 그러지 않으면 effect가 도는 한 틱 동안 "계산하는 중…"이
  // 잠깐 스쳐 지나간다.
  const [data, setData] = useState<HeatmapResponse | null>(() => cache.current.get(kind) ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [significantOnly, setSignificantOnly] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("max_eps2");
  const [expanded, setExpanded] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // Brief settle animation on the rows whenever the sort/filter controls
  // below actually change the order (spec §5-4-3) -- triggered from
  // those controls' own onChange, not derived reactively from `rows`,
  // so it never fires a setState synchronously inside an effect.
  const [sorting, setSorting] = useState(false);

  function triggerRowSettle() {
    setSorting(true);
    window.setTimeout(() => setSorting(false), 220);
    // WJ-1: 정렬/필터를 바꾸면 다시 7행으로 접힌다.
    setExpanded(false);
  }

  useEffect(() => {
    if (previousDatasetId.current === datasetId) return;
    previousDatasetId.current = datasetId;
    cache.current = new Map();
  }, [datasetId]);

  useEffect(() => {
    if (!enabled) return;
    const cacheKey = kind;
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
          const response = await getScreeningHeatmap(datasetId, kind);
          if (cancelled) return;
          cache.current.set(cacheKey, response);
          onCacheUpdate?.(Object.fromEntries(cache.current));
          setData(response);
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
  }, [datasetId, kind, enabled]);

  // TC-4: "최대/최소 ρ" 정렬의 대표값 -- 각 행에서 eps2가 가장 큰(=가장
  // 중요한) 타깃 칸의 부호 있는 rho를 그 행의 대표 rho로 쓴다. 특정
  // 타깃에 고정하지 않고(제거된 "특정 타깃 기준"), 행마다 자기 자신의
  // 지배적 관계를 대표값으로 삼는다.
  const dominantRho = useMemo(() => {
    if (!data) return [];
    return data.features.map((_, rowIndex) => {
      let bestCol = -1;
      let bestEps2 = -Infinity;
      data.values[rowIndex].forEach((v, colIndex) => {
        if (v != null && v > bestEps2) {
          bestEps2 = v;
          bestCol = colIndex;
        }
      });
      return bestCol >= 0 ? (data.rho[rowIndex]?.[bestCol] ?? null) : null;
    });
  }, [data]);

  const rows = useMemo(() => {
    if (!data) return [];
    let indices = data.features.map((_, index) => index);
    if (significantOnly) {
      indices = indices.filter((index) => data.significant[index].some(Boolean));
    }
    if (sortMode === "max_rho") {
      // 빨강(양) -> 흰색 -> 파랑(음) 순으로 배열된다 (지시서 TC-4).
      indices = [...indices].sort((a, b) => (dominantRho[b] ?? -Infinity) - (dominantRho[a] ?? -Infinity));
    } else if (sortMode === "min_rho") {
      indices = [...indices].sort((a, b) => (dominantRho[a] ?? Infinity) - (dominantRho[b] ?? Infinity));
    }
    // sortMode "max_eps2"는 서버 기본 순서(이미 max eps2 내림차순)를 그대로 쓴다.
    return indices;
  }, [data, significantOnly, sortMode, dominantRho]);

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
          <h2>{VIEW_TITLE[kind]}</h2>
          {kind === "categorical" && (
            <p className="heatmapIntro">검정 {testedCount}건 · FDR 통과 {passedCount}건</p>
          )}
        </div>
        <div className="heatmapToggleStack">
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
            {(Object.keys(SORT_OPTION_LABEL) as SortMode[]).map((mode) => (
              <option key={mode} value={mode}>{SORT_OPTION_LABEL[mode]}</option>
            ))}
          </select>
        </div>
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
                kind={kind}
                theme={theme}
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
        <span>{kind === "numeric" ? `−${scaleMax.toFixed(2)}` : scaleMin.toFixed(2)}</span>
        <div
          className="heatmapColorbarTrack"
          style={{
            background:
              kind === "numeric"
                ? `linear-gradient(to right, ${THEME_COLORS[theme].neg}, ${THEME_COLORS[theme].center}, ${THEME_COLORS[theme].pos})`
                : `linear-gradient(to right, ${THEME_COLORS[theme].center}, ${THEME_COLORS[theme].pos})`,
          }}
        />
        <span>{scaleMax.toFixed(2)}</span>
      </div>
      <p className="heatmapCaption">
        {kind === "numeric" ? (
          <>
            셀 농도는 ε²(설명력) · 색상은 ρ(스피어만 상관)의 부호(빨강=양, 파랑=음)입니다.
            {data.excluded_configs > 0 && ` Config ${data.excluded_configs}개 제외.`}
            {" "}표본이 30개 미만인 셀은 사선 패턴으로 표시됩니다. 색은 있지만 점선 테두리인 셀은 표본 게이트로 유의 인자 목록에서 빠진 셀입니다.
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
          <div className="heatmapTooltipRow"><span>ε²</span><b>{tooltip.eps2 != null ? tooltip.eps2.toFixed(3) : "표본 부족"}</b></div>
          {tooltip.rho != null && (
            <div className="heatmapTooltipRow"><span>ρ</span><b>{tooltip.rho >= 0 ? "+" : ""}{tooltip.rho.toFixed(3)}</b></div>
          )}
          <div className="heatmapTooltipRow"><span>n</span><b>{tooltip.n.toLocaleString()}</b></div>
          <div className="heatmapTooltipRow"><span>q</span><b>{formatQValue(tooltip.q)}</b></div>
          <div className="heatmapTooltipRow"><span>신뢰도</span><b>{tooltip.tier ? TIER_LABEL[tooltip.tier] : "-"}</b></div>
          <div className="heatmapTooltipRow"><span>FDR 통과</span><b>{tooltip.significant ? "예" : "아니오"}</b></div>
          {tooltip.uShape && (
            <p className="heatmapTooltipShapeNote">U자 형태 — 순위상관으로는 약해 보이지만 설명력은 높습니다</p>
          )}
          {tooltip.gateExcluded && (
            <div className="heatmapTooltipRow"><span>유의 인자</span><b className="paretoUnderSampledLabel">표본 게이트 미달로 제외</b></div>
          )}
        </div>
      )}
    </section>
  );
}

function FragmentRow({
  feature,
  rowIndex,
  data,
  kind,
  theme,
  scaleMax,
  onHover,
  onSelectCell,
}: {
  feature: string;
  rowIndex: number;
  data: HeatmapResponse;
  kind: HeatmapKind;
  theme: "light" | "dark";
  scaleMax: number;
  onHover: (tooltip: TooltipState | null) => void;
  onSelectCell: (selection: HeatmapCellSelection) => void;
}) {
  return (
    <>
      <div className="heatmapRowLabel">{feature}</div>
      {data.targets.map((target, colIndex) => {
        const eps2Value = data.values[rowIndex][colIndex];
        const rhoValue = data.rho[rowIndex]?.[colIndex] ?? null;
        const n = data.n[rowIndex][colIndex];
        const q = data.q[rowIndex][colIndex];
        const significant = data.significant[rowIndex][colIndex];
        const tier = data.tier[rowIndex][colIndex];
        const masked = eps2Value == null;
        // 범주형 보기 전용: 색은 FDR 게이트를 통과한 셀에만 칠한다 (지시서
        // E "색 스케일 규칙" -- 자동 정규화 금지와 짝을 이루는 규칙. q가
        // 없거나(=미검정) 0.05 이상이면 값은 있어도 회색 고정).
        const gated = kind === "categorical" && !masked && !significant;
        // QA-3: 상관계수는 그려지는데(n>=30) 종류별 표본 게이트(R>=100/
        // D>=40) 미달로 유의 인자 목록에는 없는 셀 -- 히트맵과 유의 인자
        // 판정이 어긋나 보이지 않도록 별도 표시한다.
        const gateExcluded = kind === "numeric" && !masked && Boolean(data.gate_excluded?.[rowIndex]?.[colIndex]);
        const uShape = kind === "numeric" && isUShape(eps2Value, rhoValue);
        const style: React.CSSProperties = {};
        if (!masked && !gated) {
          const { bg, light } = cellBackground(eps2Value, kind === "numeric" ? rhoValue : null, scaleMax, theme);
          style.background = bg;
          style.color = light ? "var(--heatmap-text-inverse)" : "var(--heatmap-text)";
        }
        const hoverPayload = { feature, target, eps2: eps2Value, rho: rhoValue, n, q, significant, tier, gateExcluded, uShape };
        return (
          <button
            key={target}
            type="button"
            // TC-2: 유의 인자 강조 테두리(노란/주황)는 제거했다 -- "significant"
            // 클래스 자체는 더 이상 시각 효과가 없으므로 붙이지 않는다.
            // 필터(유의 인자만 보기 체크박스)는 data.significant 배열을
            // 그대로 쓰므로 이 클래스와 무관하게 동작한다.
            className={`heatmapCell ${masked ? "masked" : ""} ${gated ? "gated" : ""} ${gateExcluded ? "gate-excluded" : ""}`}
            style={style}
            onMouseEnter={(event) => onHover({ x: event.clientX, y: event.clientY, ...hoverPayload })}
            onMouseMove={(event) => onHover({ x: event.clientX, y: event.clientY, ...hoverPayload })}
            onMouseLeave={() => onHover(null)}
            onTouchStart={(event) =>
              onHover({ x: event.touches[0]?.clientX ?? 0, y: event.touches[0]?.clientY ?? 0, ...hoverPayload })
            }
            onClick={() => onSelectCell({ target, feature, significant, qValue: q })}
            aria-label={`${feature}, ${target}, eps2 ${eps2Value != null ? formatEps2(eps2Value) : "표본 부족"}${gateExcluded ? ", 표본 게이트로 유의 인자 목록에서 제외" : ""}`}
          >
            {masked ? "" : formatEps2(eps2Value)}
          </button>
        );
      })}
    </>
  );
}
