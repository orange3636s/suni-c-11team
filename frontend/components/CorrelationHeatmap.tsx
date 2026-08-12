"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { flushSync } from "react-dom";
import ChartExportButton from "@/components/ChartExportButton";
import { FavoriteStarButton } from "@/components/FavoriteStarButton";
import { getScreeningHeatmap } from "@/lib/api";
import { buildExportFilename, buildHeatmapCaptionText, exportNodeAsPng } from "@/lib/chartExport";
import { TIER_LABEL } from "@/lib/confidenceTier";
import { formatAdjR2, formatQValue } from "@/lib/numberFormat";
import { useResolvedTheme } from "@/lib/useResolvedTheme";
import type { ConfidenceTier, HeatmapResponse, RelationShape } from "@/types/data";

// 기본 7행만 보여주고 "전체 N행 보기"로 펼친다 -- 스크롤 상자와 자르기를
// 함께 쓰지 않는다(펼치면 전체가 그대로 자라난다). 정렬을 바꾸면 다시
// 7행으로 접힌다(아래 sortMode/significantOnly onChange가
// setExpanded(false)를 함께 호출).
const DEFAULT_ROW_LIMIT = 7;
// 내보내기 사본의 해석 캡션 폭 상한(px) -- 아래 HeatmapLegendCaption 주석 참고.
const EXPORT_CAPTION_MAX_WIDTH = 760;
// 이 히트맵은 수치형(R, D vs Y1~Y5) 보기 하나만 그리므로 제목이 고정이다.
// Config 효과는 Config별 트리맵 탭이 담당한다.
const VIEW_TITLE = "R, D vs Y1~Y5 상관관계 히트맵";

// 정렬 드롭다운은 네 항목이며 모두 부호 없는 크기(Adj R²)나 Step 기준이다
// -- 핵심 인자 다수가 U자라 전체구간 부호(ρ)는 표본 절반에 대해 반대로
// 읽히므로, 그 값으로 행을 줄 세우면 오해를 만든다. 기본값은
// "최대 Adj R²"(서버 기본 순서와 동일).
export type SortMode = "max_adj_r2" | "min_adj_r2" | "step_desc" | "step_asc";
export const SORT_OPTION_LABEL: Record<SortMode, string> = {
  max_adj_r2: "최대 Adj R²",
  min_adj_r2: "최소 Adj R²",
  step_desc: "Step 내림차순",
  step_asc: "Step 오름차순",
};
function isSortMode(value: string | null | undefined): value is SortMode {
  return value === "max_adj_r2" || value === "min_adj_r2" || value === "step_desc" || value === "step_asc";
}

/** `Step13_R1` -> 13. 문자열 정렬로는 Step1 < Step11 < Step2가 되므로
 * 반드시 숫자로 뽑아 비교한다. 스텝 번호가 없는 이름은 맨 뒤로 민다. */
function featureStep(feature: string): number {
  const match = /^Step(\d+)_/.exec(feature);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
}

/** WCAG 상대 휘도 -- sRGB 채널을 감마 보정(선형화)한 뒤 가중합한다. 단순
 * `(0.2126R+0.7152G+0.0722B)/255`(감마 미보정 luma)는 어두운 배경 쪽에서
 * 실제 지각 밝기보다 값을 부풀려, 글자색 자동 전환 임계값 판정이
 * 어긋난다. */
function srgbChannelToLinear(channel: number): number {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
function relativeLuminance([r, g, b]: [number, number, number]): number {
  const [rl, gl, bl] = [r, g, b].map(srgbChannelToLinear);
  return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}
/** WCAG 대비비 -- 두 상대 휘도 중 밝은 쪽을 분자로 둔다. */
function contrastRatio(luminanceA: number, luminanceB: number): number {
  const lighter = Math.max(luminanceA, luminanceB);
  const darker = Math.min(luminanceA, luminanceB);
  return (lighter + 0.05) / (darker + 0.05);
}

// 단일 색상 순차(sequential) 그라데이션 -- 셀 색이 전달하는 정보는
// "설명력이 얼마나 큰가" 하나뿐이고, 방향은 툴팁의 차수/꼭짓점/증감으로만
// 말한다. Adjusted R²는 0~1 한 방향 값이라 중앙값이 없으므로 발산
// (빨강~흰색~파랑) 팔레트를 쓰지 않는다.
//
// 라이트는 흰색 -> 빨강, 다크는 패널 배경(거의 같은 어둡기) -> 빨강으로
// 서로 다른 정지점을 쓴다 -- 라이트 램프를 그대로 반전해 다크에 쓰면
// 중간 구간 색이 탁해진다. 정지점은 Adjusted R² 절대값 기준이며, 아래에서
// scale.max(=1.00)로 정규화해 쓴다(데이터셋마다 다시 스케일링하지 않는
// 고정 척도).
type GradientStop = [number, string];
const LIGHT_GRADIENT_STOPS: GradientStop[] = [
  [0.0, "#FFFFFF"],
  [0.1, "#FBE3DF"],
  [0.25, "#F2B5AC"],
  [0.45, "#E07E70"],
  [0.6, "#CF5343"],
  [0.7, "#B03024"],
];
const DARK_GRADIENT_STOPS: GradientStop[] = [
  [0.0, "#171C24"],
  [0.1, "#33222A"],
  [0.25, "#5A2E32"],
  [0.45, "#8C3A38"],
  [0.6, "#B04236"],
  [0.7, "#D14A3A"],
];
const GRADIENT_SPAN = LIGHT_GRADIENT_STOPS[LIGHT_GRADIENT_STOPS.length - 1][0];
function gradientStopsFor(theme: "light" | "dark"): GradientStop[] {
  return theme === "dark" ? DARK_GRADIENT_STOPS : LIGHT_GRADIENT_STOPS;
}

// 글자색 자동 전환에 쓰는 두 후보 잉크. 고정 임계값(예: "휘도 0.55")
// 하나로 밝은/어두운 글자를 가르면 중간 농도 구간에서 대비가 무너진다
// (라이트 램프의 0.45~0.8 구간에서 실측 2.0~3.2:1까지 떨어짐) -- 대신
// 두 후보 각각과의 실제 대비비를 계산해 더 높은 쪽을 쓴다.
const INK_DARK: [number, number, number] = hexToRgb("#1d1d1f"); // var(--heatmap-text) 라이트 값
const INK_LIGHT: [number, number, number] = hexToRgb("#ffffff"); // var(--heatmap-text-inverse)
const INK_DARK_LUMINANCE = relativeLuminance(INK_DARK);
const INK_LIGHT_LUMINANCE = relativeLuminance(INK_LIGHT);
// 이 값 미만인 다크 모드 셀은 배경이 패널색에 거의 묻히므로, 일반 규칙
// (대비 기준 자동 전환) 대신 흐리지만 읽히는 전용 톤을 쓴다 -- 흰 글자를
// 그대로 얹으면 "0에 가까운 셀이 가장 눈에 띈다"는 정보 위계 역전이
// 그대로 남는다. grade_thresholds.yaml의 "보통" 등급 하한과 같은 값.
const WEAK_CELL_ADJ_R2_THRESHOLD = 0.05;

function interpolateGradient(ratio: number, stops: GradientStop[]): [number, number, number] {
  const t = Math.min(1, Math.max(0, ratio)) * GRADIENT_SPAN;
  let lower = stops[0];
  let upper = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i += 1) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }
  const span = upper[0] - lower[0];
  const local = span > 0 ? (t - lower[0]) / span : 0;
  const from = hexToRgb(lower[1]);
  const to = hexToRgb(upper[1]);
  return [
    Math.round(from[0] + (to[0] - from[0]) * local),
    Math.round(from[1] + (to[1] - from[1]) * local),
    Math.round(from[2] + (to[2] - from[2]) * local),
  ];
}

/** 셀 배경 -- Adjusted R² 크기 하나만 본다. `scaleMax`는 서버가 내려주는
 * 고정 척도(ADJ_R2_SCALE의 최대값=1.00)라 데이터셋이 바뀌어도 같은 값이
 * 같은 농도로 보인다. */
function cellBackgroundRgb(value: number, scaleMax: number, theme: "light" | "dark"): [number, number, number] {
  return interpolateGradient(Math.max(0, value) / (scaleMax || GRADIENT_SPAN), gradientStopsFor(theme));
}

/** 셀 글자색 -- 다크 모드의 약한 셀(R²<0.05)은 흐린 전용 톤(대비 3:1
 * 이상)을, 그 외에는 배경과 실측 대비가 더 높은 쪽 잉크(대비 4.5:1
 * 안팎)를 고른다. 라이트/다크 모두 같은 로직이다. */
function cellTextColor(value: number, rgb: [number, number, number], theme: "light" | "dark"): string {
  if (theme === "dark" && value < WEAK_CELL_ADJ_R2_THRESHOLD) return "var(--heatmap-text-muted)";
  const bgLuminance = relativeLuminance(rgb);
  const contrastWithDarkInk = contrastRatio(bgLuminance, INK_DARK_LUMINANCE);
  const contrastWithLightInk = contrastRatio(bgLuminance, INK_LIGHT_LUMINANCE);
  return contrastWithDarkInk >= contrastWithLightInk ? "var(--heatmap-text)" : "var(--heatmap-text-inverse)";
}

function gradientCss(theme: "light" | "dark"): string {
  const stops = gradientStopsFor(theme).map(([at, hex]) => `${hex} ${(at / GRADIENT_SPAN) * 100}%`);
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

const DIRECTION_TEXT: Partial<Record<RelationShape, string>> = {
  monotonic_increasing: "값이 커질수록 불량 증가",
  monotonic_decreasing: "값이 커질수록 불량 감소",
};

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
  adjR2: number | null;
  degree: number | null;
  shape: RelationShape | null;
  optimalCenter: number | null;
  n: number;
  measurementRatePct: number | null;
  q: number | null;
  significant: boolean;
  tier: ConfidenceTier | null;
  // 값은 그려지지만(n>=30) 종류별 표본 게이트 미달로 유의 인자 목록에는
  // 없는 셀.
  gateExcluded: boolean;
};

export default function CorrelationHeatmap({
  datasetId,
  enabled,
  onSelectCell,
  initialCache,
  onCacheUpdate,
  initialSortMode,
  onSortModeChange,
  favorited = false,
  favoritePending = false,
  onToggleFavorite,
}: {
  datasetId: string;
  enabled: boolean;
  onSelectCell: (selection: HeatmapCellSelection) => void;
  // 탭 왕복으로 이 컴포넌트가 언마운트/리마운트돼도 재요청하지 않도록,
  // 상위(AnalysisStateProvider의 analysis.heatmap)가 들고 있는 캐시를
  // 씨앗으로 받고, 새로 조회할 때마다 갱신분을 되돌려준다. 키는 kind
  // 하나뿐이고, 그 값은 항상 "numeric"이다.
  initialCache?: Record<string, HeatmapResponse>;
  onCacheUpdate?: (cache: Record<string, HeatmapResponse>) => void;
  // 즐겨찾기에서 열었을 때 저장 시점 정렬 기준으로 복원한다 --
  // 마운트 시 한 번만 반영하고, 이후 URL과 계속 동기화하지는 않는다.
  initialSortMode?: string | null;
  // 즐겨찾기 별이 "지금 이 정렬 기준"이 이미 저장돼 있는지 보여주려면
  // 부모가 현재 sortMode를 알아야 한다 -- 이 컴포넌트가 상태를 들고
  // 있으므로(select가 여기서 바뀐다) 바뀔 때마다(마운트 포함) 알려준다.
  onSortModeChange?: (sortMode: SortMode) => void;
  favorited?: boolean;
  favoritePending?: boolean;
  onToggleFavorite?: (sortMode: SortMode) => void;
}) {
  // 셀 색은 라이트/다크가 서로 다른 램프를 쓴다(흰색~빨강 / 패널 배경~빨강)
  // -- cellBackgroundRgb/cellTextColor/gradientCss에 그대로 흘려보낸다.
  const theme = useResolvedTheme();
  const kind = "numeric" as const;
  // 마운트 시점의 initialCache로 씨앗을 뿌린다 -- 탭을 나갔다
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
  const [sortMode, setSortMode] = useState<SortMode>(isSortMode(initialSortMode) ? initialSortMode : "max_adj_r2");
  const [expanded, setExpanded] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // Brief settle animation on the rows whenever the sort/filter controls
  // below actually change the order -- triggered from
  // those controls' own onChange, not derived reactively from `rows`,
  // so it never fires a setState synchronously inside an effect.
  const [sorting, setSorting] = useState(false);
  // 이미지 저장: true인 동안에만 "전체 행" 사본이 화면 밖 컨테이너에
  // 마운트된다(아래 handleExport 주석 참고). 저장 버튼의 disabled 상태도
  // 이 값 하나로 쓴다.
  const [exporting, setExporting] = useState(false);
  const exportSurfaceRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    onSortModeChange?.(sortMode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortMode]);

  function triggerRowSettle() {
    setSorting(true);
    window.setTimeout(() => setSorting(false), 220);
    // 정렬/필터를 바꾸면 다시 7행으로 접힌다.
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
          const response = await getScreeningHeatmap(datasetId);
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

  const rows = useMemo(() => {
    if (!data) return [];
    let indices = data.features.map((_, index) => index);
    if (significantOnly) {
      indices = indices.filter((index) => data.significant[index].some(Boolean));
    }
    const rowMax = (index: number) =>
      data.values[index].reduce<number>((best, value) => (value != null && value > best ? value : best), 0);
    if (sortMode === "min_adj_r2") {
      indices = [...indices].sort((a, b) => rowMax(a) - rowMax(b));
    } else if (sortMode === "step_asc" || sortMode === "step_desc") {
      // 같은 스텝에 여러 인자가 있으면(Step14_R1/Step14_R2/Step14_D1)
      // 인자명으로 2차 정렬해 순서가 흔들리지 않게 한다.
      const direction = sortMode === "step_asc" ? 1 : -1;
      indices = [...indices].sort((a, b) => {
        const stepDelta = featureStep(data.features[a]) - featureStep(data.features[b]);
        if (stepDelta !== 0) return stepDelta * direction;
        return data.features[a].localeCompare(data.features[b]);
      });
    }
    // "max_adj_r2"는 서버 기본 순서(이미 행별 최대 Adjusted R² 내림차순)를
    // 그대로 쓴다.
    return indices;
  }, [data, significantOnly, sortMode]);

  const visibleRows = expanded ? rows : rows.slice(0, DEFAULT_ROW_LIMIT);

  /** 이미지 저장 -- 화면에는 기본 7행만 떠 있지만 내보낸 PNG에는 **전체
   * 행**(예: R+D 58행)이 들어가야 한다. 그래서 화면 상태(expanded)를
   * 건드리는 대신, 전체 행짜리 사본을 화면 밖 컨테이너
   * (`position:absolute; left:-99999px`)에 잠깐 마운트해 그 노드를 굽는다.
   *
   * `display:none`을 쓰지 않는 이유가 핵심이다 -- 그 서브트리에서는
   * getComputedStyle이 빈 값을 돌려주므로, chartExport의 "계산된 스타일을
   * 인라인으로 굳히기" 패스가 그대로 무력화된다(이 기능의 원래 색 유실
   * 버그와 똑같은 증상이 난다). 화면 밖이지만 실제로 레이아웃된, DOM에
   * 붙어 있는 노드여야 한다.
   *
   * flushSync로 마운트를 동기 확정한 뒤 rAF 한 번으로 레이아웃까지 끝난
   * 것을 보장하고 캡처한다. finally에서 반드시 언마운트한다. */
  async function handleExport() {
    if (!data || exporting) return;
    try {
      flushSync(() => setExporting(true));
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const surface = exportSurfaceRef.current;
      if (!surface) throw new Error("내보낼 히트맵을 준비하지 못했습니다.");
      await exportNodeAsPng(surface, {
        filename: buildExportFilename({ feature: null, target: "all", view: "heatmap" }),
        captionText: buildHeatmapCaptionText({
          rowCount: rows.length,
          columnCount: data.targets.length,
          sortLabel: SORT_OPTION_LABEL[sortMode],
          datasetId,
        }),
      });
    } catch (failure) {
      console.warn("히트맵 이미지 저장 실패", failure);
    } finally {
      setExporting(false);
    }
  }

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

  const scaleMax = data.scale.max;
  const gridTemplateColumns = `160px repeat(${data.targets.length}, minmax(64px, 1fr))`;

  return (
    <section id="correlationHeatmapCard" className="resultCard heatmapCard">
      <div className="heatmapHeaderRow">
        <div className="heatmapHeaderRowText">
          <span className="sectionLabel">CORRELATION OVERVIEW</span>
          <h2>{VIEW_TITLE}</h2>
        </div>
        {/* 파레토·산점도·박스플롯 카드와 같은 자리(제목 줄 우측 끝)·같은
            순서(즐겨찾기 별 -> 이미지 저장)·같은 컴포넌트를 쓴다. */}
        <div className="heatmapHeaderRowActions">
          {onToggleFavorite && (
            <FavoriteStarButton
              favorited={favorited}
              disabled={favoritePending}
              onClick={() => onToggleFavorite(sortMode)}
            />
          )}
          <ChartExportButton
            onClick={() => void handleExport()}
            busy={exporting}
            title="이미지로 저장 (PNG, 전체 행)"
          />
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
                scaleMax={scaleMax}
                theme={theme}
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

      <ColorScaleLegend scaleMax={scaleMax} theme={theme} />
      <HeatmapLegendCaption excludedConfigs={data.excluded_configs} />

      {exporting && (
        // 화면 밖(왼쪽 -99999px)에 전체 행짜리 사본을 띄운다. display:none이
        // 아니라 실제로 레이아웃되는 노드라야 getComputedStyle이 값을
        // 돌려주고, chartExport의 색 굳히기 패스가 동작한다.
        //
        // theme을 그대로 흘려보낸다 -- exportNodeAsPng이 캡처 직전
        // <html data-theme>를 light로 뒤집으면 useResolvedTheme()도 같은
        // 프레임에서 "light"로 갱신되므로, 다크 모드에서 저장을 눌러도
        // 이 사본은 항상 라이트 램프로 그려진다(흰 배경에 어울리는 색).
        <div
          aria-hidden="true"
          style={{ position: "absolute", left: -99999, top: 0, width: "max-content", pointerEvents: "none" }}
        >
          <HeatmapExportSurface
            surfaceRef={exportSurfaceRef}
            data={data}
            rows={rows}
            scaleMax={scaleMax}
            theme={theme}
          />
        </div>
      )}

      {tooltip && (
        <div className="heatmapTooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
          <strong>{tooltip.feature} × {tooltip.target}</strong>
          <div className="heatmapTooltipRow">
            <span>Adj R²</span>
            <b>
              {tooltip.adjR2 != null ? tooltip.adjR2.toFixed(3) : "표본 부족"}
              {tooltip.adjR2 != null && tooltip.degree != null && ` (${tooltip.degree}차 적합)`}
            </b>
          </div>
          <div className="heatmapTooltipRow">
            <span>n</span>
            <b>
              {tooltip.n.toLocaleString()}
              {tooltip.measurementRatePct != null && ` · 계측률 ${tooltip.measurementRatePct.toFixed(1)}%`}
            </b>
          </div>
          {tooltip.degree === 2 && tooltip.optimalCenter != null && (
            <div className="heatmapTooltipRow"><span>꼭짓점</span><b>{tooltip.optimalCenter.toFixed(1)}</b></div>
          )}
          {tooltip.degree === 1 && tooltip.shape != null && DIRECTION_TEXT[tooltip.shape] && (
            <div className="heatmapTooltipRow"><span>방향</span><b>{DIRECTION_TEXT[tooltip.shape]}</b></div>
          )}
          <div className="heatmapTooltipRow"><span>q</span><b>{formatQValue(tooltip.q)}</b></div>
          <div className="heatmapTooltipRow"><span>신뢰도</span><b>{tooltip.tier ? TIER_LABEL[tooltip.tier] : "-"}</b></div>
          <div className="heatmapTooltipRow"><span>FDR 통과</span><b>{tooltip.significant ? "예" : "아니오"}</b></div>
          {tooltip.gateExcluded && (
            <div className="heatmapTooltipRow"><span>유의 인자</span><b className="paretoUnderSampledLabel">표본 게이트 미달로 제외</b></div>
          )}
        </div>
      )}
    </section>
  );
}

/** 색 농도 범례(Adjusted R² 척도). 화면 카드와 내보내기 사본이 같은 것을
 * 쓴다 -- 내보낸 이미지에서 범례가 잘리면 셀 색을 읽을 수 없다. */
function ColorScaleLegend({
  scaleMax,
  theme,
  labelled = false,
}: {
  scaleMax: number;
  theme: "light" | "dark";
  labelled?: boolean;
}) {
  return (
    <div className="heatmapColorbar">
      {/* 화면에서는 바로 아래 캡션이 척도를 설명하므로 라벨을 생략하고,
          캡션과 떨어져 읽힐 수 있는 내보내기 사본에만 붙인다. */}
      {labelled && <span>Adjusted R²</span>}
      <span>0.00</span>
      {/* 셀과 같은 gradientCss()로 그린다 -- 하드코딩된 미리보기 색이
          아니라 실제 셀 색 함수 그대로라 늘 일치한다. 다크 모드에서는
          다크 램프(패널 배경~빨강)를, 라이트에서는 라이트 램프(흰색~빨강)를
          보여준다. */}
      <div className="heatmapColorbarTrack" style={{ background: gradientCss(theme) }} />
      <span>{scaleMax.toFixed(2)}</span>
    </div>
  );
}

/** `maxWidth`는 내보내기 사본 전용이다 -- 그 카드는 폭이 `max-content`라,
 * 캡을 씌우지 않으면 이 긴 문장이 한 줄로 펴지면서 카드(=이미지) 폭을
 * 2,000px 넘게 끌고 간다. */
function HeatmapLegendCaption({ excludedConfigs, maxWidth }: { excludedConfigs: number; maxWidth?: number }) {
  return (
    <p className="heatmapCaption" style={maxWidth != null ? { maxWidth } : undefined}>
      셀 숫자·농도는 Adjusted R²(설명력)이며, 우측 상단 배지는 적합 차수(1차/2차)입니다. 방향은 색이 아니라 툴팁의 꼭짓점·증감으로 읽습니다.
      {excludedConfigs > 0 && ` Config ${excludedConfigs}개 제외.`}
      {" "}표본이 30개 미만인 셀은 사선 패턴으로 표시됩니다. 색은 있지만 점선 테두리인 셀은 표본 게이트로 유의 인자 목록에서 빠진 셀입니다.
    </p>
  );
}

function noopHover(): void {}
function noopSelect(): void {}

/** 내보내기 전용 사본 -- 화면 카드와 같은 클래스를 쓰되
 *  1) 행을 자르지 않고(rows 전체),
 *  2) 가로 스크롤 대신 내용만큼 늘어나며(overflow: visible + max-content),
 *  3) 인자명 열을 max-content로 둬 라벨이 잘리지 않게 한다.
 * 제목 / 타깃 열 머리글(Y1~Y5) / 인자명 / 셀 값 + 차수 배지 / 색 농도
 * 범례 / 해석 캡션이 모두 한 이미지에 들어간다. */
function HeatmapExportSurface({
  surfaceRef,
  data,
  rows,
  scaleMax,
  theme,
}: {
  surfaceRef: RefObject<HTMLDivElement | null>;
  data: HeatmapResponse;
  rows: number[];
  scaleMax: number;
  theme: "light" | "dark";
}) {
  const gridTemplateColumns = `max-content repeat(${data.targets.length}, minmax(72px, 1fr))`;
  return (
    <div ref={surfaceRef} className="resultCard heatmapCard" style={{ width: "max-content", maxWidth: "none" }}>
      <div className="heatmapHeaderRow">
        <div className="heatmapHeaderRowText">
          <span className="sectionLabel">CORRELATION OVERVIEW</span>
          <h2>{VIEW_TITLE}</h2>
        </div>
      </div>
      <div className="heatmapScrollArea" style={{ overflow: "visible" }}>
        <div className="heatmapGrid" style={{ gridTemplateColumns }}>
          <div className="heatmapCornerCell heatmapColHeader" />
          {data.targets.map((target) => (
            <div key={target} className="heatmapColHeader">{target}</div>
          ))}
          {rows.map((rowIndex) => (
            <FragmentRow
              key={data.features[rowIndex]}
              feature={data.features[rowIndex]}
              rowIndex={rowIndex}
              data={data}
              scaleMax={scaleMax}
              theme={theme}
              onHover={noopHover}
              onSelectCell={noopSelect}
            />
          ))}
        </div>
      </div>
      <ColorScaleLegend scaleMax={scaleMax} theme={theme} labelled />
      <HeatmapLegendCaption excludedConfigs={data.excluded_configs} maxWidth={EXPORT_CAPTION_MAX_WIDTH} />
    </div>
  );
}

function FragmentRow({
  feature,
  rowIndex,
  data,
  scaleMax,
  theme,
  onHover,
  onSelectCell,
}: {
  feature: string;
  rowIndex: number;
  data: HeatmapResponse;
  scaleMax: number;
  theme: "light" | "dark";
  onHover: (tooltip: TooltipState | null) => void;
  onSelectCell: (selection: HeatmapCellSelection) => void;
}) {
  const totalRows = data.total_rows ?? 0;
  return (
    <>
      <div className="heatmapRowLabel">{feature}</div>
      {data.targets.map((target, colIndex) => {
        const adjR2Value = data.values[rowIndex][colIndex];
        const degree = data.degree?.[rowIndex]?.[colIndex] ?? null;
        const shape = data.shape?.[rowIndex]?.[colIndex] ?? null;
        const optimalCenter = data.optimal_center?.[rowIndex]?.[colIndex] ?? null;
        const n = data.n[rowIndex][colIndex];
        const q = data.q[rowIndex][colIndex];
        const significant = data.significant[rowIndex][colIndex];
        const tier = data.tier[rowIndex][colIndex];
        const masked = adjR2Value == null;
        // 값은 그려지는데(n>=30) 종류별 표본 게이트(R>=100/D>=40)
        // 미달로 유의 인자 목록에는 없는 셀 -- 히트맵과 유의 인자 판정이
        // 어긋나 보이지 않도록 별도 표시한다.
        const gateExcluded = !masked && Boolean(data.gate_excluded?.[rowIndex]?.[colIndex]);
        const style: React.CSSProperties = {};
        if (!masked) {
          const rgb = cellBackgroundRgb(adjR2Value, scaleMax, theme);
          style.background = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          style.color = cellTextColor(adjR2Value, rgb, theme);
        }
        const hoverPayload = {
          feature,
          target,
          adjR2: adjR2Value,
          degree,
          shape,
          optimalCenter,
          n,
          measurementRatePct: totalRows > 0 ? (n / totalRows) * 100 : null,
          q,
          significant,
          tier,
          gateExcluded,
        };
        return (
          <button
            key={target}
            type="button"
            // 유의 여부는 셀 클래스로 표시하지 않는다 -- 필터("유의 인자만
            // 보기" 체크박스)는 data.significant 배열을 직접 읽으므로
            // 클래스 없이도 동작한다.
            className={`heatmapCell ${masked ? "masked" : ""} ${gateExcluded ? "gate-excluded" : ""}`}
            style={style}
            onMouseEnter={(event) => onHover({ x: event.clientX, y: event.clientY, ...hoverPayload })}
            onMouseMove={(event) => onHover({ x: event.clientX, y: event.clientY, ...hoverPayload })}
            onMouseLeave={() => onHover(null)}
            onTouchStart={(event) =>
              onHover({ x: event.touches[0]?.clientX ?? 0, y: event.touches[0]?.clientY ?? 0, ...hoverPayload })
            }
            onClick={() => onSelectCell({ target, feature, significant, qValue: q })}
            aria-label={`${feature}, ${target}, Adjusted R 제곱 ${adjR2Value != null ? formatAdjR2(adjR2Value) : "표본 부족"}${degree != null ? `, ${degree}차 적합` : ""}${gateExcluded ? ", 표본 게이트로 유의 인자 목록에서 제외" : ""}`}
          >
            {masked ? "" : formatAdjR2(adjR2Value)}
            {!masked && degree != null && <span className="heatmapCellDegree" aria-hidden="true">{degree}</span>}
          </button>
        );
      })}
    </>
  );
}
