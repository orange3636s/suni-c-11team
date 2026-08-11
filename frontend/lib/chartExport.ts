// 차트 이미지 저장 -- 파레토·산점도·박스플롯·상관 히트맵 네 차트 카드가
// 이 한 유틸을 공유한다. 캡처 대상은 `<svg>`가 아니라 카드 루트 DOM 노드
// 전체다: 해석 배너, 범례, 축 라벨은 SVG 밖의 HTML이라 SVG만 구우면
// 이미지에서 통째로 빠진다.
import { toBlob } from "html-to-image";

// 등폭 서체가 없는 환경(다른 OS 등)에서도 캡션·눈금 숫자 정렬이 깨지지
// 않도록, 캡션 밴드와 SVG 텍스트에 이 안전한 스택을 강제로 박아 넣는다.
// 화면 쪽 --font-data와 첫 후보('JetBrains Mono')를 맞춰 두어야 내보낸
// 이미지와 화면의 자간이 어긋나지 않는다.
const EXPORT_FONT_STACK = `'JetBrains Mono', ui-monospace, "SF Mono", Consolas, monospace`;
// 선명도를 위한 최소 배율 -- 실제 devicePixelRatio가 더 크면 그 값을 쓴다.
const MIN_EXPORT_SCALE = 2;
// 내보낸 PNG는 문서/슬라이드에 붙는 그림이다 -- 화면 테마가 무엇이든
// 항상 흰 바탕으로 굽는다(다크모드 카드 배경을 그대로 구우면 문서에
// 붙였을 때 검은 사각형이 된다).
const EXPORT_BACKGROUND_COLOR = "#ffffff";
// 이 크기를 넘으면 콘솔에 경고만 남긴다(다운로드는 그대로 진행한다).
const EXPORT_BLOB_WARN_BYTES = 5 * 1024 * 1024;

// 캡처에서 제외할 인터랙션 전용 요소 -- 툴팁(마우스 위치에 종속), 저장
// 버튼 자신, 즐겨찾기 별, 산점도 그룹 강조 슬라이더는 정적 이미지에
// 의미가 없거나(저장 버튼) 캡처 시점의 우연한 마우스 위치를 그대로
// 굽는다(툴팁).
const EXPORT_EXCLUDE_SELECTORS = [".heatmapTooltip", ".chartExportButton", ".favoriteStarButton", ".scatterScrubRow"];

/* ------------------------------------------------------------------ *
 * SVG 색 유실 방지 (핵심 버그)
 *
 * html-to-image(1.11.x)의 clone-node는 `<svg>`를 만나면 `cloneNode(true)`로
 * 통째로 깊은 복사를 하고 곧바로 `cloneChildren`에서 빠져나온다. 그 결과
 * **SVG 자식 요소에는 `cloneCSSStyle`(계산된 스타일을 인라인으로 굽는
 * 단계)이 단 한 번도 적용되지 않는다** -- 인라인 style과 presentation
 * attribute만 복제된다.
 *
 * 그런데 복제본은 `<foreignObject>` 안에 담겨 독립된 SVG 문서로 직렬화된
 * 뒤 이미지로 래스터화된다. 그 문서에는 페이지의 스타일시트도, `:root`에
 * 정의된 CSS 커스텀 프로퍼티도 존재하지 않는다. 따라서
 *   - 클래스 규칙으로 칠하던 SVG 요소(.scatterPlotBg / .paretoBar.tier-* /
 *     .scatterGridLine / .scatterRecommendedBand{opacity} / .paretoThresholdLine
 *     {stroke-dasharray} 등)는 선언 자체가 사라져 SVG 기본값으로 떨어지고
 *     (fill 기본값 = 검정, opacity 기본값 = 1),
 *   - `var(--token)` / `color-mix(... var(--token) ...)`도 해석되지 않는다.
 * 반면 `fill={hex}`처럼 React prop -> presentation attribute로 나간 색은
 * 깊은 복사로 살아남는다. "빨간 추세선/기준선만 살아남고 배경은 검게,
 * 권장 구간 띠는 회색 덩어리로" 나오던 증상이 정확히 이 조합이다.
 * (범례 스와치는 SVG가 아닌 순수 HTML이라 정상 -- 이 점이 결정적 단서다.)
 *
 * 대응: 캡처 직전에 **살아 있는 DOM**의 SVG 자손을 훑어 getComputedStyle로
 * 이미 해석이 끝난 구체값을 인라인 style로 박아 넣는다(살아 있는 노드라야
 * var()/클래스 규칙이 모두 해석된다). 캡처가 끝나면 style 속성 원문을
 * 그대로 되돌려 화면은 캡처 전과 100% 동일하게 남긴다.
 * ------------------------------------------------------------------ */
const FROZEN_SVG_PROPERTIES = [
  "fill",
  "stroke",
  "stop-color",
  "flood-color",
  "color",
  "background-color",
  "stroke-width",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "opacity",
  "fill-opacity",
  "stroke-opacity",
  "font-family",
  "font-size",
  "font-style",
  "font-weight",
  "letter-spacing",
  "text-anchor",
  "dominant-baseline",
];

// 글꼴 스택을 강제로 얹을 SVG 텍스트 요소.
const SVG_TEXT_TAGS = new Set(["text", "tspan", "textPath"]);

type StyleSnapshot = { element: SVGElement; style: string | null };

/** 등폭 계열이면 통째로 내보내기용 스택으로 갈아끼우고(눈금 숫자 정렬이
 * 목적), 아니면(축 제목 등 UI 서체) 원래 스택 뒤에 폴백으로만 덧붙인다 --
 * 화면에서 본문 서체이던 글자가 이미지에서만 등폭으로 바뀌지 않게. */
function exportFontFamily(computedFamily: string): string {
  if (!computedFamily) return EXPORT_FONT_STACK;
  if (/mono|consolas|menlo|courier/i.test(computedFamily)) return EXPORT_FONT_STACK;
  return `${computedFamily}, ${EXPORT_FONT_STACK}`;
}

/** 살아 있는 노드 안의 모든 SVG 요소(루트 `<svg>` 포함)에 계산된 스타일을
 * 인라인으로 굽는다. 되돌리기용 스냅숏(style 속성 원문)을 반환한다. */
function freezeSvgStyles(node: HTMLElement): StyleSnapshot[] {
  if (typeof window === "undefined") return [];
  const snapshots: StyleSnapshot[] = [];
  node.querySelectorAll<Element>("svg, svg *").forEach((element) => {
    // `svg *`는 <foreignObject> 안의 HTML까지 걸린다 -- 그쪽은
    // html-to-image가 이미 정상 처리하므로 SVG 네임스페이스만 손댄다.
    if (!(element instanceof SVGElement)) return;
    const inline = element.style;
    if (!inline) return;
    snapshots.push({ element, style: element.getAttribute("style") });
    const computed = window.getComputedStyle(element);
    for (const property of FROZEN_SVG_PROPERTIES) {
      const value = computed.getPropertyValue(property);
      // 빈 값/none은 굳이 덮어쓰지 않는다(none은 의미가 있으므로 그대로 둔다).
      if (!value) continue;
      inline.setProperty(property, value);
    }
    if (SVG_TEXT_TAGS.has(element.tagName)) {
      inline.setProperty("font-family", exportFontFamily(computed.getPropertyValue("font-family")));
    }
  });
  return snapshots;
}

/** style 속성 원문을 정확히 복원한다 -- 원래 style이 없던 요소는 속성
 * 자체를 지운다(빈 style="" 이 남으면 화면이 같아 보여도 DOM이 달라진다). */
function restoreSvgStyles(snapshots: StyleSnapshot[]): void {
  for (const { element, style } of snapshots) {
    if (style == null) element.removeAttribute("style");
    else element.setAttribute("style", style);
  }
}

/* ------------------------------------------------------------------ *
 * 내보내기 동안만 라이트 테마 강제
 *
 * ThemeProvider는 사용자의 선택(localStorage "dashboard-theme")을 React
 * state로 들고 있고, DOM에는 `<html data-theme>`를 파생시켜 찍기만 한다.
 * 그리고 차트들이 쓰는 useResolvedTheme()은 그 속성 하나만 MutationObserver로
 * 읽는다. 따라서 이 속성만 잠깐 뒤집었다가 되돌리면
 *   - CSS 변수(html[data-theme="dark"] 규칙 전부)와
 *   - JS로 분기하던 하드코딩 hex 상수(theme === "dark" ? ... : ...)
 * 두 경로가 **동시에** 라이트값으로 맞춰진다. 사용자의 실제 테마 설정
 * (state·localStorage)은 전혀 건드리지 않는다.
 * ------------------------------------------------------------------ */
type ThemeOverride = { changed: boolean; restore: () => void };

function forceLightThemeForExport(): ThemeOverride {
  const noop: ThemeOverride = { changed: false, restore: () => {} };
  if (typeof document === "undefined") return noop;
  const root = document.documentElement;
  if (root.dataset.theme !== "dark") return noop;
  const previousTheme = root.dataset.theme;
  const previousColorScheme = root.style.colorScheme;
  root.dataset.theme = "light";
  root.style.colorScheme = "light";
  return {
    changed: true,
    restore() {
      root.dataset.theme = previousTheme;
      root.style.colorScheme = previousColorScheme;
    },
  };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame !== "function") {
      window.setTimeout(() => resolve(), 16);
      return;
    }
    requestAnimationFrame(() => resolve());
  });
}

/** data-theme 변경 -> MutationObserver(마이크로태스크) -> React 리렌더
 * (스케줄러 매크로태스크) -> 페인트 순서를 모두 통과할 때까지 기다린다. */
async function waitForThemeRepaint(): Promise<void> {
  await nextFrame();
  await new Promise<void>((resolve) => window.setTimeout(() => resolve(), 0));
  await nextFrame();
}

function resolveCaptionColor(node: HTMLElement): string {
  const token = getComputedStyle(node).getPropertyValue("--text-secondary").trim();
  return token || "#666666";
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/** 카드 루트 DOM 노드 전체를 PNG로 내보낸다.
 *
 * - 배경은 테마와 무관하게 항상 흰색이다(문서/슬라이드용 그림).
 * - 다크모드에서 눌렀다면 캡처 동안만 `<html data-theme>`를 light로 뒤집어
 *   글자·격자선이 흰 바탕에서 읽히는 색으로 렌더된 뒤 캡처된다.
 * - 캡처 직전 SVG 자손의 계산된 색/선/글꼴을 인라인 style로 굳혀
 *   html-to-image의 SVG 복제 단계에서 클래스·var() 기반 색이 날아가는
 *   문제를 막는다(위 주석 참고).
 * - 캡션은 캔버스 fillText가 아니라 임시 DOM으로 붙였다 뗀다(폭 측정이
 *   없는 fillText는 긴 캡션을 그냥 잘라먹는다).
 *
 * 성공 시 만들어진 Blob을 돌려준다(호출부가 크기를 확인할 수 있게). */
export async function exportNodeAsPng(
  node: HTMLElement,
  opts: { filename: string; captionText: string },
): Promise<Blob> {
  const dpr = typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
  const pixelRatio = Math.max(dpr, MIN_EXPORT_SCALE);

  const themeOverride = forceLightThemeForExport();
  let caption: HTMLDivElement | null = null;
  let frozen: StyleSnapshot[] = [];

  try {
    if (themeOverride.changed) await waitForThemeRepaint();

    caption = document.createElement("div");
    caption.textContent = opts.captionText;
    caption.style.cssText = [
      "margin-top:10px",
      "padding-top:8px",
      "border-top:1px solid rgba(128,128,128,0.25)",
      `font:12px ${EXPORT_FONT_STACK}`,
      `color:${resolveCaptionColor(node)}`,
    ].join(";");
    node.appendChild(caption);

    // 캡션까지 붙인 뒤에 굳힌다 -- 순서가 바뀌어도 무방하지만(캡션은 HTML),
    // "캡처 직전 상태 그대로"를 굳힌다는 규칙을 지킨다.
    frozen = freezeSvgStyles(node);

    const blob = await toBlob(node, {
      pixelRatio,
      backgroundColor: EXPORT_BACKGROUND_COLOR,
      filter: (el) => {
        if (!(el instanceof Element)) return true;
        return !EXPORT_EXCLUDE_SELECTORS.some((selector) => el.matches?.(selector));
      },
    });
    if (!blob) throw new Error("PNG 인코딩에 실패했습니다.");
    if (blob.size > EXPORT_BLOB_WARN_BYTES) {
      console.warn(
        `내보낸 PNG가 ${(blob.size / (1024 * 1024)).toFixed(1)}MB로 5MB를 넘습니다 (${opts.filename}). ` +
          "행 수를 줄이거나 배율을 낮추는 것을 검토하세요.",
      );
    }
    triggerDownload(blob, opts.filename);
    return blob;
  } finally {
    // 복원 순서가 중요하다: 인라인 style을 먼저 되돌린 뒤 테마를 되돌려야
    // React가 테마 리렌더에서 자기 style prop을 다시 쓰는 시점에 우리가
    // 덧칠한 값이 남아 있지 않다.
    restoreSvgStyles(frozen);
    caption?.remove();
    themeOverride.restore();
  }
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

/** `YYYYMMDD-HHmm` -- 클릭 시점의 로컬 시각(24h). */
function formatExportTimestamp(date: Date): string {
  return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}-${pad2(date.getHours())}${pad2(date.getMinutes())}`;
}

/** `YYYY-MM-DD` -- 캡션에 쓰는 ISO 형식 날짜(클릭 시점의 로컬 날짜). */
function formatCaptionDate(date: Date): string {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

/** 캡션에 찍는 데이터셋 식별자 -- 32자 UUID를 그대로 넣으면 한 줄 캡션의
 * 절반을 잡아먹으므로 앞 8자만 쓴다. **캡션 전용**이다: API 호출·캐시 키
 * 등 실제 식별에 쓰는 datasetId는 어디서도 자르지 않는다. */
function shortDatasetId(datasetId: string): string {
  return datasetId.slice(0, 8);
}

/** `{인자}_{타깃}_{뷰}_{YYYYMMDD-HHmm}.png`. 파레토·히트맵
 * 카드처럼 단일 인자가 없는 경우(`feature: null`)에는 뷰에 맞는 이름을 그
 * 자리에 대신 쓴다 -- 예) `Pareto_Y1_pareto_20260810-2256.png`,
 * `Heatmap_all_heatmap_20260812-1130.png`. */
export function buildExportFilename(opts: {
  feature: string | null;
  target: string;
  view: "pareto" | "scatter" | "box" | "heatmap";
}): string {
  const stamp = formatExportTimestamp(new Date());
  const fallbackFeature = opts.view === "heatmap" ? "Heatmap" : "Pareto";
  const featurePart = opts.feature ?? fallbackFeature;
  return `${featurePart}_${opts.target}_${opts.view}_${stamp}.png`;
}

/** 인자 산점도/박스플롯 카드의 캡션 한 줄. 예)
 * `Step28_R1 vs Y1 · Adj R² 0.234 (2차) · n=1,492 · a1b2c3d4 · 2026-08-10`.
 * 화면 메타 줄과 같은 서버 산출 Adjusted R²를 쓴다 -- 이미지로 내보낸
 * 캡션과 화면 숫자가 어긋나면 안 된다. */
export function buildFactorCaptionText(opts: {
  feature: string;
  target: string;
  adjR2: number;
  degree?: number | null;
  n: number;
  datasetId: string;
}): string {
  const dateStr = formatCaptionDate(new Date());
  const degreePart = opts.degree != null ? ` (${opts.degree}차)` : "";
  return `${opts.feature} vs ${opts.target} · Adj R² ${opts.adjR2.toFixed(3)}${degreePart} · n=${opts.n.toLocaleString()} · ${shortDatasetId(opts.datasetId)} · ${dateStr}`;
}

/** 파레토 카드의 캡션 한 줄 -- 단일 설명력/n이 없으므로(타깃 전체 순위
 * 차트) 인자 개수로 대신한다. */
export function buildParetoCaptionText(opts: { target: string; factorCount: number; datasetId: string }): string {
  const dateStr = formatCaptionDate(new Date());
  return `R/D/Config vs ${opts.target} · 인자 ${opts.factorCount}개 · ${shortDatasetId(opts.datasetId)} · ${dateStr}`;
}

/** 상관 히트맵 카드의 캡션 한 줄 -- 단일 (인자, 타깃) 쌍이 아니므로
 * 격자 크기와 현재 정렬 기준으로 대신한다. 행 수는 화면에 보이는 7행이
 * 아니라 **내보낸 이미지에 실제로 들어간 전체 행 수**다. */
export function buildHeatmapCaptionText(opts: {
  rowCount: number;
  columnCount: number;
  sortLabel: string;
  datasetId: string;
}): string {
  const dateStr = formatCaptionDate(new Date());
  return `R, D vs Y1~Y5 · ${opts.rowCount}행 x ${opts.columnCount}열 · 정렬 ${opts.sortLabel} · ${shortDatasetId(opts.datasetId)} · ${dateStr}`;
}
