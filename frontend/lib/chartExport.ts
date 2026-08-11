// 지시서 "차트 이미지 저장" -- 파레토·산점도·박스플롯·Config 박스플롯 네
// 차트 카드가 이 한 유틸을 공유한다. 카드 루트 DOM 노드를 통째로
// html-to-image로 캡처한다 -- SVG 하나만 굽던 이전 방식은 해석 배너, 범례,
// 축 라벨처럼 SVG 밖에 있는 HTML을 모두 놓쳤다.
import { toBlob } from "html-to-image";

// 등폭 서체가 없는 환경(다른 OS 등)에서도 캡션 정렬이 깨지지 않도록, 캡션
// 밴드에는 이 안전한 스택을 강제로 박아 넣는다.
const EXPORT_FONT_STACK = 'ui-monospace, "SF Mono", Consolas, monospace';
// 선명도를 위한 최소 배율 -- 실제 devicePixelRatio가 더 크면 그 값을 쓴다.
const MIN_EXPORT_SCALE = 2;

// 캡처에서 제외할 인터랙션 전용 요소 -- 툴팁(마우스 위치에 종속), 저장
// 버튼 자신, 즐겨찾기 별, 산점도 그룹 강조 슬라이더는 정적 이미지에
// 의미가 없거나(저장 버튼) 캡처 시점의 우연한 마우스 위치를 그대로
// 굽는다(툴팁).
const EXPORT_EXCLUDE_SELECTORS = [".heatmapTooltip", ".chartExportButton", ".favoriteStarButton", ".scatterScrubRow"];

function isTransparent(color: string): boolean {
  return !color || color === "transparent" || /rgba?\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)/.test(color);
}

// 카드 루트가 배경을 투명하게 두는 경우(테마 변수만 상위에서 상속) --
// --surface-card 토큰의 계산값으로 대신한다. 하드코딩된 흰색을 쓰지
// 않아야 다크모드에서 배경이 뒤집히지 않는다.
function resolveBackgroundColor(node: HTMLElement): string {
  const own = getComputedStyle(node).backgroundColor;
  if (!isTransparent(own)) return own;
  const token = getComputedStyle(node).getPropertyValue("--surface-card").trim();
  return token || "#ffffff";
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

/** 카드 루트 DOM 노드 전체를 PNG로 내보낸다. 배경은 카드 자신의 계산된
 * 배경색(또는 투명이면 --surface-card 토큰)을 쓴다 -- 라이트/다크 모두
 * 정상적인 대비를 유지한다. 캡처 직전에 카드 하단에 캡션 줄을 임시 DOM으로
 * 붙였다가 캡처가 끝나면 곧바로 떼어낸다(캔버스 fillText는 폭 측정이 없어
 * 긴 캡션이 잘리므로, 레이아웃이 알아서 줄바꿈하는 실제 DOM을 쓴다). */
export async function exportNodeAsPng(
  node: HTMLElement,
  opts: { filename: string; captionText: string },
): Promise<void> {
  const backgroundColor = resolveBackgroundColor(node);
  const captionColor = resolveCaptionColor(node);
  const dpr = typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
  const pixelRatio = Math.max(dpr, MIN_EXPORT_SCALE);

  const caption = document.createElement("div");
  caption.textContent = opts.captionText;
  caption.style.cssText = [
    "margin-top:10px",
    "padding-top:8px",
    "border-top:1px solid rgba(128,128,128,0.25)",
    `font:12px ${EXPORT_FONT_STACK}`,
    `color:${captionColor}`,
  ].join(";");
  node.appendChild(caption);

  try {
    const blob = await toBlob(node, {
      pixelRatio,
      backgroundColor,
      filter: (el) => {
        if (!(el instanceof Element)) return true;
        return !EXPORT_EXCLUDE_SELECTORS.some((selector) => el.matches?.(selector));
      },
    });
    if (!blob) throw new Error("PNG 인코딩에 실패했습니다.");
    triggerDownload(blob, opts.filename);
  } finally {
    caption.remove();
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

/** `{인자}_{타깃}_{뷰}_{YYYYMMDD-HHmm}.png` (지시서 WI-4). 파레토
 * 카드처럼 단일 인자가 없는 경우(`feature: null`) "Pareto"를 그 자리에
 * 대신 쓴다 -- 예) `Pareto_Y1_pareto_20260810-2256.png`. */
export function buildExportFilename(opts: {
  feature: string | null;
  target: string;
  view: "pareto" | "scatter" | "box";
}): string {
  const stamp = formatExportTimestamp(new Date());
  const featurePart = opts.feature ?? "Pareto";
  return `${featurePart}_${opts.target}_${opts.view}_${stamp}.png`;
}

/** 인자 산점도/박스플롯 카드의 캡션 한 줄 (지시서 WI-4 예시:
 * `Step28_R1 vs Y1 · ε² 0.212 · n=1,492 · test.CSV · 2026-08-10`). */
export function buildFactorCaptionText(opts: {
  feature: string;
  target: string;
  eps2: number;
  n: number;
  datasetId: string;
}): string {
  const dateStr = formatCaptionDate(new Date());
  return `${opts.feature} vs ${opts.target} · ε² ${opts.eps2.toFixed(3)} · n=${opts.n.toLocaleString()} · ${opts.datasetId} · ${dateStr}`;
}

/** 파레토 카드의 캡션 한 줄 -- 단일 ε²/n이 없으므로(타깃 전체 순위 차트)
 * 인자 개수로 대신한다. */
export function buildParetoCaptionText(opts: { target: string; factorCount: number; datasetId: string }): string {
  const dateStr = formatCaptionDate(new Date());
  return `R/D/Config vs ${opts.target} · 인자 ${opts.factorCount}개 · ${opts.datasetId} · ${dateStr}`;
}
