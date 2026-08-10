// 지시서 "원인 분석 화면 정리" WI-4 -- 파레토·산점도·박스플롯 세 차트
// 카드가 이 한 유틸을 공유한다(같은 요구사항을 화면마다 따로 만들지
// 않는다). 전부 순수 프런트 렌더링: 이미 그려진 SVG를 그대로 PNG로
// 굽는다.

// 등폭 서체가 없는 환경(다른 OS 등)에서도 정렬이 깨지지 않도록, 내보낼
// SVG에는 이 안전한 스택을 강제로 박아 넣는다 -- base64 폰트 임베드까지는
// 하지 않는다(실용적으로 이 정도면 충분하다는 것이 지시서의 판단).
const EXPORT_FONT_STACK = 'ui-monospace, "SF Mono", Consolas, monospace';
// 차트 아래에 메타 한 줄(캡션)을 그릴 여백.
const CAPTION_BAND_HEIGHT = 24;
// 선명도를 위한 최소 배율 -- 실제 devicePixelRatio가 더 크면 그 값을 쓴다.
const MIN_EXPORT_SCALE = 2;

// 차트 렌더링에 실제로 쓰이는 속성만 굽는다 -- 나머지(margin, position 등
// 레이아웃 속성)는 SVG 좌표 자체에 이미 반영돼 있어 style로 옮길 필요가
// 없다.
const STYLE_PROPERTIES = [
  "fill",
  "stroke",
  "stroke-width",
  "color",
  "font-family",
  "font-size",
  "font-weight",
  "opacity",
] as const;

/** 클론 트리를 원본과 나란히 순회하며, 각 노드에 원본의 계산된 스타일을
 * 인라인 style 속성으로 굽는다. CSS 변수(예: var(--tier-strong))는
 * 문서에서 분리된 SVG 단독으로는 해석되지 않으므로, 직렬화 전에 이미
 * 해석된(resolved) 값을 심어야 한다 -- 안 그러면 색이 통째로 빠진다. */
function inlineComputedStyles(original: Element, clone: Element): void {
  const computed = window.getComputedStyle(original);
  const declarations: string[] = [];
  for (const prop of STYLE_PROPERTIES) {
    const value = computed.getPropertyValue(prop);
    if (value) declarations.push(`${prop}:${value}`);
  }
  // 안전한 폰트 스택으로 덮어쓴다(계산된 font-family 뒤에 와서 우선한다).
  declarations.push(`font-family:${EXPORT_FONT_STACK}`);
  clone.setAttribute("style", declarations.join(";"));

  const originalChildren = Array.from(original.children);
  const cloneChildren = Array.from(clone.children);
  originalChildren.forEach((child, index) => {
    const clonedChild = cloneChildren[index];
    if (clonedChild) inlineComputedStyles(child, clonedChild);
  });
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("SVG를 이미지로 변환하지 못했습니다."));
    image.src = url;
  });
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

/** SVG 차트 하나를 PNG로 내보낸다 (지시서 WI-4). 배경은 항상 불투명
 * 흰색으로 채운다 -- 투명 PNG는 다크 배경 문서에서 안 보인다. 하단에는
 * 캡션 밴드를 덧붙여 메타(인자·타깃·통계량·데이터셋·날짜)를 한 줄로
 * 굽는다. */
export async function exportChartAsPng(
  svgElement: SVGSVGElement,
  opts: { filename: string; captionText: string },
): Promise<void> {
  const rect = svgElement.getBoundingClientRect();
  const width = Math.max(Math.round(rect.width || Number(svgElement.getAttribute("width")) || 640), 1);
  const height = Math.max(Math.round(rect.height || Number(svgElement.getAttribute("height")) || 360), 1);

  // 클론에 원본 계산 스타일을 인라인으로 구운 뒤 직렬화한다 -- 문서에
  // 붙어 있는 원본과 달리 클론은 <img>로 옮겨지는 순간 CSS 문맥을 잃는다.
  const clone = svgElement.cloneNode(true) as SVGSVGElement;
  inlineComputedStyles(svgElement, clone);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  if (!clone.getAttribute("viewBox")) clone.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const svgString = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  try {
    const image = await loadImage(svgUrl);
    const dpr = typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
    const scale = Math.max(dpr, MIN_EXPORT_SCALE);

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round((height + CAPTION_BAND_HEIGHT) * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("캔버스 컨텍스트를 생성하지 못했습니다.");
    ctx.scale(scale, scale);

    // 배경을 흰색으로 채운다(하지 말 것: 투명 배경).
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height + CAPTION_BAND_HEIGHT);
    ctx.drawImage(image, 0, 0, width, height);

    ctx.fillStyle = "#333333";
    ctx.font = `12px ${EXPORT_FONT_STACK}`;
    ctx.textBaseline = "alphabetic";
    ctx.fillText(opts.captionText, 8, height + CAPTION_BAND_HEIGHT - 8);

    const pngBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!pngBlob) throw new Error("PNG 인코딩에 실패했습니다.");
    triggerDownload(pngBlob, opts.filename);
  } finally {
    URL.revokeObjectURL(svgUrl);
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
