let measureCanvas: HTMLCanvasElement | null = null;

/** Canvas-based text width measurement -- used by tick-density overlap
 * detection to decide whether adjacent axis labels would collide before
 * actually rendering them. */
export function measureTextWidth(text: string, font: string): number {
  if (typeof document === "undefined") return text.length * 6;
  if (!measureCanvas) measureCanvas = document.createElement("canvas");
  const ctx = measureCanvas.getContext("2d");
  if (!ctx) return text.length * 6;
  ctx.font = font;
  return ctx.measureText(text).width;
}
