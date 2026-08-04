/** Heckbert's "nice numbers" step selection -- ticks land on 1/2/5 x 10^n
 * step multiples (spec §8: 1/2/5/10/20/50...) instead of naive even
 * division of the domain, so labels read as round numbers. */
function niceNum(range: number, round: boolean): number {
  if (!Number.isFinite(range) || range <= 0) return 1;
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / 10 ** exponent;
  let niceFraction: number;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  } else {
    niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}

/** Nice-stepped ticks covering `domain`, aiming for roughly `targetCount`
 * ticks (the actual count depends on where nice step boundaries fall). */
export function niceTicks(domain: readonly [number, number], targetCount: number): number[] {
  const [min, max] = domain;
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const range = niceNum(max - min, false);
  const step = niceNum(range / Math.max(targetCount - 1, 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = niceMin; value <= niceMax + step * 0.5; value += step) {
    if (value >= min - step * 0.001 && value <= max + step * 0.001) {
      ticks.push(Math.abs(value) < step * 1e-9 ? 0 : Number(value.toFixed(10)));
    }
  }
  return ticks.length > 0 ? ticks : [min];
}

/** Backs off from `maxCount` toward `minCount`, one nice-step level at a
 * time, until adjacent tick labels no longer overlap across `pixelSpan`
 * -- or `minCount` is reached regardless (spec §8: "겹치면 한 단계
 * 물러난다"). `measureLabelSize` decides what "overlap" means along this
 * axis: label *width* for a horizontal axis with centered text, or a
 * fixed label *height* for a vertical axis of stacked horizontal text --
 * the caller picks, since the two axes collide on different dimensions.
 * Density naturally shrinks as `pixelSpan` shrinks (panel open, mobile)
 * since the same labels then need more backoff steps to stop colliding. */
export function niceTicksFitted(
  domain: readonly [number, number],
  maxCount: number,
  minCount: number,
  pixelSpan: number,
  formatFn: (value: number) => string,
  measureLabelSize: (label: string) => number,
  minGapPx = 6,
): number[] {
  let best = niceTicks(domain, maxCount);
  for (let count = maxCount; count >= minCount; count -= 1) {
    const ticks = niceTicks(domain, count);
    best = ticks;
    if (ticks.length < 2) break;
    const spacingPx = pixelSpan / (ticks.length - 1);
    const maxLabelSize = Math.max(...ticks.map((tick) => measureLabelSize(formatFn(tick))));
    if (maxLabelSize + minGapPx <= spacingPx) break;
  }
  return best;
}
