/** Heckbert's "nice numbers" step selection -- ticks land on 1/2/5 x 10^n
 * step multiples (1/2/5/10/20/50...) instead of naive even division of
 * the domain, so labels read as round numbers. */
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

/** Picks a tick set whose *actual* count lands in [minCount, maxCount]
 * and whose labels don't overlap across `pixelSpan` -- or gets as close
 * to that as the 1/2/5/10 step grid allows.
 *
 * `niceTicks(domain, requestedCount)` doesn't reliably return
 * `requestedCount` ticks -- the step it rounds to comes off a coarse
 * {1,2,5,10}x10^n grid, so asking for e.g. 8 ticks across a span of 4.3
 * can round to step=1 and yield only 4 (well under the 6-8 target this
 * was meant to hit). Requesting a single fixed count is therefore not
 * enough; this searches a padded band of *requested* counts, collects
 * whichever ones land their *actual* length inside [minCount, maxCount],
 * and from those prefers the densest one that doesn't overlap. If none
 * land in-band at all (the grid can skip straight from 6 to 11 ticks for
 * some spans), it searches every candidate instead so overlap-safety
 * still wins over hitting the band exactly ("겹치면 한 단계 줄인다").
 * `measureLabelSize` decides what "overlap" means along this axis: label
 * *width* for a horizontal axis with centered text, or a fixed label
 * *height* for a vertical axis of stacked horizontal text -- the caller
 * picks, since the two axes collide on different dimensions. Density
 * naturally shrinks as `pixelSpan` shrinks (panel open, mobile) since
 * the same labels then need a sparser candidate to stop colliding. */
export function niceTicksFitted(
  domain: readonly [number, number],
  maxCount: number,
  minCount: number,
  pixelSpan: number,
  formatFn: (value: number) => string,
  measureLabelSize: (label: string) => number,
  minGapPx = 6,
): number[] {
  const candidates: number[][] = [];
  for (let requested = maxCount + 4; requested >= 2; requested -= 1) {
    candidates.push(niceTicks(domain, requested));
  }
  const inBand = candidates.filter((ticks) => ticks.length >= minCount && ticks.length <= maxCount);
  const pool = inBand.length > 0 ? inBand : candidates;
  const byDensityDesc = [...pool].sort((a, b) => b.length - a.length);
  let fallback = byDensityDesc[byDensityDesc.length - 1] ?? niceTicks(domain, minCount);
  for (const ticks of byDensityDesc) {
    if (ticks.length < 2) {
      fallback = ticks;
      continue;
    }
    const spacingPx = pixelSpan / (ticks.length - 1);
    const maxLabelSize = Math.max(...ticks.map((tick) => measureLabelSize(formatFn(tick))));
    if (maxLabelSize + minGapPx <= spacingPx) return ticks;
    fallback = ticks;
  }
  return fallback;
}
