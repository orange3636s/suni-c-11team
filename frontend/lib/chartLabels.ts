/** Both backend axis labels are one meaningful token followed by
 * descriptive filler the chart doesn't need to repeat -- the factor's
 * "{feature} (Step N · kind)" and the target's "{target} 불량률 (%)"
 * both reduce to their first space-separated word. Both axis helpers
 * route through this one primitive so the two can't drift apart -- fixing
 * a title on one axis without the other is exactly what a per-file helper
 * invites.
 */
function firstToken(raw: string): string {
  return raw.split(" ")[0];
}

/** Every scatter/box-plot entry point (root-cause main chart, quick-look,
 * Config box plot, alarm deep-links) reads the backend's
 * `axis.x_label` -- built server-side as "{feature} (Step N · kind)" --
 * through this single function instead of each display spot re-deciding
 * how to strip it down to just the factor name.
 */
export function factorAxisLabel(rawXLabel: string): string {
  return firstToken(rawXLabel);
}

/** `axis.y_label` is built server-side as "{target} 불량률 (%)" -- this
 * strips it down to just the target name (e.g. "Y5"). The unit isn't
 * lost: tick labels and hover tooltips already show the percentage
 * value itself, they just don't repeat "불량률 (%)" in the axis title
 * too. Pareto's y-axes (기여율/누적 기여율) are a different, genuinely
 * percentage-native metric -- they don't go through this helper and
 * stay untouched.
 */
export function targetAxisLabel(rawYLabel: string): string {
  return firstToken(rawYLabel);
}
