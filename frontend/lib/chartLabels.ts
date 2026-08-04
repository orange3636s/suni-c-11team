/** Every scatter/box-plot entry point (root-cause main chart, quick-look,
 * Config box plot, alarm deep-links) reads the backend's
 * `axis.x_label` -- built server-side as "{feature} (Step N · kind)" --
 * through this single function instead of each display spot re-deciding
 * how to strip it down to just the factor name. Previously only the
 * numeric ScatterChart was fixed to drop the parenthetical decomposition;
 * the categorical box-plot's own xaxis title built off the same raw
 * string was missed because it lived in a different file with no shared
 * helper to route through.
 */
export function factorAxisLabel(rawXLabel: string): string {
  return rawXLabel.split(" ")[0];
}
