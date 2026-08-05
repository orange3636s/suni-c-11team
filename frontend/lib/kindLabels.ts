/** Single source of truth for how a factor's `kind` ("R" | "D" | "Config")
 * displays on screen. The raw identifier is never changed -- it still
 * flows through the JSON report, internal state, etc. untouched; only
 * the label shown to a person goes through this map.
 */
const KIND_LABEL: Record<string, string> = { R: "Response", D: "Defect", Config: "Eq." };

export function kindLabel(kind: string): string {
  return KIND_LABEL[kind] ?? kind;
}
