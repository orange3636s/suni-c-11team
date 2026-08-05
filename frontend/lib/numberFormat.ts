/** Single source of truth for q-value/p-value display: 4 decimal places,
 * fixed notation always (never exponential) -- `toFixed` never switches
 * to exponential notation the way `toExponential`/`toPrecision` can, so
 * a very small value (e.g. 1.06e-104) reads as "0.0000" instead of an
 * exponent a non-statistician can't parse at a glance.
 */
function formatFixed4(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(4);
}

export function formatQValue(value: number | null | undefined): string {
  return formatFixed4(value);
}

export function formatPValue(value: number | null | undefined): string {
  return formatFixed4(value);
}
