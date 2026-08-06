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

// 자릿수 기준 (spec 문구 전수 검토 §A-6-2) -- 인자 값·관리한계·권장구간은
// 소수점 1자리, 불량률·수율은 2자리, ε²는 3자리, 기여율·감소율은 정수+%.
// 화면마다 각자 toFixed()를 부르면 반드시 어긋나므로(실측: 히트맵 셀은
// eps2를 2자리로, 다른 모든 곳은 3자리로 표시하고 있었다) 여기 하나로
// 모은다.
function formatFixed(value: number | null | undefined, digits: number): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

export function formatFactorValue(value: number | null | undefined): string {
  return formatFixed(value, 1);
}

export function formatDefectRate(value: number | null | undefined): string {
  return formatFixed(value, 2);
}

export function formatEps2(value: number | null | undefined): string {
  return formatFixed(value, 3);
}

export function formatContributionPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${Math.round(value)}%`;
}
