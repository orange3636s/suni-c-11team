/** Single source of truth for q-value display: 4 decimal places, fixed
 * notation always (never exponential) -- `toFixed` never switches to
 * exponential notation the way `toExponential`/`toPrecision` can, so a
 * very small value (e.g. 1.06e-104) reads as "0.0000" instead of an
 * exponent a non-statistician can't parse at a glance.
 *
 * 화면에는 p-value를 표시하지 않는다 -- 표본이 커지면(1,500장 안팎) 약한
 * 관계도 극단적으로 유의해져 변별력이 없어서, root-cause 페이지는
 * Adjusted R²(적합 차수)·파레토 기여율로 관계 강도를 보여준다. FDR 게이트
 * 자체는 유지되므로 q-value(CorrelationHeatmap의 히트맵 셀 툴팁)는 이
 * 함수로 표시한다.
 */
function formatFixed4(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(4);
}

export function formatQValue(value: number | null | undefined): string {
  return formatFixed4(value);
}

// 자릿수 기준 -- 인자 값·관리한계·권장구간은 소수점 1자리, 불량률·수율은
// 2자리, Adjusted R²는 3자리, 기여율·감소율은 정수+%. 같은 값이 화면마다
// 다른 자릿수로 보이지 않도록 각 화면에서 toFixed()를 직접 부르지 말고
// 이 함수들을 쓴다.
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

export function formatAdjR2(value: number | null | undefined): string {
  return formatFixed(value, 3);
}

export function formatContributionPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${Math.round(value)}%`;
}
