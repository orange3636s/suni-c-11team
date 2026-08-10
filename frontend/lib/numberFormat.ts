/** Single source of truth for q-value display: 4 decimal places, fixed
 * notation always (never exponential) -- `toFixed` never switches to
 * exponential notation the way `toExponential`/`toPrecision` can, so a
 * very small value (e.g. 1.06e-104) reads as "0.0000" instead of an
 * exponent a non-statistician can't parse at a glance.
 *
 * 지시서 "원인 분석 화면 정리" WI-3: p-value는 여기서 뺐다 -- 표본이
 * 커지면(1,500장 안팎) 약한 관계도 극단적으로 유의해져 변별력이 없다
 * (root-cause 페이지는 ε²·R²(적합 차수)·파레토 기여율로 대체했다). q-value
 * (CorrelationHeatmap의 히트맵 셀 툴팁)는 여전히 이 함수를 쓴다 -- FDR
 * 게이트 자체는 유지되고, 그 결과를 보여주는 자리는 남아 있다.
 */
function formatFixed4(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(4);
}

export function formatQValue(value: number | null | undefined): string {
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
