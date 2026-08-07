import type { ParetoRankingItem } from "@/types/data";

/** 차트 표시 규칙 (spec §B-2) -- 타깃당 몇 개의 인자를 차트로 그릴지 결정한다.
 * `items`는 이미 eps2 내림차순으로 정렬된 top-5 Pareto 결과이며, 각 항목의
 * `cumulative_pct`는 (top-5로 잘리기 전) 전체 후보 풀 기준 누적 기여율이다. */
export function selectDisplayFactors(items: ParetoRankingItem[]): ParetoRankingItem[] {
  if (items.length === 0) return [];

  // 1) Pareto 80% 범위 -- 80%를 넘기는 첫 인자까지 포함한다.
  const cutoffIndex = items.findIndex((item) => item.cumulative_pct >= 80);
  const pareto = cutoffIndex === -1 ? items : items.slice(0, cutoffIndex + 1);

  // 2) 강함/보통만 추출 -- 3개를 넘어도 전부 표시한다.
  const core = pareto.filter((item) => item.confidence_tier === "strong" || item.confidence_tier === "moderate");
  if (core.length >= 3) return core;

  // 3) 3개 미만이면 약함으로 보충한다 (참고 등급은 어떤 경우에도 제외).
  const fill = pareto.filter((item) => item.confidence_tier === "weak").slice(0, 3 - core.length);
  return [...core, ...fill];
}
