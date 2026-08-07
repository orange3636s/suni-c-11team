import type { ParetoRankingItem } from "@/types/data";

/** 차트 표시 규칙 (spec §A-2) -- "유의한 인자만" 모드에서 그릴 인자를
 * 고른다. `items`는 이미 eps2 내림차순으로 정렬된 top-5 Pareto 결과이며,
 * 각 항목의 `cumulative_pct`는 (top-5로 잘리기 전) 전체 후보 풀 기준
 * 누적 기여율이다.
 *
 * 강함·보통 등급만 추출한다 -- 3개를 넘어도 전부 표시하고, 3개 미만이라도
 * 약함으로 보충하지 않는다 (강함·보통이 0개면 결과도 0개). 등급 무관 상위
 * 인자를 보고 싶다면 화면의 "전체 상위 3개" 토글을 쓴다. */
export function selectDisplayFactors(items: ParetoRankingItem[]): ParetoRankingItem[] {
  if (items.length === 0) return [];

  // Pareto 80% 범위 -- 80%를 넘기는 첫 인자까지 포함한다.
  const cutoffIndex = items.findIndex((item) => item.cumulative_pct >= 80);
  const pareto = cutoffIndex === -1 ? items : items.slice(0, cutoffIndex + 1);

  return pareto.filter((item) => item.confidence_tier === "strong" || item.confidence_tier === "moderate");
}
