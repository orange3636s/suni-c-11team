import type { ParetoRankingItem } from "@/types/data";

// 지시서 "원인 분석 화면 정리" WI-2 -- 표시 기준 토글(전체 상위 3개 /
// 유의한 인자만)을 없애고 파레토 기여율 20% 이상 고정 기준 하나로
// 대체한다. 토글이 없으므로 이 임계값은 상수다: 다른 데이터셋에서 2위
// 인자가 20%를 넘으면 카드도 자동으로 늘어난다(등급/누적 80%와 무관).
export const DISPLAY_CONTRIBUTION_THRESHOLD_PCT = 20;

/** 차트 표시 규칙 (지시서 WI-2) -- 파레토 기여율이 20% 이상인 인자를
 * 전부 고른다(개수 상한 없음, 하나도 없으면 빈 배열). */
export function selectDisplayFactors(items: ParetoRankingItem[]): ParetoRankingItem[] {
  return items.filter((item) => item.contribution_pct >= DISPLAY_CONTRIBUTION_THRESHOLD_PCT);
}
