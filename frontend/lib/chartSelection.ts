import type { ParetoRankingItem } from "@/types/data";

// 원인 분석 화면은 표시 기준 토글 없이 파레토 기여율 고정 기준 하나로
// 표시할 인자를 고른다. 토글이 없으므로 이 임계값은 상수다: 다른
// 데이터셋에서 2위 인자가 이 값을 넘으면 카드도 자동으로 늘어난다
// (등급/누적 80%와 무관).
// 수율 예측·모니터링(FMEA)과 "핵심 인자" 개념을 공유하므로 백엔드
// src/analysis/thresholds.py의 CORE_FACTOR_CONTRIBUTION_MIN과 같은 값을
// 유지해야 한다 -- 언어가 달라 import는 못 하니 한쪽만 바꾸면 화면마다
// 핵심 인자 집합이 갈린다.
export const DISPLAY_CONTRIBUTION_THRESHOLD_PCT = 10;

/** 차트 표시 규칙 -- 파레토 기여율이
 * DISPLAY_CONTRIBUTION_THRESHOLD_PCT 이상인 인자를 전부 고른다(개수
 * 상한 없음, 하나도 없으면 빈 배열). */
export function selectDisplayFactors(items: ParetoRankingItem[]): ParetoRankingItem[] {
  return items.filter((item) => item.contribution_pct >= DISPLAY_CONTRIBUTION_THRESHOLD_PCT);
}
