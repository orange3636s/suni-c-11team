import type { ParetoRankingItem } from "@/types/data";

// 지시서 "원인 분석 화면 정리" WI-2 -- 표시 기준 토글(전체 상위 3개 /
// 유의한 인자만)을 없애고 파레토 기여율 고정 기준 하나로 대체한다.
// 토글이 없으므로 이 임계값은 상수다: 다른 데이터셋에서 2위 인자가
// 이 값을 넘으면 카드도 자동으로 늘어난다(등급/누적 80%와 무관).
// YG/ZF-1: 원래 20이었다 -- 수율 예측·모니터링(FMEA)과 같은 "핵심
// 인자" 개념을 공유하는 화면 전체 임계를 10으로 낮췄다(백엔드
// src/analysis/thresholds.py의 CORE_FACTOR_CONTRIBUTION_MIN과 값을
// 맞춘다 -- 언어가 달라 import는 못 하지만 반드시 같은 값으로 유지).
// 폐기된 20% 결정의 배경은 docs/decisions.md 참고.
export const DISPLAY_CONTRIBUTION_THRESHOLD_PCT = 10;

/** 차트 표시 규칙 (지시서 WI-2) -- 파레토 기여율이
 * DISPLAY_CONTRIBUTION_THRESHOLD_PCT 이상인 인자를 전부 고른다(개수
 * 상한 없음, 하나도 없으면 빈 배열). */
export function selectDisplayFactors(items: ParetoRankingItem[]): ParetoRankingItem[] {
  return items.filter((item) => item.contribution_pct >= DISPLAY_CONTRIBUTION_THRESHOLD_PCT);
}
