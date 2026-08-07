import type { ConfidenceTier } from "@/types/data";

// 신뢰 등급 라벨 단일 소스 (지시서 D) -- 예전에는 ParetoChart.tsx /
// CompareAcrossTargetsModal.tsx / root-cause/page.tsx가 각자
// 지역 TIER_LABEL을 갖고 있었고 값이 갈렸다("약함"/"참고" vs "근거
// 부족"/"관계 없음"). 보수적인 쪽(root-cause·ParetoChart가 쓰던 표현)을
// 정본으로 삼는다 -- "약함"/"참고"는 실제보다 관계가 있는 것처럼 읽힌다.
export const TIER_LABEL: Record<ConfidenceTier, string> = {
  strong: "강함",
  moderate: "보통",
  weak: "근거 부족",
  reference: "관계 없음",
};

// 등급 배지 호버 설명.
export const TIER_TOOLTIP: Record<ConfidenceTier, string> = {
  strong: "부도율 변동의 10% 이상을 설명합니다. 조치의 우선 순위입니다.",
  moderate: "5~10%를 설명합니다. 관계는 있지만 추가 확인이 필요합니다.",
  weak: "통계적 근거가 부족합니다. 조치 판단에 사용하지 마세요.",
  reference: "효과 크기가 기준(0.02)에 미치지 못합니다.",
};

export function hasReliableEvidence(tier: ConfidenceTier): boolean {
  return tier === "strong" || tier === "moderate";
}

export function confidenceBadgeClassName(tier: ConfidenceTier): string {
  return `confidenceBadge tier-${tier}`;
}

// eps2와 같은 임계값(0.02/0.05/0.10)을 rho^2("설명된 분산 비율")에 적용한
// 근사 등급 -- Trellis처럼 그룹별 eps2를 새로 계산하지 않고 Spearman
// rho만 있는 곳에서, 두 지표 모두 "설명된 분산 비율" 성격이라는 점에
// 기대어 같은 4단계 등급 언어로 표시한다 (지시서 D-③: 통계적 유의성과
// 효과크기 등급을 같은 배지에 섞지 않는다 -- 이 함수는 유의성과
// 무관하게 순수히 rho의 크기만 본다).
export function effectSizeTierFromRho(rho: number): ConfidenceTier {
  const r2 = rho * rho;
  if (r2 >= 0.10) return "strong";
  if (r2 >= 0.05) return "moderate";
  if (r2 >= 0.02) return "weak";
  return "reference";
}
