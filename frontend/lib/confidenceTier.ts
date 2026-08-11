import type { ConfidenceTier } from "@/types/data";

// 신뢰 등급 라벨 단일 소스 -- ParetoChart / CompareAcrossTargetsModal /
// root-cause 페이지가 모두 여기서 읽는다. 표현은 보수적인 쪽으로
// 고정한다: "약함"/"참고"는 실제보다 관계가 있는 것처럼 읽히므로
// "근거 부족"/"관계 없음"을 쓴다.
export const TIER_LABEL: Record<ConfidenceTier, string> = {
  strong: "강함",
  moderate: "보통",
  weak: "근거 부족",
  reference: "관계 없음",
};

// 등급 배지 호버 설명.
export const TIER_TOOLTIP: Record<ConfidenceTier, string> = {
  strong: "불량률 변동의 10% 이상을 설명합니다. 조치의 우선 순위입니다.",
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

// 서버 등급과 같은 임계값(0.02/0.05/0.10)을 rho^2("설명된 분산 비율")에
// 적용한 근사 등급. 서버가 낸 Adjusted R²가 없는 자리 -- 즉 그룹을 클라이언트에서
// 다시 쪼개 Spearman rho만 직접 계산할 수 있는 곳(장비별 비교 모달) --
// 에서만 쓴다. 두 지표 모두 "설명된 분산 비율" 성격이라 같은 4단계 등급
// 언어로 표시할 수 있다. 통계적 유의성과 효과크기 등급은 같은 배지에
// 섞지 않는다 -- 이 함수는 유의성과 무관하게 크기만 본다.
//
// 서버 응답에 adj_r2가 들어 있는 화면(히트맵/파레토/산점도/불량 유형별
// 비교)은 이 근사를 쓰지 않고 그 값을 그대로 쓴다.
export function effectSizeTierFromRho(rho: number): ConfidenceTier {
  const r2 = rho * rho;
  if (r2 >= 0.10) return "strong";
  if (r2 >= 0.05) return "moderate";
  if (r2 >= 0.02) return "weak";
  return "reference";
}
