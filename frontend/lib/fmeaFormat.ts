// 모니터링 홈 블록①(ActionPriorityBlock)과 Config 트리맵이 공유하는 표시
// 포맷터. 계산은 전부 백엔드(src/analysis/screening/fmea.py,
// src/analysis/action_priority.py)에서 끝났으므로 여기는 문자열 표현만
// 담당한다.
import type { ActionPriorityRow, RelationShape } from "@/types/data";

// MA-2/MG-4: Config(범주형) 인자를 잠재 원인 목록에서 배제한 근거 --
// README/챗봇 컨텍스트 문서가 이미 밝힌 수치를 그대로 재사용한다(하드
// 코딩된 새 값이 아니다). config-treemap 탭의 "유의 조합 없음" 안내와
// 모니터링 홈 문서화가 이 두 상수를 공유한다.
export const CONFIG_SCREENING_TEST_COUNT = 600;
export const CONFIG_SCREENING_PASS_COUNT = 0;

export const RELATION_LABEL: Record<RelationShape, string> = {
  monotonic_increasing: "단조 증가",
  monotonic_decreasing: "단조 감소",
  u_shape: "U자",
  unclear: "불명확",
};

export function isMonotonic(shape: RelationShape): boolean {
  return shape === "monotonic_increasing" || shape === "monotonic_decreasing";
}

export function formatSignedPp(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%p`;
}

export function formatPct(value: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value.toFixed(digits)}%`;
}

export function formatNumber(value: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

// 지시서 KA-2: 구분자를 en dash(–)에서 물결(~)로 바꾼다 -- en dash는
// 음수 부호와 시각적으로 혼동된다("50.7–69.4"가 스크롤/줄바꿈 등으로
// 하한이 가려지면 "–69.4"처럼 음수로 보인다).
export function rangeText(item: Pick<ActionPriorityRow, "range_lo" | "range_hi">): string {
  if (item.range_lo == null && item.range_hi == null) return "-";
  if (item.range_lo != null && item.range_hi != null) {
    return `${item.range_lo.toFixed(1)} ~ ${item.range_hi.toFixed(1)}`;
  }
  if (item.range_hi != null) return `≤ ${item.range_hi.toFixed(1)}`;
  return `≥ ${item.range_lo!.toFixed(1)}`;
}

export function isOutOfRange(item: Pick<ActionPriorityRow, "range_lo" | "range_hi" | "factor_value">): boolean {
  if (item.factor_value == null) return false;
  if (item.range_lo != null && item.factor_value < item.range_lo) return true;
  if (item.range_hi != null && item.factor_value > item.range_hi) return true;
  return false;
}
