// FMEA 분석표(monitoring/page.tsx)와 권고 조치 표가 공유하는 표시 포맷터.
// 계산은 전부 백엔드(src/analysis/screening/fmea.py)에서 끝났으므로 여기는
// 문자열 표현만 담당한다.
import type { FmeaFactorItem, RelationShape } from "@/types/data";

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

export function rangeText(item: FmeaFactorItem): string {
  if (item.range_lo == null && item.range_hi == null) return "-";
  if (item.range_lo != null && item.range_hi != null) {
    return `${item.range_lo.toFixed(1)}–${item.range_hi.toFixed(1)}`;
  }
  if (item.range_hi != null) return `≤ ${item.range_hi.toFixed(1)}`;
  return `≥ ${item.range_lo!.toFixed(1)}`;
}

export function isOutOfRange(item: FmeaFactorItem): boolean {
  if (item.factor_value == null) return false;
  if (item.range_lo != null && item.factor_value < item.range_lo) return true;
  if (item.range_hi != null && item.factor_value > item.range_hi) return true;
  return false;
}
