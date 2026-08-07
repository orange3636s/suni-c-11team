import type { CSSProperties } from "react";
import { confidenceBadgeClassName, TIER_LABEL, TIER_TOOLTIP } from "@/lib/confidenceTier";
import type { ConfidenceTier } from "@/types/data";

/** 신뢰 등급 배지 -- 3곳(ParetoChart/CompareAcrossTargetsModal/root-cause,
 * 그리고 training/CorrelationHeatmap/Trellis)이 전부 이 컴포넌트를 쓴다
 * (지시서 D-④). `title`을 넘기면 기본 툴팁 대신 그 문구를 쓴다 -- Trellis
 * 패널처럼 "이 값이 무엇을 재는지"를 문맥에 맞게 다시 설명해야 하는
 * 경우용이다. */
export default function ConfidenceBadge({
  tier,
  title,
  style,
}: {
  tier: ConfidenceTier;
  title?: string;
  style?: CSSProperties;
}) {
  return (
    <span className={confidenceBadgeClassName(tier)} title={title ?? TIER_TOOLTIP[tier]} style={style}>
      {TIER_LABEL[tier]}
    </span>
  );
}
