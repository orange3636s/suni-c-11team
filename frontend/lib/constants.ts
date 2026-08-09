// E-6: 여러 파일에 중복 선언돼 있던 상수를 여기 하나로 모은다. 각 값이
// 어디서 왔고 왜 이 모양인지는 아래 각 항목의 주석 참고.
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import type { CategoricalScatterResponse, ParetoRankingItem } from "@/types/data";

// 화면 스크리닝이 항상 다루는 5개 수율 타깃 (monitoringSource.ts / root-cause/page.tsx 중복).
export const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

// GB-2: "유의 인자 없음" 빈 상태 사유 판정에 쓰는 검정 최소 표본·설명력
// 기준 -- src/analysis/screening/selector.py DEFAULT_MIN_N_NUMERIC/
// DEFAULT_MIN_N_CATEGORICAL, config/grade_thresholds.yaml min_eps2_reference와
// 반드시 같은 값을 유지한다(서버가 실제로 쓰는 기준과 어긋나면 화면
// 사유가 거짓말이 된다).
export const MIN_N_NUMERIC = 100;
export const MIN_N_CATEGORICAL = 20;
export const MIN_EPS2_REFERENCE = 0.02;

/** 판정 순서를 정하고 가장 근본적인 원인 하나만 표시한다(GB-3) -- 계측
 * 부족(검정 자체가 최소 표본 미달)이 가장 근본적이라 먼저 보고, 그
 * 다음 FDR(유의성), 마지막 효과크기 순으로 확인한다. 어느 것도 명확히
 * 설명하지 못하면 null -- 호출부가 기존 통합 통계 문구로 폴백한다
 * (추측한 사유를 쓰지 않는다).
 */
export type NoChartReason =
  | { kind: "insufficient_measurement"; bestObserved: number; minRequired: number }
  | { kind: "fdr_not_passed"; totalTested: number }
  | { kind: "effect_size_below_threshold"; maxEps2Pct: number };

export function noChartReason(
  items: ParetoRankingItem[],
  stats: { totalTested: number; fdrPassCount: number; maxEps2: number | null },
): NoChartReason | null {
  if (items.length === 0) return null;

  const minRequiredFor = (kind: string) => (kind === "Config" ? MIN_N_CATEGORICAL : MIN_N_NUMERIC);
  const allBelowMinN = items.every((item) => item.n_observed < minRequiredFor(item.kind));
  if (allBelowMinN) {
    const best = items.reduce((a, b) => (a.n_observed >= b.n_observed ? a : b));
    return { kind: "insufficient_measurement", bestObserved: best.n_observed, minRequired: minRequiredFor(best.kind) };
  }

  if (stats.fdrPassCount === 0) return { kind: "fdr_not_passed", totalTested: stats.totalTested };

  if ((stats.maxEps2 ?? 0) < MIN_EPS2_REFERENCE) {
    return { kind: "effect_size_below_threshold", maxEps2Pct: (stats.maxEps2 ?? 0) * 100 };
  }

  return null;
}

export function noChartReasonText(reason: NoChartReason): string {
  switch (reason.kind) {
    case "insufficient_measurement":
      return `계측된 wafer가 ${reason.bestObserved.toLocaleString()}장으로 검정에 필요한 최소치(${reason.minRequired}장)에 미달합니다`;
    case "fdr_not_passed":
      return `FDR 보정 후 유의한 인자가 없습니다 (${reason.totalTested.toLocaleString()}건 검정, 통과 0건)`;
    case "effect_size_below_threshold":
      return `가장 강한 인자도 설명력이 ${reason.maxEps2Pct.toFixed(2)}%로 기준(${(MIN_EPS2_REFERENCE * 100).toFixed(0)}%)에 미달합니다`;
  }
}

// Config 값 형식 검증 전용(캡처 그룹 없음) -- monitoringSource.ts가 쓰던 용도.
export const CONFIG_FORMAT_RE = /^Step\d+_Model\d+_EQ[A-Z]_CH\d+$/;

export type ConfigParts = { step: number; model: string; eq: string; chamber: string };
const CONFIG_PARSE_RE = /^Step(\d+)_(Model\d+)_(EQ[A-Z])_(CH\d+)$/;

/** `Step16_Model2_EQB_CH3` -> step/model/eq/chamber 4계층 분해. 매치
 * 실패(형식이 다른 미지 Config)는 호출부에서 "미상"/null 그룹으로
 * 모은다 -- 조용히 버리지 않는다. */
export function parseConfig(config: string): ConfigParts | null {
  const m = CONFIG_PARSE_RE.exec(config);
  if (!m) return null;
  return { step: Number(m[1]), model: m[2], eq: m[3], chamber: m[4] };
}

/** 알람 판정 3등급 색 (spec §B-1) -- ScatterChart의 삼각형 마커, alerts
 * 페이지의 5분류 배지가 같은 값을 쓴다. "주의"는 #EAB308 대신 #CA8A04를
 * 쓴다: 빈 삼각형이라 테두리만 보이므로 밝은 노랑은 흰 배경에서 식별이
 * 어렵다. CSS 변수(--chart-alarm-*, globals.css :root / [data-theme=dark])로
 * 승격해 라이트/다크 분기를 컴포넌트가 직접 하지 않도록 한다. */
export const ALARM_GRADE_COLOR: Record<"심각" | "위험" | "주의", string> = {
  심각: "var(--chart-alarm-severe)",
  위험: "var(--chart-alarm-danger)",
  주의: "var(--chart-alarm-caution)",
};

/** 발산형(빨강=낮음/초록=높음) 팔레트 -- Treemap 색 스케일, 비교 모달의
 * "최적 중심/구간 평균 불량률" 범례 스와치가 공유한다. dataviz 스킬의
 * validate_palette.js로 확인된 값(라이트 CVD ΔE 8.6, 다크 6.5). */
export const DIVERGING_GREEN = { light: "#059669", dark: "#34D399" };
export const DIVERGING_RED = { light: "#DC2626", dark: "#F87171" };

/** 카테고리형(Config) 인자의 Box Plot 스펙 -- 원인 분석 전체 화면과
 * 즐겨찾기 썸네일이 같은 모양을 쓰되 축 제목 유무가 다르다: 썸네일은
 * 160px 높이라 제목을 넣으면 잘리므로 생략하고(J-3), 전체 화면은
 * factorAxisLabel/targetAxisLabel로 축 의미를 명시한다. 이 차이는
 * 의도된 것이므로 `compact` 옵션으로 명시적으로 고른다. */
export function buildCategoricalSpec(data: CategoricalScatterResponse, options?: { compact?: boolean }) {
  const x = data.groups.flatMap((group) => group.values.map(() => group.category));
  const y = data.groups.flatMap((group) => group.values);
  const compact = options?.compact ?? false;
  return {
    data: [
      { type: "box", x, y, boxpoints: "outliers", marker: { color: "#1D4ED8" }, line: { color: "#1D4ED8" } },
    ],
    layout: compact
      ? { xaxis: { tickangle: 0 }, yaxis: {}, margin: { t: 10, b: 40, l: 40, r: 10 } }
      : {
          xaxis: { title: { text: factorAxisLabel(data.axis.x_label) }, tickangle: 0 },
          yaxis: { title: { text: targetAxisLabel(data.axis.y_label) } },
          margin: { t: 20, b: 90 },
        },
  };
}
