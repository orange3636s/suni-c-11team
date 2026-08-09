// FMEA 표(monitoring/page.tsx)의 "권고 조치" 열과, 그 아래 별도 권고 조치
// 표(RecommendedActions.tsx)가 공유하는 규칙 기반 도출 로직 (지시서 IC).
// 하드코딩된 조치 목록이 아니라 FMEA 표 자체(FmeaTablePayload)에서
// 매번 다시 도출한다.
import type { FmeaFactorItem, FmeaTablePayload } from "@/types/data";
import { formatSignedPp, isMonotonic } from "@/lib/fmeaFormat";

export type ActionStrength = "확정" | "실험 후보" | "관찰" | "제안하지 않음";

export type InlineAction = { text: string; strength: ActionStrength };

// 계측률 20% 미만인 R 인자를 "확대 대상"으로 묶는다 (지시서 IC-2).
const LOW_MEASUREMENT_RATE_PCT = 20;
// chat_system.md 근거 수치(Step1_D1 계측군 83.7% vs 미계측군 89.6%, 갭
// 약 5.9%p)를 "경고로 볼만한" 크기의 하한으로 삼는다 -- 이보다 작은
// 갭은 표본 잡음과 구분하기 어렵다.
const MNAR_WARNING_THRESHOLD_PP = 3.0;

// IC-3: FMEA 리뷰에서 반드시 나오는 "왜 장비/랏은 안 봤나" 질문에 검정
// 결과로 답한다. 수치는 실제 분석 결과에서 가져온 상수다(README.md
// "Config 30개는 어떤 타깃에서도 BH-FDR을 통과하지 못했습니다" 절,
// prompts/chat_system.md의 데이터 한계 절) -- 재검정으로 값이 바뀌면
// 여기만 갱신한다.
const CONFIG_SCREENING_TEST_COUNT = 600;
const CONFIG_SCREENING_PASS_COUNT = 0;
const LOT_ICC = 0.005;

function hasMnarWarning(item: FmeaFactorItem): boolean {
  return item.mnar_gap_pp != null && Math.abs(item.mnar_gap_pp) >= MNAR_WARNING_THRESHOLD_PP;
}

function maxDeviationFeature(items: FmeaFactorItem[]): string | null {
  let best: FmeaFactorItem | null = null;
  for (const item of items) {
    if (item.yield_deviation == null) continue;
    if (best == null || item.yield_deviation > (best.yield_deviation ?? -Infinity)) best = item;
  }
  return best?.feature ?? null;
}

/** FMEA 표 한 행의 "권고 조치" 인라인 셀 -- 규칙 우선순위대로 첫 번째로
 * 맞는 것 하나만 보여준다(한 줄 셀이라 배지 하나). 여러 규칙에 걸리는
 * 인자는 아래 별도 권고 조치 표에서 각각 행으로 펼쳐진다. */
export function inlineFactorAction(item: FmeaFactorItem, fmea: FmeaTablePayload): InlineAction {
  if (item.feature === maxDeviationFeature(fmea.items)) {
    return { text: "계측 규칙 재검토", strength: "확정" };
  }
  if (item.kind === "R" && item.measurement_rate < LOW_MEASUREMENT_RATE_PCT) {
    return { text: "계측률 확대", strength: "확정" };
  }
  if (isMonotonic(item.relation_shape)) {
    return { text: "권장 구간 관리 스플릿랏", strength: "실험 후보" };
  }
  if (hasMnarWarning(item)) {
    return { text: "계측 규칙 재검토 (MNAR)", strength: "확정" };
  }
  return { text: "관찰", strength: "관찰" };
}

export type RecommendedActionRow = {
  key: string;
  order: number;
  action: string;
  target: string;
  expectedEffect: string;
  strength: ActionStrength;
  note: string;
};

const NOT_PROPOSED_ROWS: Omit<RecommendedActionRow, "order">[] = [
  {
    key: "not-proposed-config",
    action: "장비·챔버 조건 조정",
    target: "근거 없음",
    expectedEffect: "-",
    strength: "제안하지 않음",
    note: `제안하지 않음 — ${CONFIG_SCREENING_TEST_COUNT}건 검정 FDR 통과 ${CONFIG_SCREENING_PASS_COUNT}건`,
  },
  {
    key: "not-proposed-lot",
    action: "랏 단위 원인 귀속",
    target: "근거 없음",
    expectedEffect: "-",
    strength: "제안하지 않음",
    note: `제안하지 않음 — ICC(1,1) ${LOT_ICC.toFixed(3)}, 무효과 기대값 이하`,
  },
];

/** 권고 조치 표 (지시서 IC) -- FMEA 표에서 규칙 기반으로 도출한다
 * (하드코딩 금지, IC-2). 정렬은 RPN이 아니라 실익(수율 편차) 순이다 --
 * RPN 1위와 실익 1위가 다를 수 있고, 조치 우선순위는 실익을 따른다.
 * "제안하지 않음" 두 행은 조건과 무관하게 항상 맨 끝에 붙는다(IC-3). */
export function buildRecommendedActions(fmea: FmeaTablePayload | null): RecommendedActionRow[] {
  const derived: Array<Omit<RecommendedActionRow, "order"> & { sortValue: number }> = [];

  if (fmea && fmea.items.length > 0) {
    const topFeature = maxDeviationFeature(fmea.items);
    const top = fmea.items.find((item) => item.feature === topFeature);
    if (top) {
      derived.push({
        key: `max-deviation-${top.feature}`,
        action: `${top.feature} 계측 규칙 재검토`,
        target: `${top.feature} → ${top.target}`,
        expectedEffect: formatSignedPp(top.yield_deviation),
        strength: "확정",
        note: "실익(수율 편차) 1위",
        sortValue: top.yield_deviation ?? 0,
      });
    }

    const lowRateR = fmea.items.filter(
      (item) => item.kind === "R" && item.measurement_rate < LOW_MEASUREMENT_RATE_PCT,
    );
    if (lowRateR.length > 0) {
      const avgRate = lowRateR.reduce((sum, item) => sum + item.measurement_rate, 0) / lowRateR.length;
      const maxDeviation = Math.max(...lowRateR.map((item) => item.yield_deviation ?? 0));
      derived.push({
        key: "low-rate-bundle",
        action: `계측률 확대 — ${lowRateR.map((item) => item.feature).join(", ")}`,
        target: `R 인자 ${lowRateR.length}개`,
        expectedEffect: `평균 계측률 ${avgRate.toFixed(1)}%`,
        strength: "확정",
        note: `계측률 ${LOW_MEASUREMENT_RATE_PCT}% 미만`,
        sortValue: maxDeviation,
      });
    }

    for (const item of fmea.items) {
      if (!isMonotonic(item.relation_shape)) continue;
      derived.push({
        key: `split-lot-${item.feature}`,
        action: "권장 구간 관리 스플릿랏",
        target: `${item.feature} → ${item.target}`,
        expectedEffect: formatSignedPp(item.yield_deviation),
        strength: "실험 후보",
        note: item.relation_shape === "monotonic_increasing" ? "단조 증가 관계 (단정 아님)" : "단조 감소 관계 (단정 아님)",
        sortValue: item.yield_deviation ?? 0,
      });
    }

    for (const item of fmea.items) {
      if (!hasMnarWarning(item)) continue;
      derived.push({
        key: `mnar-${item.feature}`,
        action: "계측 규칙 재검토 — 무작위 표본 병행",
        target: `${item.feature} → ${item.target}`,
        expectedEffect: formatSignedPp(item.yield_deviation),
        strength: "확정",
        note: `MNAR 경고 — 계측군·미계측군 최종 수율 갭 ${item.mnar_gap_pp!.toFixed(1)}%p`,
        sortValue: item.yield_deviation ?? 0,
      });
    }
  }

  derived.sort((a, b) => b.sortValue - a.sortValue);
  const ordered: Array<Omit<RecommendedActionRow, "order">> = [
    ...derived.map((row) => ({
      key: row.key,
      action: row.action,
      target: row.target,
      expectedEffect: row.expectedEffect,
      strength: row.strength,
      note: row.note,
    })),
    ...NOT_PROPOSED_ROWS,
  ];
  return ordered.map((row, index) => ({ ...row, order: index + 1 }));
}
