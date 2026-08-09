import type { WaferPrediction } from "@/types/data";

// 사전 알람 로그 전면 개편 (spec §A-3/§B-1), 이후 민감도 슬라이더를 실제
// 트레이드오프로 (spec §CA-1)로 판정 기준을 교체 -- src/analysis/alarm_gbdt.py의
// classify_margin/classify_wafer와 동일한 공식이다. 서버 재호출 없이
// 목표 수율·민감도를 조절할 때마다 이 파일이 즉시 재분류한다. 두 구현이
// 갈라지면 화면(이 파일)과 알림 발송(서버 기본값)의 판정이 어긋나므로,
// 이 파일을 고칠 때는 alarm_gbdt.py의 동일 함수도 함께 확인한다.

export type AlertGrade = "심각" | "위험" | "주의" | "정상" | null; // null = 판별불가(미분류)

// 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-1) -- 민감도 0(오경보
// 최소)일 때 margin 최대(가장 보수적), 1(미탐 최소)일 때 margin 0(가장
// 민감). %p 절대값이다 -- σ(웨이퍼 수율 산포) 배수는 "예측 불확실성"과
// 무관한 값이라 폐기했다.
export const MARGIN_MAX_PP = 4.0;
// 등급(심각/위험/주의) 간 간격 -- %p.
export const GRADE_STEP_PP = 0.8;

export function classifyMargin(sensitivity: number): number {
  return (1.0 - sensitivity) * MARGIN_MAX_PP;
}

export function classifyWafer(
  predMean: number,
  predLo: number,
  opts: { target: number; sensitivity: number; gatePassed?: boolean },
): AlertGrade {
  const { target, sensitivity, gatePassed = true } = opts;
  const margin = classifyMargin(sensitivity);
  if (gatePassed) {
    if (predMean <= target - margin - 2 * GRADE_STEP_PP) return "심각";
    if (predMean <= target - margin - GRADE_STEP_PP) return "위험";
    if (predMean <= target - margin) return "주의";
  }
  if (predLo >= target) return "정상";
  return null;
}

export type ClassifiedWafer = WaferPrediction & { grade: AlertGrade };

// 예측 구간 캘리브레이션 + 미분류 사유 분리 (spec §BC-1) -- measured는 더
// 이상 판정 게이트가 아니다. 계측 개수가 예측 품질을 예고하지 못했고
// (실측 MAE: 미계측군 2.694 / 1개만 계측 2.923 -- 1개만 계측된 쪽이 더
// 부정확했다), conformal 구간이 이미 그 불확실성을 폭으로 반영한다.
// measured는 이제 사유 표시에만 쓴다 (아래 reasonFor/미분류 사유 분리).
export function classifyAll(
  predictions: WaferPrediction[],
  opts: { target: number; sensitivity: number; gatePassed: boolean },
): ClassifiedWafer[] {
  return predictions.map((p) => ({
    ...p,
    grade: classifyWafer(p.pred_mean, p.pred_lo, opts),
  }));
}

export type PrecisionRecallEstimate = {
  /** 홀드아웃 OOF 표본 중 이 컷으로 "알람"이 되는 개수. */
  oofAlarms: number;
  /** 홀드아웃 OOF 표본 크기 (0이면 홀드아웃 자체가 없다 -- 랏 수 부족). */
  oofSampleSize: number;
  /** 알람 중 실제로 목표 미달이었던 비율 (0-100). oofAlarms=0이면 null. */
  precisionPct: number | null;
  /** 실제 목표 미달 중 알람으로 잡힌 비율 (0-100). 홀드아웃에 미달 표본이
   * 없으면 null. */
  recallPct: number | null;
  /** 현재 eval 알람 수에 추정 정밀도·재현율을 적용해 "놓칠 것으로 추정"되는
   * wafer 수를 역산한다: tpEst = evalAlarms * precision, totalBadEst =
   * tpEst / recall, missed = totalBadEst - tpEst. recall이 0/null이면
   * 역산 불가(null) -- eval의 실제 미달 wafer 수는 원래 알 수 없다. */
  missedEstimate: number | null;
  /** 호출자가 넘긴 evalAlarmCount 그대로 -- 화면이 같은 자리에서 "알람
   * N장"을 함께 보여줄 때 별도 prop 없이 이 값을 쓴다. */
  evalAlarms: number;
};

/** 민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 홀드아웃(train
 * OOF) (실제 Y, 예측값) 쌍으로 현재 목표·민감도 설정의 정밀도·재현율을
 * 추정한다. eval에는 실제 정답이 없으므로 이 값은 항상 "추정치"다 --
 * 화면에 "홀드아웃 기준 추정"을 병기해야 한다. `evalAlarmCount`는 지금
 * 실제로 표시 중인 eval 알람 수(심각+위험+주의) -- "놓칠 것으로 추정"
 * 계산에만 쓴다. */
export function estimatePrecisionRecall(
  oofActual: number[],
  oofPredicted: number[],
  opts: { target: number; sensitivity: number; evalAlarmCount: number },
): PrecisionRecallEstimate {
  const { target, sensitivity, evalAlarmCount } = opts;
  const margin = classifyMargin(sensitivity);
  const cut = target - margin; // 가장 느슨한 등급(주의) 컷 -- 알람 여부 자체는 이 컷 하나로 정해진다.
  const n = Math.min(oofActual.length, oofPredicted.length);

  let oofAlarms = 0;
  let truePositives = 0;
  let actualBad = 0;
  for (let i = 0; i < n; i++) {
    const isAlarm = oofPredicted[i] <= cut;
    const isBad = oofActual[i] < target;
    if (isBad) actualBad++;
    if (isAlarm) {
      oofAlarms++;
      if (isBad) truePositives++;
    }
  }

  const precisionPct = oofAlarms > 0 ? (truePositives / oofAlarms) * 100 : null;
  const recallPct = actualBad > 0 ? (truePositives / actualBad) * 100 : null;

  let missedEstimate: number | null = null;
  if (precisionPct != null && recallPct != null && recallPct > 0) {
    const tpEst = evalAlarmCount * (precisionPct / 100);
    const totalBadEst = tpEst / (recallPct / 100);
    missedEstimate = Math.max(0, Math.round(totalBadEst - tpEst));
  }

  return { oofAlarms, oofSampleSize: n, precisionPct, recallPct, missedEstimate, evalAlarms: evalAlarmCount };
}

// spec §BC-2: 계측 없이(measured=false) 등급이 매겨진 wafer는 어느
// 선정 인자도 근거로 들 수 없다 -- 사유란에 이 문구를 대신 보여주고,
// 배지에 구분 표기를 붙이며, 자동 발송 대상에서 제외한다(서버 쪽
// compute_alarm_notification_items/refresh.py가 실제 제외를 담당한다 --
// 이 파일은 화면 표시만 맡는다).
export const NO_REASON_UNMEASURED = "사유 제시 불가 — 선정 인자 미계측";

/** 알람 목록(심각/위험/주의)에 표시할 사유. grade가 있는 wafer만 부른다 --
 * 미분류(grade=null)의 사유 분리는 `summarizeClasses`의
 * straddleCount/unmeasuredCount로 별도 처리한다. */
export function reasonFor(item: WaferPrediction): string {
  if (!item.measured) return NO_REASON_UNMEASURED;
  return item.reason ?? "-";
}

export type ClassKey = "심각" | "위험" | "주의" | "정상" | "판별불가";
export const CLASS_KEYS: ClassKey[] = ["심각", "위험", "주의", "정상", "판별불가"];

// 예측 구간 캘리브레이션 + 미분류 사유 분리 (spec §BD-1) -- 화면 표시
// 라벨만 "미분류"로 바꾼다. 내부 키("판별불가")는 백엔드 refresh
// 스냅샷(counts 딕셔너리)·자동화 코드와 공유하는 식별자라 그대로 둔다 --
// 여기서 바꾸면 스냅샷 스키마까지 전부 갈아엎어야 한다.
export const CLASS_LABELS: Record<ClassKey, string> = {
  심각: "심각", 위험: "위험", 주의: "주의", 정상: "정상", 판별불가: "미분류",
};

export type ClassSummary = {
  key: ClassKey;
  items: ClassifiedWafer[];
  count: number;
  pct: number; // 평가 wafer 전체 기준 (spec §B-3: "비율 분모는 평가 wafer 전체")
  avgPredMean: number | null;
  // "판별불가"(미분류)에서만 의미가 있다 (spec §BB-1). 성격이 다른 두
  // 사유를 나눈다 -- straddleCount("상관성 부족": 인자는 계측됐는데
  // 예측 구간이 목표를 걸침, 계측을 늘려도 해소 안 됨)와
  // unmeasuredCount("계측 부족": 선정 인자가 전부 미계측, 계측을
  // 늘리면 해소됨 -- 계측 우선순위 큐의 대상).
  straddleCount: number;
  unmeasuredCount: number;
};

/** 5분류 그룹핑 + 카드에 필요한 집계 (spec §B-1/§B-3/§BB-1) -- 다섯 분류는
 * 서로 겹치지 않고, count 합은 항상 `totalWafers`와 같다. */
export function summarizeClasses(classified: ClassifiedWafer[], totalWafers: number): Record<ClassKey, ClassSummary> {
  const groups: Record<ClassKey, ClassifiedWafer[]> = { 심각: [], 위험: [], 주의: [], 정상: [], 판별불가: [] };
  for (const wafer of classified) {
    if (wafer.grade === "심각" || wafer.grade === "위험" || wafer.grade === "주의" || wafer.grade === "정상") {
      groups[wafer.grade].push(wafer);
    } else {
      groups.판별불가.push(wafer);
    }
  }
  const result = {} as Record<ClassKey, ClassSummary>;
  for (const key of CLASS_KEYS) {
    const items = groups[key];
    const avgPredMean = items.length ? items.reduce((sum, w) => sum + w.pred_mean, 0) / items.length : null;
    result[key] = {
      key,
      items,
      count: items.length,
      pct: totalWafers > 0 ? (items.length / totalWafers) * 100 : 0,
      avgPredMean,
      straddleCount: key === "판별불가" ? items.filter((w) => w.measured).length : 0,
      unmeasuredCount: key === "판별불가" ? items.filter((w) => !w.measured).length : 0,
    };
  }
  return result;
}

/** 목표 수율이 학습 데이터 Y 분포와 맞지 않는지 (spec §A-1) -- 1~99%
 * 분위수 밖이면 경고한다. */
export function targetYieldMismatch(target: number, p1: number, p99: number): boolean {
  return target < p1 || target > p99;
}

// GB-2: 알람 0건 빈 상태는 "없음"만 말하면 정상인지 판정 불가인지 구분이
//안 된다. 게이트 미달은 alarmGateBanner가 이미 별도로 보여준다(이
// 함수는 auc_gate_passed=true일 때만 불린다) -- 여기서는 그 다음 세 가지
// 원인 중 가장 근본적인 것 하나만 고른다(GB-3: 여러 사유를 동시에
// 나열하지 않는다). 판정 순서는 데이터로 확인 가능한 순서대로다:
//   1) 미분류가 아예 없다 -- 전 wafer가 실제로 목표 이상(정상)이다.
//   2) 미분류 중 predMean이 이미 목표 이하인 wafer가 있다 -- 민감도를
//      올리면(컷이 목표에 가까워지면) 알람으로 바뀔 후보다.
//   3) 그 외 -- 미분류(계측 부족/구간이 목표를 걸침)가 대부분이라
//      점추정만으로는 알람이 안 나온 것이다.
export type AlarmEmptyReason =
  | { kind: "all_above_target"; target: number }
  | { kind: "low_sensitivity"; sensitivity: number; marginPp: number }
  | { kind: "mostly_unclassified"; judgeable: number; unclassified: number }
  | { kind: "unknown" };

export function alarmEmptyReason(
  classSummary: Record<ClassKey, ClassSummary>,
  opts: { target: number; sensitivity: number; totalWafers: number },
): AlarmEmptyReason {
  const { target, sensitivity, totalWafers } = opts;
  const unclassified = classSummary.판별불가;
  if (unclassified.count === 0) return { kind: "all_above_target", target };

  // 민감도를 최대(1.0, margin=0)로 올렸을 때 "주의" 컷이 target 자체가
  // 되므로, 계측된 미분류 wafer 중 predMean이 이미 목표 이하인 것이
  // 있으면 민감도만 올려도 알람으로 바뀐다.
  const wouldAlarmAtFullSensitivity = unclassified.items.some((item) => item.measured && item.pred_mean <= target);
  if (wouldAlarmAtFullSensitivity) {
    return { kind: "low_sensitivity", sensitivity, marginPp: classifyMargin(sensitivity) };
  }

  return { kind: "mostly_unclassified", judgeable: totalWafers - unclassified.count, unclassified: unclassified.count };
}

export function alarmEmptyReasonText(reason: AlarmEmptyReason): string {
  switch (reason.kind) {
    case "all_above_target":
      return `모든 wafer의 예측 수율이 목표(${reason.target.toFixed(1)}%) 이상입니다`;
    case "low_sensitivity":
      return `현재 민감도(${reason.sensitivity.toFixed(2)})에서는 판정 컷이 목표보다 ${reason.marginPp.toFixed(1)}%p 낮습니다. 민감도를 올리면 더 많은 wafer가 판정됩니다`;
    case "mostly_unclassified":
      return `판정 가능 ${reason.judgeable.toLocaleString()}장 중 목표 미달로 판정된 wafer가 없습니다 (미분류 ${reason.unclassified.toLocaleString()}장)`;
    case "unknown":
      return "조건에 해당하는 항목이 없습니다";
  }
}

function csvEscape(value: string): string {
  if (/[",\n\r]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

/** 알람 목록 CSV 내려받기 (spec §D-2) -- UTF-8 BOM을 붙여야 한글 엑셀에서
 * 깨지지 않는다. */
export function downloadAlarmsCsv(items: ClassifiedWafer[], filenamePrefix: string): void {
  const header = ["Wafer", "예측 수율 하한", "예측 수율 상한", "등급", "사유", "LOT"];
  const rows = items.map((item) => [
    item.lot_wafer_id,
    item.pred_lo.toFixed(1),
    item.pred_hi.toFixed(1),
    item.grade ?? "",
    reasonFor(item),
    item.lot_id ?? "",
  ]);
  const csv = [header, ...rows].map((row) => row.map(csvEscape).join(",")).join("\r\n");
  // UTF-8 BOM (spec §D-2) -- 없으면 한글 엑셀이 CSV를 깨진 인코딩으로 연다.
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filenamePrefix}_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
