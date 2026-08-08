import type { WaferPrediction } from "@/types/data";

// 사전 알람 로그 전면 개편 (spec §A-3/§B-1) -- src/analysis/alarm_gbdt.py의
// classify_offset/classify_wafer와 동일한 공식이다. 서버 재호출 없이
// 목표 수율·민감도를 조절할 때마다 이 파일이 즉시 재분류한다. 두 구현이
// 갈라지면 화면(이 파일)과 알림 발송(서버 기본값)의 판정이 어긋나므로,
// 이 파일을 고칠 때는 alarm_gbdt.py의 동일 함수도 함께 확인한다.

export type AlertGrade = "심각" | "위험" | "주의" | "정상" | null; // null = 판별불가

export function classifyOffset(sensitivity: number): number {
  return 0.6 - sensitivity * 0.8;
}

export function classifyWafer(
  predHi: number,
  predLo: number,
  opts: { target: number; sensitivity: number; sigma: number; gatePassed?: boolean },
): AlertGrade {
  const { target, sensitivity, sigma, gatePassed = true } = opts;
  const off = classifyOffset(sensitivity);
  if (gatePassed) {
    if (predHi <= target - (off + 0.4) * sigma) return "심각";
    if (predHi <= target - (off + 0.2) * sigma) return "위험";
    if (predHi <= target - off * sigma) return "주의";
  }
  if (predLo >= target) return "정상";
  return null;
}

export type ClassifiedWafer = WaferPrediction & { grade: AlertGrade };

export function classifyAll(
  predictions: WaferPrediction[],
  opts: { target: number; sensitivity: number; sigma: number; gatePassed: boolean },
): ClassifiedWafer[] {
  return predictions.map((p) => ({
    ...p,
    grade: p.measured ? classifyWafer(p.pred_hi, p.pred_lo, opts) : null,
  }));
}

export type ClassKey = "심각" | "위험" | "주의" | "정상" | "판별불가";
export const CLASS_KEYS: ClassKey[] = ["심각", "위험", "주의", "정상", "판별불가"];

export type ClassSummary = {
  key: ClassKey;
  items: ClassifiedWafer[];
  count: number;
  pct: number; // 평가 wafer 전체 기준 (spec §B-3: "비율 분모는 평가 wafer 전체")
  avgPredMean: number | null;
  // "판별불가"에서만 의미가 있다 (spec §B-2).
  straddleCount: number;
  unmeasuredCount: number;
};

/** 5분류 그룹핑 + 카드에 필요한 집계 (spec §B-1/§B-3) -- 다섯 분류는 서로
 * 겹치지 않고, count 합은 항상 `totalWafers`와 같다. */
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
    item.reason ?? "",
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
