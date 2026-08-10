// 모니터링 홈의 유일한 데이터 조회 경로 -- 페이지 컴포넌트는 이 3개
// 함수(buildMonitoringSnapshot / getTreemapData / getYieldSummary)만
// 호출하고 fetch를 직접 부르지 않는다. 지금은 세 함수 모두 기존 REST
// API를 감쌀 뿐이지만, 나중에 팹 DB(SQL)가 붙으면 이 파일 안쪽만
// 교체하면 되도록 인터페이스를 분리해 둔다 (SQL 관련 코드는 아직
// 작성하지 않는다).
"use client";

import { getConfigTreemap, getYieldPrediction } from "@/lib/api";
// W-1: analysis/alarms는 항상 호출부(모니터링 페이지)가 공용
// AnalysisStateProvider에서 이미 읽은 값을 그대로 넘긴다 -- 이 파일이
// 따로 GET /api/state/latest를 다시 부르지 않는다(지시서: "화면별로
// 개별 fetch하지 마라"). 복원된 값이 쓸 만한지(isAnalysisSnapshotUsable)
// 는 그 값을 만든 쪽(AnalysisStateProvider의 hydrate/스냅샷 대체 채움)이
// 이미 판단했으므로 여기서 다시 검사하지 않는다. 타입만 참조하므로(값은
// 쓰지 않는다) 순환 임포트가 되지 않는다.
import type { AlarmsState, AnalysisState } from "@/components/AnalysisStateProvider";
import type {
  ConfigTreemapResponse,
  FmeaTablePayload,
  LatestAlarmsRecord,
  MeasurementExpansionResponse,
  YieldSummary,
} from "@/types/data";

export type MonitoringSnapshot = {
  hasAnalysis: boolean;
  createdAt: string | null;
  dataset: string | null;
  // FMEA 분석표 (지시서 IA/JA) -- 자동 갱신 스냅샷과 수동 "다시 분석"
  // 저장 둘 다 백엔드가 채워 보낸다(JA-1). `fmea`가 null인데
  // `fmeaError`도 null이면 JA-1 배포 이전에 저장된 옛 레코드 -- 다시
  // 분석하면 채워진다. `fmeaError`가 있으면 계산이 실패한 것이다.
  fmea: FmeaTablePayload | null;
  fmeaError: string | null;
  measurementExpansion: MeasurementExpansionResponse | null;
  alarmsRecord: LatestAlarmsRecord | null;
};

/** AnalysisStateProvider가 이미 들고 있는 analysis/alarms(저장된 결과를
 * 복원했든, 아직 아무도 실행한 적 없어 자동 갱신 스냅샷으로 대체
 * 채웠든 -- W-1)를 그대로 "모니터링에 바로 쓸 수 있는" 스냅샷으로
 * 감싼다. FMEA 분석표는 백엔드가 이미 다 계산해 analysis.fmea에 실어
 * 보내므로(지시서 IA-5) 여기서 추가로 조회하지 않는다. */
export async function buildMonitoringSnapshot(
  analysis: AnalysisState,
  alarms: AlarmsState,
): Promise<MonitoringSnapshot> {
  // 이 화면은 alarms를 자기 필드명이 아니라 LatestAlarmsRecord 모양으로
  // 계속 다룬다(SummaryBlock 등 기존 렌더 코드를 건드리지 않기 위해) --
  // AnalysisStateProvider의 AlarmsState(camelCase)를 그 모양으로 얇게
  // 감싼다.
  const alarmsRecord: LatestAlarmsRecord | null = alarms
    ? {
        schema_version: 1,
        created_at: alarms.createdAt,
        train_dataset: alarms.trainDataset,
        eval_dataset: alarms.evalDataset,
        payload: { targetYield: alarms.targetYield, sensitivity: alarms.sensitivity },
      }
    : null;

  if (!analysis) {
    return {
      hasAnalysis: false,
      createdAt: null,
      dataset: null,
      fmea: null,
      fmeaError: null,
      measurementExpansion: null,
      alarmsRecord,
    };
  }

  return {
    hasAnalysis: true,
    createdAt: analysis.createdAt,
    dataset: analysis.dataset,
    fmea: analysis.fmea ?? null,
    fmeaError: analysis.fmeaError ?? null,
    measurementExpansion: analysis.measurementExpansion ?? null,
    alarmsRecord,
  };
}

export async function getTreemapData(dataset: string, step: number, target = "Y1"): Promise<ConfigTreemapResponse | null> {
  try {
    return await getConfigTreemap(dataset, step, target);
  } catch {
    // 이 데이터셋에 해당 스텝의 Config가 없는 경우 등 -- 빈 상태로 처리.
    return null;
  }
}

/** WB/WC/WD: 상단 요약 카드·모드별 손실 막대·수율 분포 히스토그램이
 * 쓰는 서버 계산 결과 -- 수율 예측(y합산 순위) API가 이미 함께 계산해
 * 내려보낸다(계산은 백엔드, 여기서는 그대로 통과만 시킨다). 조회 실패
 * 시(아직 원인 분석을 한 번도 안 돌렸거나 데이터셋이 없으면) null. */
export async function getYieldSummary(alarmsRecord: LatestAlarmsRecord | null): Promise<YieldSummary | null> {
  if (!alarmsRecord) return null;
  try {
    const response = await getYieldPrediction(alarmsRecord.train_dataset, alarmsRecord.eval_dataset);
    return response.yield_summary;
  } catch {
    return null;
  }
}
