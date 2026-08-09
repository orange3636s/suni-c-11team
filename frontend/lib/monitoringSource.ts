// 모니터링 홈의 유일한 데이터 조회 경로 -- 페이지 컴포넌트는 이 3개
// 함수(buildMonitoringSnapshot / getTreemapData / getMeasurementQueue)만
// 호출하고 fetch를 직접 부르지 않는다. 지금은 세 함수 모두 기존 REST
// API를 감쌀 뿐이지만, 나중에 팹 DB(SQL)가 붙으면 이 파일 안쪽만
// 교체하면 되도록 인터페이스를 분리해 둔다 (SQL 관련 코드는 아직
// 작성하지 않는다).
"use client";

import { getAlertsData, getConfigTreemap } from "@/lib/api";
// W-1: analysis/alarms는 항상 호출부(모니터링 페이지)가 공용
// AnalysisStateProvider에서 이미 읽은 값을 그대로 넘긴다 -- 이 파일이
// 따로 GET /api/state/latest를 다시 부르지 않는다(지시서: "화면별로
// 개별 fetch하지 마라"). 복원된 값이 쓸 만한지(isAnalysisSnapshotUsable)
// 는 그 값을 만든 쪽(AnalysisStateProvider의 hydrate/스냅샷 대체 채움)이
// 이미 판단했으므로 여기서 다시 검사하지 않는다. 타입만 참조하므로(값은
// 쓰지 않는다) 순환 임포트가 되지 않는다.
import type { AlarmsState, AnalysisState } from "@/components/AnalysisStateProvider";
import type { ConfigTreemapResponse, FmeaTablePayload, LatestAlarmsRecord, MeasurementExpansionResponse } from "@/types/data";

function average(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
}

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

export async function getTreemapData(dataset: string, step: number): Promise<ConfigTreemapResponse | null> {
  try {
    return await getConfigTreemap(dataset, step);
  } catch {
    // 이 데이터셋에 해당 스텝의 Config가 없는 경우 등 -- 빈 상태로 처리.
    return null;
  }
}

export type MeasurementQueueData = {
  yieldSummary: { predMean: number; predLo: number; predHi: number; totalWafers: number } | null;
};

/** SUMMARY의 예상 수율 구간을 계산한다 -- 기존 알람 API(getAlertsData)의
 * wafer별 원시 예측치를 평균해 점추정을 구한다. 지시서 K-5: 랏 단위
 * 집계(분산·사유·계측 권고 표)는 화면 높이만 과도하게 차지하고 홈의
 * 요약 성격에 맞지 않아 제거했다 -- 그 집계 로직도 함께 지운다(더 쓰는
 * 곳이 없다).
 *
 * 구간(predLo/predHi)은 wafer별 pred_lo/pred_hi를 평균하지 않는다 --
 * 그건 웨이퍼 한 장의 conformal 여유(interval_conformal_q)를 1,000장
 * 평균에 그대로 적용하는 셈이라 평균의 불확실성을 개별값 수준으로
 * 과대평가한다(spec GA). 서버가 랏 블록 부트스트랩으로 별도 산출한
 * 집계 여유(interval_conformal_q_agg)를 점추정 평균에 적용한다. 이
 * 값이 없으면(랏 수 부족) 구간을 내지 않는다 -- 웨이퍼 여유로 대체하면
 * 다시 같은 과대평가 버그가 된다. */
export async function getMeasurementQueue(alarmsRecord: LatestAlarmsRecord | null): Promise<MeasurementQueueData> {
  if (!alarmsRecord) return { yieldSummary: null };

  let alerts;
  try {
    alerts = await getAlertsData(alarmsRecord.train_dataset, alarmsRecord.eval_dataset);
  } catch {
    return { yieldSummary: null };
  }
  const preds = alerts.predictions;
  if (preds.length === 0) return { yieldSummary: null };

  const predMean = average(preds.map((p) => p.pred_mean));
  const qAgg = alerts.interval_conformal_q_agg;
  const { predLo, predHi } = qAgg != null ? { predLo: predMean - qAgg, predHi: predMean + qAgg } : { predLo: predMean, predHi: predMean };

  return {
    yieldSummary: { predMean, predLo, predHi, totalWafers: alerts.total_wafers },
  };
}
