// 모니터링 홈의 유일한 데이터 조회 경로 -- 페이지 컴포넌트는 이 3개
// 함수(getLatestSnapshot / getTreemapData / getMeasurementQueue)만
// 호출하고 fetch를 직접 부르지 않는다. 지금은 세 함수 모두 기존 REST
// API를 감쌀 뿐이지만, 나중에 팹 DB(SQL)가 붙으면 이 파일 안쪽만
// 교체하면 되도록 인터페이스를 분리해 둔다 (SQL 관련 코드는 아직
// 작성하지 않는다).
"use client";

import {
  getAlertsData,
  getConfigTreemap,
  getLatestState,
  getScreeningScatter,
} from "@/lib/api";
import { isAnalysisSnapshotUsable } from "@/lib/snapshotVersion";
import type {
  ConfidenceTier,
  ConfigTreemapResponse,
  LatestAlarmsRecord,
  MeasurementExpansionResponse,
  ParetoRankingItem,
  RelationShape,
} from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
// 이 미만이면 이탈률 대신 "계측 N%"를 보여준다 (지시서 §4①) -- 표본이
// 적을 때 이탈률을 크게 보여주면 근거 없는 신호를 만든다.
const LOW_MEASUREMENT_RATE_PCT = 10;
const CONFIG_FORMAT_RE = /^Step\d+_Model\d+_EQ[A-Z]_CH\d+$/;

function average(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
}

function isReliableTier(tier: ConfidenceTier): boolean {
  return tier === "strong" || tier === "moderate";
}

/** 카드 헤더의 유의 인자 한 줄. `feature: null`은 "유의 인자 없음"(빈칸이
 * 정보다 -- 없는 근거를 채우지 않는다). `unknownConfigCount`는 이 인자가
 * 속한 스텝의 Config 값 중 Model/EQ/Chamber 형식으로 파싱되지 않는
 * 값의 개수 -- "확인이 필요한 것" 액션의 근거로 쓴다. */
export type SignificantFactorDetail = {
  target: string;
  feature: string | null;
  kind: string | null;
  step: number | null;
  confidenceTier: ConfidenceTier | null;
  rangeText: string | null;
  deviationText: string | null;
  relationShape: RelationShape | null;
  optimalCenter: number | null;
  unknownConfigCount: number;
};

function emptyFactorDetail(target: string, item: ParetoRankingItem | undefined): SignificantFactorDetail {
  return {
    target,
    feature: null,
    kind: item?.kind ?? null,
    step: item?.step ?? null,
    confidenceTier: item?.confidence_tier ?? null,
    rangeText: null,
    deviationText: null,
    relationShape: null,
    optimalCenter: null,
    unknownConfigCount: 0,
  };
}

async function buildSignificantFactor(
  dataset: string,
  target: string,
  item: ParetoRankingItem | undefined,
  totalWafers: number | null,
): Promise<SignificantFactorDetail> {
  // "유의 인자 없음": 1위조차 없거나, 있어도 근거 부족/관계 없음 등급이면
  // 인자명을 쓰지 않는다 (지시서 §4①).
  if (!item || !isReliableTier(item.confidence_tier)) return emptyFactorDetail(target, item);

  // Config(범주형) 인자는 권장구간/이탈 개념이 없다 -- 인자명·등급만 표시.
  if (item.kind === "Config") {
    return {
      target, feature: item.feature, kind: item.kind, step: item.step,
      confidenceTier: item.confidence_tier, rangeText: null, deviationText: null,
      relationShape: null, optimalCenter: null, unknownConfigCount: 0,
    };
  }

  try {
    const data = await getScreeningScatter(dataset, target, item.feature);
    // "권장구간"은 normal_range다 -- reference_lines의 iqr_lo/hi는 다른
    // 개념(관리한계 LCL/UCL, CompareAcrossTargetsModal 등이 쓰는 것)이라
    // 여기 쓰면 항상 null이 나온다 (Step1_D1 실측: reference_lines에는
    // iqr_lo/hi 자체가 없고 mean/q1/q3/s3_*/s6_*/warning_*만 있음).
    const { lo, hi } = data.normal_range;
    let rangeText: string | null = null;
    if (lo != null && hi != null) rangeText = `${lo.toFixed(1)}–${hi.toFixed(1)}`;
    else if (hi != null) rangeText = `≤ ${hi.toFixed(1)}`;
    else if (lo != null) rangeText = `≥ ${lo.toFixed(1)}`;

    const measurementRatePct = totalWafers ? (data.n / totalWafers) * 100 : null;
    let deviationText: string | null = null;
    if (measurementRatePct != null && measurementRatePct < LOW_MEASUREMENT_RATE_PCT) {
      deviationText = `계측 ${measurementRatePct.toFixed(1)}%`;
    } else if (data.points.length > 0) {
      const outCount = data.points.filter((p) => !p.in_range).length;
      deviationText = `이탈 ${((outCount / data.points.length) * 100).toFixed(0)}%`;
    }

    // 이 인자가 속한 스텝의 Config만 본다(다른 스텝 Config가 섞이면
    // 의미가 없다) -- points[].config는 이미 그 스텝의 Config 컬럼
    // 값이다 (src/analysis/scatter.py의 _config_column_for).
    const unknownConfigs = new Set(
      data.points
        .map((p) => p.config)
        .filter((c): c is string => c != null && !CONFIG_FORMAT_RE.test(c)),
    );

    return {
      target, feature: item.feature, kind: item.kind, step: item.step,
      confidenceTier: item.confidence_tier, rangeText, deviationText,
      relationShape: data.relation_shape, optimalCenter: data.optimal_center,
      unknownConfigCount: unknownConfigs.size,
    };
  } catch {
    // 개별 인자 조회 실패는 그 줄만 비워두고 나머지 타깃은 정상 렌더한다.
    return {
      target, feature: item.feature, kind: item.kind, step: item.step,
      confidenceTier: item.confidence_tier, rangeText: null, deviationText: null,
      relationShape: null, optimalCenter: null, unknownConfigCount: 0,
    };
  }
}

export type MonitoringSnapshot = {
  hasAnalysis: boolean;
  createdAt: string | null;
  dataset: string | null;
  significantFactors: SignificantFactorDetail[];
  measurementExpansion: MeasurementExpansionResponse | null;
  alarmsRecord: LatestAlarmsRecord | null;
};

/** GET /api/state/latest를 감싸고, 타깃 5개의 1위 인자만 개별 조회해
 * (최대 5회) 권장구간·이탈률을 채운 "모니터링에 바로 쓸 수 있는" 스냅샷을
 * 반환한다. 인자별 산점도 상세는 상태 스냅샷에 실려 있지 않으므로
 * (types/data.ts LatestAnalysisPayload 주석 참고) 여기서 보강한다 --
 * 전체 인자를 다시 부르지 않는다. */
export async function getLatestSnapshot(): Promise<MonitoringSnapshot> {
  const state = await getLatestState();
  const analysis = state.analysis;
  // 지시서 AJ-4: 원인 분석 화면(AnalysisStateProvider)과 같은 판정 함수를
  // 써서 낡은 스냅샷(구버전 또는 재학습 이후)을 "분석 없음"으로 취급한다
  // -- 이 판정을 따로 하면 원인 분석은 재실행을 안내하는데 모니터링
  // 홈만 옛 결과를 계속 보여주는 어긋남이 생긴다.
  if (!analysis || !isAnalysisSnapshotUsable(analysis, state.training?.created_at ?? null)) {
    return {
      hasAnalysis: false,
      createdAt: null,
      dataset: null,
      significantFactors: [],
      measurementExpansion: null,
      alarmsRecord: state.alarms,
    };
  }

  const totalWafers = analysis.payload.measurementExpansion?.total_wafers ?? null;
  const significantFactors = await Promise.all(
    TARGETS.map((target) =>
      buildSignificantFactor(analysis.dataset, target, analysis.payload.paretoByTarget[target]?.items[0], totalWafers),
    ),
  );

  return {
    hasAnalysis: true,
    createdAt: analysis.created_at,
    dataset: analysis.dataset,
    significantFactors,
    measurementExpansion: analysis.payload.measurementExpansion ?? null,
    alarmsRecord: state.alarms,
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
 * wafer별 원시 예측치를 평균해 구한다. 지시서 K-5: 랏 단위 집계(분산·
 * 사유·계측 권고 표)는 화면 높이만 과도하게 차지하고 홈의 요약 성격에
 * 맞지 않아 제거했다 -- 그 집계 로직도 함께 지운다(더 쓰는 곳이 없다). */
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

  return {
    yieldSummary: {
      predMean: average(preds.map((p) => p.pred_mean)),
      predLo: average(preds.map((p) => p.pred_lo)),
      predHi: average(preds.map((p) => p.pred_hi)),
      totalWafers: alerts.total_wafers,
    },
  };
}
