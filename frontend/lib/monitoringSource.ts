// 모니터링 홈의 유일한 데이터 조회 경로 -- 페이지 컴포넌트는 이 함수만
// 호출하고 fetch를 직접 부르지 않는다.
"use client";

// analysis/alarms는 항상 호출부(모니터링 페이지)가 공용
// AnalysisStateProvider에서 이미 읽은 값을 그대로 넘긴다 -- 이 파일이
// 따로 GET /api/state/latest를 다시 부르지 않는다(화면별 개별 fetch
// 금지). 복원된 값이 쓸 만한지(isAnalysisSnapshotUsable)
// 는 그 값을 만든 쪽(AnalysisStateProvider의 hydrate/스냅샷 대체 채움)이
// 이미 판단했으므로 여기서 다시 검사하지 않는다. 타입만 참조하므로(값은
// 쓰지 않는다) 순환 임포트가 되지 않는다.
import type { AlarmsState, AnalysisState } from "@/components/AnalysisStateProvider";
import type { ActionPriorityPayload, FmeaTablePayload, LatestAlarmsRecord } from "@/types/data";

export type MonitoringSnapshot = {
  hasAnalysis: boolean;
  createdAt: string | null;
  dataset: string | null;
  // 블록③(데이터 한계) 원천 -- MNAR·분산 분해만 담는다. 자동 갱신
  // 스냅샷과 수동 "다시 분석" 저장 둘 다 백엔드가 채워 보낸다. 셋을
  // 구분해야 한다: 값이 있으면 정상, `fmeaError`가 있으면 계산 실패,
  // 둘 다 null이면 FMEA가 채워지기 전에 저장된 레코드다(다시 분석하면
  // 채워진다).
  fmea: FmeaTablePayload | null;
  fmeaError: string | null;
  // 블록①②(조치 우선순위·조치 가능 범위) 원천 -- 항상 train.CSV
  // 기준이라 eval 데이터셋(위 dataset)과 무관하다. fmea와 같은 null
  // 규칙을 따른다.
  actionPriority: ActionPriorityPayload | null;
  actionPriorityError: string | null;
  alarmsRecord: LatestAlarmsRecord | null;
};

/** AnalysisStateProvider가 이미 들고 있는 analysis/alarms(저장된 결과를
 * 복원했든, 아직 아무도 실행한 적 없어 자동 갱신 스냅샷으로 대체
 * 채웠든)를 그대로 "모니터링에 바로 쓸 수 있는" 스냅샷으로 감싼다.
 * 블록①②③에 필요한 값은 모두 백엔드가 이미 다 계산해
 * analysis.fmea/analysis.actionPriority에 실어 보내므로 여기서 추가로
 * 조회하지 않는다 -- 화면은 읽어서 그리기만 한다.
 */
export async function buildMonitoringSnapshot(
  analysis: AnalysisState,
  alarms: AlarmsState,
): Promise<MonitoringSnapshot> {
  // 이 화면의 렌더 코드는 alarms를 서버 응답 모양(LatestAlarmsRecord,
  // snake_case)으로 다루므로 AnalysisStateProvider의
  // AlarmsState(camelCase)를 그 모양으로 얇게 감싼다.
  const alarmsRecord: LatestAlarmsRecord | null = alarms
    ? {
        schema_version: 1,
        created_at: alarms.createdAt,
        train_dataset: alarms.trainDataset,
        eval_dataset: alarms.evalDataset,
        payload: {},
      }
    : null;

  if (!analysis) {
    return {
      hasAnalysis: false,
      createdAt: null,
      dataset: null,
      fmea: null,
      fmeaError: null,
      actionPriority: null,
      actionPriorityError: null,
      alarmsRecord,
    };
  }

  return {
    hasAnalysis: true,
    createdAt: analysis.createdAt,
    dataset: analysis.dataset,
    fmea: analysis.fmea ?? null,
    fmeaError: analysis.fmeaError ?? null,
    actionPriority: analysis.actionPriority ?? null,
    actionPriorityError: analysis.actionPriorityError ?? null,
    alarmsRecord,
  };
}
