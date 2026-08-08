"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { getLatestState } from "@/lib/api";
import type { MeasurementQueueData, MonitoringSnapshot } from "@/lib/monitoringSource";
import { isAnalysisSnapshotUsable } from "@/lib/snapshotVersion";
import type {
  AlarmGrade,
  AlertsDataResponse,
  CategoricalScatterResponse,
  MeasurementExpansionResponse,
  ModelPerformanceResponse,
  NotificationSettingsSummary,
  ParetoRankingResponse,
  ScreeningScatterResponse,
} from "@/types/data";

// 사전 알람 로그 전면 개편 (spec §A-1/§A-2) -- 사용자가 설정한 적 없거나
// 첫 접속이면 이 값이 기본이다.
export const DEFAULT_TARGET_YIELD = 85.0;
export const DEFAULT_SENSITIVITY = 0.5;

const DEFAULT_NOTIFICATIONS: NotificationSettingsSummary = {
  slack: { connected: false, target: null, webhook_masked: null, verified_at: null },
  telegram: { connected: false, target: null, chat_id_masked: null, verified_at: null },
  gmail: { connected: false, pending: false, email: null, verified_at: null },
  conditions: { grades: ["심각"], timing: "on_analysis" },
};

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Two layers, per spec §4:
//   1. Tab switching: this Context holds the FULL result (including
//      scatter points) for the lifetime of the browser tab -- switching
//      away and back is a plain unmount/remount of the page component,
//      never a network call, since the data never left memory.
//   2. Reload/reconnect: on first mount, this provider calls
//      GET /api/state/latest exactly once and seeds whatever the server
//      had -- a narrower payload (no scatter points, spec §3-1) than a
//      live run produces. Pages that see a restored `analysis` entry are
//      responsible for re-fetching full per-factor scatter data in the
//      background (same as a live run's own fetch loop) to fill in points.

export type TrainingState = {
  dataset: string;
  createdAt: string;
  performance: ModelPerformanceResponse;
  // 지시서 I-2/I-3: 모델 학습 팝업의 SQL 연결 정보(비밀번호 제외)·Refresh
  // 주기 -- 서버 재접속 시 팝업이 다시 채워 보여준다.
  sqlHost: string;
  sqlPort: string;
  sqlDb: string;
  sqlUser: string;
  refreshIntervalMinutes: number | null;
} | null;

export type AnalysisState = {
  dataset: string;
  createdAt: string;
  activeTarget: string;
  paretoByTarget: Record<string, ParetoRankingResponse>;
  scatterByKey: Record<string, ScreeningScatterResponse>;
  categoricalByKey: Record<string, CategoricalScatterResponse>;
  // True once scatterByKey/categoricalByKey are known-complete (either a
  // live run just finished, or a restored entry finished its background
  // point-fill fetch) -- false right after a server restore, before that
  // fetch resolves.
  pointsComplete: boolean;
  // '계측 확대 제안' 카드 (spec 문구 전수 검토 PART B) -- 분석 실행 시 한
  // 번만 계산되어 여기 저장된다. null은 "아직 계산되지 않음"과 "계산에
  // 실패함"을 구분하지 않는다 -- 두 경우 모두 카드를 그리지 않는다.
  measurementExpansion: MeasurementExpansionResponse | null;
  // 알람 판정 GBDT 전환 (spec §B) -- wafer_id -> 등급. 분석 실행 시 한 번만
  // 가져와 모든 산점도/Box Plot 카드가 공유한다 (§B-2: 카드마다 재요청하지
  // 않는다).
  alarmGradeByWaferId: Record<string, AlarmGrade> | null;
} | null;

export type AlarmsState = {
  trainDataset: string;
  evalDataset: string;
  createdAt: string;
  // 사전 알람 로그 전면 개편 (spec §A-1/§A-2) -- 사용자가 조절한 설정.
  // 서버에도 이 두 값만 저장된다(가벼움).
  targetYield: number;
  sensitivity: number;
  // wafer 수만큼 커질 수 있어(spec §A/analysis의 alarmGradeByWaferId와
  // 동일한 원칙) 서버에는 저장하지 않는다 -- 재접속 직후에는 null이고,
  // 페이지가 배경에서 다시 불러와 채운다.
  data: AlertsDataResponse | null;
} | null;

// 지시서 K-3: 모니터링 홈이 마지막으로 렌더한 결과를 탭 이동에도 살아남는
// 이 컨텍스트에 보관한다(페이지 컴포넌트의 useState에만 두면 언마운트마다
// 날아간다). `cacheKey`가 마지막 계산 시점의 analysis/training
// createdAt과 같으면 재조회 없이 그대로 쓴다 -- 무효화 조건은 원인 분석
// 재실행·새 학습·명시적 새로고침(하드 리로드로 이 컨텍스트 자체가
// 재생성됨) 셋뿐이다.
export type MonitoringHomeState = {
  cacheKey: string;
  snapshot: MonitoringSnapshot;
  queue: MeasurementQueueData;
} | null;

type AnalysisStateValue = {
  // True once the one-time GET /api/state/latest has settled (success or
  // failure) -- pages use this to decide "still restoring, show a
  // skeleton" vs "nothing saved, show the empty state" (spec §4-2).
  hydrated: boolean;
  training: TrainingState;
  setTraining: (value: TrainingState | ((previous: TrainingState) => TrainingState)) => void;
  analysis: AnalysisState;
  setAnalysis: (value: AnalysisState | ((previous: AnalysisState) => AnalysisState)) => void;
  // 지시서 AJ: 서버에 저장된 analysis 레코드가 있었지만
  // isAnalysisSnapshotUsable이 거부해(구버전/재학습 이후) 복원하지 않았다는
  // 신호 -- true면 원인 분석 화면이 "저장된 결과가 이전 버전이라 불러오지
  // 않았습니다" 안내 + 재실행 버튼을 보여준다. 조용히 빈 화면을 만들지
  // 않기 위한 것이지, `analysis`가 null인 다른 이유(애초에 실행한 적
  // 없음)와는 구분되어야 한다.
  analysisSnapshotStale: boolean;
  // 지시서 CB: 저장된 학습/분석/알람 결과 중 하나 이상이 이미 삭제된
  // 데이터셋(구버전 내장 데이터셋 등)을 가리켜 서버가 통째로 버렸다는
  // 신호. true면 "이전에 선택한 데이터셋이 더 이상 없어 train으로
  // 전환했습니다" 안내를 보여준다.
  datasetFallbackNotice: boolean;
  alarms: AlarmsState;
  setAlarms: (value: AlarmsState | ((previous: AlarmsState) => AlarmsState)) => void;
  monitoringHome: MonitoringHomeState;
  setMonitoringHome: (value: MonitoringHomeState) => void;
  // 설정 패널 신설 §D-3: state/latest가 마운트 시 1번에 실어 온 알림 설정.
  // 설정 패널의 각 변경 액션(연결/해제/조건 저장)은 자기 응답으로 이 값을
  // 갱신할 뿐, 별도 GET을 새로 만들지 않는다.
  notifications: NotificationSettingsSummary;
  setNotifications: (value: NotificationSettingsSummary) => void;
};

const AnalysisStateContext = createContext<AnalysisStateValue | null>(null);

export default function AnalysisStateProvider({ children }: { children: ReactNode }) {
  const [hydrated, setHydrated] = useState(false);
  const [training, setTraining] = useState<TrainingState>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>(null);
  const [analysisSnapshotStale, setAnalysisSnapshotStale] = useState(false);
  const [datasetFallbackNotice, setDatasetFallbackNotice] = useState(false);
  const [alarms, setAlarms] = useState<AlarmsState>(null);
  const [monitoringHome, setMonitoringHome] = useState<MonitoringHomeState>(null);
  const [notifications, setNotifications] = useState<NotificationSettingsSummary>(DEFAULT_NOTIFICATIONS);
  const hydrationStarted = useRef(false);

  useEffect(() => {
    if (hydrationStarted.current) return;
    hydrationStarted.current = true;
    let cancelled = false;
    getLatestState()
      .then((state) => {
        if (cancelled) return;
        if (state.training) {
          setTraining({
            dataset: state.training.dataset,
            createdAt: state.training.created_at,
            performance: state.training.payload.performance,
            sqlHost: state.training.payload.sqlHost ?? "",
            sqlPort: state.training.payload.sqlPort ?? "",
            sqlDb: state.training.payload.sqlDb ?? "",
            sqlUser: state.training.payload.sqlUser ?? "",
            refreshIntervalMinutes: state.training.payload.refreshIntervalMinutes ?? null,
          });
        }
        if (state.analysis) {
          // 지시서 AJ: 저장된 스냅샷이 지금 화면이 이해하는 형태·모델과
          // 여전히 맞는지 프론트가 직접 검사한다 -- 서버는 봉투 형식
          // (schema_version)만 보고 이미 걸러줬을 수 있지만, 그 필터는
          // PARETO_TOP_N처럼 payload *내용* 규칙이 바뀐 경우를 모른다.
          // false면 절대 부분 복원하지 않고(신·구 데이터가 섞이면 안 된다)
          // 통째로 버린 뒤 재실행을 안내한다.
          if (isAnalysisSnapshotUsable(state.analysis, state.training?.created_at ?? null)) {
            // Server-persisted analysis payload is Pareto-only (spec §6
            // size budget) -- scatterByKey/categoricalByKey always start
            // empty on restore and get filled by the page's own background
            // fetchAllScatterData pass (pointsComplete: false triggers it).
            setAnalysis({
              dataset: state.analysis.dataset,
              createdAt: state.analysis.created_at,
              activeTarget: state.analysis.payload.activeTarget,
              paretoByTarget: state.analysis.payload.paretoByTarget,
              scatterByKey: {},
              categoricalByKey: {},
              pointsComplete: false,
              // 좌표와 달리 이 카드는 그 자체로 작아 재계산 없이 그대로
              // 복원한다 (spec §B-7: "카드를 열 때마다 재계산하지 마라").
              measurementExpansion: state.analysis.payload.measurementExpansion ?? null,
              // wafer 수만큼 커질 수 있어(예: train.CSV 1만 행) 서버에 저장하지
              // 않는다 -- scatterByKey와 같은 방식으로 복원 직후 배경에서
              // 다시 채운다 (root-cause/page.tsx의 fetchAllScatterData 이펙트).
              alarmGradeByWaferId: null,
            });
          } else {
            setAnalysisSnapshotStale(true);
          }
        }
        if (state.alarms) {
          setAlarms({
            trainDataset: state.alarms.train_dataset,
            evalDataset: state.alarms.eval_dataset,
            createdAt: state.alarms.created_at,
            targetYield: state.alarms.payload.targetYield ?? DEFAULT_TARGET_YIELD,
            sensitivity: state.alarms.payload.sensitivity ?? DEFAULT_SENSITIVITY,
            data: null,
          });
        }
        if (state.notifications) {
          setNotifications(state.notifications);
        }
        if (state.dataset_fallback_applied) {
          setDatasetFallbackNotice(true);
        }
      })
      .catch(() => {
        // spec: 복원 실패가 앱을 막으면 안 된다 -- start empty, silently.
      })
      .finally(() => {
        if (!cancelled) setHydrated(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<AnalysisStateValue>(
    () => ({
      hydrated,
      training,
      setTraining,
      analysis,
      setAnalysis,
      analysisSnapshotStale,
      datasetFallbackNotice,
      alarms,
      setAlarms,
      monitoringHome,
      setMonitoringHome,
      notifications,
      setNotifications,
    }),
    [hydrated, training, analysis, analysisSnapshotStale, datasetFallbackNotice, alarms, monitoringHome, notifications],
  );

  return <AnalysisStateContext.Provider value={value}>{children}</AnalysisStateContext.Provider>;
}

export function useAnalysisState(): AnalysisStateValue {
  const context = useContext(AnalysisStateContext);
  if (!context) {
    throw new Error("useAnalysisState must be used within AnalysisStateProvider.");
  }
  return context;
}
