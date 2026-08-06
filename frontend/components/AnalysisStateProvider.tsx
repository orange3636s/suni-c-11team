"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { getLatestState } from "@/lib/api";
import type {
  AlarmListResponse,
  AlarmSummaryResponse,
  CategoricalScatterResponse,
  MeasurementExpansionResponse,
  ModelPerformanceResponse,
  ParetoRankingResponse,
  RecommendationListResponse,
  ScreeningScatterResponse,
} from "@/types/data";

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
  paretoByTarget: Record<string, ParetoRankingResponse>;
  analysisReady: boolean;
  activeTarget: string;
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
  // '계측 확대 권고' 카드 (spec 문구 전수 검토 PART B) -- 분석 실행 시 한
  // 번만 계산되어 여기 저장된다. null은 "아직 계산되지 않음"과 "계산에
  // 실패함"을 구분하지 않는다 -- 두 경우 모두 카드를 그리지 않는다.
  measurementExpansion: MeasurementExpansionResponse | null;
} | null;

export type AlarmsState = {
  trainDataset: string;
  evalDataset: string;
  createdAt: string;
  summary: AlarmSummaryResponse;
  alarms: AlarmListResponse;
  recommendations: RecommendationListResponse;
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
  alarms: AlarmsState;
  setAlarms: (value: AlarmsState | ((previous: AlarmsState) => AlarmsState)) => void;
};

const AnalysisStateContext = createContext<AnalysisStateValue | null>(null);

export default function AnalysisStateProvider({ children }: { children: ReactNode }) {
  const [hydrated, setHydrated] = useState(false);
  const [training, setTraining] = useState<TrainingState>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>(null);
  const [alarms, setAlarms] = useState<AlarmsState>(null);
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
            paretoByTarget: {},
            analysisReady: false,
            activeTarget: "Y1",
          });
        }
        if (state.analysis) {
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
          });
        }
        if (state.alarms) {
          setAlarms({
            trainDataset: state.alarms.train_dataset,
            evalDataset: state.alarms.eval_dataset,
            createdAt: state.alarms.created_at,
            summary: state.alarms.payload.summary,
            alarms: state.alarms.payload.alarms,
            recommendations: state.alarms.payload.recommendations,
          });
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
    () => ({ hydrated, training, setTraining, analysis, setAnalysis, alarms, setAlarms }),
    [hydrated, training, analysis, alarms],
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
