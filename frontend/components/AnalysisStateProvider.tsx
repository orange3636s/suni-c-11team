"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { getLatestState, getSnapshot, getSnapshotMeta } from "@/lib/api";
import type { MonitoringSnapshot } from "@/lib/monitoringSource";
import { isAnalysisSnapshotUsable } from "@/lib/snapshotVersion";
import type {
  AlertsDataResponse,
  ActionPriorityPayload,
  AnalysisProgress,
  BootstrapStatus,
  CategoricalScatterResponse,
  FmeaTablePayload,
  HeatmapResponse,
  LastRunStatus,
  ManualEvalOverride,
  ModelPerformanceResponse,
  NotificationSettingsSummary,
  ParetoRankingResponse,
  RefreshSnapshot,
  ScreeningScatterResponse,
  TargetProvenance,
} from "@/types/data";

// J-4: 60초 폴링 간격 -- "탭을 오가는 동안 갱신이 없으면 네트워크 요청이
// 발생하지 않는다"는 meta 조회에는 예외이지만(가벼운 엔드포인트), 굳이
// 더 짧게 잡을 이유는 없다(자동 갱신 주기 자체가 보통 분 단위다).
const SNAPSHOT_META_POLL_MS = 60_000;
// W-4: 첫 기동 부트스트랩이 진행 중인 동안에는 체감 대기를 줄이기 위해
// 더 짧게 돈다 -- 완료(done/failed)되면 다시 60초로 돌아간다.
const SNAPSHOT_META_POLL_BOOTSTRAP_MS = 10_000;
// SF-3: "분석 시작"(refresh_running)이 도는 동안에는 네 화면이 진행
// 표시("분석 진행 중… (2/4) 원인 분석")를 실시간에 가깝게 봐야 한다 --
// 부트스트랩보다도 더 짧게 돈다.
const SNAPSHOT_META_POLL_ANALYSIS_RUNNING_MS = 2_000;
// "방금 갱신됨" 배지를 몇 초간 보여줄지.
const SNAPSHOT_JUST_UPDATED_MS = 5_000;

// 사전 알람 로그 전면 개편 (spec §A-1/§A-2) -- 사용자가 설정한 적 없거나
// 첫 접속이면 이 값이 기본이다. HD그룹: 88.0 -- src/analysis/alarm_gbdt.py의
// DEFAULT_TARGET_YIELD와 반드시 같은 값을 유지한다.
export const DEFAULT_TARGET_YIELD = 88.0;
// AA-4: "오경보 최소" 프리셋(alerts/page.tsx SENSITIVITY_PRESETS.low_fp)과
// 같은 값이어야 한다 -- 다르면 "기본 상태인데 어느 프리셋도 활성이 아닌"
// 어색한 상태가 된다. src/analysis/alarm_gbdt.py의 DEFAULT_SENSITIVITY와도
// 반드시 같은 값을 유지한다(첫 로딩과 서버 판정 기준 불일치 방지).
export const DEFAULT_SENSITIVITY = 0.2;

const DEFAULT_NOTIFICATIONS: NotificationSettingsSummary = {
  slack: { connected: false, target: null, webhook_masked: null, verified_at: null },
  telegram: { connected: false, target: null, chat_id_masked: null, verified_at: null },
  gmail: { connected: false, pending: false, email: null, verified_at: null },
  conditions: { grades: ["심각"], timing: ["on_analysis"] },
  automation: {
    enabled: false,
    sql_host: "",
    sql_port: "",
    sql_db: "",
    sql_user: "",
    refresh_interval_minutes: 60,
    last_run_at: null,
    last_run_status: null,
    last_run_sent_count: null,
  },
  telegram_bot_username: null,
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
  targetProvenance: TargetProvenance | null;
  // FMEA 분석표 (모니터링 홈 블록③, 지시서 IA/JA) -- 계산은 백엔드 전용
  // (`src/analysis/screening/fmea.py`), 프런트는 절대 계산하지 않는다.
  // 자동 갱신 스냅샷(`src/automation/refresh.py`)과 수동 "다시 분석"
  // 저장(`POST /api/state/analysis`, `api/routes/state.py`의 `_with_fmea`)
  // 두 경로 모두 저장 시점에 채워 넣으므로 어느 쪽으로 복원해도 값이
  // 있다. `fmea`가 null인데 `fmeaError`도 null이면 이 저장 경로가
  // 갱신되기 전(JA-1 배포 이전)에 저장된 옛 레코드라는 뜻 -- 다시
  // 분석하면 채워진다. `fmeaError`가 있으면 계산이 실패한 것이다.
  fmea?: FmeaTablePayload | null;
  fmeaError?: string | null;
  // MB/MC: 모니터링 홈 블록①·② -- fmea와 같은 두 경로·같은 null 규칙을
  // 따르지만 항상 train.CSV 기준이라 eval 데이터셋과 무관하다
  // (`_with_action_priority`).
  actionPriority?: ActionPriorityPayload | null;
  actionPriorityError?: string | null;
  // TA그룹: 상관관계 히트맵 캐시 -- 탭을 오가는 동안(CorrelationHeatmap이
  // 언마운트/리마운트돼도) 재요청하지 않도록 scatterByKey와 같은 원리로
  // 이 컨텍스트에 보관한다. 키는 CorrelationHeatmap 내부와 동일한
  // `numeric:${metric}` / `categorical:${configLevel}` 형식 -- 서버에는
  // 저장하지 않는다(스냅샷 예산은 Pareto만으로도 이미 거의 다 쓴다;
  // scatterByKey처럼 재접속 시에는 다시 조회한다). 분석을 새로 실행하면
  // 이 객체 자체가 새로 만들어지므로 자연히 빈 캐시로 시작한다.
  heatmap: Record<string, HeatmapResponse>;
} | null;

export type AlarmsState = {
  trainDataset: string;
  evalDataset: string;
  createdAt: string;
  // 사전 알람 로그 전면 개편 (spec §A-1/§A-2) -- 사용자가 조절한 설정.
  // 서버에도 이 두 값만 저장된다(가벼움).
  targetYield: number;
  sensitivity: number;
  // wafer 수만큼 커질 수 있어 서버에는 저장하지 않는다 -- 재접속 직후에는
  // null이고, 페이지가 배경에서 다시 불러와 채운다.
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
  // 블록①②③(조치 우선순위·조치 가능 범위·데이터 한계)에 필요한 값은
  // 전부 `snapshot.fmea`/`snapshot.actionPriority`에 이미 있다 -- MA-3
  // 재설계 이전에는 수율 예측 API를 별도로 불러 yieldSummary를 함께
  // 캐시했지만, 그 API 호출도(delete 대상 블록만 쓰던 필드였다) 더는
  // 필요 없다.
  snapshot: MonitoringSnapshot;
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
  // D-2: 이전 결과 복원(GET /api/state/latest) 자체가 실패했다는 신호
  // (DB 손상, 네트워크 오류 등) -- "저장된 결과 없음"과 구분해야
  // 사용자가 결과가 사라진 줄 알고 재분석을 다시 돌리지 않는다.
  degraded: boolean;
  retryHydration: () => void;
  alarms: AlarmsState;
  setAlarms: (value: AlarmsState | ((previous: AlarmsState) => AlarmsState)) => void;
  monitoringHome: MonitoringHomeState;
  setMonitoringHome: (value: MonitoringHomeState) => void;
  // 설정 패널 신설 §D-3: state/latest가 마운트 시 1번에 실어 온 알림 설정.
  // 설정 패널의 각 변경 액션(연결/해제/조건 저장)은 자기 응답으로 이 값을
  // 갱신할 뿐, 별도 GET을 새로 만들지 않는다.
  notifications: NotificationSettingsSummary;
  setNotifications: (value: NotificationSettingsSummary) => void;
  // J-3/J-4: 자동 갱신 파이프라인이 저장한 단일 스냅샷 -- 모니터링 홈은
  // 이 값만 읽고 자기 스스로 API를 부르지 않는다(지시서: "탭이 개별로
  // fetch하지 않는다"). null은 "아직 한 번도 갱신되지 않음"(최초 기동)과
  // "복원 실패"를 구분하지 않는다 -- 둘 다 화면은 "첫 갱신 대기 중"으로
  // 같게 취급한다.
  snapshot: RefreshSnapshot | null;
  // schema_version이 바뀐 뒤 남은 옛 스냅샷이 있었다는 신호 -- true면
  // "저장된 결과가 이전 버전이라 불러오지 않았습니다. 다음 갱신
  // 주기에 자동으로 채워집니다." 안내를 보여준다.
  snapshotStaleVersion: boolean;
  // 방금(수 초 이내) 새 스냅샷으로 갱신됐다는 신호 -- 상단에 "방금
  // 갱신됨 · HH:MM"을 잠깐 보여주는 데 쓴다.
  snapshotJustUpdated: boolean;
  // W-4: 첫 기동 부트스트랩 진행 상태. null이면 부트스트랩이 아직 한
  // 번도 보고되지 않은 것(구버전 배포, 또는 메타를 아직 못 받아옴)이고
  // 화면은 이를 "부트스트랩 없음"과 동일하게 취급한다.
  bootstrapStatus: BootstrapStatus | null;
  // 실패 배너의 "다시 시도"용 -- 폴링 타이머를 기다리지 않고 즉시
  // meta/snapshot을 다시 확인한다.
  refreshSnapshotNow: () => void;
  // SC-3: "모델 분석"(부트스트랩·학습 후 자동 복구 포함)이 지금 실행
  // 중인지.
  refreshRunning: boolean;
  // SF-3: 네 화면이 공유하는 "분석 시작" 진행 표시 -- 실행 중이 아니면
  // null.
  analysisProgress: AnalysisProgress | null;
  // SC-2: 등록된 활성 분석 데이터가 업로드/DB 불러오기로 설정된 것이면
  // 그 정보. null이면 내장(기본) 데이터.
  manualEvalOverride: ManualEvalOverride | null;
  // 작업지시(Config 하이드레이션 실패 수정) T4: "분석 시작"의 최근 실행
  // 결과 -- 백그라운드 실행이 조용히 실패했을 때 원인을 보여주는 데 쓴다.
  lastRun: LastRunStatus | null;
};

// W-1: 부트스트랩/주기 자동 갱신이 채운 스냅샷을, 한 번도 저장된 적
// 없는(=클릭한 적 없는) analysis/alarms의 대체 데이터로 쓴다. 화면들이
// 이미 알고 있는 "복원됐지만 아직 좌표/원시 예측치는 없는" 모양
// (pointsComplete:false, data:null)으로 채우므로, 각 화면의 기존
// 배경-채움 이펙트(fetchAllScatterData 등)가 그대로 나머지를 채운다 --
// 화면 쪽 렌더링 코드는 손대지 않는다. `paretoByTarget`/`fmea`/
// `actionPriority`는 스냅샷과 저장된 결과 둘 다 같은 백엔드 함수
// (`_pareto_payload`/`_fmea_payload`/`_action_priority_payload`)로
// 만들어지므로 모양이 같다.
function synthesizeAnalysisFromSnapshot(snap: RefreshSnapshot): AnalysisState {
  const targets = Object.keys(snap.analysis.paretoByTarget ?? {});
  return {
    dataset: snap.source.eval_dataset,
    createdAt: snap.created_at,
    activeTarget: targets[0] ?? "Y1",
    paretoByTarget: snap.analysis.paretoByTarget as Record<string, ParetoRankingResponse>,
    scatterByKey: {},
    categoricalByKey: {},
    pointsComplete: false,
    targetProvenance: snap.analysis.target_provenance ?? null,
    fmea: (snap.analysis.fmea as FmeaTablePayload | null) ?? null,
    fmeaError: snap.analysis.fmeaError ?? null,
    actionPriority: (snap.analysis.actionPriority as ActionPriorityPayload | null) ?? null,
    actionPriorityError: snap.analysis.actionPriorityError ?? null,
    heatmap: {},
  };
}

function synthesizeAlarmsFromSnapshot(snap: RefreshSnapshot): AlarmsState {
  // 옛 알람 등급/게이트 판정 파이프라인이 폐기되면서 백엔드 스냅샷의
  // `alarms` 블록은 항상 null이다(src/automation/refresh.py) -- 여기서
  // 읽던 target_yield/sensitivity는 더 이상 서버가 채우지 않으므로
  // 기본값으로 폴백한다.
  return {
    trainDataset: snap.source.train_dataset,
    evalDataset: snap.source.eval_dataset,
    createdAt: snap.created_at,
    targetYield: snap.alarms?.target_yield ?? DEFAULT_TARGET_YIELD,
    sensitivity: snap.alarms?.sensitivity ?? DEFAULT_SENSITIVITY,
    data: null,
  };
}

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
  // D-2: 복원 실패(서버가 degraded:true를 보고했거나, 이 요청 자체가
  // 실패했거나)를 "저장된 결과 없음"과 구분해 보여준다 -- 안 그러면
  // 사용자가 결과가 사라진 줄 알고 (비싼) 재분석을 다시 돌린다.
  const [degraded, setDegraded] = useState(false);
  const hydrationStarted = useRef(false);
  const [snapshot, setSnapshot] = useState<RefreshSnapshot | null>(null);
  const [snapshotStaleVersion, setSnapshotStaleVersion] = useState(false);
  const [snapshotJustUpdated, setSnapshotJustUpdated] = useState(false);
  const [bootstrapStatus, setBootstrapStatus] = useState<BootstrapStatus | null>(null);
  // SC-3: "모델 분석" 파이프라인(부트스트랩·[분석 시작]·학습 후 자동
  // 복구)이 지금 실행 중인지 -- [분석 시작] 버튼 disabled 여부에 쓴다.
  const [refreshRunning, setRefreshRunning] = useState(false);
  // SF-3: 네 화면이 공유하는 진행 표시.
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null);
  // 작업지시(Config 하이드레이션 실패 수정) T4: "분석 시작"의 최근 실행
  // 결과 -- 실행이 끝난 뒤에도 남아 백그라운드 실패를 화면에 드러낸다.
  const [lastRun, setLastRun] = useState<LastRunStatus | null>(null);
  // SC-2: 업로드/DB 불러오기로 등록된 활성 분석 데이터 -- 있으면 셸
  // 레벨 배너가 "수동 · {filename}"을 보여준다.
  const [manualEvalOverride, setManualEvalOverride] = useState<ManualEvalOverride | null>(null);
  // 폴링 콜백이 매번 최신 snapshot을 읽어야 하는데, setInterval에 넘긴
  // 클로저는 등록 시점 값을 붙잡는다 -- ref로 최신 값을 따라간다.
  const snapshotRef = useRef<RefreshSnapshot | null>(null);
  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);
  // W-4: 다음 폴링 간격을 고르는 재귀 setTimeout이 매 실행 시점의 최신
  // 부트스트랩 상태를 읽어야 한다 -- 위 snapshotRef와 같은 이유.
  const bootstrapStatusRef = useRef<BootstrapStatus | null>(null);
  useEffect(() => {
    bootstrapStatusRef.current = bootstrapStatus;
  }, [bootstrapStatus]);
  // SF-3: refreshRunning도 같은 이유로 ref가 필요하다 -- scheduleNext가
  // 다음 간격을 고를 때 실행 중이면 2초로 바짝 좁힌다.
  const refreshRunningRef = useRef(false);
  useEffect(() => {
    refreshRunningRef.current = refreshRunning;
  }, [refreshRunning]);

  function hydrate() {
    let cancelled = false;
    setDegraded(false);
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
              targetProvenance: state.analysis.payload.targetProvenance ?? null,
              // 지시서 JA-1: 저장 시점(POST /api/state/analysis)에 서버가
              // 채워 넣으므로 그대로 복원한다 -- 이 값이 undefined인 것은
              // JA-1 배포 이전에 저장된 옛 레코드뿐이다. actionPriority도
              // 같은 규칙(`_with_action_priority`).
              fmea: state.analysis.payload.fmea ?? null,
              fmeaError: state.analysis.payload.fmeaError ?? null,
              actionPriority: state.analysis.payload.actionPriority ?? null,
              actionPriorityError: state.analysis.payload.actionPriorityError ?? null,
              heatmap: {},
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
        if (state.degraded) {
          setDegraded(true);
        }
      })
      .catch(() => {
        // spec: 복원 실패가 앱을 막으면 안 된다 -- start empty. D-2: 하지만
        // 조용히 넘어가지는 않는다 -- degraded로 표시해 "결과 없음"과
        // 구분하고 재시도할 수 있게 한다.
        if (!cancelled) setDegraded(true);
      })
      .finally(() => {
        if (!cancelled) setHydrated(true);
      });
    return () => {
      cancelled = true;
    };
  }

  useEffect(() => {
    if (hydrationStarted.current) return;
    hydrationStarted.current = true;
    return hydrate();
  }, []);

  // D-2: 재시도 버튼이 부른다 -- 마운트 가드(hydrationStarted)를 우회해
  // 다시 조회한다. 반환되는 정리 함수는 이 수동 호출에서는 쓰지 않는다
  // (프로바이더는 앱 생명주기 내내 마운트돼 있으므로 위 useEffect의
  // 언마운트 케이스와 달리 실질적 경쟁 조건이 없다).
  function retryHydration() {
    hydrate();
  }

  // J-4: 스냅샷 전체를 받아 상태를 갱신한다. `announce`가 true면(폴링이
  // 감지한 새 스냅샷) "방금 갱신됨" 배지를 잠깐 띄운다 -- 최초 로드
  // 시점에는 "갱신"이 아니라 "복원"이므로 배지를 띄우지 않는다.
  const applySnapshot = useCallback((full: Awaited<ReturnType<typeof getSnapshot>>, announce: boolean) => {
    setSnapshot(full.snapshot);
    setSnapshotStaleVersion(full.stale_version || full.stale_model);
    if (announce && full.snapshot) {
      setSnapshotJustUpdated(true);
      window.setTimeout(() => setSnapshotJustUpdated(false), SNAPSHOT_JUST_UPDATED_MS);
    }
    // W-1: analysis/alarms가 아직 아무것도 없으면(저장된 결과도, 이전에
    // 채운 대체값도 없으면) 스냅샷으로 채운다 -- 함수형 업데이터의 이전
    // 값 검사로 채우므로, 이 콜백이 GET /api/state/latest 복원보다 먼저
    // 끝나든 나중에 끝나든(둘은 별도 요청이라 순서가 보장되지 않는다)
    // 실제 저장된 결과가 있으면 그쪽이 항상 이긴다(hydrate가 무조건
    // setAnalysis/setAlarms로 덮어쓴다).
    if (full.snapshot) {
      const snap = full.snapshot;
      setAnalysis((prev) => prev ?? synthesizeAnalysisFromSnapshot(snap));
      setAlarms((prev) => prev ?? synthesizeAlarmsFromSnapshot(snap));
    }
  }, []);

  // W-4: 실패 배너의 "다시 시도"가 부른다 -- 폴링 타이머를 기다리지 않고
  // 즉시 한 번 meta/snapshot을 다시 확인한다(서버 쪽 자동 갱신이 그 사이
  // 이미 성공했을 수 있다). 아래 effect가 실제 구현을 채워 넣는다.
  const checkForUpdateRef = useRef<() => Promise<void>>(async () => {});
  const refreshSnapshotNow = useCallback(() => {
    void checkForUpdateRef.current();
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timeoutId: number | undefined;
    getSnapshot()
      .then((full) => {
        if (!cancelled) applySnapshot(full, false);
      })
      .catch(() => {
        // spec: 복원 실패가 앱을 막으면 안 된다 -- 다음 폴링/갱신 주기가
        // 다시 채운다.
      });

    // 갱신 감지: 윈도우 포커스 복귀 시 + 주기 폴링. `created_at`만 보는
    // 가벼운 엔드포인트라 실제로 더 최신인 경우에만 전체를 다시 받는다
    // (지시서: "같으면 아무것도 하지 않는다 -- 무조건 재조회 금지").
    // 같은 응답의 `bootstrap` 필드로 첫 기동 진행 상태도 함께 갱신한다.
    async function checkForUpdate() {
      try {
        const meta = await getSnapshotMeta();
        setBootstrapStatus(meta.bootstrap ?? null);
        setRefreshRunning(meta.refresh_running);
        setAnalysisProgress(meta.analysis_progress ?? null);
        setManualEvalOverride(meta.manual_eval_override ?? null);
        setLastRun(meta.last_run ?? null);
        const cachedCreatedAt = snapshotRef.current?.created_at ?? null;
        if (meta.created_at && meta.created_at !== cachedCreatedAt) {
          const full = await getSnapshot();
          if (!cancelled) applySnapshot(full, true);
        }
      } catch {
        // best-effort -- 다음 폴링에서 다시 시도한다.
      }
    }
    checkForUpdateRef.current = checkForUpdate;

    // W-4: 고정 setInterval 대신 재귀 setTimeout을 쓴다 -- 매 회차가
    // 끝난 뒤 그 시점의 최신 부트스트랩 상태(bootstrapStatusRef)를 보고
    // 다음 간격(부트스트랩 중 10초 / 평소 60초)을 새로 고른다.
    function scheduleNext() {
      const delay = refreshRunningRef.current
        ? SNAPSHOT_META_POLL_ANALYSIS_RUNNING_MS
        : bootstrapStatusRef.current?.status === "running"
          ? SNAPSHOT_META_POLL_BOOTSTRAP_MS
          : SNAPSHOT_META_POLL_MS;
      timeoutId = window.setTimeout(() => {
        void checkForUpdate().then(() => {
          if (!cancelled) scheduleNext();
        });
      }, delay);
    }
    // 첫 기동 부트스트랩 배너를 새로고침 없이 바로 보여주려면 진행 상태를
    // 마운트 즉시 한 번 확인해야 한다(그 전까지는 60초를 기다려야 했다).
    void checkForUpdate();
    scheduleNext();

    function handleFocus() {
      void checkForUpdate();
    }
    function handleVisibility() {
      if (document.visibilityState === "visible") void checkForUpdate();
    }
    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      cancelled = true;
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      degraded,
      retryHydration,
      alarms,
      setAlarms,
      monitoringHome,
      setMonitoringHome,
      notifications,
      setNotifications,
      snapshot,
      snapshotStaleVersion,
      snapshotJustUpdated,
      bootstrapStatus,
      refreshSnapshotNow,
      refreshRunning,
      analysisProgress,
      manualEvalOverride,
      lastRun,
    }),
    [
      hydrated, training, analysis, analysisSnapshotStale, datasetFallbackNotice, degraded, alarms, monitoringHome,
      notifications, snapshot, snapshotStaleVersion, snapshotJustUpdated, bootstrapStatus, refreshSnapshotNow,
      refreshRunning, analysisProgress, manualEvalOverride, lastRun,
    ],
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
