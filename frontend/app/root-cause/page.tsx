"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { DEFAULT_SENSITIVITY, DEFAULT_TARGET_YIELD, useAnalysisState } from "@/components/AnalysisStateProvider";
import CompareAcrossConfigsModal from "@/components/CompareAcrossConfigsModal";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import FallbackModeBadge from "@/components/FallbackModeBadge";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import ParetoChart, { buildParetoSummaryText } from "@/components/ParetoChart";
import { usePanelState } from "@/components/PanelStateProvider";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type QuickLookView, type ScatterColorMode, type ScatterView } from "@/components/ScatterChart";
import { selectDisplayFactors } from "@/lib/chartSelection";
import { hasReliableEvidence, TIER_LABEL } from "@/lib/confidenceTier";
import { buildCategoricalSpec, noChartReason, noChartReasonText, TARGETS } from "@/lib/constants";
import { formatPValue } from "@/lib/numberFormat";
import { ANALYSIS_SNAPSHOT_VERSION } from "@/lib/snapshotVersion";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import {
  activateDataset,
  ApiNetworkError,
  ApiResponseError,
  ApiTimeoutError,
  createFavorite,
  deleteFavorite,
  dispatchAlarmNotifications,
  getAlarms,
  getDatasetSchema,
  getFavorites,
  getMeasurementExpansion,
  getScreeningHeatmap,
  getScreeningPareto,
  getScreeningScatter,
  getScreeningScatterCategorical,
  saveAnalysisState,
} from "@/lib/api";
import type {
  AlarmGrade,
  CategoricalScatterResponse,
  ConfidenceTier,
  DatasetSchemaResponse,
  FavoriteSnapshot,
  MethodComparison,
  ParetoRankingItem,
  ParetoRankingResponse,
  ScatterPoint,
  ScreeningScatterResponse,
  WindowMethod,
} from "@/types/data";

// 알람 판정 GBDT 전환 (spec §B) -- 산점도/Box Plot이 그리는 wafer는 이
// 데이터셋 자기 자신(build_scatter_data(df, df, factor)와 동일하게
// train=eval=datasetId)이므로, 마커도 같은 데이터셋을 자기 자신에 대해
// 판정한 결과를 쓴다. train≠eval로 판정하면(예: 항상 eval="test") 두
// 데이터셋의 분포가 다를 때 알람이 사실상 0건으로 사라질 수 있다(스키마가
// 크게 다른 데이터셋 조합에서 실측: 다른 데이터셋을 test.csv로 판정하면
// 0건, 자기 자신으로 판정하면 수백 건).
async function fetchAlarmGradeByWaferId(
  datasetId: string,
  target?: number,
  sensitivity?: number,
): Promise<Record<string, AlarmGrade>> {
  // 지시서: 알림 기록에서 저장한 목표 수율·민감도를 그대로 넘겨 두 화면의
  // 알람 판정 기준을 일치시킨다. 저장된 값이 없으면(최초 실행 등)
  // undefined로 넘어가 백엔드 기본값(85.0/0.5)을 쓴다.
  const response = await getAlarms(datasetId, datasetId, { target, sensitivity });
  const map: Record<string, AlarmGrade> = {};
  for (const item of response.items) {
    map[item.lot_wafer_id] = item.grade;
  }
  return map;
}

/** "알람 마커 기준: 목표 91.0% · 민감도 0.50 · eval=자기 자신" -- 원인
 * 분석과 알림 기록의 판정 기준이 일치하는지, 그리고 이 화면의 알람이
 * (알림 발송/알림 기록과 달리) 항상 자기 자신을 eval로 판정한다는 것을
 * 화면에서 바로 확인할 수 있게 한다(A-3). 아직 한 번도 계산되지
 * 않았으면(카드 로딩 중 등) null. */
function formatAlarmCriteria(criteria: { target: number; sensitivity: number } | null | undefined): string | null {
  if (!criteria) return null;
  return `목표 ${criteria.target.toFixed(1)}% · 민감도 ${criteria.sensitivity.toFixed(2)} · eval=자기 자신`;
}

// Stable empty-object fallbacks (spec: avoid a fresh `{}` literal every
// render feeding a useMemo/useEffect dependency array, which would defeat
// memoization and refire effects needlessly).
const EMPTY_PARETO_BY_TARGET: Record<string, ParetoRankingResponse> = {};
const EMPTY_SCATTER_BY_KEY: Record<string, ScreeningScatterResponse> = {};
const EMPTY_CATEGORICAL_BY_KEY: Record<string, CategoricalScatterResponse> = {};
const RUN_STAGES = ["인자 스크리닝 중 (5개 타깃)", "Pareto 집계 중", "산점도 준비 중", "히트맵 집계 중", "계측 확대 시뮬레이션 중"];

type ColorMode = ScatterColorMode;
type RunState = "idle" | "running" | "error" | "done";
type ChartCriterion = "significant" | "all";

type AnalysisFailureKind = "network" | "timeout" | "server" | "model_not_ready" | "unknown";

const ANALYSIS_FAILURE_MESSAGE: Record<AnalysisFailureKind, string> = {
  network: "서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  timeout: "분석 시간이 초과되었습니다. 다시 시도해 주세요.",
  server: "분석 중 오류가 발생했습니다. 다시 시도해 주세요.",
  model_not_ready: "모델 학습이 완료되지 않았습니다. 사이드바 하단의 모델 학습에서 먼저 학습을 실행해 주세요.",
  unknown: "분석을 완료하지 못했습니다. 다시 시도해 주세요.",
};

/** Turns a raw fetch/HTTP failure into a screen-safe message -- never
 * shows "Failed to fetch"/"500 Internal Server Error" verbatim (spec
 * §5-2). The raw detail is kept separately for the collapsible
 * "자세히 보기" section and the console, not shown by default. */
function classifyAnalysisFailure(error: unknown): { kind: AnalysisFailureKind; detail: string } {
  if (error instanceof ApiTimeoutError) return { kind: "timeout", detail: error.message };
  if (error instanceof ApiNetworkError) return { kind: "network", detail: error.message };
  if (error instanceof ApiResponseError) {
    if (error.status >= 500) return { kind: "server", detail: `HTTP ${error.status}: ${error.message}` };
    if (/모델|학습/.test(error.message)) return { kind: "model_not_ready", detail: `HTTP ${error.status}: ${error.message}` };
    return { kind: "unknown", detail: `HTTP ${error.status}: ${error.message}` };
  }
  if (error instanceof Error) return { kind: "unknown", detail: error.message };
  return { kind: "unknown", detail: String(error) };
}

/** `보통` 등급 인자의 설명력이 낮은 편임을 알리는 한 줄 캡션 (spec §C-4).
 * train.CSV의 Step24_R1 → Y4(ε² 0.073)가 여기 해당한다. */
/** DE그룹: 즐겨찾기 스냅샷 저장 시 재사용하기 위해 텍스트 생성 로직을
 * 분리했다 -- `보통` 등급이 아니면 해석 문구 자체가 없다(빈 문자열). */
function buildModerateInterpretation(tier: ConfidenceTier, eps2: number): string {
  if (tier !== "moderate") return "";
  return `이 인자의 설명력은 ${(eps2 * 100).toFixed(1)}%로 낮은 편입니다. 다른 요인의 영향이 더 클 수 있습니다.`;
}

function ModerateTierCaption({ tier, eps2 }: { tier: ConfidenceTier; eps2: number }) {
  const text = buildModerateInterpretation(tier, eps2);
  if (!text) return null;
  // DC그룹: 해석 문구를 옅은 카드(.interpretCard)에 담는다 -- 메타 줄
  // 바로 아래 텍스트로 붙어 겹쳐 보이던 것을 시각적으로 분리한다.
  // Pareto·Scatter·Box 세 뷰가 전부 이 컴포넌트/클래스를 공유한다.
  return <p className="interpretCard">{text}</p>;
}


/** Step 2 of a run (or a restore's background point-fill, spec §3-1/§4-2):
 * fetch every displayed factor's full scatter/categorical data for all 5
 * targets' Pareto items. Shared so a live run and a restored-but-lean
 * result refill through the exact same code path. */
async function fetchAllScatterData(
  dataset: string,
  paretoByTarget: Record<string, ParetoRankingResponse>,
): Promise<{
  scatterMap: Record<string, ScreeningScatterResponse>;
  categoricalMap: Record<string, CategoricalScatterResponse>;
}> {
  const fetched = await Promise.all(
    TARGETS.flatMap((t) =>
      (paretoByTarget[t]?.items ?? []).map(async (item) => {
        const key = `${t}::${item.feature}`;
        if (item.kind === "Config") {
          return { key, type: "categorical" as const, data: await getScreeningScatterCategorical(dataset, t, item.feature) };
        }
        return { key, type: "numeric" as const, data: await getScreeningScatter(dataset, t, item.feature) };
      }),
    ),
  );
  const scatterMap: Record<string, ScreeningScatterResponse> = {};
  const categoricalMap: Record<string, CategoricalScatterResponse> = {};
  for (const result of fetched) {
    if (result.type === "categorical") categoricalMap[result.key] = result.data;
    else scatterMap[result.key] = result.data;
  }
  return { scatterMap, categoricalMap };
}

export default function RootCausePage() {
  return (
    <Suspense fallback={null}>
      <RootCauseContent />
    </Suspense>
  );
}

function RootCauseContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setAnalysisDataset, setTrainingPanelOpen } = usePanelState();
  // AG-1/AG-2: 새 파일을 업로드하면 활성 평가 데이터셋으로 전환하고
  // 스냅샷 파이프라인을 1회 실행한다(화면별 개별 재분석은 걸지 않는다).
  // Y 계열이 감지되면 자동 학습은 절대 걸지 않고 안내만 띄운다.
  const [showTrainingSuggestion, setShowTrainingSuggestion] = useState(false);
  const [activateError, setActivateError] = useState("");
  function handleDatasetUploaded(uploadedId: string, hasTargetColumns: boolean) {
    setActivateError("");
    void activateDataset(uploadedId)
      .then(() => refreshSnapshotNow())
      .catch((failure) => {
        setActivateError(failure instanceof Error ? failure.message : "활성 평가 데이터셋 전환에 실패했습니다.");
      });
    setShowTrainingSuggestion(hasTargetColumns);
  }
  // 원인 분석 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- the actual
  // result (Pareto/스크리닝/산점도) lives in the shared AnalysisStateProvider
  // context, not local useState, so tab switching renders it from memory
  // with zero network calls (checklist §탭 이동 #1/#4), and a page
  // reload/reconnect restores a lean (points-less) version of it via
  // GET /api/state/latest.
  const {
    analysis, setAnalysis, hydrated, analysisSnapshotStale, datasetFallbackNotice, alarms,
    snapshot: automationSnapshot, refreshSnapshotNow, training,
  } = useAnalysisState();
  // DE그룹: 즐겨찾기 스냅샷에 저장 시점의 활성 모델(챔피언)을 함께
  // 담는다 -- 이후 재학습/재승격으로 model_id가 바뀌면 "이전 분석 기준"
  // 배지를 붙일 수 있다. 학습 기록이 아직 없으면 null.
  const championVersion = training?.performance.model_id ?? null;
  // ≤767px: 산점도/박스플롯 높이 240px (spec §B-6).
  const isMobileLayout = useIsMobileLayout();
  const chartHeight = isMobileLayout ? 240 : 420;
  // B-5: 즐겨찾기 딥링크(`?dataset=&target=&feature=`)가 저장해 둔
  // 데이터셋으로 연다 -- 이게 없으면 즐겨찾기가 항상 "현재 선택된
  // 데이터셋"으로 열려서, train에서 저장한 카드를 test 상태에서 열면
  // 같은 인자명의 다른 데이터셋 차트가 경고 없이 표시된다. 이미 이
  // 페이지가 마운트된 채로 다른 즐겨찾기를 또 여는 경우는(같은 라우트라
  // 리마운트가 안 됨) target/feature 딥링크와 동일한 기존 한계다.
  const [datasetId, setDatasetId] = useState(searchParams.get("dataset") || "train");
  // hasConfig 판단(Eq. 색상 옵션·팝오버 행 노출 여부)에 쓰는 데이터셋 스키마.
  const [analysisSchema, setAnalysisSchema] = useState<DatasetSchemaResponse | null>(null);
  const [activeTarget, setActiveTarget] = useState(searchParams.get("target") || "Y1");
  const [selectedWafer, setSelectedWafer] = useState<ScatterPoint | null>(null);
  const [compareFeature, setCompareFeature] = useState<string | null>(null);
  // Y1~Y5 비교 모달과 서로 배타적으로 열린다 (아래 openCompare/openTrellis).
  const [trellisFactor, setTrellisFactor] = useState<{ feature: string; step: number } | null>(null);
  // 판정 기준 토글 (spec §B-6) -- 기본값은 "유의한 인자만" (§B-2 규칙 적용).
  // 타깃 선택·데이터셋 변경 시 기본값으로 되돌아간다 (아래 selectTarget과
  // datasetId 변경 이펙트에서 초기화).
  const [chartCriterion, setChartCriterion] = useState<ChartCriterion>("significant");

  const [runState, setRunState] = useState<RunState>("idle");
  // Bumped once per "원인 분석 실행"/"다시 실행" -- folded into each factor
  // card's React `key` so every chart's per-card Color By state (spec
  // §5-3) is forced back to "기본" on a fresh run, even when the same
  // feature reappears at the same list position.
  const [runGeneration, setRunGeneration] = useState(0);
  const [runStageIndex, setRunStageIndex] = useState(0);
  const [runError, setRunError] = useState("");
  const [runErrorDetail, setRunErrorDetail] = useState("");
  const [runElapsedSeconds, setRunElapsedSeconds] = useState(0);
  // A-1: 분석 실행 직후 알림 발송(fire-and-forget)이 실패했을 때 조용히
  // 삼키지 않고 사용자에게 알린다 -- 이전에는 .catch(() => {})로 무시돼
  // 발송 경로가 죽어 있어도 아무도 몰랐다. 분석 결과 자체는 이미 표시된
  // 뒤라 이 실패로 화면을 막지는 않는다.
  const [dispatchNotifyError, setDispatchNotifyError] = useState("");

  const paretoByTarget = analysis?.paretoByTarget ?? EMPTY_PARETO_BY_TARGET;
  const scatterByKey = analysis?.scatterByKey ?? EMPTY_SCATTER_BY_KEY;
  const categoricalByKey = analysis?.categoricalByKey ?? EMPTY_CATEGORICAL_BY_KEY;
  // 셀렉터를 바꿨는데 화면은 이전 데이터셋 결과인 경우 (spec §5-3).
  const datasetMismatch = Boolean(analysis && analysis.dataset !== datasetId);

  useEffect(() => {
    let cancelled = false;
    getDatasetSchema(datasetId)
      .then((result) => {
        if (!cancelled) setAnalysisSchema(result);
      })
      .catch(() => {
        if (!cancelled) setAnalysisSchema(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // 즐겨찾기 (지시서 J) -- `${dataset}::${target}::${feature}::${viewType}` ->
  // favorite_id. D-1: viewType을 키에 포함해야 한다 -- 안 그러면 같은
  // 인자를 Box 뷰로 저장하려는 별 클릭이 기존 Scatter 즐겨찾기와 같은
  // 키로 잡혀 그것을 지워버린다. 목록은 마운트 시 한 번만 불러온다
  // (브라우저 저장소 금지, 서버가 유일한 출처). 별 버튼은 이 맵에 키가
  // 있는지로만 채움 여부를 판단한다.
  const [favoriteIdByKey, setFavoriteIdByKey] = useState<Record<string, string>>({});
  function favoriteKeyOf(s: { dataset: string; target: string; feature: string; viewType: string }): string {
    return `${s.dataset}::${s.target}::${s.feature}::${s.viewType}`;
  }
  useEffect(() => {
    let cancelled = false;
    getFavorites()
      .then((response) => {
        if (cancelled) return;
        const map: Record<string, string> = {};
        for (const item of response.items) {
          map[favoriteKeyOf(item.snapshot)] = item.favorite_id;
        }
        setFavoriteIdByKey(map);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // D-1: 생성/삭제 요청이 아직 끝나지 않은 키는 다시 받지 않는다 --
  // 빠른 더블클릭 시 두 호출 모두 같은(아직 갱신 전) favoriteIdByKey를
  // 보고 둘 다 "생성" 경로로 들어가 중복 레코드가 생기고, 클라이언트는
  // 마지막 id만 기억해 나머지 하나는 지울 수 없는 좀비로 남았다. 이
  // 가드는 useState가 아니라 ref다 -- 같은 렌더에서 연달아 호출되면
  // useState는 아직 반영 전(stale)이라 막지 못한다.
  const pendingFavoriteKeysRef = useRef<Set<string>>(new Set());
  const [pendingFavoriteKeys, setPendingFavoriteKeys] = useState<Set<string>>(new Set());

  async function toggleFavorite(snapshot: FavoriteSnapshot) {
    const key = favoriteKeyOf(snapshot);
    if (pendingFavoriteKeysRef.current.has(key)) return;
    pendingFavoriteKeysRef.current.add(key);
    setPendingFavoriteKeys(new Set(pendingFavoriteKeysRef.current));
    try {
      const existingId = favoriteIdByKey[key];
      if (existingId) {
        setFavoriteIdByKey((previous) => {
          const next = { ...previous };
          delete next[key];
          return next;
        });
        try {
          await deleteFavorite(existingId);
        } catch {
          // Best-effort -- a failed unfavorite just leaves the star filled;
          // the user can retry.
          setFavoriteIdByKey((previous) => ({ ...previous, [key]: existingId }));
        }
        return;
      }
      try {
        const created = await createFavorite(snapshot);
        setFavoriteIdByKey((previous) => ({ ...previous, [key]: created.favorite_id }));
      } catch {
        // Best-effort -- 저장 실패 시 별은 그대로 빈 채로 남는다.
      }
    } finally {
      pendingFavoriteKeysRef.current.delete(key);
      setPendingFavoriteKeys(new Set(pendingFavoriteKeysRef.current));
    }
  }

  const [pendingScrollFeature, setPendingScrollFeature] = useState<string | null>(null);
  const [quickLook, setQuickLook] = useState<{ target: string; feature: string; isConfig: boolean } | null>(null);
  const [quickLookData, setQuickLookData] = useState<ScreeningScatterResponse | CategoricalScatterResponse | null>(null);
  const [quickLookError, setQuickLookError] = useState("");
  const [quickLookColorMode, setQuickLookColorMode] = useState<ColorMode>("default");
  const [quickLookView, setQuickLookView] = useState<QuickLookView>("scatter");
  const initialDeepLinkHandled = useRef(false);

  // Unlike the main 5-card grid (each NumericFactorCard remounts on a new
  // target/feature via its own `key`), this quick-look card is a single
  // persistent instance reused across every heatmap-cell/Pareto-bar/alarm
  // deep-link click -- so its view state needs an explicit reset back to
  // Scatter Plot whenever the selected factor changes (spec §2-2/§8).
  // Adjusting state during render (React's documented alternative to an
  // effect for "reset when a prop changes") instead of useEffect, so it
  // doesn't cause an extra cascading render pass.
  const quickLookKey = quickLook ? `${quickLook.target}::${quickLook.feature}` : "";
  const [prevQuickLookKey, setPrevQuickLookKey] = useState(quickLookKey);
  if (quickLookKey !== prevQuickLookKey) {
    setPrevQuickLookKey(quickLookKey);
    setQuickLookView("scatter");
  }

  // A dataset change no longer wipes the displayed result (spec §5-3:
  // "결과를 자동으로 지우지 마라") -- only the quick-look popover, which is
  // scoped to whatever factor/dataset it was opened for and would show a
  // stale chart otherwise. The mismatch banner (datasetMismatch, above)
  // is what tells the user the selector and the result have diverged.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQuickLook(null);
      setQuickLookData(null);
      // 데이터셋 변경 시 판정 기준 토글도 기본값으로 되돌린다 (spec §B-6).
      setChartCriterion("significant");
    }, 0);
    return () => window.clearTimeout(timer);
  }, [datasetId]);

  // 재접속/새로고침 복원, 그리고 탭을 옮겼다 돌아온 경우 모두 이 마운트
  // 이펙트가 처리한다 -- `hydrated`는 앱 전체에서 한 번만 false->true로
  // 바뀌므로, 이미 하이드레이션이 끝난 뒤에 이 페이지가 (재)마운트되면
  // 즉시 실행된다. 셀렉터/타깃/실행 상태를 컨텍스트의 결과에 맞춰 한 번만
  // 동기화한다 (spec §4-3) -- 이후 사용자가 셀렉터를 바꿔도 다시 개입하지
  // 않는다.
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current) return;
    syncedFromRestore.current = true;
    if (!analysis) return;
    const timer = window.setTimeout(() => {
      // B-5: 즐겨찾기 딥링크가 dataset을 지정했으면 복원된 결과의
      // dataset으로 덮어쓰지 않는다 -- 안 그러면 URL이 가리키는(favorite이
      // 저장된) 데이터셋이 이 타이머 한 번으로 조용히 되돌아가, 곧 열릴
      // quickLook이 엉뚱한 데이터셋의 동명 인자를 보여주게 된다.
      if (!searchParams.get("dataset")) setDatasetId(analysis.dataset);
      if (!searchParams.get("target")) setActiveTarget(analysis.activeTarget);
      setRunState("done");
      setAnalysisDataset(analysis.dataset);
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, analysis]);

  // 복원된 결과는 산점도 좌표를 담고 있지 않다 (spec §3-1) -- 배경에서 한
  // 번 다시 채운다. 채우는 동안에도 스크리닝 표/Pareto/비교 카드 등 좌표가
  // 필요 없는 부분은 이미 즉시 보인다.
  useEffect(() => {
    if (!analysis || analysis.pointsComplete) return;
    let cancelled = false;
    const { dataset, paretoByTarget: restoredPareto } = analysis;
    void (async () => {
      try {
        const { scatterMap, categoricalMap } = await fetchAllScatterData(dataset, restoredPareto);
        if (cancelled) return;
        setAnalysis((previous) =>
          previous && previous.dataset === dataset
            ? { ...previous, scatterByKey: scatterMap, categoricalByKey: categoricalMap, pointsComplete: true }
            : previous,
        );
      } catch {
        // Best-effort background fill -- the user can always click "다시
        // 실행" if this silently fails.
      }
    })();
    // 알람 삼각형(spec §B)은 부트스트랩 앙상블이라 수십 초가 걸릴 수 있다
    // (§A-1) -- 산점도 좌표 복원과 묶어 기다리게 하면 그 시간만큼 차트
    // 전체가 "불러오는 중"에 갇힌다. 별도 요청으로 분리해 늦게 도착해도
    // 삼각형만 나중에 얹히게 한다. 알림 기록에 저장된 목표·민감도를 이
    // 순간의 값으로 한 번 읽어 넘긴다 -- alarms가 나중에 바뀌면 별도의
    // "기준 변경 감지" 이펙트(아래)가 다시 불러온다.
    const targetAtRestore = alarms?.targetYield ?? DEFAULT_TARGET_YIELD;
    const sensitivityAtRestore = alarms?.sensitivity ?? DEFAULT_SENSITIVITY;
    const criteriaAtRestore = alarms?.createdAt ?? null;
    void fetchAlarmGradeByWaferId(dataset, alarms?.targetYield, alarms?.sensitivity)
      .then((alarmGradeByWaferId) => {
        if (cancelled) return;
        setAnalysis((previous) =>
          previous && previous.dataset === dataset
            ? {
                ...previous,
                alarmGradeByWaferId,
                alarmCriteria: { appliedAt: criteriaAtRestore, target: targetAtRestore, sensitivity: sensitivityAtRestore },
              }
            : previous,
        );
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, setAnalysis]);

  // 알림 기록에서 목표 수율·민감도를 바꿔 새로 저장하면(alarms.createdAt이
  // 갱신됨) 이미 그려진 삼각형이 낡은 기준을 쓰고 있는 셈이다 -- 원인
  // 분석에 들어와 있는 동안이든, 다른 탭에 있다가 돌아왔든, createdAt이
  // 마지막으로 적용한 값과 달라진 순간 다시 불러온다. 매번 무조건
  // 재조회하지 않는다: 기준이 바뀌지 않았으면(가장 흔한 경우) 이 조건이
  // 곧장 거짓이 되어 네트워크 요청이 없다. 부트스트랩 앙상블이라 수십
  // 초 걸릴 수 있으므로(§A-1) 로딩 중(alarmGradeByWaferId===null)에는
  // 한 번 더 발사하지 않는다.
  useEffect(() => {
    if (!analysis || !analysis.pointsComplete) return;
    if (analysis.alarmGradeByWaferId === null) return;
    const currentCriteriaAt = alarms?.createdAt ?? null;
    if (analysis.alarmCriteria?.appliedAt === currentCriteriaAt) return;
    let cancelled = false;
    const dataset = analysis.dataset;
    const targetNow = alarms?.targetYield ?? DEFAULT_TARGET_YIELD;
    const sensitivityNow = alarms?.sensitivity ?? DEFAULT_SENSITIVITY;
    setAnalysis((previous) => (previous && previous.dataset === dataset ? { ...previous, alarmGradeByWaferId: null } : previous));
    void fetchAlarmGradeByWaferId(dataset, alarms?.targetYield, alarms?.sensitivity)
      .then((alarmGradeByWaferId) => {
        if (cancelled) return;
        setAnalysis((previous) =>
          previous && previous.dataset === dataset
            ? {
                ...previous,
                alarmGradeByWaferId,
                alarmCriteria: { appliedAt: currentCriteriaAt, target: targetNow, sensitivity: sensitivityNow },
              }
            : previous,
        );
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // 개별 필드만 의존성으로 둔다 -- analysis 객체 전체를 넣으면
    // scatterByKey 등 이 이펙트와 무관한 필드가 바뀔 때마다(배경 채우기
    // 등) 다시 실행된다. 아래 가드들이 불필요한 재실행을 이미 안전하게
    // no-op으로 걸러낸다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    analysis?.dataset,
    analysis?.pointsComplete,
    analysis?.alarmGradeByWaferId,
    analysis?.alarmCriteria,
    alarms?.createdAt,
    alarms?.targetYield,
    alarms?.sensitivity,
    setAnalysis,
  ]);

  async function runAnalysis() {
    setRunState("running");
    setRunError("");
    setRunErrorDetail("");
    setDispatchNotifyError("");
    setRunStageIndex(0);
    setRunElapsedSeconds(0);
    setRunGeneration((generation) => generation + 1);
    const startedAt = Date.now();
    const elapsedTimer = window.setInterval(() => {
      setRunElapsedSeconds(Math.round((Date.now() - startedAt) / 1000));
    }, 1000);
    try {
      setRunStageIndex(1);
      const paretoResults = await Promise.all(
        TARGETS.map((t) => getScreeningPareto(datasetId, t).then((response) => [t, response] as const)),
      );
      const paretoMap: Record<string, ParetoRankingResponse> = Object.fromEntries(paretoResults);

      setRunStageIndex(2);
      const { scatterMap, categoricalMap } = await fetchAllScatterData(datasetId, paretoMap);

      setRunStageIndex(3);
      // Warms the server-side cache with the same computation the
      // heatmap will read -- not a second independent calculation, just
      // a second (cheap, cached) round trip.
      await getScreeningHeatmap(datasetId, "spearman").catch(() => {});

      setRunStageIndex(4);
      // '계측 확대 권고' 카드 (spec §B-7) -- 분석 실행 시 한 번만 계산해
      // 결과에 포함시킨다. 실패해도 분석 자체는 이미 성공했으므로 카드
      // 없이 나머지를 보여준다.
      const measurementExpansion = await getMeasurementExpansion(datasetId).catch(() => null);

      setAnalysis({
        dataset: datasetId,
        createdAt: new Date().toISOString(),
        activeTarget,
        paretoByTarget: paretoMap,
        scatterByKey: scatterMap,
        categoricalByKey: categoricalMap,
        pointsComplete: true,
        measurementExpansion,
        alarmGradeByWaferId: null,
        alarmCriteria: null,
      });
      setRunState("done");
      setAnalysisDataset(datasetId);
      // 알람 심각도 삼각형 (spec §B) -- 부트스트랩 앙상블이라 수십 초가
      // 걸릴 수 있다(§A-1). 위 setAnalysis를 붙잡아 두면 이미 준비된
      // 산점도/Pareto까지 그만큼 늦게 보이므로, 별도 요청으로 분리해
      // 나중에 도착하는 대로 삼각형만 얹는다. 알림 기록에 저장된 목표·
      // 민감도를 그대로 써서 두 화면의 판정 기준을 일치시킨다.
      const targetAtRun = alarms?.targetYield ?? DEFAULT_TARGET_YIELD;
      const sensitivityAtRun = alarms?.sensitivity ?? DEFAULT_SENSITIVITY;
      const criteriaAtRun = alarms?.createdAt ?? null;
      void fetchAlarmGradeByWaferId(datasetId, alarms?.targetYield, alarms?.sensitivity)
        .then((alarmGradeByWaferId) => {
          setAnalysis((previous) =>
            previous && previous.dataset === datasetId
              ? {
                  ...previous,
                  alarmGradeByWaferId,
                  alarmCriteria: { appliedAt: criteriaAtRun, target: targetAtRun, sensitivity: sensitivityAtRun },
                }
              : previous,
          );
        })
        .catch(() => {});
      // 알림 연동 §C-4 "분석 실행 직후" -- fire-and-forget. 신뢰도 게이트·
      // 중복 발송 방지·연결된 채널 유무는 전부 서버(dispatch_alarm_notifications)
      // 가 판단한다: 이 호출은 그저 "지금 막 분석이 끝났다"는 신호일 뿐이고,
      // 실패해도 분석 결과 화면에는 아무 영향이 없어야 한다.
      void dispatchAlarmNotifications(datasetId, datasetId).catch(() => {
        setDispatchNotifyError("알림 발송에 실패했습니다. 알림 기록 탭에서 채널 연결 상태를 확인해 주세요.");
      });
      // 성공 직후 저장 (spec §3-4) -- paretoByTarget만 보낸다. 인자별
      // 산점도 상세(관리한계·권장구간·최적중심 등, 좌표 제외)까지 25개
      // 인자 전부 실으면 그것만으로 ~105KB라 100KB 예산(spec §6)을
      // 넘는다 -- 어차피 복원 직후 배경에서 fetchAllScatterData로 다시
      // 채우므로(위 useEffect), 서버에는 화면 목록 구성에 꼭 필요한
      // Pareto와, 그 자체로 작은 계측 확대 권고 결과만 남긴다.
      void saveAnalysisState(datasetId, {
        activeTarget,
        paretoByTarget: paretoMap,
        measurementExpansion,
        snapshotVersion: ANALYSIS_SNAPSHOT_VERSION,
      }).catch(() => {});
    } catch (failure) {
      // Never leave a stale result on screen after a failure -- it could
      // be mistaken for the new run's output (spec §5-2).
      setAnalysis(null);
      const { kind, detail } = classifyAnalysisFailure(failure);
      console.error("원인 분석 실행 실패:", failure);
      setRunError(ANALYSIS_FAILURE_MESSAGE[kind]);
      setRunErrorDetail(detail);
      setRunState("error");
      setAnalysisDataset(null);
    } finally {
      window.clearInterval(elapsedTimer);
    }
  }

  // "알람 마커 기준: 목표 91.0% · 민감도 0.50" -- 모든 ScatterChart 인스턴스가
  // 같은 문자열을 공유한다(카드마다 다시 계산하지 않는다).
  const alarmCriteriaLabel = formatAlarmCriteria(analysis?.alarmCriteria);

  const activeParetoResponse = paretoByTarget[activeTarget];
  const activeParetoItems: ParetoRankingItem[] = useMemo(
    () => activeParetoResponse?.items ?? [],
    [activeParetoResponse],
  );
  // 차트 표시 규칙 (spec §A-2) -- "유의한 인자만"은 강함·보통만, 3개를
  // 넘어도 전부 표시하고 약함으로 보충하지 않는다 (selectDisplayFactors).
  // "전체 상위 3개"는 등급 무관 ε² 순위대로 정확히 3개다 (items는 이미
  // 백엔드에서 ε² 내림차순 top-5로 내려온다).
  const significantDisplayFactors = useMemo(() => selectDisplayFactors(activeParetoItems), [activeParetoItems]);
  const displayFactors = chartCriterion === "significant" ? significantDisplayFactors : activeParetoItems.slice(0, 3);
  // 이 타깃에서 그릴 차트가 0개인지 (spec §A-4) -- "유의한 인자만" 필터
  // 결과 자체가 비어있는지로 판단한다 (약함만 있고 강함·보통이 0개인
  // 경우도 포함해야 하므로, effect_size_pass_count처럼 약함까지 세는
  // 값에 기대면 안 된다).
  const activeTargetHasNoChart =
    chartCriterion === "significant" && activeParetoItems.length > 0 && significantDisplayFactors.length === 0;
  // 5개 타깃 전부 차트가 0개인지 (spec §A-4) -- killing_event처럼 전 타깃에서
  // 계측된 인자로 불량률이 설명되지 않는 경우, 타깃별 안내 문구를 5번
  // 반복하지 않고 통합 안내 하나만 보여준다.
  const allTargetsHaveNoChart =
    chartCriterion === "significant" &&
    runState === "done" &&
    TARGETS.every((t) => {
      const items = paretoByTarget[t]?.items ?? [];
      return items.length > 0 && selectDisplayFactors(items).length === 0;
    });
  // §B-5 통합 안내에 쓰는 5개 타깃 합산 통계 (검정 건수/효과 크기 통과/최대
  // 설명력).
  const datasetNoChartStats = useMemo(() => {
    if (!allTargetsHaveNoChart) return null;
    let totalTested = 0;
    let effectSizePass = 0;
    let fdrPass = 0;
    let maxEps2 = 0;
    const allItems: ParetoRankingItem[] = [];
    for (const t of TARGETS) {
      const response = paretoByTarget[t];
      if (!response) continue;
      totalTested += response.total_factor_count;
      effectSizePass += response.effect_size_pass_count;
      fdrPass += response.fdr_pass_count;
      maxEps2 = Math.max(maxEps2, response.max_eps2 ?? 0);
      allItems.push(...response.items);
    }
    return { totalTested, effectSizePass, maxEps2, reason: noChartReason(allItems, { totalTested, fdrPassCount: fdrPass, maxEps2 }) };
  }, [allTargetsHaveNoChart, paretoByTarget]);

  /** 인자 카드 하나를 그린다 (numeric -> ScatterChart, Config -> Box Plot) --
   * 메인 그리드가 이 함수를 쓴다. `target`을 클로저의 activeTarget에 기대지
   * 않고 인자로 받는다. */
  // Y1~Y5 비교 / 장비별 Trellis 모달은 서로 배타적으로 열린다 -- 하나를
  // 열 때 다른 쪽을 닫아 동시에 뜨는 일이 없게 한다.
  function openCompare(feature: string) {
    setTrellisFactor(null);
    setCompareFeature(feature);
  }
  function openTrellis(feature: string, step: number) {
    setCompareFeature(null);
    setTrellisFactor({ feature, step });
  }

  function renderFactorCard(target: string, item: ParetoRankingItem, index: number) {
    const isConfig = item.kind === "Config";
    const key = `${target}::${item.feature}`;
    // D-1: viewType별로 별도 즐겨찾기이므로, 채움 여부도 viewType별로
    // 따로 물어야 한다 -- NumericFactorCard는 자기 view 상태를 알므로
    // 함수로 넘겨 카드 내부에서 평가하게 한다.
    const isFavorited = (viewType: string) => Boolean(favoriteIdByKey[`${datasetId}::${target}::${item.feature}::${viewType}`]);
    const isFavoritePending = (viewType: string) => pendingFavoriteKeys.has(`${datasetId}::${target}::${item.feature}::${viewType}`);
    if (!isConfig) {
      return (
        <NumericFactorCard
          key={`${runGeneration}-${target}-${item.feature}`}
          item={item}
          index={index}
          dataset={datasetId}
          activeTarget={target}
          numericData={scatterByKey[key]}
          onSelectWafer={setSelectedWafer}
          onCompare={openCompare}
          onTrellis={openTrellis}
          hasConfig={(analysisSchema?.config_columns.length ?? 0) > 0}
          alarmGradeByWaferId={analysis?.alarmGradeByWaferId}
          alarmCriteriaLabel={alarmCriteriaLabel}
          paretoItems={paretoByTarget[target]?.items ?? []}
          paretoN80={paretoByTarget[target]?.n80 ?? null}
          onParetoBarClick={handleParetoBarClick}
          isFavorited={isFavorited}
          isFavoritePending={isFavoritePending}
          onToggleFavorite={toggleFavorite}
          championVersion={championVersion}
        />
      );
    }
    return (
      <CategoricalFactorCard
        key={`${target}-${item.feature}`}
        item={item}
        index={index}
        dataset={datasetId}
        activeTarget={target}
        championVersion={championVersion}
        categoricalData={categoricalByKey[key]}
        chartHeight={chartHeight}
        isFavorited={isFavorited("box")}
        isFavoritePending={isFavoritePending("box")}
        onToggleFavorite={toggleFavorite}
      />
    );
  }

  function updateUrl(target: string, feature?: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("target", target);
    if (feature) params.set("feature", feature);
    else params.delete("feature");
    router.replace(`/root-cause?${params.toString()}`, { scroll: false });
  }

  function selectTarget(target: string) {
    setActiveTarget(target);
    updateUrl(target);
    // 타깃 선택 시 판정 기준 토글도 기본값으로 되돌린다 (spec §B-6).
    setChartCriterion("significant");
    // Keeps the persisted-for-restore activeTarget in sync with whatever
    // the user is actually looking at, not frozen at whatever it was
    // when the run/restore first completed.
    setAnalysis((previous) => (previous ? { ...previous, activeTarget: target } : previous));
  }

  function openFactor(target: string, feature: string) {
    const isConfig = /_Config$/.test(feature);
    setActiveTarget(target);
    updateUrl(target, feature);
    setQuickLook(null);
    setQuickLookData(null);
    setQuickLookError("");
    // Every newly opened quick-look factor starts at 기본, same as the
    // main list's per-card Color By (spec §5-3).
    setQuickLookColorMode("default");
    const isDisplayed = (paretoByTarget[target]?.items ?? []).some((f) => f.feature === feature);
    if (isDisplayed) {
      setPendingScrollFeature(feature);
    } else {
      setPendingScrollFeature(null);
      setQuickLook({ target, feature, isConfig });
    }
  }

  function handleHeatmapSelect(selection: HeatmapCellSelection) {
    openFactor(selection.target, selection.feature);
  }

  function handleParetoBarClick(item: ParetoRankingItem) {
    openFactor(activeTarget, item.feature);
  }

  // Deep-link support: `?target=&feature=` resolves once the execution's
  // results are available.
  useEffect(() => {
    if (initialDeepLinkHandled.current || runState !== "done") return;
    initialDeepLinkHandled.current = true;
    const featureFromUrl = searchParams.get("feature");
    if (!featureFromUrl) return;
    const timer = window.setTimeout(() => openFactor(activeTarget, featureFromUrl), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runState]);

  useEffect(() => {
    if (!pendingScrollFeature) return;
    const timer = window.setTimeout(() => {
      const displayed = paretoByTarget[activeTarget]?.items ?? [];
      if (!displayed.some((f) => f.feature === pendingScrollFeature)) {
        setPendingScrollFeature(null);
        return;
      }
      const element = document.getElementById(`factor-${pendingScrollFeature}`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
        setPendingScrollFeature(null);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [pendingScrollFeature, activeTarget, paretoByTarget]);

  useEffect(() => {
    if (!quickLook) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const result = quickLook.isConfig
            ? await getScreeningScatterCategorical(datasetId, quickLook.target, quickLook.feature)
            : await getScreeningScatter(datasetId, quickLook.target, quickLook.feature);
          if (!cancelled) setQuickLookData(result);
        } catch (failure) {
          if (!cancelled) setQuickLookError(failure instanceof Error ? failure.message : "산점도를 불러오지 못했습니다.");
        }
      })();
    }, 0);
    document.getElementById("heatmapQuickLook")?.scrollIntoView({ behavior: "smooth", block: "start" });
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quickLook]);

  const quickLookNumeric = quickLook && !quickLook.isConfig ? (quickLookData as ScreeningScatterResponse | null) : null;
  const quickLookCategorical = quickLook && quickLook.isConfig ? (quickLookData as CategoricalScatterResponse | null) : null;
  // 모니터링 트리맵 타일 클릭 딥링크 (`?feature=Step7_Config&config=...`).
  const configFromTreemap = searchParams.get("config");

  return (
    <DashboardShell activeItem="원인 분석">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">인자 진단</span>
        <h1>원인 분석</h1>
        {/* 지시서 U-1: 기능 안내 문구(계측 확대 시 기대 효과…)는 삭제했다
            (이전 지시서 G의 삭제 기준 -- 기능 설명이지 수치 정보가 아니다).
            첫 줄은 1280px에서 한 줄에 들어가도록 (spec §A-1) word-break:
            keep-all만 쓴다 -- white-space: nowrap과 함께 쓰지 않는다(좁은
            화면에서 잘림). */}
        <p className="rootCauseIntro">
          타깃별 Pareto와 강함·보통 등급 인자의 산점도·Box Plot을 확인합니다.
          <br />
          권장 구간은 통계(SPC)와 학습(ML) 두 방식을 비교해 나은 쪽을 채택합니다.
        </p>
        {/* 지시서 U-2: 마지막 분석 실행 시각 -- 회색 설명 바로 아래, 이력이
            없으면(analysis가 null) LastRunNote가 스스로 아무것도 렌더하지
            않는다. */}
        <LastRunNote createdAt={analysis?.createdAt} />
        <FallbackModeBadge />
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(220px,1fr)" }}>
          <DatasetSelector label="분석 대상" value={datasetId} onChange={setDatasetId} onUploaded={handleDatasetUploaded} />
        </div>
        <DatasetMismatchWarning mismatch={datasetMismatch} />
        {activateError && <p className="notifyFieldError">{activateError}</p>}
        {/* AG-2: Y 계열이 감지돼도 자동 학습은 걸지 않는다 -- 안내 +
            모델 학습·자동화 팝업을 여는 링크만 제공한다. */}
        {showTrainingSuggestion && (
          <p className="analysisFallbackNotice" role="status">
            이 파일에는 Y 계열이 있습니다. 학습에 사용하려면 모델 학습·자동화에서 실행하세요.{" "}
            <button type="button" className="linkButton" onClick={() => setTrainingPanelOpen(true)}>
              열기
            </button>
          </p>
        )}
      </section>

      <section className="uploadCard">
        <div className="paretoRunBar">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: "var(--fs-title)" }}>원인 분석 실행</h2>
              <LastRunNote createdAt={analysis?.createdAt} />
            </div>
            {/* 기능 설명("타깃 5개 각각의…")은 삭제했다(이전 지시서 G의
                삭제 기준 -- 지워도 화면의 숫자를 못 읽게 되지 않는다).
                "완료된 결과입니다…"는 실행 전/후를 구분해주는 상태
                안내라 유지한다. 삼항이 아니라 조건부 렌더인 이유: 실행
                전에는 <p> 자체가 없어야 빈 여백이 남지 않는다. */}
            {runState === "done" && (
              <p style={{ margin: "4px 0 0", fontSize: "var(--fs-body)", color: "var(--text-secondary)" }}>
                완료된 결과입니다. 데이터셋을 바꾸면 다시 실행해야 합니다.
              </p>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center" }}>
            {/* W-5: 스냅샷(자동 갱신 파이프라인 산출물)이 이미 있으면 "안
                누르면 아무것도 없다"는 인상을 주지 않도록 보조 액션(테두리
                버튼)으로 낮춘다 -- 화면당 채워진 버튼은 1개만 유지한다.
                반대로 스냅샷도 결과도 없으면(부트스트랩 실패 등) 이
                버튼이 유일한 복구 경로이므로 채워진 스타일로 눈에 띄게
                둔다. */}
            <button
              type="button"
              className={automationSnapshot || runState === "done" ? "button secondary" : "button"}
              disabled={runState === "running"}
              onClick={() => void runAnalysis()}
            >
              {runState === "running" ? "분석 중..." : runState === "error" ? "다시 시도" : "다시 분석"}
            </button>
          </div>
        </div>
        {dispatchNotifyError && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">{dispatchNotifyError}</p>
            </div>
            <button type="button" className="button" onClick={() => setDispatchNotifyError("")}>닫기</button>
          </div>
        )}
        {runState === "running" && (
          <div className="paretoRunProgress" style={{ marginTop: 12 }}>
            <div className="paretoRunProgressTrack">
              <span style={{ width: `${((runStageIndex + 1) / RUN_STAGES.length) * 100}%` }} />
            </div>
            <span className="paretoRunStage">{RUN_STAGES[runStageIndex]} · 경과 {runElapsedSeconds}초</span>
          </div>
        )}
        {runState === "error" && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">{runError}</p>
              {runErrorDetail && (
                <details className="analysisErrorDetail">
                  <summary>자세히 보기</summary>
                  <code>{runErrorDetail}</code>
                </details>
              )}
            </div>
            <button type="button" className="button" onClick={() => void runAnalysis()}>다시 시도</button>
          </div>
        )}
        {/* 지시서 AJ: 서버에 저장된 분석 결과가 있었지만 낡은 버전이거나
            (예: PARETO_TOP_N 5->10) 그 이후 모델이 재학습돼 폐기된 경우 --
            runState는 여전히 "idle"이라 위 두 블록과는 겹치지 않는다.
            조용히 빈 화면만 두지 않고 이유와 재실행 경로를 알려준다. */}
        {runState === "idle" && analysisSnapshotStale && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">
                저장된 분석 결과가 이전 버전이라 불러오지 않았습니다. 원인 분석을 다시 실행해 주세요.
              </p>
            </div>
            <button type="button" className="button" onClick={() => void runAnalysis()}>원인 분석 실행</button>
          </div>
        )}
        {/* 지시서 CB: 저장된 학습/분석/알람 결과가 이미 삭제된 데이터셋을
            가리켜 통째로 버려진 경우 -- 조용히 train으로 바꿔치기하지
            않고(다른 스키마의 옛 payload가 train 라벨을 달고 뜨면 더
            나쁘다) 이유를 안내하고 재실행을 유도한다. */}
        {runState === "idle" && datasetFallbackNotice && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">
                이전에 선택한 데이터셋이 더 이상 없어 train으로 전환했습니다. 원인 분석을 다시 실행해 주세요.
              </p>
            </div>
            <button type="button" className="button" onClick={() => void runAnalysis()}>원인 분석 실행</button>
          </div>
        )}
      </section>

      <HeatmapParetoSection
        datasetId={datasetId}
        enabled={runState === "done"}
        activeTarget={activeTarget}
        onActiveTargetChange={selectTarget}
        onHeatmapCellSelect={handleHeatmapSelect}
        criterionControl={<ChartCriterionToggle value={chartCriterion} onChange={setChartCriterion} />}
      />

      {runState === "done" && (
        <>
          {quickLook && (
            <article id="heatmapQuickLook" className="resultCard factorChartCard">
              <div className="factorChartHeader">
                <div className="factorChartHeaderRow">
                  <div className="factorChartTitleRow">
                    <h2>{quickLook.feature} vs {quickLook.target}</h2>
                    {!quickLook.isConfig && (
                      <ColorBySelect
                        value={quickLookColorMode}
                        onChange={setQuickLookColorMode}
                        hasConfig={(analysisSchema?.config_columns.length ?? 0) > 0}
                      />
                    )}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    {!quickLook.isConfig && <QuickLookViewToggle value={quickLookView} onChange={setQuickLookView} />}
                    <button
                      className="button"
                      type="button"
                      onClick={() => {
                        setQuickLook(null);
                        setQuickLookData(null);
                      }}
                    >
                      닫기
                    </button>
                  </div>
                </div>
                <div className="factorChartHeaderRow meta">
                  <span className="sectionLabel">선택한 인자</span>
                  {quickLookNumeric && (
                    <div className="factorChartMetaLine">
                      <span className="metaItem">n={quickLookNumeric.n.toLocaleString()}</span>
                      <span className="metaItem">ε²={quickLookNumeric.eps2.toFixed(3)}</span>
                      <span className="metaItem">p-value {formatPValue(quickLookNumeric.p_value)}</span>
                      <span className="metaItem">등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}</span>
                    </div>
                  )}
                </div>
              </div>
              {/* 모니터링 트리맵 타일 클릭으로 들어온 경우 (지시서 §4③
                  "Config 필터 적용") -- 어떤 Config 값을 보고 왔는지
                  알려준다. Box Plot 자체는 항상 전체 카테고리를 함께
                  보여주는 게 맞으므로(비교가 목적), 특정 카테고리만
                  숨기는 대신 배너로 표시한다. */}
              {quickLook.isConfig && configFromTreemap && (
                <p className="sectionCaption">트리맵에서 선택: {configFromTreemap}</p>
              )}
              {quickLookError && <p className="errorMessage">{quickLookError}</p>}
              {quickLookNumeric && <ModerateTierCaption tier={quickLookNumeric.confidence_tier} eps2={quickLookNumeric.eps2} />}
              {quickLookCategorical && <ModerateTierCaption tier={quickLookCategorical.confidence_tier} eps2={quickLookCategorical.eps2} />}
              {!quickLookError && quickLookNumeric && !hasReliableEvidence(quickLookNumeric.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  이 인자와 {quickLook.target}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(quickLookNumeric.p_value)}, 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}).
                  아래 경고선은 예측 수율을 기준으로 별도 산출된 것으로 별개이지만, 원인으로 단정할 근거는 부족합니다.
                </p>
              )}
              {!quickLookError && quickLookCategorical && !hasReliableEvidence(quickLookCategorical.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  통계적 신뢰도가 낮습니다 (p = {formatPValue(quickLookCategorical.p_value)}, 등급 {TIER_LABEL[quickLookCategorical.confidence_tier]}).
                </p>
              )}
              {quickLookNumeric ? (
                <ScatterChart
                  data={quickLookNumeric}
                  colorMode={quickLookColorMode}
                  view={quickLookView}
                  onSelectWafer={setSelectedWafer}
                  height={chartHeight}
                  alarmGradeByWaferId={analysis?.alarmGradeByWaferId}
                  alarmCriteriaLabel={alarmCriteriaLabel}
                />
              ) : quickLookCategorical ? (
                <PlotlyChart spec={buildCategoricalSpec(quickLookCategorical)} height={chartHeight} />
              ) : !quickLookError ? (
                <p className="emptyMessage">불러오는 중…</p>
              ) : null}
            </article>
          )}

          {allTargetsHaveNoChart && datasetNoChartStats ? (
            // 전 타깃이 0개(spec §A-4) -- killing_event처럼 타깃별 안내를
            // 5번 반복하지 않고 상단에 한 번만 표시한다.
            <section className="resultCard noChartMessage">
              <h2>강함·보통 등급 인자가 없습니다</h2>
              <p className="noChartStats">
                {/* GB-2: 계측 부족/FDR 미통과/효과크기 미달 중 가장 근본적인
                    사유 하나를 우선 보여준다 -- 셋 다 판정 불가면(드묾)
                    기존 통합 통계 문구로 폴백한다. */}
                {datasetNoChartStats.reason
                  ? noChartReasonText(datasetNoChartStats.reason)
                  : `검정 ${datasetNoChartStats.totalTested.toLocaleString()}건 · 효과 크기 조건 통과 ${datasetNoChartStats.effectSizePass.toLocaleString()}건 · 최대 설명력 ${datasetNoChartStats.maxEps2.toFixed(4)}`}
                <br />
                불량률 변동의 대부분이 계측되지 않은 공정 변수로 설명됩니다. 안전율 예측 기반 알람은 계속 동작합니다.
              </p>
              <p className="noChartStats">전체 상위 3개로 전환하면 순위대로 확인할 수 있습니다.</p>
              <button type="button" className="button" onClick={() => setChartCriterion("all")}>전체 상위 3개로 전환</button>
            </section>
          ) : activeTargetHasNoChart && activeParetoResponse ? (
            <section className="resultCard noChartMessage">
              <h2>{activeTarget} · 강함·보통 등급 인자가 없습니다</h2>
              <p className="noChartStats">
                {(() => {
                  const reason = noChartReason(activeParetoResponse.items, {
                    totalTested: activeParetoResponse.total_factor_count,
                    fdrPassCount: activeParetoResponse.fdr_pass_count,
                    maxEps2: activeParetoResponse.max_eps2,
                  });
                  return reason ? (
                    noChartReasonText(reason)
                  ) : (
                    <>
                      검정 {activeParetoResponse.total_factor_count.toLocaleString()}건 · 효과 크기 조건 통과 {activeParetoResponse.effect_size_pass_count.toLocaleString()}건
                      <br />
                      최대 설명력 {(activeParetoResponse.max_eps2 ?? 0).toFixed(4)}
                    </>
                  );
                })()}
              </p>
              <p className="noChartStats">전체 상위 3개로 전환하면 순위대로 확인할 수 있습니다.</p>
              <button type="button" className="button" onClick={() => setChartCriterion("all")}>전체 상위 3개로 전환</button>
            </section>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
              {displayFactors.map((item, index) => renderFactorCard(activeTarget, item, index))}
            </div>
          )}

        </>
      )}

      {selectedWafer && (
        <WaferDetailPopover
          point={selectedWafer}
          target={activeTarget}
          onClose={() => setSelectedWafer(null)}
          hasConfig={(analysisSchema?.config_columns.length ?? 0) > 0}
        />
      )}
      {compareFeature && (
        <CompareAcrossTargetsModal
          feature={compareFeature}
          originTarget={activeTarget}
          datasetId={datasetId}
          onClose={() => setCompareFeature(null)}
          onSelectTarget={(target) => selectTarget(target)}
        />
      )}
      {trellisFactor && (
        <CompareAcrossConfigsModal
          feature={trellisFactor.feature}
          step={trellisFactor.step}
          target={activeTarget}
          datasetId={datasetId}
          onClose={() => setTrellisFactor(null)}
        />
      )}
    </DashboardShell>
  );
}

/** 표시 기준 토글 (spec §A-2/§A-3) -- Y1~Y5 세그먼트와 같은 sticky 줄에
 * 우측 정렬로 얹힌다 (HeatmapParetoSection의 criterionControl 슬롯).
 * 전환은 이미 받은 Pareto 결과를 클라이언트에서 다시 필터링할 뿐, API를
 * 재호출하지 않는다. "판정 기준"이 아니라 "표시 기준"이라 이름 붙인 것은
 * 이 토글이 등급 판정 자체가 아니라 화면에 그릴 차트만 거르기 때문이다. */
function ChartCriterionToggle({ value, onChange }: { value: ChartCriterion; onChange: (criterion: ChartCriterion) => void }) {
  return (
    <div className="chartCriterionBar">
      <span className="chartCriterionLabel">표시 기준</span>
      <div className="scatterViewToggle" role="group" aria-label="차트 표시 기준">
        <button
          type="button"
          className={`scatterViewToggleBtn ${value === "significant" ? "active" : ""}`}
          onClick={() => onChange("significant")}
        >
          유의한 인자만
        </button>
        <button
          type="button"
          className={`scatterViewToggleBtn ${value === "all" ? "active" : ""}`}
          onClick={() => onChange("all")}
        >
          전체 상위 3개
        </button>
      </div>
    </div>
  );
}

/** Pareto/Scatter/Box view toggle (spec "Pareto를 산점도 카드로 병합") --
 * lives in the card header, same row/height as the title, not inside
 * ScatterChart/ParetoChart themselves: the toggle state is owned by
 * whichever card renders the chart (spec §2-2: "산점도마다 독립적인
 * 상태"), purely a client-side re-render of already-fetched
 * points/bins/pareto items, no new API call on switch. Pareto sits first
 * (탐색 뷰 -> 검증 뷰 순서) and is also the *default* view -- 처음 열었을
 * 때는 "어느 인자를 볼까" 전체 조망(Pareto)이 먼저 오고, 개별 인자를
 * 골라 들어간 뒤에야 "그 인자가 실제로 어떤가" 검증(산점도)으로
 * 넘어가는 게 순서에 맞다. 카드별 독립 상태이므로(spec §2-2) 다른 인자
 * 카드를 열어도 각자 Pareto로 시작한다 -- see NumericFactorCard's
 * `useState<ScatterView>("pareto")`. Quick Look은 별개 용도라 그대로
 * `"scatter"` 기본값을 쓴다(아래 quickLookView). */
/** "비교" 줄 -- Y1~Y5 비교/장비별 Trellis 모달을 여는 트리거다. `보기`
 * 토글과 같은 `.scatterViewToggle*` 마크업을 쓰지만 상태 토글이 아니므로
 * (지시서 A: "눌리면 모달이 열리고 토글은 선택 상태로 남지 않는다")
 * `active` 클래스를 절대 붙이지 않는다. `onTrellis`가 없으면(Config 인자)
 * 버튼 하나만 남는다. */
function CompareToggleRow({ onCompare, onTrellis }: { onCompare: () => void; onTrellis: (() => void) | null }) {
  return (
    <div className="scatterViewToggleRow">
      <span className="scatterViewToggleLabel">비교</span>
      <div className="scatterViewToggle" role="group" aria-label="비교 보기">
        <button
          type="button"
          className="scatterViewToggleBtn"
          title="이 인자가 다른 불량 유형에도 영향을 주는지 확인"
          onClick={onCompare}
        >
          Y1~Y5 비교
        </button>
        {onTrellis && (
          <button
            type="button"
            className="scatterViewToggleBtn"
            title="이 인자의 효과가 장비에 따라 달라지는지 확인"
            onClick={onTrellis}
          >
            장비별 Trellis
          </button>
        )}
      </div>
    </div>
  );
}

function ViewToggle({ value, onChange }: { value: ScatterView; onChange: (view: ScatterView) => void }) {
  return (
    <div className="scatterViewToggleRow">
      <span className="scatterViewToggleLabel">보기</span>
      <div className="scatterViewToggle" role="group" aria-label="차트 보기 방식">
        <button type="button" className={`scatterViewToggleBtn ${value === "pareto" ? "active" : ""}`} onClick={() => onChange("pareto")}>
          Pareto
        </button>
        <button type="button" className={`scatterViewToggleBtn ${value === "scatter" ? "active" : ""}`} onClick={() => onChange("scatter")}>
          Scatter Plot
        </button>
        <button type="button" className={`scatterViewToggleBtn ${value === "box" ? "active" : ""}`} onClick={() => onChange("box")}>
          Box Plot
        </button>
      </div>
    </div>
  );
}

/** Scatter/Box-only toggle for the heatmap/Pareto-bar Quick Look panel
 * (spec: Quick Look에는 Pareto 옵션이 없다 -- 이미 Pareto에서 골라 연 인자를
 * 보는 자리이므로 다시 Pareto로 돌아갈 이유가 없다). Kept as a separate
 * component (rather than reusing ViewToggle with a shown/hidden Pareto
 * button) so quickLookView's QuickLookView type never has to accept
 * "pareto" at the type level. */
function QuickLookViewToggle({ value, onChange }: { value: QuickLookView; onChange: (view: QuickLookView) => void }) {
  return (
    <div className="scatterViewToggleRow">
      <span className="scatterViewToggleLabel">보기</span>
      <div className="scatterViewToggle" role="group" aria-label="차트 보기 방식">
        <button type="button" className={`scatterViewToggleBtn ${value === "scatter" ? "active" : ""}`} onClick={() => onChange("scatter")}>
          Scatter Plot
        </button>
        <button type="button" className={`scatterViewToggleBtn ${value === "box" ? "active" : ""}`} onClick={() => onChange("box")}>
          Box Plot
        </button>
      </div>
    </div>
  );
}

/** SPC/ML 권장구간 산출 방식 토글 -- 보기 토글 바로 아래, 같은 좌측 라벨
 * 폭(스타일 재사용)으로 세로 정렬된다. 전환은 산점도/박스플롯의 보기
 * 전용이며 (spec §2-2/§3-3) 알람 로그·개선 권장 목록은 절대 건드리지
 * 않는다 -- `adopted` 쪽에는 작은 채택 배지를 붙여 기본 선택이 왜 그
 * 값인지 알 수 있게 한다. */
function MethodToggle({
  value,
  adopted,
  onChange,
}: {
  value: WindowMethod;
  adopted: WindowMethod | null;
  onChange: (method: WindowMethod) => void;
}) {
  return (
    <div className="scatterViewToggleRow">
      <span className="scatterViewToggleLabel">방식</span>
      <div className="scatterViewToggle" role="group" aria-label="권장 구간 산출 방식">
        <button type="button" className={`scatterViewToggleBtn ${value === "spc" ? "active" : ""}`} onClick={() => onChange("spc")}>
          SPC{adopted === "spc" && <span className="methodAdoptedBadge" title="채택된 방식"><CheckGlyph /></span>}
        </button>
        <button type="button" className={`scatterViewToggleBtn ${value === "ml" ? "active" : ""}`} onClick={() => onChange("ml")}>
          ML{adopted === "ml" && <span className="methodAdoptedBadge" title="채택된 방식"><CheckGlyph /></span>}
        </button>
      </div>
    </div>
  );
}

// U-5: 텍스트 글리프(✓)는 폰트마다 모양·정렬이 달라진다 -- 장식용 문자
// 대신 크기가 고정된 SVG를 쓴다.
function CheckGlyph() {
  return (
    <svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5 6.5 12 13 4" />
    </svg>
  );
}

/** 권장 구간 산출 방식 비교 카드 (spec §4) -- 표시 전용, 여기서 방식을
 * 선택할 수 없다 (전환은 오직 위 MethodToggle로만). SPC가 채택돼도 배지는
 * 같은 자리에 붙는다. */
function MethodComparisonCard({ methods }: { methods: MethodComparison }) {
  const rows: { key: WindowMethod; title: string; subtitle: string }[] = [
    { key: "spc", title: "SPC", subtitle: "12분위 규칙" },
    { key: "ml", title: "ML", subtitle: "결정트리 학습" },
  ];
  return (
    <div className="methodComparisonCard">
      <div className="methodComparisonHeader">
        <span className="methodComparisonKicker">RECOMMENDED RANGE</span>
        <h3>권장 구간 산출 방식</h3>
        <span className="methodComparisonSource">train.CSV 기준 · 재현율 2배 가중 F2 × 안정성</span>
      </div>
      <div className="methodComparisonGrid">
        {rows.map(({ key, title, subtitle }) => {
          const m = methods[key];
          const isAdopted = methods.adopted === key;
          return (
            <div key={key} className={`methodComparisonCell ${isAdopted ? `adopted-${key}` : ""}`}>
              <div className="methodComparisonCellTitle">
                <span className={`methodComparisonName method-${key}`}>{title}</span>
                <span className="methodComparisonSubtitle">{subtitle}</span>
                {isAdopted && <span className="methodComparisonBadge">채택</span>}
              </div>
              {m ? (
                <>
                  <div className="methodComparisonRange">{formatNum1(m.window[0])} ~ {formatNum1(m.window[1])}</div>
                  <div className="methodComparisonStats">
                    <div><b>{m.recall.toFixed(1)}%</b><span>재현율</span></div>
                    <div><b>{m.precision.toFixed(1)}%</b><span>정밀도</span></div>
                    <div><b>{m.f2.toFixed(1)}</b><span>F2</span></div>
                    <div><b>{m.stability.toFixed(2)}</b><span>안정성</span></div>
                  </div>
                  <div className="methodComparisonScore">점수 {m.score.toFixed(1)}</div>
                </>
              ) : (
                <div className="methodComparisonRange methodComparisonUnavailable">산출 불가</div>
              )}
            </div>
          );
        })}
      </div>
      <p className="methodComparisonReason">{methods.adopted_reason}</p>
      <p className="methodComparisonFootnote">점수는 재현율에 2배 가중한 F2에 구간 안정성을 반영한 값입니다.</p>
    </div>
  );
}

function formatNum1(value: number): string {
  return value.toFixed(1);
}

/** One dropdown per scatter card (spec §5-3) -- no server round-trip on
 * change, `lot_id`/`config` already ride along in the point data
 * ScatterChart already has. */
function ColorBySelect({
  value,
  onChange,
  hasConfig = true,
}: {
  value: ColorMode;
  onChange: (mode: ColorMode) => void;
  // Config 컬럼이 0개인 데이터셋(업로드 데이터셋 등)에서는
  // "Config별" 색상 옵션이 고를 수 있는 값 자체가 없으므로 숨긴다 (spec
  // 문구 전수 검토 §A-5).
  hasConfig?: boolean;
}) {
  // 저장된 즐겨찾기 스냅샷에 옛 값(예: 삭제된 "alarm")이 남아 있어도 화면이
  // 깨지지 않도록 알 수 없는 값은 기본으로 떨어뜨린다 -- value는 항상
  // ColorMode 타입으로 좁혀지지만, 향후 즐겨찾기 복원 경로가 생기면 이
  // 방어가 실제로 쓰인다.
  const knownValues: ColorMode[] = hasConfig ? ["default", "config_model", "lot"] : ["default", "lot"];
  const safeValue = knownValues.includes(value) ? value : "default";
  return (
    <label className="colorBySelectField">
      <span>색상</span>
      <select
        className="colorBySelect"
        value={safeValue}
        onChange={(event) => onChange(event.target.value as ColorMode)}
      >
        <option value="default">기본</option>
        {hasConfig && <option value="config_model">Config별</option>}
        <option value="lot">LOT별</option>
      </select>
    </label>
  );
}

/** Owns its own Color By state locally (spec §5-3: "전역 store에 넣지
 * 마라") -- the parent forces a reset to 기본 by changing this
 * component's `key` (remount) on every new analysis run or target
 * switch, rather than lifting the state up. */
function NumericFactorCard({
  item,
  index,
  dataset,
  activeTarget,
  numericData,
  onSelectWafer,
  onCompare,
  onTrellis,
  hasConfig,
  alarmGradeByWaferId,
  alarmCriteriaLabel,
  paretoItems,
  paretoN80,
  onParetoBarClick,
  isFavorited,
  isFavoritePending,
  onToggleFavorite,
  championVersion,
}: {
  item: ParetoRankingItem;
  index: number;
  dataset: string;
  activeTarget: string;
  numericData: ScreeningScatterResponse | undefined;
  onSelectWafer: (point: ScatterPoint) => void;
  onCompare: (feature: string) => void;
  onTrellis: (feature: string, step: number) => void;
  hasConfig: boolean;
  alarmGradeByWaferId?: Record<string, AlarmGrade> | null;
  alarmCriteriaLabel: string | null;
  // Pareto 보기(spec "Pareto를 산점도 카드로 병합")용 데이터 -- 카드마다
  // 다시 fetch하지 않고 부모(RootCauseContent)가 이미 갖고 있는
  // paretoByTarget에서 한 번만 내려준다.
  paretoItems: ParetoRankingItem[];
  paretoN80: number | null;
  onParetoBarClick: (item: ParetoRankingItem) => void;
  // D-1: viewType(Scatter/Box/Pareto)별로 별도 즐겨찾기다 -- 이 카드가
  // 지금 어떤 view인지는 카드 자신만 아므로, 부모가 boolean이 아니라
  // 함수를 내려줘 카드 내부에서 현재 view로 평가한다.
  isFavorited: (viewType: string) => boolean;
  isFavoritePending: (viewType: string) => boolean;
  onToggleFavorite: (snapshot: FavoriteSnapshot) => void;
  // DE그룹: 즐겨찾기 스냅샷에 저장 시점의 활성 모델(챔피언) id를 함께
  // 담는다.
  championVersion: string | null;
}) {
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  // View state lives per-card (spec §2-2: "산점도마다 독립적인 상태"), never
  // in a shared store/URL/localStorage -- resets for free whenever this
  // card remounts on a new run/target (see its `key` at the call site).
  const [view, setView] = useState<ScatterView>("pareto");
  // ≤767px: 산점도 높이 240px (spec: JSON 보고서 버튼 제거 · 모바일 레이아웃
  // 전환 §B-6).
  const isMobileLayout = useIsMobileLayout();
  const chartHeight = isMobileLayout ? 240 : 480;
  // SPC/ML 토글 상태 (spec §3-2): 기본 선택은 이 인자의 `methods.adopted`를
  // 따른다. `numericData`는 비동기로 한 번만 채워지므로 (같은 카드 인스턴스가
  // 다른 인자 데이터로 바뀌는 일은 없다 -- 위 key가 매 실행/타깃/인자 조합마다
  // 새로 발급된다) "처음 도착했을 때 한 번 반영" 패턴을 useEffect 대신
  // 렌더 중 상태 조정으로 처리한다 (quickLookView가 쓰는 것과 같은 패턴).
  const [method, setMethod] = useState<WindowMethod>("spc");
  const [methodInitialized, setMethodInitialized] = useState(false);
  if (!methodInitialized && numericData?.methods) {
    setMethodInitialized(true);
    setMethod(numericData.methods.adopted);
  }
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`}>
      <div className="factorChartHeader">
        <div className="factorChartHeaderRow">
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
            <FavoriteStarButton
              favorited={isFavorited(view)}
              disabled={isFavoritePending(view)}
              onClick={() =>
                onToggleFavorite({
                  dataset,
                  target: activeTarget,
                  feature: item.feature,
                  viewType: view,
                  colorBy: colorMode,
                  method,
                  isConfig: false,
                  // DE그룹: 해석 문구는 저장 시점 값을 문자열로 스냅샷한다
                  // -- Pareto 보기는 종합 문구, Scatter/Box는 "보통" 등급
                  // 설명력 캡션(둘 다 없으면 빈 문자열).
                  interpretation:
                    view === "pareto"
                      ? buildParetoSummaryText(paretoItems, paretoN80)
                      : buildModerateInterpretation(item.confidence_tier, item.eps2),
                  championVersion,
                })
              }
            />
            <ColorBySelect value={colorMode} onChange={setColorMode} hasConfig={hasConfig} />
          </div>
          <div className="factorChartToggleStack">
            <CompareToggleRow onCompare={() => onCompare(item.feature)} onTrellis={() => onTrellis(item.feature, item.step)} />
            <ViewToggle value={view} onChange={setView} />
            {/* SPC/ML 방식 토글은 Scatter/Box 보기 전용이다 (spec §2-2/§3-3)
                -- Pareto 보기에서는 아무 것도 바꾸지 않으므로 숨긴다. */}
            {view !== "pareto" && numericData?.methods && (
              <MethodToggle value={method} adopted={numericData.methods.adopted} onChange={setMethod} />
            )}
          </div>
        </div>
        <div className="factorChartHeaderRow meta">
          <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
          {numericData && (
            <div className="factorChartMetaLine">
              <span className="metaItem">n={numericData.n.toLocaleString()}</span>
              <span className="metaItem">기여율 {item.contribution_pct.toFixed(1)}%</span>
              <span className="metaItem metaCumulative">누적 {item.cumulative_pct.toFixed(1)}%</span>
              <span className="metaItem">p-value {formatPValue(item.p_value)}</span>
              <span className="metaItem">등급 {TIER_LABEL[item.confidence_tier]}</span>
            </div>
          )}
        </div>
      </div>
      <ModerateTierCaption tier={item.confidence_tier} eps2={item.eps2} />
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(item.p_value)}). 아래 경고선은 예측 수율을 기준으로 별도 산출된 것이라 별개이지만, 원인으로 단정할 근거는 부족합니다.
        </p>
      )}
      {view === "pareto" ? (
        // Pareto는 paretoItems(부모의 paretoByTarget)만 있으면 그릴 수 있다
        // -- numericData(산점도 좌표)를 기다릴 이유가 없다. 복원 직후처럼
        // Pareto는 이미 채워졌는데 좌표만 배경에서 다시 채워지는 중일 때도
        // (spec §3-1) Pareto 보기는 즉시 보여야 한다.
        <ParetoChart
          target={activeTarget}
          items={paretoItems}
          n80={paretoN80}
          activeFeature={item.feature}
          onBarClick={onParetoBarClick}
          embedded
          height={chartHeight}
        />
      ) : numericData ? (
        <>
          <ScatterChart
            data={numericData}
            colorMode={colorMode}
            view={view}
            method={method}
            onSelectWafer={onSelectWafer}
            height={chartHeight}
            alarmGradeByWaferId={alarmGradeByWaferId}
            alarmCriteriaLabel={alarmCriteriaLabel}
          />
          {numericData.methods && <MethodComparisonCard methods={numericData.methods} />}
        </>
      ) : (
        <p className="emptyMessage">불러오는 중…</p>
      )}
    </article>
  );
}

/** Config(범주형) 인자 카드 -- 원래는 root-cause 렌더 루프에 인라인돼
 * 있었으나 컴포넌트로 뺐다. */
function CategoricalFactorCard({
  item,
  index,
  dataset,
  activeTarget,
  categoricalData,
  chartHeight,
  isFavorited,
  isFavoritePending,
  onToggleFavorite,
  championVersion,
}: {
  item: ParetoRankingItem;
  index: number;
  dataset: string;
  activeTarget: string;
  categoricalData: CategoricalScatterResponse | undefined;
  chartHeight: number;
  isFavorited: boolean;
  isFavoritePending: boolean;
  onToggleFavorite: (snapshot: FavoriteSnapshot) => void;
  championVersion: string | null;
}) {
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`}>
      <div className="factorChartMeta">
        <div className="factorChartTitleBlock">
          <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
            <FavoriteStarButton
              favorited={isFavorited}
              disabled={isFavoritePending}
              onClick={() =>
                onToggleFavorite({
                  dataset,
                  target: activeTarget,
                  feature: item.feature,
                  viewType: "box",
                  isConfig: true,
                  interpretation: buildModerateInterpretation(item.confidence_tier, item.eps2),
                  championVersion,
                })
              }
            />
          </div>
        </div>
        {categoricalData && (
          <small className="factorChartStats">
            <span>n={categoricalData.n}</span>
            <span>기여율={item.contribution_pct.toFixed(1)}%</span>
            <span className="metaCumulative">누적={item.cumulative_pct.toFixed(1)}%</span>
            <span>p-value {formatPValue(item.p_value)}</span>
          </small>
        )}
      </div>
      <ModerateTierCaption tier={item.confidence_tier} eps2={item.eps2} />
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (p = {formatPValue(item.p_value)}). 원인으로 단정할 근거는 부족합니다.
        </p>
      )}
      {categoricalData ? (
        <PlotlyChart spec={buildCategoricalSpec(categoricalData)} height={chartHeight} />
      ) : (
        <p className="emptyMessage">불러오는 중…</p>
      )}
    </article>
  );
}

/** 즐겨찾기 별 토글 (지시서 J-2) -- 저장 시점 상태 스냅샷만 넘긴다, 점
 * 데이터는 절대 포함하지 않는다. */
function FavoriteStarButton({
  favorited,
  disabled,
  onClick,
}: {
  favorited: boolean;
  // D-1: 생성/삭제 요청이 진행 중인 동안 버튼을 막는다 -- 빠른 더블클릭이
  // 중복 즐겨찾기(좀비 레코드)를 만드는 걸 막는 시각적 짝.
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`favoriteStarButton ${favorited ? "active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={favorited}
      aria-label={favorited ? "즐겨찾기 해제" : "즐겨찾기 추가"}
      title={favorited ? "즐겨찾기 해제" : "즐겨찾기 추가"}
    >
      {favorited ? "★" : "☆"}
    </button>
  );
}

function WaferDetailPopover({
  point,
  target,
  onClose,
  hasConfig,
}: {
  point: ScatterPoint;
  target: string;
  onClose: () => void;
  // 데이터셋에 Config 컬럼이 아예 없으면(업로드 데이터셋 등)
  // 모든 wafer가 "미계측"으로만 표시되어 "계측 안 됨"인지 "그런 항목 자체가
  // 없음"인지 구분이 안 되므로, 행 자체를 숨긴다 (spec 문구 전수 검토 §A-5).
  hasConfig: boolean;
}) {
  return (
    <div className="waferDetailPopover" style={{ right: 24, bottom: 24 }} role="dialog" aria-label="wafer 상세">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">WAFER</span>
          <h2>{point.lot_wafer_id ?? "미상"}</h2>
        </div>
        <button className="button" type="button" onClick={onClose}>닫기</button>
      </div>
      <dl>
        <dt>LOT</dt><dd>{point.lot_id ?? "-"}</dd>
        <dt>{target}</dt><dd>{point.y.toFixed(2)}</dd>
        <dt>인자값</dt><dd>{point.x.toFixed(2)}</dd>
        {hasConfig && (
          <>
            <dt>Eq.</dt><dd>{point.config ?? "미계측"}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
