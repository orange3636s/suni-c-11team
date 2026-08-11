"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import CompareAcrossConfigsModal from "@/components/CompareAcrossConfigsModal";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import { DatasetMismatchWarning, LastRunNote, TrainingAnalysisDataNote } from "@/components/LastRunNote";
import ParetoChart from "@/components/ParetoChart";
import { usePanelState } from "@/components/PanelStateProvider";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type QuickLookView, type ScatterColorMode } from "@/components/ScatterChart";
import { buildExportFilename, buildFactorCaptionText, buildParetoCaptionText, exportChartAsPng } from "@/lib/chartExport";
import { selectDisplayFactors } from "@/lib/chartSelection";
import { hasReliableEvidence, TIER_LABEL } from "@/lib/confidenceTier";
import { buildCategoricalSpec, TARGETS } from "@/lib/constants";
import { fitDefectRateCurve } from "@/lib/defectRateCurve";
import { ANALYSIS_SNAPSHOT_VERSION } from "@/lib/snapshotVersion";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import {
  ApiNetworkError,
  ApiResponseError,
  ApiTimeoutError,
  createFavorite,
  deleteFavorite,
  dispatchAlarmNotifications,
  getDatasetSchema,
  getFavorites,
  getScreeningPareto,
  getScreeningScatter,
  getScreeningScatterCategorical,
  saveAnalysisState,
  triggerRefresh,
} from "@/lib/api";
import type {
  CategoricalScatterResponse,
  ConfidenceTier,
  DatasetSchemaResponse,
  FavoriteSnapshot,
  HeatmapResponse,
  MethodComparison,
  ParetoRankingItem,
  ParetoRankingResponse,
  ScatterPoint,
  ScreeningScatterResponse,
  WindowMethod,
} from "@/types/data";

// Stable empty-object fallbacks (spec: avoid a fresh `{}` literal every
// render feeding a useMemo/useEffect dependency array, which would defeat
// memoization and refire effects needlessly).
const EMPTY_PARETO_BY_TARGET: Record<string, ParetoRankingResponse> = {};
const EMPTY_SCATTER_BY_KEY: Record<string, ScreeningScatterResponse> = {};
const EMPTY_CATEGORICAL_BY_KEY: Record<string, CategoricalScatterResponse> = {};
const EMPTY_HEATMAP_CACHE: Record<string, HeatmapResponse> = {};
// TA그룹: 히트맵은 더 이상 이 진행 단계를 막지 않는다 -- CorrelationHeatmap이
// 카드 자체의 로딩 상태로 독립적으로 조회하고(analysis.heatmap 캐시에
// 저장), 여기서 미리 fire-and-forget으로 불러 두던 호출은 결과를 어디에도
// 쓰지 않는 중복 요청이었다.
const RUN_STAGES = ["요청 준비 중", "서버 준비 중 · Pareto 집계 중", "산점도·Box Plot 준비 중"];

type ColorMode = ScatterColorMode;
type RunState = "idle" | "running" | "error" | "done";

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
  onResult?: (result: {
    key: string;
    type: "numeric" | "categorical";
    data: ScreeningScatterResponse | CategoricalScatterResponse;
  }) => void,
): Promise<{
  scatterMap: Record<string, ScreeningScatterResponse>;
  categoricalMap: Record<string, CategoricalScatterResponse>;
}> {
  const jobs = TARGETS.flatMap((t) =>
    (paretoByTarget[t]?.items ?? []).map((item) => ({ target: t, item })),
  );
  const scatterMap: Record<string, ScreeningScatterResponse> = {};
  const categoricalMap: Record<string, CategoricalScatterResponse> = {};
  let nextJob = 0;
  const worker = async () => {
    while (nextJob < jobs.length) {
      const job = jobs[nextJob++];
      const key = `${job.target}::${job.item.feature}`;
      try {
        if (job.item.kind === "Config") {
          const data = await getScreeningScatterCategorical(dataset, job.target, job.item.feature);
          categoricalMap[key] = data;
          onResult?.({ key, type: "categorical", data });
        } else {
          const data = await getScreeningScatter(dataset, job.target, job.item.feature);
          scatterMap[key] = data;
          onResult?.({ key, type: "numeric", data });
        }
      } catch (error) {
        console.warn(`개별 차트 로드 실패: ${key}`, error);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, Math.max(jobs.length, 1)) }, () => worker()));
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
  // RD-1: 원인분석 화면의 자체 업로드 경로를 제거했다 -- 데이터셋
  // 전환은 이제 모델 분석 팝업에서만 일어난다.
  const { setAnalysisDataset, setAnalysisPanelOpen } = usePanelState();
  // 원인 분석 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- the actual
  // result (Pareto/스크리닝/산점도) lives in the shared AnalysisStateProvider
  // context, not local useState, so tab switching renders it from memory
  // with zero network calls (checklist §탭 이동 #1/#4), and a page
  // reload/reconnect restores a lean (points-less) version of it via
  // GET /api/state/latest.
  const {
    analysis, setAnalysis, hydrated, analysisSnapshotStale, datasetFallbackNotice,
    snapshot: automationSnapshot, training,
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
  // 지시서 JC-1: 폴백 모드(SQL 미연결)의 원인 분석 기본 데이터셋은
  // train.CSV가 아니라 test.CSV다 -- 상단 배지(FallbackModeBadge: "학습
  // train.CSV → 평가 test.CSV")와 일치시킨다(JC-2). URL의 즐겨찾기
  // 딥링크(dataset 파라미터)나 복원된 분석 결과(아래 syncedFromRestore
  // 이펙트)가 있으면 이 기본값보다 항상 우선한다 -- SQL 연결 상태에서는
  // 그 복원 값이 자동화가 실제로 정한 평가 데이터셋이므로 그대로 따른다.
  const [datasetId, setDatasetId] = useState(searchParams.get("dataset") || "test");
  // hasConfig 판단(Eq. 색상 옵션·팝오버 행 노출 여부)에 쓰는 데이터셋 스키마.
  const [analysisSchema, setAnalysisSchema] = useState<DatasetSchemaResponse | null>(null);
  const [activeTarget, setActiveTarget] = useState(searchParams.get("target") || "Y1");
  const [selectedWafer, setSelectedWafer] = useState<ScatterPoint | null>(null);
  const [compareFeature, setCompareFeature] = useState<string | null>(null);
  // Y1~Y5 비교 모달과 서로 배타적으로 열린다 (아래 openCompare/openTrellis).
  const [trellisFactor, setTrellisFactor] = useState<{ feature: string; step: number } | null>(null);

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
    }, 0);
    return () => window.clearTimeout(timer);
  }, [datasetId]);

  // 재접속/새로고침 복원, 그리고 탭을 옮겼다 돌아온 경우 모두 이 마운트
  // 이펙트가 처리한다 -- `hydrated`는 앱 전체에서 한 번만 false->true로
  // 바뀌므로, 이미 하이드레이션이 끝난 뒤에 이 페이지가 (재)마운트되면
  // 즉시 실행된다. 셀렉터/타깃/실행 상태를 컨텍스트의 결과에 맞춰 한 번만
  // 동기화한다 (spec §4-3) -- 이후 사용자가 셀렉터를 바꿔도 다시 개입하지
  // 않는다.
  //
  // 지시서 JC-1 진단: `hydrated`(GET /api/state/latest)와 `analysis`가
  // 채워지는 시점(저장된 결과가 있으면 즉시, 없으면 자동 갱신 스냅샷의
  // W-1 대체 채움이 나중에 도착)은 서로 다른 요청이라 순서가 보장되지
  // 않는다. 예전 코드는 `hydrated`가 먼저 되면(=이 시점엔 `analysis`가
  // 아직 null) `syncedFromRestore.current`를 바로 true로 세워버려, 이후
  // W-1이 `analysis`를 채워도 이 이펙트가 다시 돌지 않았다 -- 그 결과
  // 데이터셋 셀렉터가 초기값(폴백 모드에서 마땅히 test.CSV여야 할 값)에
  // 영원히 멈춰 있었다. `analysis`가 실제로 왔을 때만 ref를 세워
  // "한 번만 동기화"를 지킨다.
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current || !analysis) return;
    syncedFromRestore.current = true;
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

  // NB-2: "다시 분석" 버튼을 없앤 뒤에도 이 화면은 스스로 채워져야 한다
  // -- 보통은 부트스트랩/자동 갱신 스냅샷이 채운 analysis(W-1 대체
  // 채움, AnalysisStateProvider 참고)가 위 이펙트로 즉시 반영되지만,
  // 그마저 아직 없는 진짜 콜드 상태(스냅샷도 저장된 결과도 없음)에서는
  // 이 화면이 직접 한 번 실행해 채운다. 그동안 진행 바(진행 상태 표시,
  // NB-2 유지 대상)가 보인다. syncedFromRestore와 마찬가지로 한 번만.
  const autoRanOnce = useRef(false);
  useEffect(() => {
    if (!hydrated || autoRanOnce.current || analysis || runState !== "idle") return;
    autoRanOnce.current = true;
    void runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, analysis, runState]);

  // RD-2: 모델 분석 팝업에서 데이터셋을 바꾸면(activate-dataset ->
  // 스냅샷 갱신 -> analysis.dataset 변경) 이 화면의 라벨도 따라간다 --
  // 위 syncedFromRestore와 달리 최초 1회가 아니라 analysis.dataset이
  // 바뀔 때마다 계속 동작한다. 즐겨찾기 딥링크(?dataset=)가 있으면
  // 그 값을 유지한다(위와 같은 이유).
  useEffect(() => {
    if (!analysis || searchParams.get("dataset")) return;
    setDatasetId(analysis.dataset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis?.dataset]);

  // 복원된 결과는 산점도 좌표를 담고 있지 않다 (spec §3-1) -- 배경에서 한
  // 번 다시 채운다. 채우는 동안에도 스크리닝 표/Pareto/비교 카드 등 좌표가
  // 필요 없는 부분은 이미 즉시 보인다.
  useEffect(() => {
    if (!analysis || analysis.pointsComplete) return;
    let cancelled = false;
    const { dataset, paretoByTarget: restoredPareto } = analysis;
    void (async () => {
      try {
        await fetchAllScatterData(dataset, restoredPareto, (result) => {
          if (cancelled) return;
          setAnalysis((previous) => {
            if (!previous || previous.dataset !== dataset) return previous;
            return result.type === "categorical"
              ? {
                  ...previous,
                  categoricalByKey: {
                    ...previous.categoricalByKey,
                    [result.key]: result.data as CategoricalScatterResponse,
                  },
                }
              : {
                  ...previous,
                  scatterByKey: {
                    ...previous.scatterByKey,
                    [result.key]: result.data as ScreeningScatterResponse,
                  },
                };
          });
        });
        if (cancelled) return;
        setAnalysis((previous) =>
          previous && previous.dataset === dataset
            ? { ...previous, pointsComplete: true }
            : previous,
        );
      } catch {
        // Best-effort background fill -- the user can always click "다시
        // 실행" if this silently fails.
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, setAnalysis]);

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
      const paretoResults = await Promise.allSettled(
        TARGETS.map((t) => getScreeningPareto(datasetId, t).then((response) => [t, response] as const)),
      );
      const fulfilledPareto: Array<readonly [string, ParetoRankingResponse]> = [];
      for (const result of paretoResults) {
        if (result.status === "fulfilled") fulfilledPareto.push(result.value);
      }
      if (fulfilledPareto.length === 0) {
        const firstFailure = paretoResults.find((result) => result.status === "rejected");
        throw firstFailure?.reason ?? new Error("모든 타깃의 Pareto 분석이 실패했습니다.");
      }
      const paretoMap: Record<string, ParetoRankingResponse> = Object.fromEntries(fulfilledPareto);
      const targetProvenance = fulfilledPareto.find(([, response]) => response.target_provenance)?.[1].target_provenance ?? null;

      // Pareto가 준비되는 즉시 화면 골격을 연다. 이후 차트는 제한된
      // 동시성으로 하나씩 도착하며, 개별 실패가 나머지 차트를 막지 않는다.
      setAnalysis({
        dataset: datasetId,
        createdAt: new Date().toISOString(),
        activeTarget,
        paretoByTarget: paretoMap,
        scatterByKey: {},
        categoricalByKey: {},
        pointsComplete: false,
        targetProvenance,
        heatmap: {},
      });
      setAnalysisDataset(datasetId);

      setRunStageIndex(2);
      const scatterPromise = fetchAllScatterData(datasetId, paretoMap, (result) => {
        setAnalysis((previous) => {
          if (!previous || previous.dataset !== datasetId) return previous;
          return result.type === "categorical"
            ? {
                ...previous,
                categoricalByKey: {
                  ...previous.categoricalByKey,
                  [result.key]: result.data as CategoricalScatterResponse,
                },
              }
            : {
                ...previous,
                scatterByKey: {
                  ...previous.scatterByKey,
                  [result.key]: result.data as ScreeningScatterResponse,
                },
              };
        });
      }).then(() => {
        setAnalysis((previous) => previous && previous.dataset === datasetId ? { ...previous, pointsComplete: true } : previous);
      });

      await scatterPromise;
      setRunState("done");
      // 알림 연동 §C-4 "분석 실행 직후" -- fire-and-forget. 발송 시점 설정
      // 일치 여부·시간당 예산·연결된 채널 유무는 전부 서버(수율 예측 갱신
      // 파이프라인, dispatch_yield_update)가 판단한다: 이 호출은 그저
      // "지금 막 분석이 끝났다"는 신호일 뿐이고, 실패해도 분석 결과
      // 화면에는 아무 영향이 없어야 한다.
      void dispatchAlarmNotifications(datasetId, datasetId).catch(() => {
        setDispatchNotifyError("알림 발송에 실패했습니다. 수율 예측 탭에서 채널 연결 상태를 확인해 주세요.");
      });
      // 성공 직후 저장 (spec §3-4) -- paretoByTarget만 보낸다. 인자별
      // 산점도 상세(관리한계·권장구간·최적중심 등, 좌표 제외)까지 25개
      // 인자 전부 실으면 그것만으로 ~105KB라 100KB 예산(spec §6)을
      // 넘는다 -- 어차피 복원 직후 배경에서 fetchAllScatterData로 다시
      // 채우므로(위 useEffect), 서버에는 화면 목록 구성에 꼭 필요한
      // Pareto만 남긴다. fmea/actionPriority는 프런트가 보내지 않는다
      // -- 저장 시점에 서버(api/routes/state.py)가 채운다(JA-1).
      void saveAnalysisState(datasetId, {
        activeTarget,
        paretoByTarget: paretoMap,
        targetProvenance,
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

  const analysisVisible = analysis != null && (runState === "running" || runState === "done");

  const activeParetoResponse = paretoByTarget[activeTarget];
  const activeParetoItems: ParetoRankingItem[] = useMemo(
    () => activeParetoResponse?.items ?? [],
    [activeParetoResponse],
  );
  // 지시서 WI-2/YG: 표시 기준은 파레토 기여율 10% 이상 고정(토글 없음) --
  // selectDisplayFactors가 그 필터를 적용한다.
  const displayFactors = useMemo(() => selectDisplayFactors(activeParetoItems), [activeParetoItems]);
  const activeTargetIsEmpty = Boolean(
    analysisVisible && activeParetoResponse && activeParetoResponse.items.length === 0,
  );
  // 이 타깃에서 그릴 차트가 0개인지 -- 순위 자체는 있지만(activeParetoItems
  // 비어있지 않음) 그중 기여율 10% 이상인 인자가 하나도 없는 경우.
  const activeTargetHasNoChart = activeParetoItems.length > 0 && displayFactors.length === 0;
  // WI-2 안내 문구("Y4는 기여율 10% 이상 인자가 없습니다 (최대 8.3%)")의
  // 최대값 -- items는 이미 ε² 내림차순이라 기여율도 같은 순서이지만,
  // 그 가정에 기대지 않고 직접 최댓값을 구한다.
  const maxContributionPct = useMemo(
    () => (activeParetoItems.length === 0 ? 0 : Math.max(...activeParetoItems.map((item) => item.contribution_pct))),
    [activeParetoItems],
  );

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
    // Keeps the persisted-for-restore activeTarget in sync with whatever
    // the user is actually looking at, not frozen at whatever it was
    // when the run/restore first completed.
    setAnalysis((previous) => (previous ? { ...previous, activeTarget: target } : previous));
  }

  // QB-1: Config 여부는 스키마(config_columns)로 판정한다 -- 예전에는
  // `/_Config$/` 이름 패턴에 기대는 휴리스틱이었다. 데이터셋 컬럼명
  // 규칙(Step{n}_Config)이 우연히 일치해 지금까지는 사고가 나지 않았지만,
  // 400 응답을 받고서야 분기하는 방식이 아니라 요청 전에 권위 있는
  // 근거(스키마)로 미리 분기해야 한다(하지 말 것: 400 왕복 후 재호출).
  // 스키마가 아직 없는 극히 짧은 창(딥링크 최초 진입)에서만 이름 패턴을
  // 폴백으로 쓴다.
  function isConfigFeature(feature: string): boolean {
    if (analysisSchema) return analysisSchema.config_columns.includes(feature);
    return /_Config$/.test(feature);
  }

  function openFactor(target: string, feature: string) {
    const isConfig = isConfigFeature(feature);
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
  // 지시서 WI-3: p-value 대신 표시할 R²(차수) -- Quick Look도 같은
  // fitDefectRateCurve(ScatterChart가 추세선을 그릴 때 쓰는 것과 동일
  // 함수)를 이 인자의 산점도 좌표 위에서 그대로 계산한다.
  const quickLookCurveFit = useMemo(
    () => (quickLookNumeric ? fitDefectRateCurve(quickLookNumeric.points) : null),
    [quickLookNumeric],
  );
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
        {/* TD-2: "세 화면 상단" -- 페이지 제목 바로 아래 있는 이 인스턴스에만
            붙인다(837행 부근의 실행 바 안 LastRunNote는 상단이 아니라 실행
            버튼 옆의 보조 표시라 중복 렌더하지 않는다). */}
        <TrainingAnalysisDataNote
          trainFilename={training?.performance?.source_filename ?? null}
          evalFilename={automationSnapshot?.source?.eval_dataset_filename ?? null}
        />
      </section>

      {/* NB-1/NB-2: "분석 대상 [변경]" 카드와 "다시 분석" 실행 버튼을
          제거했다 -- 파일 첨부·분석 실행은 모델 분석·자동화 팝업으로
          일원화됐다. datasetMismatch는 즐겨찾기 딥링크(?dataset=)가 현재
          분석과 다른 데이터셋을 가리키는 경우에도 발생하므로(선택
          카드와 무관) 계속 보여준다. */}
      <DatasetMismatchWarning mismatch={datasetMismatch} />

      <section className="uploadCard">
        <div className="paretoRunBar">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: "var(--fs-title)" }}>원인 분석</h2>
              <LastRunNote createdAt={analysis?.createdAt} />
            </div>
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
            <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
          </div>
        )}
        {/* 지시서 AJ: 서버에 저장된 분석 결과가 있었지만 낡은 버전이거나
            (예: PARETO_TOP_N 5->10) 그 이후 모델이 재학습돼 폐기된 경우 --
            runState는 여전히 "idle"이라 위 두 블록과는 겹치지 않는다.
            조용히 빈 화면만 두지 않고 이유와 재실행 경로를 알려준다.
            NB-3: 이 화면에는 더 이상 재실행 버튼이 없으므로, 모델
            분석·자동화 팝업으로 안내한다. */}
        {runState === "idle" && analysisSnapshotStale && (
          <div className="analysisErrorBox" role="alert">
            <span className="analysisErrorIcon" aria-hidden="true">⚠</span>
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">
                저장된 분석 결과가 이전 버전이라 불러오지 않았습니다. 모델 분석·자동화에서 파일을 업로드하거나 SQL을 갱신하세요.
              </p>
            </div>
            <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
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
                이전에 선택한 데이터셋이 더 이상 없어 train으로 전환했습니다. 모델 분석·자동화에서 파일을 업로드하거나 SQL을 갱신하세요.
              </p>
            </div>
            <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
          </div>
        )}
      </section>

      {analysisVisible && analysis.targetProvenance?.uses_predictions && (
        <p className="analysisDataNotice" role="note">
          이 분석은 실측값이 없는 항목을 모델 예측값으로 보완해 계산했습니다. 예측값 기반 관계는 실제 공정 원인과 다를 수 있으므로 공정 검증과 함께 사용해 주세요. 모델 {analysis.targetProvenance.model_version ?? analysis.targetProvenance.model_id ?? "정보 없음"} · 예측 {analysis.targetProvenance.predicted_target_cells.toLocaleString()}셀 · 실측/예측 혼합 행 {analysis.targetProvenance.mixed_rows.toLocaleString()}개
        </p>
      )}

      <HeatmapParetoSection
        datasetId={datasetId}
        enabled={analysisVisible}
        activeTarget={activeTarget}
        onActiveTargetChange={selectTarget}
        onHeatmapCellSelect={handleHeatmapSelect}
        heatmapInitialCache={analysis?.heatmap ?? EMPTY_HEATMAP_CACHE}
        onHeatmapCacheUpdate={(cache) => setAnalysis((previous) => (previous ? { ...previous, heatmap: cache } : previous))}
      />

      {analysisVisible && (
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
                      <span className="metaItem">ε² {quickLookNumeric.eps2.toFixed(3)}</span>
                      {quickLookCurveFit && (
                        <span className="metaItem">R² {quickLookCurveFit.r2.toFixed(3)} ({quickLookCurveFit.degree}차)</span>
                      )}
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
              {quickLookCategorical && <ModerateTierCaption tier={quickLookCategorical.confidence_tier} eps2={quickLookCategorical.eps2} />}
              {!quickLookError && quickLookNumeric && !hasReliableEvidence(quickLookNumeric.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  이 인자와 {quickLook.target}의 통계적 연관성은 신뢰도가 낮습니다 (ε² = {quickLookNumeric.eps2.toFixed(3)}, 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}).
                  원인으로 단정할 근거는 부족합니다.
                </p>
              )}
              {!quickLookError && quickLookCategorical && !hasReliableEvidence(quickLookCategorical.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  통계적 신뢰도가 낮습니다 (ε² = {quickLookCategorical.eps2.toFixed(3)}, 등급 {TIER_LABEL[quickLookCategorical.confidence_tier]}).
                </p>
              )}
              {quickLookNumeric ? (
                <ScatterChart
                  data={quickLookNumeric}
                  colorMode={quickLookColorMode}
                  view={quickLookView}
                  onSelectWafer={setSelectedWafer}
                  height={chartHeight}
                  reliabilityText={buildModerateInterpretation(quickLookNumeric.confidence_tier, quickLookNumeric.eps2)}
                />
              ) : quickLookCategorical ? (
                <PlotlyChart spec={buildCategoricalSpec(quickLookCategorical)} height={chartHeight} />
              ) : !quickLookError ? (
                <p className="emptyMessage">불러오는 중…</p>
              ) : null}
            </article>
          )}

          {activeTargetIsEmpty && activeParetoResponse ? (
            <section className="resultCard noChartMessage">
              <h2>{activeTarget} 분석 인자가 없습니다</h2>
              <p className="noChartStats">
                분석 가능한 타깃 표본 {activeParetoResponse.analyzable_target_samples?.toLocaleString() ?? "0"}개 · 모델 {activeParetoResponse.model_available ? "사용 가능" : "사용 불가"}
                <br />
                {activeParetoResponse.factor_measurement_insufficient
                  ? "공정 인자의 유효 계측 표본이 부족합니다."
                  : "현재 데이터에서 순위를 계산할 수 있는 인자가 없습니다."}
              </p>
              <p className="noChartStats">모델 상태와 데이터의 R/D/Config 계측값을 확인한 뒤 모델 분석·자동화에서 파일을 업로드하거나 SQL을 갱신하세요.</p>
              <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
            </section>
          ) : (
            <>
              {/* 지시서 WI-1: 파레토는 타깃당 1개, 화면 상단에 고정 -- 인자
                  카드마다 반복해 그리던 것을 걷어냈다. 인자 카드는 그
                  아래에 나열된다. 표시 기준(WI-2/YG, 기여율 10% 이상)과
                  무관하게 이 타깃에 순위가 있는 한(activeParetoItems가
                  비어있지 않은 한) 항상 보인다 -- "이 타깃의 인자 순위"
                  전체를 보여주는 차트라 10% 미만 인자도 막대로는 남는다. */}
              {activeParetoResponse && activeParetoItems.length > 0 && (
                <PinnedParetoCard
                  target={activeTarget}
                  items={activeParetoItems}
                  n80={activeParetoResponse.n80}
                  datasetId={datasetId}
                  onBarClick={handleParetoBarClick}
                />
              )}

              {activeTargetHasNoChart ? (
                // 지시서 WI-2/YG: 기여율 10% 이상 인자가 하나도 없으면
                // 카드 대신 이 안내를 띄운다(문구 형식은 지시서 예시 그대로).
                <section className="resultCard noChartMessage">
                  <h2>{activeTarget}는 기여율 10% 이상 인자가 없습니다 (최대 {maxContributionPct.toFixed(1)}%)</h2>
                </section>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
                  {displayFactors.map((item, index) => renderFactorCard(activeTarget, item, index))}
                </div>
              )}
            </>
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

/** 인자 카드 "보기" 토글 (지시서 WI-1) -- Pareto 옵션을 없앴다(파레토는
 * 타깃당 1개, 화면 상단에 고정된 PinnedParetoCard가 전담). 이제 Quick
 * Look의 QuickLookViewToggle과 옵션이 같아졌지만, 상태를 소유하는 카드가
 * 다르므로(이 토글은 NumericFactorCard, 저쪽은 root-cause 페이지의
 * quickLookView) 두 컴포넌트를 그대로 둔다. */
function ViewToggle({ value, onChange }: { value: QuickLookView; onChange: (view: QuickLookView) => void }) {
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

/** 지시서 WI-1/WI-4: 파레토는 타깃당 1개, 화면 상단에 고정된다 --
 * ParetoChart를 non-embedded(기본) 모드로 그린다 -- 그쪽이 이미 소유한
 * 제목("R/D/Config vs {target}")·등급 범례 헤더를 그대로 쓰고,
 * `headerActions`로 이미지 저장 버튼만 그 헤더 안에 얹는다(이전에는 이
 * non-embedded 경로를 실제로 쓰는 곳이 없었다 -- 인자 카드는 항상
 * embedded로, 즐겨찾기 썸네일은 embedded+thumbnail로 불렀다). 인자
 * 카드와 달리 즐겨찾기 별은 없다 -- 이 카드는 특정 (타깃, 인자) 조합이
 * 아니라 타깃 전체의 순위를 보여줘 즐겨찾기 스냅샷(dataset+target+feature)
 * 모델과 맞지 않는다. */
function PinnedParetoCard({
  target,
  items,
  n80,
  datasetId,
  onBarClick,
}: {
  target: string;
  items: ParetoRankingItem[];
  n80: number | null;
  datasetId: string;
  onBarClick: (item: ParetoRankingItem) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  return (
    <ParetoChart
      target={target}
      items={items}
      n80={n80}
      onBarClick={onBarClick}
      svgExportRef={svgRef}
      headerActions={
        <ExportPngButton
          svgRef={svgRef}
          buildOptions={() => ({
            filename: buildExportFilename({ feature: null, target, view: "pareto" }),
            captionText: buildParetoCaptionText({ target, factorCount: items.length, datasetId }),
          })}
        />
      }
    />
  );
}

/** 지시서 WI-4: 파레토·산점도·박스플롯 카드 우상단, 즐겨찾기(☆) 옆에
 * 붙는 이미지 저장 버튼. SVG를 클론해 계산된 스타일을 인라인으로 굽고
 * (lib/chartExport.ts) canvas에 그려 PNG로 내려받는다 -- 세 차트가 이
 * 컴포넌트 하나를 공유한다. `buildOptions`를 클릭 시점에 지연 평가하는
 * 것은 파일명·캡션에 들어가는 시각을 "카드가 열린 시각"이 아니라
 * "다운로드를 누른 시각"으로 맞추기 위함이다. */
function ExportPngButton({
  svgRef,
  buildOptions,
}: {
  svgRef: React.RefObject<SVGSVGElement | null>;
  buildOptions: () => { filename: string; captionText: string };
}) {
  const [busy, setBusy] = useState(false);
  async function handleClick() {
    const svg = svgRef.current;
    if (!svg || busy) return;
    setBusy(true);
    try {
      await exportChartAsPng(svg, buildOptions());
    } catch (error) {
      console.warn("차트 이미지 저장 실패", error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <button
      type="button"
      className="chartExportButton"
      onClick={() => void handleClick()}
      disabled={busy}
      title="이미지로 저장 (PNG)"
    >
      ⬇ 이미지 저장
    </button>
  );
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
  // D-1: viewType(Scatter/Box)별로 별도 즐겨찾기다 -- 이 카드가 지금 어떤
  // view인지는 카드 자신만 아므로, 부모가 boolean이 아니라 함수를 내려줘
  // 카드 내부에서 현재 view로 평가한다.
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
  // 지시서 WI-1: Pareto가 빠졌으므로 기본값은 Scatter Plot이다.
  const [view, setView] = useState<QuickLookView>("scatter");
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
  // 지시서 WI-3: p-value 대신 R²(차수) -- ScatterChart가 추세선을 그릴 때
  // 쓰는 것과 같은 fitDefectRateCurve를 이미 받아온 산점도 좌표 위에서
  // 그대로 다시 계산한다(같은 순수 함수·같은 입력이라 값이 어긋나지 않는다).
  const curveFit = useMemo(() => (numericData ? fitDefectRateCurve(numericData.points) : null), [numericData]);
  // 지시서 WI-4: 이미지 저장 버튼이 이 SVG를 직렬화한다.
  const svgRef = useRef<SVGSVGElement | null>(null);
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`}>
      <div className="factorChartHeader">
        <div className="factorChartHeaderRow">
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
            {/* QA-2: 배제 대신 하한(30) 이상 표본을 그대로 판정하되, 종류별
                (R/D) 정상 판정 임계 미만이면 등급을 낮추고(위 배지에 이미
                반영됨) 이유를 별도로 밝힌다 -- 등급만 봐서는 "왜 낮은지"가
                안 보인다. */}
            {item.under_sampled && (
              <span className="underSampledBadge" title={`계측 n=${item.n_observed} -- 표본이 정상 판정 임계에 못 미쳐 신뢰 등급을 한 단계 낮췄습니다.`}>
                표본 부족
              </span>
            )}
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
                  interpretation: buildModerateInterpretation(item.confidence_tier, item.eps2),
                  championVersion,
                })
              }
            />
            {/* 지시서 WI-4: 이미지 저장 버튼은 즐겨찾기 별 바로 옆. */}
            <ExportPngButton
              svgRef={svgRef}
              buildOptions={() => ({
                filename: buildExportFilename({ feature: item.feature, target: activeTarget, view }),
                captionText: buildFactorCaptionText({
                  feature: item.feature,
                  target: activeTarget,
                  eps2: item.eps2,
                  n: numericData?.n ?? item.n_observed,
                  datasetId: dataset,
                }),
              })}
            />
            <ColorBySelect value={colorMode} onChange={setColorMode} hasConfig={hasConfig} />
          </div>
          <div className="factorChartToggleStack">
            <CompareToggleRow onCompare={() => onCompare(item.feature)} onTrellis={() => onTrellis(item.feature, item.step)} />
            <ViewToggle value={view} onChange={setView} />
            {numericData?.methods && (
              <MethodToggle value={method} adopted={numericData.methods.adopted} onChange={setMethod} />
            )}
          </div>
        </div>
        <div className="factorChartHeaderRow meta">
          <span className="sectionLabel">{index + 1}위</span>
          {/* 지시서 WI-3: p-value·q-value는 화면에서 뺀다(FDR 게이트 자체는
              백엔드에서 계속 동작 -- 표시만 안 하는 것). ε²·R²(적합
              차수)·파레토 기여율 셋이 서로를 보완한다(표본이 커지면 p는
              약한 관계도 극단적으로 유의해져 변별력이 없다). */}
          {numericData && (
            <div className="factorChartMetaLine">
              <span className="metaItem">n={numericData.n.toLocaleString()}</span>
              <span className="metaItem">ε² {item.eps2.toFixed(3)}</span>
              {curveFit && <span className="metaItem">R² {curveFit.r2.toFixed(3)} ({curveFit.degree}차)</span>}
              <span className="metaItem">기여율 {item.contribution_pct.toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (ε² = {item.eps2.toFixed(3)}, 등급 {TIER_LABEL[item.confidence_tier]}). 원인으로 단정할 근거는 부족합니다.
        </p>
      )}
      {numericData ? (
        <>
          <ScatterChart
            data={numericData}
            colorMode={colorMode}
            view={view}
            method={method}
            onSelectWafer={onSelectWafer}
            height={chartHeight}
            reliabilityText={buildModerateInterpretation(item.confidence_tier, item.eps2)}
            svgExportRef={svgRef}
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
          <span className="sectionLabel">{index + 1}위</span>
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
        {/* 지시서 WI-3: p-value 제거, ε²·기여율만 남긴다 -- R²(적합 곡선)는
            숫자 x축이 있는 산점도에서만 의미가 있어(회귀 곡선을 그 위에
            적합한다) Config(범주형) 카드에는 없다. */}
        {categoricalData && (
          <small className="factorChartStats">
            <span>n={categoricalData.n}</span>
            <span>ε²={item.eps2.toFixed(3)}</span>
            <span>기여율={item.contribution_pct.toFixed(1)}%</span>
          </small>
        )}
      </div>
      <ModerateTierCaption tier={item.confidence_tier} eps2={item.eps2} />
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (ε² = {item.eps2.toFixed(3)}, 등급 {TIER_LABEL[item.confidence_tier]}). 원인으로 단정할 근거는 부족합니다.
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
