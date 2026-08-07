"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import CompareAcrossConfigsModal from "@/components/CompareAcrossConfigsModal";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import MeasurementExpansionCard from "@/components/MeasurementExpansionCard";
import ParetoChart from "@/components/ParetoChart";
import { usePanelState } from "@/components/PanelStateProvider";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type QuickLookView, type ScatterColorMode, type ScatterView } from "@/components/ScatterChart";
import { factorAxisLabel, targetAxisLabel } from "@/lib/chartLabels";
import { selectDisplayFactors } from "@/lib/chartSelection";
import { measurementRateDisclaimer } from "@/lib/measurementDisclaimer";
import { formatPValue } from "@/lib/numberFormat";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import {
  ApiNetworkError,
  ApiResponseError,
  ApiTimeoutError,
  dispatchAlarmNotifications,
  getAlarms,
  getDatasetSchema,
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
// 데이터셋의 분포가 다를 때 알람이 사실상 0건으로 사라질 수 있다(실측:
// mentorship_dataset_final을 test.csv로 판정하면 0건, 자기 자신으로
// 판정하면 634건).
async function fetchAlarmGradeByWaferId(datasetId: string): Promise<Record<string, AlarmGrade>> {
  const response = await getAlarms(datasetId, datasetId);
  const map: Record<string, AlarmGrade> = {};
  for (const item of response.items) {
    map[item.lot_wafer_id] = item.grade;
  }
  return map;
}

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;
// Stable empty-object fallbacks (spec: avoid a fresh `{}` literal every
// render feeding a useMemo/useEffect dependency array, which would defeat
// memoization and refire effects needlessly).
const EMPTY_PARETO_BY_TARGET: Record<string, ParetoRankingResponse> = {};
const EMPTY_SCATTER_BY_KEY: Record<string, ScreeningScatterResponse> = {};
const EMPTY_CATEGORICAL_BY_KEY: Record<string, CategoricalScatterResponse> = {};
// 등급 라벨 정비 (spec §C-2) -- "약함"/"참고"는 이름만으로 차이가 드러나지
// 않아 "근거 부족"/"관계 없음"으로 바꾼다. 내부 코드의 tier 값(strong/
// moderate/weak/reference)과 JSON `grade` 필드는 그대로 두고, 화면 표시
// 문자열만 바꾼다 (spec §C-2/§21).
const TIER_LABEL: Record<ConfidenceTier, string> = { strong: "강함", moderate: "보통", weak: "근거 부족", reference: "관계 없음" };
// 등급 배지 호버 설명 (spec §C-3).
const TIER_TOOLTIP: Record<ConfidenceTier, string> = {
  strong: "부도율 변동의 10% 이상을 설명합니다. 조치의 우선 순위입니다.",
  moderate: "5~10%를 설명합니다. 관계는 있지만 추가 확인이 필요합니다.",
  weak: "통계적 근거가 부족합니다. 조치 판단에 사용하지 마세요.",
  reference: "효과 크기가 기준(0.02)에 미치지 못합니다.",
};
const RUN_STAGES = ["인자 스크리닝 중 (5개 타깃)", "Pareto 집계 중", "산점도 준비 중", "히트맵 집계 중", "계측 확대 시뮬레이션 중"];

type ColorMode = ScatterColorMode;
type RunState = "idle" | "running" | "error" | "done";
type ChartCriterion = "significant" | "all";

function hasReliableEvidence(tier: ConfidenceTier): boolean {
  return tier === "strong" || tier === "moderate";
}

type AnalysisFailureKind = "network" | "timeout" | "server" | "model_not_ready" | "unknown";

const ANALYSIS_FAILURE_MESSAGE: Record<AnalysisFailureKind, string> = {
  network: "서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.",
  timeout: "분석 시간이 초과되었습니다. 다시 시도해 주세요.",
  server: "분석 중 오류가 발생했습니다. 다시 시도해 주세요.",
  model_not_ready: "모델 학습이 완료되지 않았습니다. 모델 학습 탭에서 먼저 학습을 실행해 주세요.",
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

function ConfidenceBadge({ tier }: { tier: ConfidenceTier }) {
  return (
    <span className={`confidenceBadge tier-${tier}`} title={TIER_TOOLTIP[tier]}>
      {TIER_LABEL[tier]}
    </span>
  );
}

/** `보통` 등급 인자의 설명력이 낮은 편임을 알리는 한 줄 캡션 (spec §C-4).
 * train.CSV의 Step24_R1 → Y4(ε² 0.073)가 여기 해당한다. */
function ModerateTierCaption({ tier, eps2 }: { tier: ConfidenceTier; eps2: number }) {
  if (tier !== "moderate") return null;
  return (
    <p className="moderateTierCaption">
      이 인자의 설명력은 {(eps2 * 100).toFixed(1)}%로 낮은 편입니다. 다른 요인의 영향이 더 클 수 있습니다.
    </p>
  );
}

function buildCategoricalSpec(data: CategoricalScatterResponse) {
  const x = data.groups.flatMap((group) => group.values.map(() => group.category));
  const y = data.groups.flatMap((group) => group.values);
  return {
    data: [
      {
        type: "box",
        x,
        y,
        boxpoints: "outliers",
        marker: { color: "#1D4ED8" },
        line: { color: "#1D4ED8" },
      },
    ],
    layout: {
      xaxis: { title: { text: factorAxisLabel(data.axis.x_label) }, tickangle: 0 },
      yaxis: { title: { text: targetAxisLabel(data.axis.y_label) } },
      margin: { t: 20, b: 90 },
    },
  };
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
  const { setAnalysisDataset } = usePanelState();
  // 원인 분석 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- the actual
  // result (Pareto/스크리닝/산점도) lives in the shared AnalysisStateProvider
  // context, not local useState, so tab switching renders it from memory
  // with zero network calls (checklist §탭 이동 #1/#4), and a page
  // reload/reconnect restores a lean (points-less) version of it via
  // GET /api/state/latest.
  const { analysis, setAnalysis, hydrated } = useAnalysisState();
  // ≤767px: 산점도/박스플롯 높이 240px (spec §B-6).
  const isMobileLayout = useIsMobileLayout();
  const chartHeight = isMobileLayout ? 240 : 420;
  const [datasetId, setDatasetId] = useState("train");
  // 해석 시 한계 문구의 계측률 수치는 데이터셋마다 다르므로 (spec 문구 전수
  // 검토 §A-1) 하드코딩 대신 실제 스키마를 불러와 반영한다.
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
      setDatasetId(analysis.dataset);
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
    // 삼각형만 나중에 얹히게 한다.
    void fetchAlarmGradeByWaferId(dataset)
      .then((alarmGradeByWaferId) => {
        if (cancelled) return;
        setAnalysis((previous) => (previous && previous.dataset === dataset ? { ...previous, alarmGradeByWaferId } : previous));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [analysis, setAnalysis]);

  async function runAnalysis() {
    setRunState("running");
    setRunError("");
    setRunErrorDetail("");
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
      });
      setRunState("done");
      setAnalysisDataset(datasetId);
      // 알람 심각도 삼각형 (spec §B) -- 부트스트랩 앙상블이라 수십 초가
      // 걸릴 수 있다(§A-1). 위 setAnalysis를 붙잡아 두면 이미 준비된
      // 산점도/Pareto까지 그만큼 늦게 보이므로, 별도 요청으로 분리해
      // 나중에 도착하는 대로 삼각형만 얹는다.
      void fetchAlarmGradeByWaferId(datasetId)
        .then((alarmGradeByWaferId) => {
          setAnalysis((previous) => (previous && previous.dataset === datasetId ? { ...previous, alarmGradeByWaferId } : previous));
        })
        .catch(() => {});
      // 알림 연동 §C-4 "분석 실행 직후" -- fire-and-forget. 신뢰도 게이트·
      // 중복 발송 방지·연결된 채널 유무는 전부 서버(dispatch_alarm_notifications)
      // 가 판단한다: 이 호출은 그저 "지금 막 분석이 끝났다"는 신호일 뿐이고,
      // 실패해도 분석 결과 화면에는 아무 영향이 없어야 한다.
      void dispatchAlarmNotifications(datasetId, datasetId).catch(() => {});
      // 성공 직후 저장 (spec §3-4) -- paretoByTarget만 보낸다. 인자별
      // 산점도 상세(관리한계·권장구간·최적중심 등, 좌표 제외)까지 25개
      // 인자 전부 실으면 그것만으로 ~105KB라 100KB 예산(spec §6)을
      // 넘는다 -- 어차피 복원 직후 배경에서 fetchAllScatterData로 다시
      // 채우므로(위 useEffect), 서버에는 화면 목록 구성에 꼭 필요한
      // Pareto와, 그 자체로 작은 계측 확대 권고 결과만 남긴다.
      void saveAnalysisState(datasetId, { activeTarget, paretoByTarget: paretoMap, measurementExpansion }).catch(() => {});
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
  // 계측된 인자로 부도율이 설명되지 않는 경우, 타깃별 안내 문구를 5번
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
    let maxEps2 = 0;
    for (const t of TARGETS) {
      const response = paretoByTarget[t];
      if (!response) continue;
      totalTested += response.total_factor_count;
      effectSizePass += response.effect_size_pass_count;
      maxEps2 = Math.max(maxEps2, response.max_eps2 ?? 0);
    }
    return { totalTested, effectSizePass, maxEps2 };
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
    if (!isConfig) {
      return (
        <NumericFactorCard
          key={`${runGeneration}-${target}-${item.feature}`}
          item={item}
          index={index}
          activeTarget={target}
          numericData={scatterByKey[key]}
          onSelectWafer={setSelectedWafer}
          onCompare={openCompare}
          onTrellis={openTrellis}
          hasConfig={(analysisSchema?.config_columns.length ?? 0) > 0}
          alarmGradeByWaferId={analysis?.alarmGradeByWaferId ?? undefined}
          paretoItems={paretoByTarget[target]?.items ?? []}
          paretoN80={paretoByTarget[target]?.n80 ?? null}
          onParetoBarClick={handleParetoBarClick}
        />
      );
    }
    return (
      <CategoricalFactorCard
        key={`${target}-${item.feature}`}
        item={item}
        index={index}
        activeTarget={target}
        categoricalData={categoricalByKey[key]}
        chartHeight={chartHeight}
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

  return (
    <DashboardShell activeItem="원인 분석">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">ROOT CAUSE</span>
        <h1>원인 분석</h1>
        {/* 3줄 고정 (spec 문구 전수 검토 PART C) -- Box Plot, SPC/ML 비교,
            계측 확대 제안을 언급한다. 배치는 히트맵 → Pareto → 산점도 →
            계측 확대 제안 순으로 유지한다 (spec §A-0/§A-2: 발표 흐름상
            계측 확대 제안은 결론이므로 가장 아래). 원인 분석 탭에 새 기능이
            추가되거나 제거되면 이 문구도 함께 갱신해야 한다. 첫 줄은
            1280px에서 한 줄에 들어가도록 (spec §A-1) word-break: keep-all만
            쓴다 -- white-space: nowrap과 함께 쓰지 않는다 (좁은 화면에서
            잘림). */}
        <p className="rootCauseIntro">
          타깃별 Pareto와 강함·보통 등급 인자의 산점도·Box Plot을 확인합니다.
          <br />
          권장 구간은 통계(SPC)와 학습(ML) 두 방식을 비교해 나은 쪽을 채택합니다.
          <br />
          산점도·Box Plot 아래에서 계측 확대 시 기대 효과를 확인할 수 있습니다.
        </p>
      </section>

      <section className="uploadCard">
        <div className="rcControlBar" style={{ gridTemplateColumns: "minmax(220px,1fr)" }}>
          <DatasetSelector label="분석 데이터셋" value={datasetId} onChange={setDatasetId} />
        </div>
        <DatasetMismatchWarning mismatch={datasetMismatch} />
      </section>

      <section className="uploadCard">
        <div className="paretoRunBar">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: "var(--text-section, 17px)" }}>원인 분석 실행</h2>
              <LastRunNote createdAt={analysis?.createdAt} />
            </div>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--text-secondary)" }}>
              {runState === "done"
                ? "완료된 결과입니다. 데이터셋을 바꾸면 다시 실행해야 합니다."
                : "타깃 5개 각각의 전체 인자 풀(R+D+Eq.) 기준 Pareto 상위 10개를 계산합니다."}
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center" }}>
            <button type="button" className="button" disabled={runState === "running"} onClick={() => void runAnalysis()}>
              {runState === "running" ? "원인 분석 중..." : runState === "done" ? "다시 실행" : runState === "error" ? "다시 시도" : "원인 분석 실행"}
            </button>
          </div>
        </div>
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
                    <div className="factorChartMetaTwoLine">
                      <span>n={quickLookNumeric.n.toLocaleString()} · ε²={quickLookNumeric.eps2.toFixed(3)}</span>
                      <span>p-value {formatPValue(quickLookNumeric.p_value)} · 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}</span>
                    </div>
                  )}
                </div>
              </div>
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
                  alarmGradeByWaferId={analysis?.alarmGradeByWaferId ?? undefined}
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
                검정 {datasetNoChartStats.totalTested.toLocaleString()}건 · 효과 크기 조건 통과 {datasetNoChartStats.effectSizePass.toLocaleString()}건 · 최대 설명력 {datasetNoChartStats.maxEps2.toFixed(4)}
                <br />
                부도율 변동의 대부분이 계측되지 않은 공정 변수로 설명됩니다. 안전율 예측 기반 알람은 계속 동작합니다.
              </p>
              <p className="noChartStats">전체 상위 3개로 전환하면 순위대로 확인할 수 있습니다.</p>
              <button type="button" className="button" onClick={() => setChartCriterion("all")}>전체 상위 3개로 전환</button>
            </section>
          ) : activeTargetHasNoChart && activeParetoResponse ? (
            <section className="resultCard noChartMessage">
              <h2>{activeTarget} · 강함·보통 등급 인자가 없습니다</h2>
              <p className="noChartStats">
                검정 {activeParetoResponse.total_factor_count.toLocaleString()}건 · 효과 크기 조건 통과 {activeParetoResponse.effect_size_pass_count.toLocaleString()}건
                <br />
                최대 설명력 {(activeParetoResponse.max_eps2 ?? 0).toFixed(4)}
              </p>
              <p className="noChartStats">전체 상위 3개로 전환하면 순위대로 확인할 수 있습니다.</p>
              <button type="button" className="button" onClick={() => setChartCriterion("all")}>전체 상위 3개로 전환</button>
            </section>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
              {displayFactors.map((item, index) => renderFactorCard(activeTarget, item, index))}
            </div>
          )}

          {/* 배치는 히트맵 → Pareto → 산점도/Box Plot → 계측 확대 제안
              순으로 고정한다 (spec §A-0) -- 발표 흐름상 인자를 찾고 근거를
              본 뒤 결론(계측을 늘려라)으로 이어지므로 가장 아래에 온다. */}
          <MeasurementExpansionCard data={analysis?.measurementExpansion ?? null} />
        </>
      )}

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>{measurementRateDisclaimer(analysisSchema)}</li>
          <li>ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행/후행 관계나 교락 인자는 반영되지 않았습니다.</li>
          <li>&ldquo;근거 부족&rdquo;·&ldquo;관계 없음&rdquo; 등급 인자는 통계적 신뢰도가 낮아 원인으로 단정할 근거가 부족합니다. 사전 알람은 p&lt;0.05(강함·보통 등급) 인자에서만 생성됩니다.</li>
        </ul>
      </section>

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
 * (탐색 뷰 -> 검증 뷰 순서) but is never the *default* view -- see
 * NumericFactorCard's `useState<ScatterView>("scatter")`. */
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
          SPC{adopted === "spc" && <span className="methodAdoptedBadge" title="채택된 방식">✓</span>}
        </button>
        <button type="button" className={`scatterViewToggleBtn ${value === "ml" ? "active" : ""}`} onClick={() => onChange("ml")}>
          ML{adopted === "ml" && <span className="methodAdoptedBadge" title="채택된 방식">✓</span>}
        </button>
      </div>
    </div>
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
  // Config 컬럼이 0개인 데이터셋(mentorship_dataset_v7_killing_event)에서는
  // "Eq. 모델별" 색상 옵션이 고를 수 있는 값 자체가 없으므로 숨긴다 (spec
  // 문구 전수 검토 §A-5).
  hasConfig?: boolean;
}) {
  return (
    <label className="colorBySelectField">
      <span>색상</span>
      <select
        className="colorBySelect"
        value={value}
        onChange={(event) => onChange(event.target.value as ColorMode)}
      >
        <option value="default">기본</option>
        {hasConfig && <option value="config_model">Eq. 모델별</option>}
        <option value="lot">LOT별</option>
        <option value="alarm">알람 여부</option>
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
  activeTarget,
  numericData,
  onSelectWafer,
  onCompare,
  onTrellis,
  hasConfig,
  alarmGradeByWaferId,
  paretoItems,
  paretoN80,
  onParetoBarClick,
}: {
  item: ParetoRankingItem;
  index: number;
  activeTarget: string;
  numericData: ScreeningScatterResponse | undefined;
  onSelectWafer: (point: ScatterPoint) => void;
  onCompare: (feature: string) => void;
  onTrellis: (feature: string, step: number) => void;
  hasConfig: boolean;
  alarmGradeByWaferId?: Record<string, AlarmGrade>;
  // Pareto 보기(spec "Pareto를 산점도 카드로 병합")용 데이터 -- 카드마다
  // 다시 fetch하지 않고 부모(RootCauseContent)가 이미 갖고 있는
  // paretoByTarget에서 한 번만 내려준다.
  paretoItems: ParetoRankingItem[];
  paretoN80: number | null;
  onParetoBarClick: (item: ParetoRankingItem) => void;
}) {
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  // View state lives per-card (spec §2-2: "산점도마다 독립적인 상태"), never
  // in a shared store/URL/localStorage -- resets for free whenever this
  // card remounts on a new run/target (see its `key` at the call site).
  const [view, setView] = useState<ScatterView>("scatter");
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
            <div className="factorCompareGroup" role="group" aria-label="비교 보기">
              <button
                type="button"
                className="compareTriggerButton"
                title="이 인자가 다른 불량 유형에도 영향을 주는지 확인"
                onClick={() => onCompare(item.feature)}
              >
                ⊞ Y1~Y5 비교
              </button>
              <button
                type="button"
                className="compareTriggerButton"
                title="이 인자의 효과가 장비에 따라 달라지는지 확인"
                onClick={() => onTrellis(item.feature, item.step)}
              >
                ⊞ 장비별 Trellis
              </button>
            </div>
            <ColorBySelect value={colorMode} onChange={setColorMode} hasConfig={hasConfig} />
          </div>
          <div className="factorChartToggleStack">
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
            <div className="factorChartMetaTwoLine">
              <span>n={numericData.n.toLocaleString()} · 기여율 {item.contribution_pct.toFixed(1)}% · <span className="metaCumulative">누적 {item.cumulative_pct.toFixed(1)}%</span></span>
              <span>p-value {formatPValue(item.p_value)} · 등급 {TIER_LABEL[item.confidence_tier]}</span>
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
  activeTarget,
  categoricalData,
  chartHeight,
}: {
  item: ParetoRankingItem;
  index: number;
  activeTarget: string;
  categoricalData: CategoricalScatterResponse | undefined;
  chartHeight: number;
}) {
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`}>
      <div className="factorChartMeta">
        <div className="factorChartTitleBlock">
          <span className="sectionLabel">{index + 1}위 · ε² {item.eps2.toFixed(3)}</span>
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
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

function WaferDetailPopover({
  point,
  target,
  onClose,
  hasConfig,
}: {
  point: ScatterPoint;
  target: string;
  onClose: () => void;
  // 데이터셋에 Config 컬럼이 아예 없으면(mentorship_dataset_v7_killing_event)
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
        <dt>정상범위 내</dt><dd>{point.in_range ? "예" : "아니오 (알람)"}</dd>
        {hasConfig && (
          <>
            <dt>Eq.</dt><dd>{point.config ?? "미계측"}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
