"use client";

import { Suspense, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import CompareAcrossConfigsModal from "@/components/CompareAcrossConfigsModal";
import CompareAcrossTargetsModal from "@/components/CompareAcrossTargetsModal";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import { DatasetMismatchWarning, PageHeaderMeta } from "@/components/LastRunNote";
import SampleNotice from "@/components/SampleNotice";
import ParetoChart from "@/components/ParetoChart";
import { usePanelState } from "@/components/PanelStateProvider";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart, { type QuickLookView, type ScatterColorMode } from "@/components/ScatterChart";
import { buildExportFilename, buildFactorCaptionText, buildParetoCaptionText, exportNodeAsPng } from "@/lib/chartExport";
import { selectDisplayFactors } from "@/lib/chartSelection";
import { hasReliableEvidence, TIER_LABEL } from "@/lib/confidenceTier";
import { buildCategoricalSpec, TARGETS } from "@/lib/constants";
import { fitDefectRateCurve } from "@/lib/defectRateCurve";
import { useIsMobileLayout } from "@/lib/useMediaQuery";
import {
  createFavorite,
  deleteFavorite,
  getDatasetSchema,
  getFavorites,
  getScreeningScatter,
  getScreeningScatterCategorical,
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

// Stable empty-object fallbacks -- a fresh `{}` literal on every render would
// feed a useMemo/useEffect dependency array, defeating memoization and
// refiring effects needlessly.
const EMPTY_PARETO_BY_TARGET: Record<string, ParetoRankingResponse> = {};
const EMPTY_SCATTER_BY_KEY: Record<string, ScreeningScatterResponse> = {};
const EMPTY_CATEGORICAL_BY_KEY: Record<string, CategoricalScatterResponse> = {};
const EMPTY_HEATMAP_CACHE: Record<string, HeatmapResponse> = {};

type ColorMode = ScatterColorMode;
// 이 화면은 스스로 분석을 실행하지 않는다 -- 복원/스냅샷 동기화만 이
// 상태를 움직이므로 "idle" -> "done" 두 값이면 충분하다.
type RunState = "idle" | "done";

/** `보통` 등급 인자의 설명력이 낮은 편임을 알리는 한 줄 캡션.
 * train.CSV의 Step24_R1 → Y4(Adj R² 0.094)가 여기 해당한다.
 * 즐겨찾기 스냅샷 저장 때도 같은 문구를 쓰므로 컴포넌트 밖에 둔다 --
 * `보통` 등급이 아니면 해석 문구 자체가 없다(빈 문자열). */
function buildModerateInterpretation(tier: ConfidenceTier, adjR2: number): string {
  if (tier !== "moderate") return "";
  return `이 인자의 설명력은 ${(adjR2 * 100).toFixed(1)}%로 낮은 편입니다. 다른 요인의 영향이 더 클 수 있습니다.`;
}

function ModerateTierCaption({ tier, adjR2 }: { tier: ConfidenceTier; adjR2: number }) {
  const text = buildModerateInterpretation(tier, adjR2);
  if (!text) return null;
  // 해석 문구를 옅은 카드(.interpretCard)에 담아 바로 위 메타 줄과
  // 시각적으로 분리한다. Pareto·Scatter·Box 세 뷰가 전부 이
  // 컴포넌트/클래스를 공유한다.
  return <p className="interpretCard">{text}</p>;
}


/** Step 2 of a run (or a restore's background point-fill):
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
  // 호출부(useEffect cleanup)가 취소를 알리는 신호. 이게 없으면
  // 이펙트가 재실행돼 새 워커 배치가 시작된 뒤에도, 취소된 이전 배치의
  // 워커가 남은 잡을 끝까지 쏘아 보낸다(결과만 버려질 뿐 요청은 계속
  // 나간다) -- in-flight 요청이 실행마다 계속 누적된다.
  isCancelled?: () => boolean,
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
      if (isCancelled?.()) return;
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
  // 데이터셋 전환은 모델 분석 팝업에서만 일어난다.
  const { setAnalysisDataset, setAnalysisPanelOpen } = usePanelState();
  // 원인 분석 결과 상태 유지 -- the actual result (Pareto/스크리닝/산점도)
  // lives in the shared AnalysisStateProvider context, not local useState,
  // so tab switching renders it from memory with zero network calls, and a
  // page reload/reconnect restores a lean (points-less) version of it via
  // GET /api/state/latest.
  const {
    analysis, setAnalysis, hydrated, analysisSnapshotStale, datasetFallbackNotice,
    training,
  } = useAnalysisState();
  // 즐겨찾기 스냅샷에 저장 시점의 활성 모델(챔피언)을 함께
  // 담는다 -- 이후 재학습/재승격으로 model_id가 바뀌면 "이전 분석 기준"
  // 배지를 붙일 수 있다. 학습 기록이 아직 없으면 null.
  const championVersion = training?.performance.model_id ?? null;
  // ≤767px: 산점도/박스플롯 높이 240px.
  const isMobileLayout = useIsMobileLayout();
  const chartHeight = isMobileLayout ? 240 : 420;
  // 즐겨찾기 딥링크(`?dataset=&target=&feature=`)가 저장해 둔
  // 데이터셋으로 연다 -- 이게 없으면 즐겨찾기가 항상 "현재 선택된
  // 데이터셋"으로 열려서, train에서 저장한 카드를 test 상태에서 열면
  // 같은 인자명의 다른 데이터셋 차트가 경고 없이 표시된다. 이미 이
  // 페이지가 마운트된 채로 다른 즐겨찾기를 또 여는 경우는(같은 라우트라
  // 리마운트가 안 됨) target/feature 딥링크와 동일한 기존 한계다.
  // 폴백 모드(SQL 미연결)의 원인 분석 기본 데이터셋은 train.CSV가
  // 아니라 test.CSV다 -- 헤더의 "훈련 데이터 train.CSV ·
  // 분석 데이터 test.CSV" 표기(PageHeaderMeta)와 일치시킨다. URL의 즐겨찾기
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
  // factor 카드 key에 접어 넣는 세대 번호. 이 화면은 스스로 분석을
  // 실행하지 않으므로 값은 늘 0이며, 카드 key의 안정적인 상수 접두사
  // 역할만 한다.
  const runGeneration = 0;

  const paretoByTarget = analysis?.paretoByTarget ?? EMPTY_PARETO_BY_TARGET;
  const scatterByKey = analysis?.scatterByKey ?? EMPTY_SCATTER_BY_KEY;
  const categoricalByKey = analysis?.categoricalByKey ?? EMPTY_CATEGORICAL_BY_KEY;
  // 셀렉터를 바꿨는데 화면은 이전 데이터셋 결과인 경우.
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

  // 즐겨찾기 -- `${dataset}::${target}::${feature}::${viewType}` ->
  // favorite_id. viewType을 키에 포함해야 한다 -- 안 그러면 같은
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

  // 생성/삭제 요청이 아직 끝나지 않은 키는 다시 받지 않는다 --
  // 빠른 더블클릭 시 두 호출 모두 같은(아직 갱신 전) favoriteIdByKey를
  // 보고 둘 다 "생성" 경로로 들어가 중복 레코드가 생기고, 클라이언트는
  // 마지막 id만 기억해 나머지 하나는 지울 수 없는 좀비로 남는다. 이
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
  // Scatter Plot whenever the selected factor changes.
  // Adjusting state during render (React's documented alternative to an
  // effect for "reset when a prop changes") instead of useEffect, so it
  // doesn't cause an extra cascading render pass.
  const quickLookKey = quickLook ? `${quickLook.target}::${quickLook.feature}` : "";
  const [prevQuickLookKey, setPrevQuickLookKey] = useState(quickLookKey);
  if (quickLookKey !== prevQuickLookKey) {
    setPrevQuickLookKey(quickLookKey);
    setQuickLookView("scatter");
  }

  // A dataset change must not wipe the displayed result -- it only closes
  // the quick-look popover, which is scoped to whatever factor/dataset it
  // was opened for and would show a stale chart otherwise. The mismatch
  // banner (datasetMismatch, above) is what tells the user the selector and
  // the result have diverged.
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
  // 동기화한다 -- 이후 사용자가 셀렉터를 바꿔도 다시 개입하지 않는다.
  //
  // `hydrated`(GET /api/state/latest)와 `analysis`가 채워지는 시점(저장된
  // 결과가 있으면 즉시, 없으면 자동 갱신 스냅샷의 대체 채움이 나중에
  // 도착)은 서로 다른 요청이라 순서가 보장되지 않는다. 그래서 ref는
  // `analysis`가 실제로 왔을 때만 세운다 -- `hydrated`만 보고 세우면
  // (그 시점엔 `analysis`가 아직 null) 나중에 `analysis`가 채워져도 이
  // 이펙트가 다시 돌지 않아, 데이터셋 셀렉터가 초기값에 영원히 멈춘다.
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current || !analysis) return;
    syncedFromRestore.current = true;
    const timer = window.setTimeout(() => {
      // 즐겨찾기 딥링크가 dataset을 지정했으면 복원된 결과의
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

  // 분석 실행은 전부 모델 분석 팝업의 [분석 시작]에서 일어나고, 이
  // 화면은 그 결과를 받아 그리기만 한다 -- 부트스트랩/[분석 시작]이 채운
  // 스냅샷이 AnalysisStateProvider를 거쳐 위 syncedFromRestore 이펙트로
  // 반영된다. 스냅샷도 저장된 결과도 없는 진짜 콜드 상태에서는 아래
  // 렌더가 "분석 결과가 없습니다 -- 모델 분석에서 분석을 시작하세요"
  // 안내를 보여준다.

  // 모델 분석 팝업에서 데이터셋을 바꾸면(activate-dataset ->
  // 스냅샷 갱신 -> analysis.dataset 변경) 이 화면의 라벨도 따라간다 --
  // 위 syncedFromRestore와 달리 최초 1회가 아니라 analysis.dataset이
  // 바뀔 때마다 계속 동작한다. 즐겨찾기 딥링크(?dataset=)가 있으면
  // 그 값을 유지한다(위와 같은 이유).
  useEffect(() => {
    if (!analysis || searchParams.get("dataset")) return;
    setDatasetId(analysis.dataset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis?.dataset]);

  // 복원된 결과는 산점도 좌표를 담고 있지 않다 -- 배경에서 한
  // 번 다시 채운다. 채우는 동안에도 스크리닝 표/Pareto/비교 카드 등 좌표가
  // 필요 없는 부분은 이미 즉시 보인다.
  //
  // 의존성을 `analysis` 객체 전체가 아니라 이 이펙트가 실제로 반응
  // 해야 하는 스칼라 값만으로 좁힌다. onResult 콜백이 매 응답마다
  // setAnalysis로 scatterByKey/categoricalByKey를 새 객체로 갈아끼우는데,
  // `analysis`를 의존성에 두면 그 새 객체 참조 자체가 이펙트를 다시
  // 실행시킨다 -- 실행 #2가 같은 50개 잡을 처음부터 다시 발사하고,
  // cleanup으로 취소된 실행 #1은 (위 isCancelled 없이는) 남은 잡을 계속
  // 쏘며 결과만 버려서, in-flight 요청이 계속 쌓이며 무한 반복된다.
  // dataset/createdAt/pointsComplete 세 값이 실제로 바뀔 때만
  // (새 실행이 복원되거나 채움이 끝났을 때만) 다시 돌면 충분하다.
  useEffect(() => {
    if (!analysis || analysis.pointsComplete) return;
    let cancelled = false;
    const { dataset, paretoByTarget: restoredPareto } = analysis;
    void (async () => {
      try {
        await fetchAllScatterData(
          dataset,
          restoredPareto,
          (result) => {
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
          },
          () => cancelled,
        );
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
  }, [analysis?.dataset, analysis?.createdAt, analysis?.pointsComplete, setAnalysis]);

  const analysisVisible = analysis != null && runState === "done";

  const activeParetoResponse = paretoByTarget[activeTarget];
  const activeParetoItems: ParetoRankingItem[] = useMemo(
    () => activeParetoResponse?.items ?? [],
    [activeParetoResponse],
  );
  // 표시 기준은 파레토 기여율 10% 이상 고정(토글 없음) --
  // selectDisplayFactors가 그 필터를 적용한다.
  const displayFactors = useMemo(() => selectDisplayFactors(activeParetoItems), [activeParetoItems]);
  const activeTargetIsEmpty = Boolean(
    analysisVisible && activeParetoResponse && activeParetoResponse.items.length === 0,
  );
  // 이 타깃에서 그릴 차트가 0개인지 -- 순위 자체는 있지만(activeParetoItems
  // 비어있지 않음) 그중 기여율 10% 이상인 인자가 하나도 없는 경우.
  const activeTargetHasNoChart = activeParetoItems.length > 0 && displayFactors.length === 0;
  // 안내 문구("Y4는 기여율 10% 이상 인자가 없습니다 (최대 8.3%)")의
  // 최대값 -- items는 이미 Adjusted R² 내림차순이라 기여율도 같은 순서이지만,
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
    // viewType별로 별도 즐겨찾기이므로, 채움 여부도 viewType별로
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

  // Config 여부는 스키마(config_columns)로 판정한다 -- 요청 전에 권위
  // 있는 근거로 미리 분기해야 한다(400 응답을 받고서야 재호출하는 왕복은
  // 금지). 이름 패턴 `/_Config$/`은 데이터셋 컬럼명 규칙(Step{n}_Config)에
  // 우연히 기댄 휴리스틱이라, 스키마가 아직 도착하지 않은 극히 짧은
  // 창(딥링크 최초 진입)에서만 폴백으로 쓴다.
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
    // main list's per-card Color By.
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
  // 화면에 표시하는 R²(차수) -- Quick Look도 같은
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
        {/* 첫 줄은 1280px에서 한 줄에 들어가도록 word-break: keep-all만
            쓴다 -- white-space: nowrap과 함께 쓰지 않는다(좁은 화면에서
            잘림). */}
        <p className="rootCauseIntro">
          타깃별 Pareto와 강함·보통 등급 인자의 산점도·Box Plot을 확인합니다.
          <br />
          권장 구간은 통계(SPC)와 학습(ML) 두 방식을 비교해 나은 쪽을 채택합니다.
        </p>
        {/* 여섯 화면 공통 헤더 메타(마지막 실행·데이터셋·학습/분석 데이터
            파일명). 화면 하단에 같은 정보를 중복 렌더하지 않는다. */}
        <PageHeaderMeta />
      </section>

      {/* 즐겨찾기 딥링크(?dataset=)가 현재 분석과 다른 데이터셋을 가리키는
          경우에도 불일치가 발생하므로 항상 렌더한다. */}
      <DatasetMismatchWarning mismatch={datasetMismatch} />

      {/* 분석 결과가 아예 없을 때만 렌더하는 빈 상태 안내. 저장된 결과가
          낡은 버전이거나(analysisSnapshotStale) 가리키던 데이터셋이 삭제돼
          버려진 경우(datasetFallbackNotice)도 같은 안내로 합친다 -- 사유가
          달라도 사용자가 취할 다음 행동은 같다. [열기]는 항상 같은 모델
          분석 팝업을 연다. */}
      {runState === "idle" && !analysis && (
        <section className="uploadCard">
          <div className="analysisErrorBox" role="status">
            <div className="analysisErrorBody">
              <p className="analysisErrorMessage">
                분석 결과가 없습니다. 모델 분석에서 분석을 시작하세요.
                {analysisSnapshotStale && " (저장된 결과가 이전 버전이라 불러오지 않았습니다.)"}
                {datasetFallbackNotice && " (이전에 선택한 데이터셋이 더 이상 없어 train으로 전환했습니다.)"}
              </p>
            </div>
            <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
          </div>
        </section>
      )}

      {analysisVisible && <SampleNotice sampleInfo={activeParetoResponse?.sample_info} />}

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
                      {/* 서버가 낸 Adjusted R² 하나만 쓴다 -- 히트맵 셀·파레토
                          막대와 같은 값이라 세 화면이 절대 어긋나지 않는다
                          (예전에는 ε²와 클라이언트 재적합 R²를 나란히
                          띄워 두 숫자가 서로 달랐다). */}
                      <span className="metaItem">
                        R² {quickLookNumeric.adj_r2.toFixed(3)}
                        {quickLookNumeric.degree != null && ` (${quickLookNumeric.degree}차)`}
                      </span>
                      <span className="metaItem">등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}</span>
                    </div>
                  )}
                </div>
              </div>
              {/* 모니터링 트리맵 타일 클릭으로 들어온 경우 -- 어떤 Config
                  값을 보고 왔는지 알려준다. Box Plot 자체는 항상 전체 카테고리를 함께
                  보여주는 게 맞으므로(비교가 목적), 특정 카테고리만
                  숨기는 대신 배너로 표시한다. */}
              {quickLook.isConfig && configFromTreemap && (
                <p className="sectionCaption">트리맵에서 선택: {configFromTreemap}</p>
              )}
              {quickLookError && <p className="errorMessage">{quickLookError}</p>}
              {quickLookCategorical && <ModerateTierCaption tier={quickLookCategorical.confidence_tier} adjR2={quickLookCategorical.adj_r2} />}
              {!quickLookError && quickLookNumeric && !hasReliableEvidence(quickLookNumeric.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  이 인자와 {quickLook.target}의 통계적 연관성은 신뢰도가 낮습니다 (Adj R² = {quickLookNumeric.adj_r2.toFixed(3)}, 등급 {TIER_LABEL[quickLookNumeric.confidence_tier]}).
                  원인으로 단정할 근거는 부족합니다.
                </p>
              )}
              {!quickLookError && quickLookCategorical && !hasReliableEvidence(quickLookCategorical.confidence_tier) && (
                <p className="heatmapSignificanceBanner">
                  통계적 신뢰도가 낮습니다 (Adj R² = {quickLookCategorical.adj_r2.toFixed(3)}, 등급 {TIER_LABEL[quickLookCategorical.confidence_tier]}).
                </p>
              )}
              {quickLookNumeric ? (
                <ScatterChart
                  data={quickLookNumeric}
                  colorMode={quickLookColorMode}
                  view={quickLookView}
                  onSelectWafer={setSelectedWafer}
                  height={chartHeight}
                  reliabilityText={buildModerateInterpretation(quickLookNumeric.confidence_tier, quickLookNumeric.adj_r2)}
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
              <p className="noChartStats">모델 상태와 데이터의 R/D/Config 계측값을 확인한 뒤 모델 분석에서 분석 데이터를 바꾸거나 다시 실행하세요.</p>
              <button type="button" className="button" onClick={() => setAnalysisPanelOpen(true)}>열기</button>
            </section>
          ) : (
            <>
              {/* 파레토는 타깃당 1개, 화면 상단에 고정하고 인자 카드는 그
                  아래에 나열된다. 인자 카드 표시 기준(기여율 10% 이상)과
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
                // 기여율 10% 이상 인자가 하나도 없으면 카드 대신 이 안내를
                // 띄운다.
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
 * 토글과 같은 `.scatterViewToggle*` 마크업을 쓰지만 상태 토글이 아니라
 * 모달을 여는 트리거이므로 `active` 클래스를 절대 붙이지 않는다(눌러도
 * 선택 상태로 남지 않는다). `onTrellis`가 없으면(Config 인자)
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

/** 인자 카드 "보기" 토글 -- Pareto 옵션은 없다(파레토는 타깃당 1개,
 * 화면 상단에 고정된 PinnedParetoCard가 전담). Quick Look의
 * QuickLookViewToggle과 옵션은 같지만 상태를 소유하는 쪽이 다르므로
 * (이 토글은 NumericFactorCard, 저쪽은 root-cause 페이지의 quickLookView)
 * 두 컴포넌트를 따로 둔다. */
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
 * 전용이며 알람 로그·개선 권장 목록은 절대 건드리지
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

// 텍스트 글리프(✓)는 폰트마다 모양·정렬이 달라진다 -- 장식용 문자
// 대신 크기가 고정된 SVG를 쓴다.
function CheckGlyph() {
  return (
    <svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5 6.5 12 13 4" />
    </svg>
  );
}

/** 권장 구간 산출 방식 비교 카드 -- 표시 전용, 여기서 방식을
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

/** 타깃당 1개, 화면 상단에 고정되는 파레토 카드.
 * ParetoChart를 non-embedded(기본) 모드로 그린다 -- 그쪽이 이미 소유한
 * 제목("R/D/Config vs {target}")·등급 범례 헤더를 그대로 쓰고,
 * `headerActions`로 이미지 저장 버튼만 그 헤더 안에 얹는다. 이 경로를 쓰는
 * 곳은 여기뿐이다(인자 카드는 embedded, 즐겨찾기 썸네일은
 * embedded+thumbnail로 부른다). 인자 카드와 달리 즐겨찾기 별은 없다 --
 * 이 카드는 특정 (타깃, 인자) 조합이 아니라 타깃 전체의 순위를 보여줘
 * 즐겨찾기 스냅샷(dataset+target+feature) 모델과 맞지 않는다. */
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
  const cardRef = useRef<HTMLElement | null>(null);
  return (
    <ParetoChart
      target={target}
      items={items}
      n80={n80}
      onBarClick={onBarClick}
      cardRef={cardRef}
      headerActions={
        <ExportPngButton
          nodeRef={cardRef}
          buildOptions={() => ({
            filename: buildExportFilename({ feature: null, target, view: "pareto" }),
            captionText: buildParetoCaptionText({ target, factorCount: items.length, datasetId }),
          })}
        />
      }
    />
  );
}

/** 파레토·산점도·박스플롯·Config 박스플롯 카드
 * 우상단, 즐겨찾기(☆) 옆에 붙는 이미지 저장 버튼. 카드 루트 DOM 노드를
 * 통째로 html-to-image로 캡처한다(lib/chartExport.ts) -- 네 차트가 이
 * 컴포넌트 하나를 공유한다. `buildOptions`를 클릭 시점에 지연 평가하는
 * 것은 파일명·캡션에 들어가는 시각을 "카드가 열린 시각"이 아니라
 * "다운로드를 누른 시각"으로 맞추기 위함이다. */
function ExportPngButton({
  nodeRef,
  buildOptions,
}: {
  nodeRef: RefObject<HTMLElement | null>;
  buildOptions: () => { filename: string; captionText: string };
}) {
  const [busy, setBusy] = useState(false);
  async function handleClick() {
    const node = nodeRef.current;
    if (!node || busy) return;
    setBusy(true);
    try {
      await exportNodeAsPng(node, buildOptions());
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

/** One dropdown per scatter card -- no server round-trip on
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
  // "Config별" 색상 옵션이 고를 수 있는 값 자체가 없으므로 숨긴다.
  hasConfig?: boolean;
}) {
  // 저장된 즐겨찾기 스냅샷에 지금은 없는 값(예: "alarm")이 남아 있어도 화면이
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

/** Owns its own Color By state locally, never in a global store -- the
 * parent forces a reset to 기본 by changing this
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
  // viewType(Scatter/Box)별로 별도 즐겨찾기다 -- 이 카드가 지금 어떤
  // view인지는 카드 자신만 아므로, 부모가 boolean이 아니라 함수를 내려줘
  // 카드 내부에서 현재 view로 평가한다.
  isFavorited: (viewType: string) => boolean;
  isFavoritePending: (viewType: string) => boolean;
  onToggleFavorite: (snapshot: FavoriteSnapshot) => void;
  // 즐겨찾기 스냅샷에 저장 시점의 활성 모델(챔피언) id를 함께 담는다.
  championVersion: string | null;
}) {
  const [colorMode, setColorMode] = useState<ColorMode>("default");
  // View state lives per-card (산점도마다 독립적인 상태), never in a shared
  // store/URL/localStorage -- resets for free whenever this card remounts
  // on a new run/target (see its `key` at the call site).
  const [view, setView] = useState<QuickLookView>("scatter");
  // ≤767px: 산점도 높이 240px.
  const isMobileLayout = useIsMobileLayout();
  const chartHeight = isMobileLayout ? 240 : 480;
  // SPC/ML 토글 상태 -- 기본 선택은 이 인자의 `methods.adopted`를
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
  // 화면에 표시하는 R²(차수) -- ScatterChart가 추세선을 그릴 때
  // 쓰는 것과 같은 fitDefectRateCurve를 이미 받아온 산점도 좌표 위에서
  // 그대로 다시 계산한다(같은 순수 함수·같은 입력이라 값이 어긋나지 않는다).
  const curveFit = useMemo(() => (numericData ? fitDefectRateCurve(numericData.points) : null), [numericData]);
  // 이미지 저장 버튼이 이 카드 루트를 DOM 캡처한다.
  const cardRef = useRef<HTMLElement | null>(null);
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`} ref={cardRef}>
      <div className="factorChartHeader">
        <div className="factorChartHeaderRow">
          <div className="factorChartTitleRow">
            <h2>{item.feature} vs {activeTarget}</h2>
            <ConfidenceBadge tier={item.confidence_tier} />
            {/* 표본이 하한(30) 이상이면 배제하지 않고 그대로 판정하되,
                종류별(R/D) 정상 판정 임계 미만이면 등급을 낮추고(위 배지에
                이미 반영됨) 이유를 별도로 밝힌다 -- 등급만 봐서는 "왜
                낮은지"가 안 보인다. */}
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
                  interpretation: buildModerateInterpretation(item.confidence_tier, item.adj_r2),
                  championVersion,
                })
              }
            />
            {/* 이미지 저장 버튼은 즐겨찾기 별 바로 옆. */}
            <ExportPngButton
              nodeRef={cardRef}
              buildOptions={() => ({
                filename: buildExportFilename({ feature: item.feature, target: activeTarget, view }),
                captionText: buildFactorCaptionText({
                  feature: item.feature,
                  target: activeTarget,
                  adjR2: item.adj_r2,
                  degree: item.degree,
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
          {/* p-value·q-value는 화면에 내보내지 않는다 -- FDR 게이트는
              백엔드에서 계속 동작하지만, 표본이 커지면 p는 약한 관계도
              극단적으로 유의해져 변별력이 없다. 대신 Adjusted R²(적합
              차수)와 파레토 기여율이 서로를 보완한다. 설명력 숫자는
              서버가 낸 adj_r2 하나뿐이다 -- 히트맵 셀·파레토 막대와 같은
              값이라 화면끼리 어긋나지 않는다. */}
          {numericData && (
            <div className="factorChartMetaLine">
              <span className="metaItem">n={numericData.n.toLocaleString()}</span>
              <span className="metaItem">
                R² {item.adj_r2.toFixed(3)}
                {item.degree != null && ` (${item.degree}차)`}
              </span>
              <span className="metaItem">기여율 {item.contribution_pct.toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (Adj R² = {item.adj_r2.toFixed(3)}, 등급 {TIER_LABEL[item.confidence_tier]}). 원인으로 단정할 근거는 부족합니다.
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
            reliabilityText={buildModerateInterpretation(item.confidence_tier, item.adj_r2)}
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
  const cardRef = useRef<HTMLElement | null>(null);
  return (
    <article className="resultCard factorChartCard" id={`factor-${item.feature}`} ref={cardRef}>
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
                  interpretation: buildModerateInterpretation(item.confidence_tier, item.adj_r2),
                  championVersion,
                })
              }
            />
            <ExportPngButton
              nodeRef={cardRef}
              buildOptions={() => ({
                filename: buildExportFilename({ feature: item.feature, target: activeTarget, view: "box" }),
                captionText: buildFactorCaptionText({
                  feature: item.feature,
                  target: activeTarget,
                  adjR2: item.adj_r2,
                  degree: item.degree,
                  n: categoricalData?.n ?? item.n_observed,
                  datasetId: dataset,
                }),
              })}
            />
          </div>
        </div>
        {/* Config의 Adjusted R²는 더미회귀에서 나오므로 적합 차수(1차/2차)가
            없다 -- 차수 배지는 수치형 카드에만 붙는다. */}
        {categoricalData && (
          <small className="factorChartStats">
            <span>n={categoricalData.n}</span>
            <span>Adj R²={item.adj_r2.toFixed(3)}</span>
            <span>기여율={item.contribution_pct.toFixed(1)}%</span>
          </small>
        )}
      </div>
      <ModerateTierCaption tier={item.confidence_tier} adjR2={item.adj_r2} />
      {!hasReliableEvidence(item.confidence_tier) && (
        <p className="heatmapSignificanceBanner">
          이 인자와 {activeTarget}의 통계적 연관성은 신뢰도가 낮습니다 (Adj R² = {item.adj_r2.toFixed(3)}, 등급 {TIER_LABEL[item.confidence_tier]}). 원인으로 단정할 근거는 부족합니다.
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

/** 즐겨찾기 별 토글 -- 저장 시점 상태 스냅샷만 넘긴다, 점
 * 데이터는 절대 포함하지 않는다. */
function FavoriteStarButton({
  favorited,
  disabled,
  onClick,
}: {
  favorited: boolean;
  // 생성/삭제 요청이 진행 중인 동안 버튼을 막는다 -- 빠른 더블클릭이
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
  // 없음"인지 구분이 안 되므로, 행 자체를 숨긴다.
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
