"use client";

import { useEffect, useState, type KeyboardEvent } from "react";
import { useRouter } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { SORT_OPTION_LABEL } from "@/components/CorrelationHeatmap";
import DashboardShell from "@/components/DashboardShell";
import { PageHeader } from "@/components/PageHeader";
import ParetoChart from "@/components/ParetoChart";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart from "@/components/ScatterChart";
import { deleteFavorite, getFavorites, getScreeningPareto, getScreeningScatter, getScreeningScatterCategorical } from "@/lib/api";
import { buildCategoricalSpec } from "@/lib/constants";
import { formatLastRun } from "@/lib/timeFormat";
import type {
  CategoricalScatterResponse,
  FavoriteRecord,
  FavoriteSnapshot,
  ParetoRankingResponse,
  ScreeningScatterResponse,
} from "@/types/data";

const VIEW_LABEL: Record<string, string> = { scatter: "Scatter Plot", box: "Box Plot", pareto: "Pareto" };
// 카드 상단 배지 -- 히트맵·파레토·트리맵은 산점도·박스플롯과 저장 단위
// 자체가 달라(특정 인자가 아니라 화면 전체/타깃 전체) 썸네일 미리보기가
// 없거나 얕다. 배지로 종류를 먼저 밝혀 구분한다.
const KIND_BADGE: Record<FavoriteSnapshot["viewType"], string> = {
  scatter: "산점도",
  box: "박스플롯",
  pareto: "파레토",
  heatmap: "히트맵",
  treemap: "트리맵",
};
const THUMBNAIL_HEIGHT = 160;

/** 카드 제목 -- scatter/box는 인자명이 핵심이라 기존 형식을 유지하고,
 * pareto/heatmap/treemap은 특정 인자가 아니라 화면 단위라 각자의
 * 식별자(정렬 기준, 스텝)로 제목을 만든다. */
function cardTitle(snapshot: FavoriteRecord["snapshot"]): string {
  switch (snapshot.viewType) {
    case "heatmap": {
      const sortLabel = snapshot.sort ? (SORT_OPTION_LABEL[snapshot.sort as keyof typeof SORT_OPTION_LABEL] ?? snapshot.sort) : "기본";
      return `R, D vs Y1~Y5 상관관계 히트맵 · ${sortLabel} 순`;
    }
    case "pareto":
      return `R/D/Config vs ${snapshot.target} 파레토`;
    case "treemap":
      return `Config vs ${snapshot.target} 트리맵 · Step${snapshot.step ?? "?"}`;
    default:
      return `${snapshot.feature} vs ${snapshot.target} · ${VIEW_LABEL[snapshot.viewType] ?? snapshot.viewType}`;
  }
}

function noop() {}

export default function FavoritesPage() {
  const router = useRouter();
  const { training, snapshot } = useAnalysisState();
  // 카드에 저장된 championVersion과 비교할 "현재" 챔피언 -- 학습
  // 기록이 아직 없으면 null이라, 저장된 값이 있어도 비교 대상이 없으므로
  // 배지를 붙이지 않는다(둘 다 모르는 상태와 "달라졌다"는 다르다).
  const currentChampionVersion = training?.performance.model_id ?? null;
  // 저장된 카드는 저장 시점의 데이터셋으로 만들어졌지만 썸네일은 항상
  // "현재" datasetId로 다시 조회한다(알려진 한계) -- 헤더에 "현재 분석
  // 기준" 데이터셋을 보여주면 이 불일치가 카드별로 눈에 보이게 된다.
  const currentDatasetId = snapshot?.source.eval_dataset ?? null;
  const [items, setItems] = useState<FavoriteRecord[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getFavorites()
      .then((response) => {
        if (!cancelled) setItems(response.items);
      })
      .catch((failure) => {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "즐겨찾기를 불러오지 못했습니다.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleDelete(favoriteId: string) {
    setItems((previous) => previous?.filter((item) => item.favorite_id !== favoriteId) ?? previous);
    try {
      await deleteFavorite(favoriteId);
    } catch {
      // Best-effort -- 다음 새로고침에서 다시 나타날 수 있다.
    }
  }

  // dataset을 항상 함께 실어야 한다 -- 안 실으면 대상 화면은 현재
  // 선택된(즐겨찾기와 무관한) 데이터셋으로 조회해, train에서 저장한
  // 카드를 test가 선택된 상태에서 열면 같은 인자명의 다른 데이터셋
  // 차트가 경고 없이 표시된다. 종류별로 저장 시점 상태(정렬·스텝·타깃)를
  // 함께 실어 그 상태로 복원한다 -- 화면만 여는 것으로는 부족하다.
  function openFavorite(item: FavoriteRecord) {
    const { snapshot } = item;
    if (snapshot.viewType === "heatmap") {
      const params = new URLSearchParams({ dataset: snapshot.dataset });
      if (snapshot.sort) params.set("heatmapSort", snapshot.sort);
      router.push(`/root-cause?${params.toString()}`);
      return;
    }
    if (snapshot.viewType === "treemap") {
      const params = new URLSearchParams({ dataset: snapshot.dataset, target: snapshot.target });
      if (snapshot.step != null) params.set("step", String(snapshot.step));
      router.push(`/config-treemap?${params.toString()}`);
      return;
    }
    // pareto: feature 없이 target만 -- root-cause는 target 하나로 그
    // 타깃의 파레토 카드를 그대로 연다. scatter/box: 기존과 동일.
    const params = new URLSearchParams({ dataset: snapshot.dataset, target: snapshot.target });
    if (snapshot.feature) params.set("feature", snapshot.feature);
    router.push(`/root-cause?${params.toString()}`);
  }

  return (
    <DashboardShell activeItem="즐겨찾기">
      <PageHeader
        eyebrow="저장된 그래프"
        title="즐겨찾기"
        description="원인 분석에서 ☆로 저장한 그래프를 최신순으로 모아 봅니다."
        metaLabel="현재 분석 기준"
      />

      {error && <p className="errorMessage">{error}</p>}

      {items && items.length === 0 && (
        <p className="emptyMessage">저장된 그래프가 없습니다. 원인 분석에서 ☆ 버튼으로 추가하세요.</p>
      )}

      {items && items.length > 0 && (
        <div className="favoritesGrid">
          {items.map((item) => (
            <FavoriteCard
              key={item.favorite_id}
              item={item}
              currentChampionVersion={currentChampionVersion}
              currentDatasetId={currentDatasetId}
              onOpen={() => openFavorite(item)}
              onDelete={() => handleDelete(item.favorite_id)}
            />
          ))}
        </div>
      )}
    </DashboardShell>
  );
}

function FavoriteCard({
  item,
  currentChampionVersion,
  currentDatasetId,
  onOpen,
  onDelete,
}: {
  item: FavoriteRecord;
  currentChampionVersion: string | null;
  currentDatasetId: string | null;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const { snapshot } = item;
  // 저장 시점 챔피언이 있고(이 필드가 없는 오래된 카드는 null), 현재
  // 챔피언도 알고 있는데, 둘이 다르면 저장된 해석이 최신 모델과 어긋날
  // 수 있다는 뜻이다.
  const isStale = Boolean(
    snapshot.championVersion && currentChampionVersion && snapshot.championVersion !== currentChampionVersion,
  );
  // 저장 시점 데이터셋과 현재 분석 기준 데이터셋이 다르면, 이
  // 썸네일은 카드가 저장될 때와 다른 데이터를 다시 조회해 그리고 있다는
  // 뜻이다(위 currentDatasetId 주석 참고).
  const isDatasetStale = Boolean(currentDatasetId && snapshot.dataset !== currentDatasetId);
  // 카드 전체(썸네일+본문)를 하나의 키보드 조작 대상으로 합친다 --
  // 삭제 버튼과 형제로 두어야 하므로(버튼 중첩 불가) <button>이 아니라
  // div + role="button"을 쓴다. Enter/Space 둘 다 처리한다.
  function handleOpenKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen();
    }
  }
  return (
    <article className="resultCard favoriteCard">
      <button type="button" className="favoriteCardDelete" onClick={(event) => { event.stopPropagation(); onDelete(); }} aria-label="삭제">
        ✕
      </button>
      <div
        className="favoriteCardOpenArea"
        role="button"
        tabIndex={0}
        onClick={onOpen}
        onKeyDown={handleOpenKeyDown}
      >
        <div className="favoriteCardThumb">
          <FavoriteThumbnail snapshot={snapshot} />
        </div>
        {/* 카드에 남기는 정보는 배지·제목·시각·해석 넷뿐이다 -- 인자
            메타(n·기여율·p-value 등)는 넣지 않는다. */}
        <div className="favoriteCardBody">
          <span className="favoriteCardKindBadge">{KIND_BADGE[snapshot.viewType] ?? snapshot.viewType}</span>
          <h3>{cardTitle(snapshot)}</h3>
          <span className="favoriteCardTime">{formatLastRun(item.created_at)}</span>
          {snapshot.interpretation && <p className="favoriteCardInterpretation">{snapshot.interpretation}</p>}
          {isStale && <span className="favoriteCardStaleBadge">이전 분석 기준</span>}
          {isDatasetStale && (
            <span className="favoriteCardStaleBadge" title={`저장 시점 데이터셋: ${snapshot.dataset}`}>
              다른 데이터셋으로 저장됨
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

function FavoriteThumbnail({ snapshot }: { snapshot: FavoriteRecord["snapshot"] }) {
  const [numericData, setNumericData] = useState<ScreeningScatterResponse | null>(null);
  const [categoricalData, setCategoricalData] = useState<CategoricalScatterResponse | null>(null);
  const [paretoData, setParetoData] = useState<ParetoRankingResponse | null>(null);
  const [failed, setFailed] = useState(false);

  // 히트맵·트리맵은 화면 전체(또는 스텝 전체)를 다시 그려야 해서 이
  // 작은 썸네일에 맞는 축약 렌더가 없다 -- 조회 자체를 건너뛰고 종류
  // 배지 + 아이콘만 보여준다(위 KIND_BADGE 참고).
  const skipsThumbnail = snapshot.viewType === "heatmap" || snapshot.viewType === "treemap";

  useEffect(() => {
    if (skipsThumbnail) return;
    let cancelled = false;
    setFailed(false);
    if (snapshot.isConfig) {
      getScreeningScatterCategorical(snapshot.dataset, snapshot.target, snapshot.feature)
        .then((result) => !cancelled && setCategoricalData(result))
        .catch(() => !cancelled && setFailed(true));
    } else if (snapshot.viewType === "pareto") {
      getScreeningPareto(snapshot.dataset, snapshot.target)
        .then((result) => !cancelled && setParetoData(result))
        .catch(() => !cancelled && setFailed(true));
    } else {
      getScreeningScatter(snapshot.dataset, snapshot.target, snapshot.feature)
        .then((result) => !cancelled && setNumericData(result))
        .catch(() => !cancelled && setFailed(true));
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot.dataset, snapshot.target, snapshot.feature, snapshot.isConfig, snapshot.viewType, skipsThumbnail]);

  if (snapshot.viewType === "heatmap") {
    return <div className="favoriteCardKindPlaceholder" aria-hidden="true">▤</div>;
  }
  if (snapshot.viewType === "treemap") {
    return <div className="favoriteCardKindPlaceholder" aria-hidden="true">▦</div>;
  }
  if (failed) return <p className="emptyMessage">미리보기를 불러오지 못했습니다.</p>;
  if (categoricalData) return <PlotlyChart spec={buildCategoricalSpec(categoricalData, { compact: true })} height={THUMBNAIL_HEIGHT} />;
  if (paretoData) {
    return (
      <ParetoChart
        target={snapshot.target}
        items={paretoData.items}
        n80={paretoData.n80}
        activeFeature={snapshot.feature}
        onBarClick={noop}
        embedded
        height={THUMBNAIL_HEIGHT}
        thumbnail
      />
    );
  }
  if (numericData) {
    return (
      <ScatterChart
        data={numericData}
        colorMode="default"
        view={snapshot.viewType === "box" ? "box" : "scatter"}
        onSelectWafer={noop}
        height={THUMBNAIL_HEIGHT}
        thumbnail
      />
    );
  }
  return <p className="emptyMessage">불러오는 중…</p>;
}
