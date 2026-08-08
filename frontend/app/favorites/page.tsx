"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import DashboardShell from "@/components/DashboardShell";
import ParetoChart from "@/components/ParetoChart";
import PlotlyChart from "@/components/PlotlyChart";
import ScatterChart from "@/components/ScatterChart";
import { deleteFavorite, getFavorites, getScreeningPareto, getScreeningScatter, getScreeningScatterCategorical } from "@/lib/api";
import { buildCategoricalSpec } from "@/lib/constants";
import { formatLastRun } from "@/lib/timeFormat";
import type {
  CategoricalScatterResponse,
  FavoriteRecord,
  ParetoRankingResponse,
  ScreeningScatterResponse,
} from "@/types/data";

const VIEW_LABEL: Record<string, string> = { scatter: "Scatter Plot", box: "Box Plot", pareto: "Pareto" };
const THUMBNAIL_HEIGHT = 160;

function noop() {}

export default function FavoritesPage() {
  const router = useRouter();
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

  function openInRootCause(item: FavoriteRecord) {
    const { snapshot } = item;
    // B-5: dataset을 함께 실어야 한다 -- 안 실으면 원인 분석은 현재
    // 선택된(즐겨찾기와 무관한) 데이터셋으로 조회해, train에서 저장한
    // 카드를 test가 선택된 상태에서 열면 같은 인자명의 다른 데이터셋
    // 차트가 경고 없이 표시된다.
    const params = new URLSearchParams({ dataset: snapshot.dataset, target: snapshot.target, feature: snapshot.feature });
    router.push(`/root-cause?${params.toString()}`);
  }

  return (
    <DashboardShell activeItem="즐겨찾기">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">FAVORITES</span>
        <h1>즐겨찾기</h1>
        <p>원인 분석에서 ☆로 저장한 그래프를 최신순으로 모아 봅니다.</p>
      </section>

      {error && <p className="errorMessage">{error}</p>}

      {items && items.length === 0 && (
        <p className="emptyMessage">저장된 그래프가 없습니다. 원인 분석에서 ☆ 버튼으로 추가하세요.</p>
      )}

      {items && items.length > 0 && (
        <div className="favoritesGrid">
          {items.map((item) => (
            <FavoriteCard key={item.favorite_id} item={item} onOpen={() => openInRootCause(item)} onDelete={() => handleDelete(item.favorite_id)} />
          ))}
        </div>
      )}
    </DashboardShell>
  );
}

function FavoriteCard({ item, onOpen, onDelete }: { item: FavoriteRecord; onOpen: () => void; onDelete: () => void }) {
  const { snapshot } = item;
  return (
    <article className="resultCard favoriteCard">
      <button type="button" className="favoriteCardDelete" onClick={(event) => { event.stopPropagation(); onDelete(); }} aria-label="삭제">
        ✕
      </button>
      <div className="favoriteCardThumb" onClick={onOpen} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter") onOpen(); }}>
        <FavoriteThumbnail snapshot={snapshot} />
      </div>
      <div className="favoriteCardBody" onClick={onOpen}>
        <h3>{snapshot.feature} vs {snapshot.target}</h3>
        <div className="favoriteCardMeta">
          <span className="favoriteCardBadge">{VIEW_LABEL[snapshot.viewType] ?? snapshot.viewType}</span>
          <span className="favoriteCardTime">{formatLastRun(item.created_at)}</span>
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

  useEffect(() => {
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
  }, [snapshot.dataset, snapshot.target, snapshot.feature, snapshot.isConfig, snapshot.viewType]);

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
      />
    );
  }
  return <p className="emptyMessage">불러오는 중…</p>;
}
