"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import CorrelationHeatmap, { type HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import type { HeatmapResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

// Matches --header-height (60px) + its 1px bottom border, i.e. the same
// `top` offset the sticky segment bar docks at in globals.css -- keeps the
// "stuck" class flipping at exactly the moment position:sticky engages.
const STICKY_OFFSET_PX = 61;

/** The shared "전체 히트맵 → 타깃 세그먼트" block used identically by the
 * 모델 학습 tab and the 원인 분석 tab -- 두 탭이 같은 컴포넌트를
 * 재사용한다. Pareto는 여기서 그리지 않는다: 원인 분석 탭은 각 인자
 * 카드의 보기 토글 안에서 Pareto를 그리고, 모델 학습 탭은 인자 스크리닝
 * 테이블만 보여준다.
 */
export default function HeatmapParetoSection({
  datasetId,
  enabled,
  activeTarget,
  onActiveTargetChange,
  onHeatmapCellSelect,
  // 표시 기준 토글 -- Y 세그먼트와 같은 줄, 우측 정렬로 얹는
  // 선택적 슬롯. 모델 학습 탭은 이 토글이 없으므로 넘기지 않으면 아무것도
  // 렌더되지 않는다 (이 컴포넌트는 두 탭이 공유한다).
  criterionControl,
  // CorrelationHeatmap으로 그대로 흘려보낸다.
  heatmapInitialCache,
  onHeatmapCacheUpdate,
  heatmapInitialSortMode,
  onHeatmapSortModeChange,
  heatmapFavorited,
  heatmapFavoritePending,
  onHeatmapToggleFavorite,
}: {
  datasetId: string;
  enabled: boolean;
  activeTarget: string;
  onActiveTargetChange: (target: string) => void;
  onHeatmapCellSelect: (selection: HeatmapCellSelection) => void;
  criterionControl?: ReactNode;
  heatmapInitialCache?: Record<string, HeatmapResponse>;
  onHeatmapCacheUpdate?: (cache: Record<string, HeatmapResponse>) => void;
  heatmapInitialSortMode?: string | null;
  onHeatmapSortModeChange?: (sortMode: string) => void;
  heatmapFavorited?: boolean;
  heatmapFavoritePending?: boolean;
  onHeatmapToggleFavorite?: (sortMode: string) => void;
}) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(false);

  // Purely a visual toggle (glass background on/off) -- position:sticky
  // itself never changes, so there is no jump. The sentinel sits right
  // where the bar's normal-flow position is; once it scrolls past the
  // header, the bar is sticking.
  useEffect(() => {
    const node = sentinelRef.current;
    if (!node || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setStuck(!entry.isIntersecting),
      { rootMargin: `-${STICKY_OFFSET_PX}px 0px 0px 0px`, threshold: 0 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [enabled]);

  return (
    <>
      <CorrelationHeatmap
        datasetId={datasetId}
        enabled={enabled}
        onSelectCell={onHeatmapCellSelect}
        initialCache={heatmapInitialCache}
        onCacheUpdate={onHeatmapCacheUpdate}
        initialSortMode={heatmapInitialSortMode}
        onSortModeChange={onHeatmapSortModeChange}
        favorited={heatmapFavorited}
        favoritePending={heatmapFavoritePending}
        onToggleFavorite={onHeatmapToggleFavorite}
      />

      {enabled && (
        <>
          <div ref={sentinelRef} aria-hidden="true" style={{ height: 0 }} />
          <div className={`targetSegmentBar ${stuck ? "stuck" : ""}`}>
            <div className="targetSegmentGroup">
              {TARGETS.map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`targetSegment ${activeTarget === t ? "active" : ""}`}
                  onClick={() => onActiveTargetChange(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            {criterionControl && <div className="targetSegmentCriterion">{criterionControl}</div>}
          </div>
        </>
      )}
    </>
  );
}
