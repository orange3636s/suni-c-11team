"use client";

import CorrelationHeatmap, { type HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import ParetoChart from "@/components/ParetoChart";
import type { ParetoRankingItem, ParetoRankingResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

/** The one shared "전체 히트맵 → 타깃 세그먼트 → Pareto" block used
 * identically by the 모델 학습 tab and the 원인 분석 tab (see the
 * "원인 분석 단순화" prompt §2-2: "히트맵·세그먼트·Pareto는 동일
 * 컴포넌트를 재사용한다"). Each caller runs its own execution flow that
 * fetches `getScreeningPareto(dataset, target)` for all 5 targets and
 * hands the results in as `paretoByTarget` -- the backend LRU cache
 * (keyed by dataset+target) guarantees both tabs see byte-identical
 * numbers for the same target even though each tab fetches
 * independently. Only the scatter section (원인 분석 전용) lives outside
 * this component.
 */
export default function HeatmapParetoSection({
  datasetId,
  enabled,
  paretoByTarget,
  activeTarget,
  onActiveTargetChange,
  onBarClick,
  onHeatmapCellSelect,
}: {
  datasetId: string;
  enabled: boolean;
  paretoByTarget: Record<string, ParetoRankingResponse>;
  activeTarget: string;
  onActiveTargetChange: (target: string) => void;
  onBarClick: (item: ParetoRankingItem) => void;
  onHeatmapCellSelect: (selection: HeatmapCellSelection) => void;
}) {
  const activeResponse = paretoByTarget[activeTarget];

  return (
    <>
      <CorrelationHeatmap datasetId={datasetId} enabled={enabled} onSelectCell={onHeatmapCellSelect} />

      {enabled && (
        <>
          <div className="targetSegmentBar">
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

          {activeResponse ? (
            <ParetoChart
              target={activeTarget}
              items={activeResponse.items}
              n80={activeResponse.n80}
              onBarClick={onBarClick}
            />
          ) : (
            <section className="resultCard">
              <p className="emptyMessage">불러오는 중…</p>
            </section>
          )}
        </>
      )}
    </>
  );
}
