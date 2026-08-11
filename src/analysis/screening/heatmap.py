"""Full factor x target correlation heatmap ("why aren't the other factors
picked" view, complementing the per-factor scatter's "why this factor").

Every cell reports the same Adjusted R² + polynomial degree the scatter
chart and Pareto chart use (`adj_r2_numeric`, shared via `score_all_factors`)
-- clicking a cell into its scatter view never disagrees with the number
shown here. Cell color is a single-hue gradient keyed only to Adjusted R²
magnitude, with no signed-correlation ("rho") channel: a single factor is
frequently U-shaped (see `relation_shape`), where a whole-range Spearman
sign is actively misleading for roughly half the sample. Direction is
therefore only ever shown as "which side of the vertex" (2nd-degree fits)
or "increasing/decreasing" (1st-degree fits) in the per-cell tooltip
payload (`relation_shape`/`optimal_center`), never as a scalar sign
driving cell color.

For a gate-excluded cell (n>=MIN_CELL_N but below the R/D-specific
significance gate in `score_all_factors`), adj_r2 is still computed
directly here (bypassing that gate) so the cell still shows a real number
instead of going blank -- only the `gate_excluded`/`tier` flags communicate
the gate outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.analysis.screening.effect_size import adj_r2_numeric
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import (
    DEFAULT_FDR_ALPHA,
    effective_confidence_tier,
    score_all_factors,
)
from src.analysis.screening.shape import classify_shape

MIN_CELL_N = 30
# Adjusted R² concentration scale for the heatmap's color gradient -- fixed
# so cell shading is comparable across screening runs/datasets instead of
# rescaling to whatever the current dataset's max happens to be.
ADJ_R2_SCALE = (0.0, 0.7)


# Config 계층은 src/config_parser.py의 YAML 기반 공통 파서를 사용한다.
# 원본 Config 범주는 그대로 두고 히트맵 집계용 Series만 파생한다.
@dataclass
class HeatmapData:
    features: list[str]
    targets: list[str]
    values: list[list[float | None]] = field(default_factory=list)  # Adjusted R²
    degree: list[list[int | None]] = field(default_factory=list)  # 1 | 2 | None
    shape: list[list[str | None]] = field(default_factory=list)
    optimal_center: list[list[float | None]] = field(default_factory=list)
    n: list[list[int]] = field(default_factory=list)
    q: list[list[float | None]] = field(default_factory=list)
    significant: list[list[bool]] = field(default_factory=list)
    tier: list[list[str | None]] = field(default_factory=list)
    # 상관계수는 그려지는데(n>=MIN_CELL_N) 유의 인자 목록에서는 종류별
    # 표본 게이트 미달로 빠지는 셀 -- 히트맵과 유의 인자 판정 정합용.
    gate_excluded: list[list[bool]] = field(default_factory=list)
    scale: dict[str, float] = field(default_factory=dict)
    excluded_configs: int = 0
    # 표본으로 계산됐으면 채워진다(호출부가 build_heatmap 이후 덧붙인다 --
    # 이 모듈 자체는 표본 여부를 모른다).
    sample_info: dict[str, Any] | None = None
    # 셀 툴팁의 "계측률 %"(= n / total_rows)를 프런트가 낼 수 있도록 하는
    # 분모. 그리드의 최대 n으로는 대신할 수 없다 -- R 인자는 전체의 15%만
    # 계측되므로 최대 n(≈1,500)을 분모로 쓰면 계측률이 항상 100%로 나온다.
    # sample_info와 같은 규칙으로 호출부가 채운다.
    total_rows: int = 0


def _pairwise_cell(
    df: pd.DataFrame, feature: str, target: str
) -> tuple[int, float | None, int | None, str | None, float | None]:
    frame = df[[feature, target]].dropna()
    n = len(frame)
    if n < MIN_CELL_N:
        return n, None, None, None, None
    # 종류별 게이트(R>=100/D>=40)와 무관하게 hard_min_n(30)만 넘으면 항상
    # 계산한다 -- 게이트 미달 셀도 진짜 Adjusted R²를 보여주고, 그 셀이
    # "유의 인자"인지 여부는 별도의 gate_excluded/tier 플래그로만 표시한다.
    result = adj_r2_numeric(frame[feature], frame[target], min_n=MIN_CELL_N)
    if result is None:
        return n, None, None, None, None
    shape_result = classify_shape(frame[feature], frame[target])
    return n, result.adj_r2, result.degree, shape_result.shape, shape_result.optimal_center


def build_heatmap(
    df: pd.DataFrame,
    schema: Schema,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> HeatmapData:
    """Full R+D x Y1~Y5 correlation heatmap. Config is always excluded --
    a polynomial fit isn't defined for an unordered categorical, and
    Config's Adjusted R² scale (dummy-regression) is incomparable to R/D's
    (see the Config treemap's own significance badge instead).
    """
    features = [*schema.r_cols, *schema.d_cols]
    targets = schema.target_cols

    scored_by_target = {
        target: {row["feature"]: row for row in score_all_factors(df, schema, target, fdr_alpha=fdr_alpha)}
        for target in targets
    }

    values: list[list[float | None]] = []
    degree_grid: list[list[int | None]] = []
    shape_grid: list[list[str | None]] = []
    center_grid: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    q_grid: list[list[float | None]] = []
    sig_grid: list[list[bool]] = []
    tier_grid: list[list[str | None]] = []
    gate_grid: list[list[bool]] = []

    for feature in features:
        value_row: list[float | None] = []
        degree_row: list[int | None] = []
        shape_row: list[str | None] = []
        center_row: list[float | None] = []
        n_row: list[int] = []
        q_row: list[float | None] = []
        sig_row: list[bool] = []
        tier_row: list[str | None] = []
        gate_row: list[bool] = []
        for target in targets:
            n, adj_r2, degree, shape, center = _pairwise_cell(df, feature, target)
            scored = scored_by_target.get(target, {}).get(feature)
            value_row.append(adj_r2)
            degree_row.append(degree)
            shape_row.append(shape)
            center_row.append(center)
            n_row.append(n)
            q_row.append(scored["q_value"] if scored else None)
            sig_row.append(bool(scored["significant"]) if scored else False)
            tier_row.append(
                effective_confidence_tier(scored["adj_r2"], scored["p_value"], under_sampled=scored.get("under_sampled", False))
                if (scored and n >= MIN_CELL_N)
                else None
            )
            # 상관계수가 그려지는데(n>=MIN_CELL_N) 유의 인자 풀에는
            # 없는(scored=None) 셀 -- 종류별 표본 게이트 미달 등으로 두
            # 화면이 어긋나는 지점.
            gate_row.append(bool(n >= MIN_CELL_N and scored is None))
        values.append(value_row)
        degree_grid.append(degree_row)
        shape_grid.append(shape_row)
        center_grid.append(center_row)
        n_grid.append(n_row)
        q_grid.append(q_row)
        sig_grid.append(sig_row)
        tier_grid.append(tier_row)
        gate_grid.append(gate_row)

    # 기본 정렬은 항상 "최대 Adjusted R²" -- 다른 정렬(최소/Step 오름/Step
    # 내림)은 프런트가 같은 응답 안의 그리드·인자명으로 클라이언트에서
    # 다시 정렬한다.
    order = sorted(
        range(len(features)),
        key=lambda i: max((v for v in values[i] if v is not None), default=0.0),
        reverse=True,
    )

    return HeatmapData(
        features=[features[i] for i in order],
        targets=targets,
        values=[values[i] for i in order],
        degree=[degree_grid[i] for i in order],
        shape=[shape_grid[i] for i in order],
        optimal_center=[center_grid[i] for i in order],
        n=[n_grid[i] for i in order],
        q=[q_grid[i] for i in order],
        significant=[sig_grid[i] for i in order],
        tier=[tier_grid[i] for i in order],
        gate_excluded=[gate_grid[i] for i in order],
        scale={"min": ADJ_R2_SCALE[0], "max": ADJ_R2_SCALE[1]},
        excluded_configs=len(schema.config_cols),
    )
