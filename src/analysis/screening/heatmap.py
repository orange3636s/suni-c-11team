"""Full factor x target correlation heatmap ("why aren't the other factors
picked" view, complementing the per-factor scatter's "why this factor").

Metric modes:
  - spearman: rank correlation, pairwise deletion, no imputation. Catches
    monotonic non-linear relationships Pearson misses.
  - eps2: reuses `score_all_factors` (the same bias-corrected effect size +
    BH-FDR family used by the Pareto selector) so this view never disagrees
    with what actually got selected in the screening tab.

Config columns are excluded entirely -- correlation isn't defined for an
unordered categorical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import DEFAULT_FDR_ALPHA, confidence_tier, score_all_factors

MIN_CELL_N = 30
SPEARMAN_SCALE = (-0.5, 0.5)
EPS2_SCALE = (0.0, 0.7)

Metric = Literal["spearman", "eps2"]


@dataclass
class HeatmapData:
    features: list[str]
    targets: list[str]
    values: list[list[float | None]] = field(default_factory=list)
    n: list[list[int]] = field(default_factory=list)
    q: list[list[float | None]] = field(default_factory=list)
    significant: list[list[bool]] = field(default_factory=list)
    tier: list[list[str | None]] = field(default_factory=list)
    scale: dict[str, float] = field(default_factory=dict)
    excluded_configs: int = 0


def _pairwise_n_and_rho(df: pd.DataFrame, feature: str, target: str, categorical: bool) -> tuple[int, float | None]:
    frame = df[[feature, target]].dropna()
    n = len(frame)
    if categorical or n < MIN_CELL_N:
        # Spearman rho isn't defined for a categorical (Config) factor --
        # only n (for the "표본 부족" mask) is meaningful here.
        return n, None
    rho = frame[feature].corr(frame[target], method="spearman")
    return n, (float(rho) if pd.notna(rho) else None)


def _features_for_kind(schema: Schema, kind: str) -> list[str]:
    if kind == "R":
        return list(schema.r_cols)
    if kind == "D":
        return list(schema.d_cols)
    if kind == "Config":
        return list(schema.config_cols)
    return [*schema.r_cols, *schema.d_cols]  # "all" -- Config stays excluded (no rho)


def build_heatmap(
    df: pd.DataFrame,
    schema: Schema,
    metric: Metric = "spearman",
    kind: str = "all",
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> HeatmapData:
    features = _features_for_kind(schema, kind)
    categorical = kind == "Config"
    # A categorical factor only has an effect-size scale -- rho is not
    # defined for it regardless of what the caller asked for.
    effective_metric: Metric = "eps2" if categorical else metric
    targets = schema.target_cols

    scored_by_target = {
        target: {row["feature"]: row for row in score_all_factors(df, schema, target, fdr_alpha=fdr_alpha)}
        for target in targets
    }

    values: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    q_grid: list[list[float | None]] = []
    sig_grid: list[list[bool]] = []
    tier_grid: list[list[str | None]] = []
    rho_for_sort: list[list[float | None]] = []

    for feature in features:
        value_row: list[float | None] = []
        n_row: list[int] = []
        q_row: list[float | None] = []
        sig_row: list[bool] = []
        tier_row: list[str | None] = []
        rho_row: list[float | None] = []
        for target in targets:
            n, rho = _pairwise_n_and_rho(df, feature, target, categorical)
            scored = scored_by_target.get(target, {}).get(feature)
            if n < MIN_CELL_N:
                value = None
            elif effective_metric == "eps2":
                value = scored["eps2"] if scored else None
            else:
                value = rho
            value_row.append(value)
            n_row.append(n)
            q_row.append(scored["q_value"] if scored else None)
            sig_row.append(bool(scored["significant"]) if scored else False)
            tier_row.append(confidence_tier(scored["p_value"]) if (scored and n >= MIN_CELL_N) else None)
            rho_row.append(rho if rho is not None else (scored["eps2"] if scored else None))
        values.append(value_row)
        n_grid.append(n_row)
        q_grid.append(q_row)
        sig_grid.append(sig_row)
        tier_grid.append(tier_row)
        rho_for_sort.append(rho_row)

    order = sorted(
        range(len(features)),
        key=lambda i: max((abs(v) for v in rho_for_sort[i] if v is not None), default=0.0),
        reverse=True,
    )
    scale_min, scale_max = EPS2_SCALE if effective_metric == "eps2" else SPEARMAN_SCALE

    return HeatmapData(
        features=[features[i] for i in order],
        targets=targets,
        values=[values[i] for i in order],
        n=[n_grid[i] for i in order],
        q=[q_grid[i] for i in order],
        significant=[sig_grid[i] for i in order],
        tier=[tier_grid[i] for i in order],
        scale={"min": scale_min, "max": scale_max},
        excluded_configs=len(schema.config_cols) if kind != "Config" else 0,
    )
