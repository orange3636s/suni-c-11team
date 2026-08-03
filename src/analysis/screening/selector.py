"""Factor scoring and selection: effect size -> BH-FDR q-value -> ranking.

Everything here operates on the FULL R+D+Config pool -- there is no "kind"
split anymore (see the "원인 분석 단순화" prompt: R/D/Config tabs were
removed, both the root-cause and training screens show one unified view).

Three selection concepts, each serving a different caller:
  - `select_top_factors`: fixed top-N (5) by eps2, full-pool contribution.
    Feeds the Pareto chart / factor cards. Count never varies by target --
    layout stability matters more than a cumulative-contribution cutoff.
  - `select_primary_factor`: the single strongest factor for a target,
    returned regardless of its p-value (confidence is communicated via a
    tier badge, not by hiding the factor). Only `None` when every
    candidate fails its own minimum-sample-size gate -- feeds the model
    training pipeline and the "1위 인자" summary card.
  - `select_alarm_factor`: `select_primary_factor` gated by p<0.05. Alarm
    generation is the one place significance still filters what's shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.screening.effect_size import eps2_categorical, eps2_numeric
from src.analysis.screening.schema import Schema
from src.analysis.screening.shape import classify_shape

DEFAULT_FDR_ALPHA = 0.05
DEFAULT_MIN_N_NUMERIC = 100
DEFAULT_MIN_N_CATEGORICAL = 20
DEFAULT_TOP_N = 5

CONFIDENCE_TIERS = ("strong", "moderate", "weak", "reference")


def confidence_tier(p_value: float) -> str:
    """p-value -> a display-confidence label, independent of FDR.

    The FDR gate (q < 0.05) no longer decides what feeds the alarm engine
    either (see select_alarm_factor, which gates on raw p instead) -- it
    is purely informational now, surfaced in q_value for tooltips.
    Nothing is hidden because of low confidence; it just reads as a
    weaker tier.
    """
    if p_value < 0.01:
        return "strong"
    if p_value < 0.05:
        return "moderate"
    if p_value < 0.20:
        return "weak"
    return "reference"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH step-up FDR correction. Returns q-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q.tolist()


@dataclass
class ParetoFactor:
    target: str
    feature: str
    kind: str  # "R" | "D" | "Config"
    step: int
    eps2: float
    p_value: float
    q_value: float
    pearson_r: float | None
    spearman_r: float | None
    n_observed: int
    contribution_pct: float
    cumulative_pct: float
    significant: bool
    relation_shape: str
    optimal_center: float | None


def _evaluate_all_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    min_n_numeric: int,
    min_n_categorical: int,
) -> list[dict]:
    rows: list[dict] = []
    y = df[target]

    for feature in [*schema.r_cols, *schema.d_cols]:
        result = eps2_numeric(df[feature], y, min_n=min_n_numeric)
        if result is None:
            continue
        kind = schema.kind_of(feature)
        rows.append(
            {
                "feature": feature,
                "kind": kind,
                "step": schema.step_of(feature) or 0,
                "eps2": result.eps2,
                "p_value": result.p_value,
                "n_observed": result.n_observed,
                "pearson_r": result.pearson_r,
                "spearman_r": result.spearman_r,
            }
        )

    for feature in schema.config_cols:
        result = eps2_categorical(df[feature], y, min_n=min_n_categorical)
        if result is None:
            continue
        rows.append(
            {
                "feature": feature,
                "kind": "Config",
                "step": schema.step_of(feature) or 0,
                "eps2": result.eps2,
                "p_value": result.p_value,
                "n_observed": result.n_observed,
                "pearson_r": None,
                "spearman_r": None,
            }
        )

    return rows


def _relation_shape(df: pd.DataFrame, target: str, feature: str, kind: str) -> tuple[str, float | None]:
    if kind == "Config":
        return "unclear", None
    result = classify_shape(df[feature], df[target])
    return result.shape, result.optimal_center


def score_all_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> list[dict]:
    """Score every R/D/Config candidate factor against `target`: eps2, BH-FDR
    q-value (informational only -- see confidence_tier's docstring) -- one
    target is one FDR family. Shared by every selection function below and
    by the correlation heatmap so all surfaces report identical q-values
    for the same factor/target pair.
    """
    rows = _evaluate_all_factors(df, schema, target, min_n_numeric, min_n_categorical)
    if not rows:
        return []

    p_values = [r["p_value"] for r in rows]
    q_values = benjamini_hochberg(p_values)
    for row, q in zip(rows, q_values):
        row["q_value"] = float(q)
        row["significant"] = bool(q < fdr_alpha)
    return rows


def _ranked_rows_with_contribution(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    fdr_alpha: float,
    min_n_numeric: int,
    min_n_categorical: int,
) -> list[dict]:
    """Every candidate factor for `target`, sorted by eps2 descending, with
    contribution_pct/cumulative_pct populated against the FULL pool's eps2
    sum. The single ranked list every selection function below slices.
    """
    rows = score_all_factors(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    if not rows:
        return []
    rows.sort(key=lambda r: r["eps2"], reverse=True)
    total_eps2 = sum(r["eps2"] for r in rows)
    cumulative = 0.0
    for row in rows:
        row["contribution_pct"] = (row["eps2"] / total_eps2 * 100.0) if total_eps2 > 0 else 0.0
        cumulative += row["contribution_pct"]
        row["cumulative_pct"] = cumulative
    return rows


def _row_to_factor(df: pd.DataFrame, target: str, row: dict) -> ParetoFactor:
    shape, center = _relation_shape(df, target, row["feature"], row["kind"])
    return ParetoFactor(
        target=target,
        feature=row["feature"],
        kind=row["kind"],
        step=row["step"],
        eps2=row["eps2"],
        p_value=row["p_value"],
        q_value=row["q_value"],
        pearson_r=row["pearson_r"],
        spearman_r=row["spearman_r"],
        n_observed=row["n_observed"],
        contribution_pct=row["contribution_pct"],
        cumulative_pct=row["cumulative_pct"],
        significant=row["significant"],
        relation_shape=shape,
        optimal_center=center,
    )


def select_top_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    limit: int = DEFAULT_TOP_N,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> list[ParetoFactor]:
    """Fixed top-`limit` factors by eps2 across the full R+D+Config pool,
    contribution denominated by that same full pool. The count is always
    `limit` (or fewer if the pool itself has fewer candidates) regardless
    of whether cumulative contribution reaches 80% -- an 80%-cumulative
    cutoff would make the displayed count vary per target, which is
    exactly the layout instability this replaces.
    """
    rows = _ranked_rows_with_contribution(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    return [_row_to_factor(df, target, row) for row in rows[:limit]]


def select_primary_factor(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> ParetoFactor | None:
    """The single strongest (highest-eps2) factor for `target`, regardless
    of p-value -- confidence is a tier badge, not a display filter. Only
    `None` when nothing in the pool clears its own min-n gate (every
    candidate's measured sample is too small to score at all), which is
    the sole "분석 불가" condition.
    """
    rows = _ranked_rows_with_contribution(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    if not rows:
        return None
    return _row_to_factor(df, target, rows[0])


def select_fdr_significant_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> list[ParetoFactor]:
    """Every factor for `target` that passes BH-FDR (q<fdr_alpha) -- the
    alarm engine's factor set. This is the one place significance still
    filters what's used: alarm generation was explicitly kept unchanged
    (still q<0.05-gated, still possibly more than one factor per target,
    e.g. Y2 -> Step16_R1 AND Step24_R1) even though display everywhere
    else (Pareto/heatmap/training cards) no longer gates on significance
    at all. Changing this would change the golden 19-alarm-wafer count.
    """
    rows = _ranked_rows_with_contribution(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    return [_row_to_factor(df, target, row) for row in rows if row["significant"]]


def find_factor(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    feature: str,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> ParetoFactor | None:
    """Score a single named factor against `target`, regardless of its
    Pareto rank -- lets a heatmap cell for any of the 88 factors still
    resolve to a (clearly-tiered) scatter view.
    """
    rows = _ranked_rows_with_contribution(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    row = next((r for r in rows if r["feature"] == feature), None)
    if row is None:
        return None
    return _row_to_factor(df, target, row)
