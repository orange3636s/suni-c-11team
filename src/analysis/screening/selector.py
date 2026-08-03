"""Per-target Pareto selection: effect size -> BH-FDR -> select all survivors.

Order matters and must not be reshuffled:
  1. compute eps2/p for every factor (R + D + Config)
  2. BH-FDR correct p-values within the target (one target = one family)
  3. drop anything with q >= fdr_alpha
  4. every remaining (significant) factor is selected -- no further
     cumulative-contribution cut. `contribution_pct`/`cumulative_pct` are
     still reported on every factor (selected or not), denominated by the
     FULL candidate pool's eps2 sum, for the Pareto chart/heatmap display.
     Significance and contribution are different axes; neither filters
     the other.
  5. if nothing survives FDR, return an empty list with
     no_significant_factor=True -- never backfill with insignificant factors
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.screening.effect_size import eps2_categorical, eps2_numeric
from src.analysis.screening.schema import Schema
from src.analysis.screening.shape import classify_shape

DEFAULT_CUTOFF = 0.8
DEFAULT_FDR_ALPHA = 0.05
DEFAULT_MIN_N_NUMERIC = 100
DEFAULT_MIN_N_CATEGORICAL = 20
REFERENCE_ONLY_LIMIT = 10
DEFAULT_MAX_DISPLAY = 10

CONFIDENCE_TIERS = ("strong", "moderate", "weak", "reference")


def confidence_tier(p_value: float) -> str:
    """p-value -> a display-confidence label, independent of FDR.

    The FDR gate (q < 0.05) decides what feeds the alarm engine; it does
    NOT decide what's allowed on screen. Everything gets shown, tiered
    by raw p-value, so a factor never simply vanishes because it missed
    a threshold -- it just reads as less trustworthy.
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


@dataclass
class TargetParetoResult:
    target: str
    factors: list[ParetoFactor] = field(default_factory=list)
    reference_only: list[ParetoFactor] = field(default_factory=list)
    excluded_count: int = 0
    no_significant_factor: bool = False


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
    q-value, significance -- one target is one FDR family. Shared by the
    Pareto selector and the correlation heatmap so both surfaces report
    identical q-values for the same factor/target pair; do not duplicate
    this FDR-application step elsewhere.
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


def select_pareto_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    cutoff: float = DEFAULT_CUTOFF,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> TargetParetoResult:
    del cutoff  # kept for API compatibility; selection no longer applies a cumulative cut (see below)
    rows = score_all_factors(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)

    if not rows:
        return TargetParetoResult(target=target, no_significant_factor=True)

    rows.sort(key=lambda r: r["eps2"], reverse=True)

    # contribution_pct/cumulative_pct are reported against the FULL candidate
    # pool (every R+D+Config factor evaluated for this target, e.g. 88 for
    # the bundled dataset) -- not just the FDR-significant subset. FDR
    # answers "is this factor trustworthy"; contribution answers "how much
    # of total explanatory power is this" -- these are different axes, and
    # denominating contribution by the significant-only sum made a single
    # significant factor read as 100% of "everything", which is never true.
    total_eps2 = sum(r["eps2"] for r in rows)
    cumulative = 0.0
    for row in rows:
        row["contribution_pct"] = (row["eps2"] / total_eps2 * 100.0) if total_eps2 > 0 else 0.0
        cumulative += row["contribution_pct"]
        row["cumulative_pct"] = cumulative

    significant_rows = [r for r in rows if r["significant"]]

    if not significant_rows:
        reference_only = _build_reference_only(df, target, rows)
        return TargetParetoResult(
            target=target,
            factors=[],
            reference_only=reference_only,
            excluded_count=len(rows),
            no_significant_factor=True,
        )

    # Every FDR-significant factor is selected here -- no further
    # cumulative-contribution cut. Applying one would let contribution
    # filter significance again (the same bug in the other direction): a
    # factor can pass FDR yet still get silently dropped just because a
    # stronger factor already accounts for most of the pool's total eps2.
    factors: list[ParetoFactor] = []
    for row in significant_rows:
        shape, center = _relation_shape(df, target, row["feature"], row["kind"])
        factors.append(
            ParetoFactor(
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
                significant=True,
                relation_shape=shape,
                optimal_center=center,
            )
        )

    reference_only = _build_reference_only(df, target, rows, exclude={f.feature for f in factors})

    return TargetParetoResult(
        target=target,
        factors=factors,
        reference_only=reference_only,
        excluded_count=len(rows) - len(factors),
        no_significant_factor=False,
    )


def _build_reference_only(
    df: pd.DataFrame,
    target: str,
    rows: list[dict],
    exclude: set[str] | None = None,
) -> list[ParetoFactor]:
    exclude = exclude or set()
    candidates = [r for r in rows if r["feature"] not in exclude]
    candidates.sort(key=lambda r: r["eps2"], reverse=True)
    reference: list[ParetoFactor] = []
    for row in candidates[:REFERENCE_ONLY_LIMIT]:
        shape, center = _relation_shape(df, target, row["feature"], row["kind"])
        reference.append(
            ParetoFactor(
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
                contribution_pct=row.get("contribution_pct", 0.0),
                cumulative_pct=row.get("cumulative_pct", 0.0),
                significant=row["significant"],
                relation_shape=shape,
                optimal_center=center,
            )
        )
    return reference


def select_pareto_factors_all_targets(
    df: pd.DataFrame,
    schema: Schema,
    targets: list[str] | None = None,
    cutoff: float = DEFAULT_CUTOFF,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> dict[str, TargetParetoResult]:
    targets = targets or schema.target_cols
    return {
        target: select_pareto_factors(df, schema, target, cutoff=cutoff, fdr_alpha=fdr_alpha)
        for target in targets
    }


def select_display_factors(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    kind: str = "all",
    cutoff: float = DEFAULT_CUTOFF,
    max_display: int = DEFAULT_MAX_DISPLAY,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> TargetParetoResult:
    """Kind-scoped display selection for the root-cause browsing view
    (Pareto chart, scatter cards, heatmap) -- NOT gated by FDR
    significance. `kind` restricts the candidate pool to "R"/"D"/"Config"
    (or "all" for the combined pool), and the denominator for
    contribution_pct is the eps2 sum within that pool only, not the full
    88-factor total.

    Selection rule: walk the pool in eps2-descending order until
    cumulative contribution first reaches `cutoff`. If that takes more
    than `max_display` factors, the signal is too diffuse to usefully
    chart -- show only the single strongest factor instead of
    `max_display` mostly-noise charts.
    """
    rows = score_all_factors(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    if kind != "all":
        rows = [r for r in rows if r["kind"] == kind]
    if not rows:
        return TargetParetoResult(target=target, no_significant_factor=True)

    rows.sort(key=lambda r: r["eps2"], reverse=True)
    total_eps2 = sum(r["eps2"] for r in rows)
    cumulative = 0.0
    n80: int | None = None
    for index, row in enumerate(rows):
        row["contribution_pct"] = (row["eps2"] / total_eps2 * 100.0) if total_eps2 > 0 else 0.0
        cumulative += row["contribution_pct"]
        row["cumulative_pct"] = cumulative
        if n80 is None and cumulative >= cutoff * 100.0:
            n80 = index + 1

    display_rows = rows[:n80] if (n80 is not None and n80 <= max_display) else rows[:1]

    factors: list[ParetoFactor] = []
    for row in display_rows:
        shape, center = _relation_shape(df, target, row["feature"], row["kind"])
        factors.append(
            ParetoFactor(
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
        )

    return TargetParetoResult(
        target=target,
        factors=factors,
        reference_only=[],
        excluded_count=len(rows) - len(factors),
        no_significant_factor=False,
    )


def find_factor(
    df: pd.DataFrame,
    schema: Schema,
    target: str,
    feature: str,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    min_n_numeric: int = DEFAULT_MIN_N_NUMERIC,
    min_n_categorical: int = DEFAULT_MIN_N_CATEGORICAL,
) -> ParetoFactor | None:
    """Score a single named factor against `target`, regardless of whether it
    passed the Pareto 80% cutoff -- lets a heatmap cell for a
    non-significant factor still resolve to a (clearly-labeled) scatter
    view. `q_value`/`significant` still come from the full FDR family, so
    they agree with what select_pareto_factors would report.
    """
    rows = score_all_factors(df, schema, target, fdr_alpha, min_n_numeric, min_n_categorical)
    row = next((r for r in rows if r["feature"] == feature), None)
    if row is None:
        return None
    total_eps2 = sum(r["eps2"] for r in rows)
    rows_by_eps2 = sorted(rows, key=lambda r: r["eps2"], reverse=True)
    cumulative = 0.0
    contribution_pct = 0.0
    for ranked_row in rows_by_eps2:
        pct = (ranked_row["eps2"] / total_eps2 * 100.0) if total_eps2 > 0 else 0.0
        cumulative += pct
        if ranked_row["feature"] == feature:
            contribution_pct = pct
            break
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
        contribution_pct=contribution_pct,
        cumulative_pct=cumulative,
        significant=row["significant"],
        relation_shape=shape,
        optimal_center=center,
    )
