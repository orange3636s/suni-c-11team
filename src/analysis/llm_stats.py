"""Extra statistics computed only for the LLM report/chat context (spec
"SUNI 챗봇 LLM 보고서 생성" §1) -- control-band bootstrap stability, per-factor
chamber interaction, the Config main-effect detection limit, and the
measurement-bias check. Purely additive: nothing here changes the output of
select_*/compute_control_range, it only computes extra numbers report.py
layers on top for judge_confidence and the narrative sections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.recommendations import _recommended_range_raw
from src.analysis.screening.effect_size import adj_r2_categorical
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import benjamini_hochberg
from src.config_parser import config_hierarchy_series

DEFAULT_N_BOOT = 100


def band_stability(x: pd.Series, n_boot: int = DEFAULT_N_BOOT) -> float:
    """Bootstrap std of the IQR*1.5 control-band center: how much the band
    would move under resampling. Each draw's `random_state` is fixed to its
    own index so the same input always reproduces the same value.
    """
    values = pd.to_numeric(x, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    centers = []
    for b in range(n_boot):
        r = values.sample(len(values), replace=True, random_state=b)
        q1, q3 = r.quantile([0.25, 0.75])
        iqr = q3 - q1
        centers.append(((q1 - 1.5 * iqr) + (q3 + 1.5 * iqr)) / 2)
    return float(np.std(centers))


def _chamber_column(df: pd.DataFrame, config_col: str) -> pd.Series:
    """The chamber token (last '_'-separated segment) of a Config column's
    value, e.g. 'Step16_Model3_EQB_CH4' -> 'CH4'.
    """
    return config_hierarchy_series(config_col, df[config_col], "chamber")


def chamber_interaction_p(df: pd.DataFrame, feature: str, target: str, config_col: str) -> float | None:
    """p-value for whether feature's relationship with target differs by
    chamber. None when the config column is missing or fewer than 2
    chambers have usable data (interaction term can't be estimated).
    """
    if config_col not in df.columns or feature not in df.columns or target not in df.columns:
        return None
    d = df[[feature, target, config_col]].dropna().rename(columns={feature: "x", target: "yv"})
    if d.empty:
        return None
    d = d.assign(CH=_chamber_column(d, config_col))
    if d["CH"].nunique() < 2:
        return None
    import statsmodels.formula.api as smf  # deferred: ~1s import cost, only paid when this factor actually needs it

    try:
        model = smf.ols("yv ~ x * C(CH)", data=d).fit()
        param_index = list(model.params.index)
        terms = [t for t in param_index if ":" in t]
        if not terms:
            return None
        a = np.zeros((len(terms), len(param_index)))
        for i, t in enumerate(terms):
            a[i, param_index.index(t)] = 1
        return float(model.f_test(a).pvalue)
    except (ValueError, np.linalg.LinAlgError):
        return None


def per_chamber_window(df: pd.DataFrame, feature: str, target: str, config_col: str) -> dict[str, dict]:
    """Per-chamber recommended window, using the same contiguous-quantile-bin
    rule as the factor's own window (recommendations._recommended_range_raw)
    restricted to each chamber's rows. Only meaningful for factors flagged
    by chamber_interaction_p.
    """
    if config_col not in df.columns:
        return {}
    d = df[[feature, target, config_col]].dropna().rename(columns={feature: "x", target: "yv"})
    if d.empty:
        return {}
    d = d.assign(CH=_chamber_column(d, config_col))
    out: dict[str, dict] = {}
    for chamber, group in d.groupby("CH"):
        raw = _recommended_range_raw(group["x"], group["yv"])
        if raw is None:
            continue
        lo, hi = raw
        overall_mean = float(group["yv"].mean())
        in_window = group[(group["x"] >= lo) & (group["x"] <= hi)]
        ratio = float(in_window["yv"].mean() / overall_mean) if len(in_window) and overall_mean else None
        out[str(chamber)] = {"lo": lo, "hi": hi, "ratio": ratio, "n": int(len(group))}
    return dict(sorted(out.items()))


@dataclass
class ConfigScreeningResult:
    n_tested: int
    n_significant_fdr: int
    max_observed_adj_r2: float | None
    max_observed_feature: str | None
    max_observed_target: str | None
    mde_adj_r2: float | None
    median_n_per_group: int | None


def config_main_effect_screening(
    df: pd.DataFrame,
    schema: Schema,
    fdr_alpha: float = 0.05,
    min_n: int = 20,
) -> ConfigScreeningResult:
    """Config-only main-effect screen: every Config column x every target
    (30 x 5 = 150 for train.CSV), BH-FDR corrected across the whole 150-cell
    family (a separate FDR family from the per-target R+D+Config screen in
    selector.py -- this one asks "does equipment configuration alone matter
    anywhere", not "what's the best factor for this target"). Reports the
    minimum-detectable-effect (MDE) size this sample could have caught, so a
    0-significant result can be read as "no effect above the detection
    floor" rather than "no effect at all".
    """
    rows = []
    for target in schema.target_cols:
        if target not in df.columns:
            continue
        y = df[target]
        for config_col in schema.config_cols:
            result = adj_r2_categorical(df[config_col], y, min_n=min_n)
            if result is None:
                continue
            rows.append(
                {
                    "feature": config_col,
                    "target": target,
                    "adj_r2": result.adj_r2,
                    "p_value": result.p_value,
                    "n": result.n_observed,
                    "k": result.k_groups,
                }
            )

    if not rows:
        return ConfigScreeningResult(0, 0, None, None, None, None, None)

    q_values = benjamini_hochberg([r["p_value"] for r in rows])
    n_significant = sum(1 for q in q_values if q < fdr_alpha)

    best = max(rows, key=lambda r: r["adj_r2"])
    median_n = int(np.median([r["n"] for r in rows]))
    median_k = int(np.median([r["k"] for r in rows]))
    median_n_per_group = median_n // median_k if median_k else None

    mde_adj_r2: float | None = None
    if median_k and median_k >= 2 and median_n > median_k:
        from statsmodels.stats.power import FTestAnovaPower  # deferred: see chamber_interaction_p

        try:
            f_req = FTestAnovaPower().solve_power(
                effect_size=None, nobs=median_n, alpha=0.05, power=0.8, k_groups=median_k
            )
            mde_adj_r2 = float(f_req**2 / (1 + f_req**2))
        except Exception:
            mde_adj_r2 = None

    return ConfigScreeningResult(
        n_tested=len(rows),
        n_significant_fdr=n_significant,
        max_observed_adj_r2=best["adj_r2"],
        max_observed_feature=best["feature"],
        max_observed_target=best["target"],
        mde_adj_r2=mde_adj_r2,
        median_n_per_group=median_n_per_group,
    )


@dataclass
class FactorMeasurementBias:
    feature: str
    target: str
    mean_measured: float
    mean_unmeasured: float
    diff: float  # mean_measured - mean_unmeasured
    p_value: float
    q_value: float


def per_factor_measurement_bias(
    df: pd.DataFrame, factors: list, fdr_alpha: float = 0.05
) -> list[FactorMeasurementBias]:
    """Per-primary-factor measurement-selection check -- for each of the
    report's primary factors, does that factor's *own target* defect rate
    differ between wafers where the factor was measured vs not?

    The check must stay per-factor: a whole-wafer "any R/D reading at all
    vs none at all" aggregate averages away a real per-factor selection
    effect (every individual factor biased low, but not all in the same
    wafers). On train.CSV the aggregate finds no significant difference
    while every one of the 5 primary factors does. `factors` is the
    caller's own list of ParetoFactor (feature/target pairs) -- the same
    primary-factor set report.py narrates, not recomputed here.
    """
    rows: list[dict] = []
    for factor in factors:
        feature, target = factor.feature, factor.target
        if feature not in df.columns or target not in df.columns:
            continue
        measured = df[feature].notna()
        y = pd.to_numeric(df[target], errors="coerce")
        group_measured = y[measured & y.notna()]
        group_unmeasured = y[~measured & y.notna()]
        if len(group_measured) < 2 or len(group_unmeasured) < 2:
            continue
        _, p_value = stats.ttest_ind(group_measured, group_unmeasured, equal_var=False)
        rows.append(
            {
                "feature": feature,
                "target": target,
                "mean_measured": float(group_measured.mean()),
                "mean_unmeasured": float(group_unmeasured.mean()),
                "p_value": float(p_value),
            }
        )
    if not rows:
        return []
    q_values = benjamini_hochberg([r["p_value"] for r in rows])
    return [
        FactorMeasurementBias(
            feature=r["feature"],
            target=r["target"],
            mean_measured=r["mean_measured"],
            mean_unmeasured=r["mean_unmeasured"],
            diff=r["mean_measured"] - r["mean_unmeasured"],
            p_value=r["p_value"],
            q_value=q,
        )
        for r, q in zip(rows, q_values)
    ]


def summarize_measurement_bias(rows: list[FactorMeasurementBias]) -> dict | None:
    """Collapses `per_factor_measurement_bias`'s per-factor rows into the
    3 numbers a caller actually narrates: how many factors were
    testable, how many showed a significant
    (q<0.05) measured-vs-unmeasured gap, and whether that gap points the
    same direction everywhere. `None` when there was nothing testable at
    all (not the same as "tested and found no bias" -- callers should
    keep those two cases visually distinct).
    """
    if not rows:
        return None
    significant = [r for r in rows if r.q_value < 0.05]
    direction: str | None = None
    if significant:
        directions = {"low" if r.diff < 0 else "high" for r in significant}
        direction = directions.pop() if len(directions) == 1 else "mixed"
    return {
        "tested_count": len(rows),
        "significant_count": len(significant),
        "direction": direction,
    }


def judge_confidence(adj_r2: float, p_value: float, band_stability_value: float | None, band_width: float | None) -> str:
    """LLM-context confidence gate -- deliberately separate from
    screening.selector.confidence_tier, which drives the 4-tier badge shown
    on-screen. This one folds in band stability (screen badges don't) and
    only has 3 outcomes because the report only ever says "present a band"
    or "don't": 약함/참고 both collapse to 근거부족.
    """
    if p_value > 0.01 or adj_r2 < 0.02:
        return "근거부족"
    if band_width and band_stability_value is not None and band_stability_value > band_width * 0.5:
        return "근거부족"
    if adj_r2 >= 0.10 and band_width and band_stability_value is not None and band_stability_value <= band_width * 0.2:
        return "강함"
    return "보통"
