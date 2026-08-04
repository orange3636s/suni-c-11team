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
from src.analysis.screening.effect_size import eps2_categorical
from src.analysis.screening.schema import Schema
from src.analysis.screening.selector import benjamini_hochberg

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
    return df[config_col].astype(str).str.split("_").str[-1]


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
    max_observed_eps2: float | None
    max_observed_feature: str | None
    max_observed_target: str | None
    mde_eps2: float | None
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
            result = eps2_categorical(df[config_col], y, min_n=min_n)
            if result is None:
                continue
            rows.append(
                {
                    "feature": config_col,
                    "target": target,
                    "eps2": result.eps2,
                    "p_value": result.p_value,
                    "n": result.n_observed,
                    "k": result.k_groups,
                }
            )

    if not rows:
        return ConfigScreeningResult(0, 0, None, None, None, None, None)

    q_values = benjamini_hochberg([r["p_value"] for r in rows])
    n_significant = sum(1 for q in q_values if q < fdr_alpha)

    best = max(rows, key=lambda r: r["eps2"])
    median_n = int(np.median([r["n"] for r in rows]))
    median_k = int(np.median([r["k"] for r in rows]))
    median_n_per_group = median_n // median_k if median_k else None

    mde_eps2: float | None = None
    if median_k and median_k >= 2 and median_n > median_k:
        from statsmodels.stats.power import FTestAnovaPower  # deferred: see chamber_interaction_p

        try:
            f_req = FTestAnovaPower().solve_power(
                effect_size=None, nobs=median_n, alpha=0.05, power=0.8, k_groups=median_k
            )
            mde_eps2 = float(f_req**2 / (1 + f_req**2))
        except Exception:
            mde_eps2 = None

    return ConfigScreeningResult(
        n_tested=len(rows),
        n_significant_fdr=n_significant,
        max_observed_eps2=best["eps2"],
        max_observed_feature=best["feature"],
        max_observed_target=best["target"],
        mde_eps2=mde_eps2,
        median_n_per_group=median_n_per_group,
    )


def measurement_bias_p(df: pd.DataFrame, schema: Schema, target_col: str = "Y") -> float | None:
    """t-test (spec §1-8): does final yield differ between wafers with at
    least one R or D sensor reading present vs. wafers with none at all?
    Scoped to the full R+D pool (not just the report's 5 narrative factors)
    -- each individual R/D column is itself only ~5-15% populated, so
    almost every wafer has *some* reading; this asks whether the tiny
    "nothing at all measured" group's yield looks different. None when the
    final-yield column is absent or either group is too small to test.
    """
    if target_col not in df.columns:
        return None
    factor_cols = [c for c in (*schema.r_cols, *schema.d_cols) if c in df.columns]
    if not factor_cols:
        return None
    measured = df[factor_cols].notna().any(axis=1)
    y = pd.to_numeric(df[target_col], errors="coerce")
    group_measured = y[measured & y.notna()]
    group_unmeasured = y[~measured & y.notna()]
    if len(group_measured) < 2 or len(group_unmeasured) < 2:
        return None
    _, p_value = stats.ttest_ind(group_measured, group_unmeasured)
    return float(p_value)


def judge_confidence(eps2: float, p_value: float, band_stability_value: float | None, band_width: float | None) -> str:
    """LLM-context confidence gate (spec §2-2) -- deliberately separate from
    screening.selector.confidence_tier, which drives the 4-tier badge shown
    on-screen. This one folds in band stability (screen badges don't) and
    only has 3 outcomes because the report only ever says "present a band"
    or "don't": 약함/참고 both collapse to 근거부족.
    """
    if p_value > 0.01 or eps2 < 0.02:
        return "근거부족"
    if band_width and band_stability_value is not None and band_stability_value > band_width * 0.5:
        return "근거부족"
    if eps2 >= 0.10 and band_width and band_stability_value is not None and band_stability_value <= band_width * 0.2:
        return "강함"
    return "보통"
