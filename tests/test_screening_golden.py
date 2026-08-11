"""Golden-value regression tests for the factor scoring/selection module.

These tests only run against the real train.CSV, which is deliberately
gitignored under data/raw/ (raw process data is not committed). They skip
gracefully when the file is absent so CI and other clones aren't broken;
run them locally with the file in place to verify against the reference
values from the module spec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import (
    score_all_factors,
    select_fdr_significant_factors,
    select_primary_factor,
)

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

# Per-target: the top-adj_r2 factor (regardless of significance), plus its shape/center.
# `center` values were re-derived when optimal_center switched from an
# unrelated linear grid-search (src/analysis/screening/shape.py's old
# `_best_center` value) to the x_mean of the quantile-bin with the lowest
# y_mean (see screening/quantile_profile.py) -- the same bin the
# recommended window and 구간 평균 불량률 curve already used, so a factor's
# "최적 중심" can no longer land outside its own "권장구간" the way it used
# to for some factor/target pairs before this fix.
## Regenerated for the epsilon-squared -> Adjusted R² swap: numeric factors
## now take their R² from the same degree-1/degree-2 polynomial fit the
## scatter chart's curve overlay uses (`curve_fit.fit_defect_rate_curve`)
## and convert it to Adjusted R², instead of a quantile-binned ANOVA's
## bias-corrected epsilon-squared. `degree` is new (1 or 2 for numeric,
## None for Config). shape/center are unaffected -- they still come from
## quantile_profile.py's own bin count, pinned at 12 for anything
## alarm/recommendation-adjacent.
##
## Invariant deliberately preserved by the swap (and asserted below): the
## #1-ranked feature per target is byte-identical to the epsilon-squared
## era, and so is the count of factors clearing
## CORE_FACTOR_CONTRIBUTION_MIN.
GOLDEN_TABLE = {
    "Y1": {"feature": "Step28_R1", "adj_r2": 0.234, "degree": 2, "n": 1492, "shape": "u_shape", "center": 57.9},
    "Y2": {"feature": "Step16_R1", "adj_r2": 0.345, "degree": 2, "n": 1470, "shape": "u_shape", "center": 56.5},
    "Y3": {"feature": "Step1_D1", "adj_r2": 0.708, "degree": 1, "n": 479, "shape": "monotonic_increasing", "center": None},
    "Y4": {"feature": "Step24_R1", "adj_r2": 0.094, "degree": 2, "n": 1512, "shape": "u_shape", "center": 56.7},
    "Y5": {"feature": "Step18_R1", "adj_r2": 0.371, "degree": 2, "n": 1479, "shape": "u_shape", "center": 55.9},
}

# contribution_pct/cumulative_pct denominated by the FULL candidate pool
# (all R+D+Config factors evaluated for the target, 88 for train.CSV) --
# not just the FDR-significant subset. This is the fix for the "single
# significant factor reads as 100% contribution" bug.
#
# The percentages rose across the board versus the epsilon-squared era
# (e.g. Y1 62.5% -> 86.2%): Adjusted R² subtracts the parameter-count bias
# that used to inflate every also-ran factor's score, so the denominator
# (the full pool's summed effect size) shrank much more than the #1
# factor's own numerator did. Rankings are unchanged -- only the spread
# between #1 and the rest widened.
CONTRIBUTION_TABLE = {
    "Y1": {"top1_pct": 86.2, "cum10_pct": 96.1, "n80": 1, "fdr_count": 1},
    "Y2": {"top1_pct": 87.6, "cum10_pct": 94.0, "n80": 1, "fdr_count": 1},
    "Y3": {"top1_pct": 95.7, "cum10_pct": 98.9, "n80": 1, "fdr_count": 1},
    "Y4": {"top1_pct": 65.5, "cum10_pct": 88.4, "n80": 5, "fdr_count": 1},
    "Y5": {"top1_pct": 90.9, "cum10_pct": 97.9, "n80": 1, "fdr_count": 1},
}

ADJ_R2_TOLERANCE = 0.01
CENTER_TOLERANCE = 0.5
PCT_TOLERANCE = 0.5

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def schema(train_df: pd.DataFrame):
    return parse_schema(train_df)


@pytest.fixture(scope="module")
def primary_factors(train_df: pd.DataFrame, schema):
    return {target: select_primary_factor(train_df, schema, target) for target in schema.target_cols}


@pytest.mark.parametrize("target", list(GOLDEN_TABLE))
def test_golden_top_factor(primary_factors, target):
    expected = GOLDEN_TABLE[target]
    factor = primary_factors[target]
    assert factor is not None

    assert factor.feature == expected["feature"]
    assert factor.adj_r2 == pytest.approx(expected["adj_r2"], abs=ADJ_R2_TOLERANCE)
    assert factor.degree == expected["degree"]
    assert factor.n_observed == expected["n"]
    assert factor.relation_shape == expected["shape"]
    if expected["center"] is None:
        assert factor.optimal_center is None
    else:
        assert factor.optimal_center == pytest.approx(expected["center"], abs=CENTER_TOLERANCE)


@pytest.mark.parametrize("target", list(CONTRIBUTION_TABLE))
def test_golden_contribution_denominator_is_full_pool(train_df, schema, target):
    """Contribution_pct/cumulative_pct/80%-reach rank/FDR-pass count, all
    computed against the full ~88-factor pool.
    """
    expected = CONTRIBUTION_TABLE[target]
    rows = score_all_factors(train_df, schema, target)
    rows.sort(key=lambda r: r["adj_r2"], reverse=True)
    total_adj_r2 = sum(r["adj_r2"] for r in rows)

    top1_pct = rows[0]["adj_r2"] / total_adj_r2 * 100.0
    cum10_pct = sum(r["adj_r2"] for r in rows[:10]) / total_adj_r2 * 100.0
    fdr_count = sum(1 for r in rows if r["significant"])

    cumulative = 0.0
    n80 = None
    for index, row in enumerate(rows):
        cumulative += row["adj_r2"] / total_adj_r2 * 100.0
        if n80 is None and cumulative >= 80.0:
            n80 = index + 1

    assert top1_pct == pytest.approx(expected["top1_pct"], abs=PCT_TOLERANCE)
    assert cum10_pct == pytest.approx(expected["cum10_pct"], abs=PCT_TOLERANCE)
    assert n80 == expected["n80"]
    assert fdr_count == expected["fdr_count"]


def test_top_factor_contribution_never_reads_100_percent(primary_factors):
    """The bug: denominating by the significant-only sum made a lone
    significant factor read as 100% of "everything." A single factor
    covering 100% of 88 candidates' explanatory power is never plausible.
    """
    for target, factor in primary_factors.items():
        assert factor.contribution_pct < 100.0, (
            f"{target}/{factor.feature}: contribution_pct={factor.contribution_pct} "
            "looks like the old significant-only-denominator bug"
        )


def test_fdr_significant_factors_for_y2(train_df, schema):
    """The alarm engine's factor set (select_fdr_significant_factors) keeps
    every BH-FDR-significant factor, not just the strongest.

    Only Step16_R1 clears q<0.05 for Y2. That was already true under the
    binned-ANOVA epsilon-squared scoring and stays true under the
    polynomial-fit Adjusted R² scoring: the p-value source changed (an
    F-test on the degree-1/degree-2 regression rather than on quantile
    bins) but Step24_R1 still doesn't clear the FDR gate for this target.
    """
    y2_factors = select_fdr_significant_factors(train_df, schema, "Y2")
    assert {f.feature for f in y2_factors} == {"Step16_R1"}


def test_no_config_factor_passes_fdr(train_df, schema):
    for target in schema.target_cols:
        factors = select_fdr_significant_factors(train_df, schema, target)
        assert "Config" not in {f.kind for f in factors}, f"{target}: Config factor should never pass FDR"


def test_step1_d1_observed_count(primary_factors):
    y3_factor = primary_factors["Y3"]
    assert y3_factor.feature == "Step1_D1"
    assert y3_factor.n_observed == 479


def test_step18_r1_adj_r2_beats_plain_linear_r2(primary_factors, train_df):
    """The point the old `eps2 > pearson_r**2` assertion made, restated for
    the polynomial fit that replaced the binned ANOVA: Y5's driver is
    U-shaped, so a whole-range straight line badly under-reports it. The
    reported Adjusted R² comes from a degree-2 fit and must beat the
    plain linear r² a Pearson correlation would give.
    """
    factor = primary_factors["Y5"]
    assert factor.feature == "Step18_R1"
    assert factor.degree == 2

    valid = train_df[["Step18_R1", "Y5"]].dropna()
    linear_r, _ = stats.pearsonr(valid["Step18_R1"], valid["Y5"])
    assert factor.adj_r2 > linear_r**2
