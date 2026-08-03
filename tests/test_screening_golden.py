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

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import (
    score_all_factors,
    select_fdr_significant_factors,
    select_primary_factor,
)

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

# Per-target: the top-eps2 factor (regardless of significance), plus its shape/center.
GOLDEN_TABLE = {
    "Y1": {"feature": "Step28_R1", "eps2": 0.192, "shape": "u_shape", "center": 57.4},
    "Y2": {"feature": "Step16_R1", "eps2": 0.159, "shape": "u_shape", "center": 58.1},
    "Y3": {"feature": "Step1_D1", "eps2": 0.660, "shape": "monotonic_increasing", "center": None},
    "Y4": {"feature": "Step24_R1", "eps2": 0.073, "shape": "u_shape", "center": 56.9},
    "Y5": {"feature": "Step18_R1", "eps2": 0.287, "shape": "u_shape", "center": 56.1},
}

# contribution_pct/cumulative_pct denominated by the FULL candidate pool
# (all R+D+Config factors evaluated for the target, 88 for train.CSV) --
# not just the FDR-significant subset. This is the fix for the "single
# significant factor reads as 100% contribution" bug.
CONTRIBUTION_TABLE = {
    "Y1": {"top1_pct": 63.9, "cum10_pct": 85.0, "n80": 7, "fdr_count": 1},
    "Y2": {"top1_pct": 63.1, "cum10_pct": 84.8, "n80": 6, "fdr_count": 2},
    "Y3": {"top1_pct": 92.6, "cum10_pct": 97.5, "n80": 1, "fdr_count": 1},
    "Y4": {"top1_pct": 41.2, "cum10_pct": 74.5, "n80": 13, "fdr_count": 1},
    "Y5": {"top1_pct": 76.7, "cum10_pct": 95.5, "n80": 2, "fdr_count": 1},
}

EPS2_TOLERANCE = 0.01
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
    assert factor.eps2 == pytest.approx(expected["eps2"], abs=EPS2_TOLERANCE)
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
    rows.sort(key=lambda r: r["eps2"], reverse=True)
    total_eps2 = sum(r["eps2"] for r in rows)

    top1_pct = rows[0]["eps2"] / total_eps2 * 100.0
    cum10_pct = sum(r["eps2"] for r in rows[:10]) / total_eps2 * 100.0
    fdr_count = sum(1 for r in rows if r["significant"])

    cumulative = 0.0
    n80 = None
    for index, row in enumerate(rows):
        cumulative += row["eps2"] / total_eps2 * 100.0
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


def test_fdr_significant_factors_include_y2_second_factor(train_df, schema):
    """The alarm engine's factor set (select_fdr_significant_factors) keeps
    every BH-FDR-significant factor, not just the strongest. Step24_R1
    passes FDR (q<0.05) for Y2 but isn't Y2's strongest factor; it must
    still appear here even though the display-only `select_primary_factor`
    (used for Pareto/training cards) never surfaces it for Y2.
    """
    y2_factors = select_fdr_significant_factors(train_df, schema, "Y2")
    assert {f.feature for f in y2_factors} == {"Step16_R1", "Step24_R1"}


def test_no_config_factor_passes_fdr(train_df, schema):
    for target in schema.target_cols:
        factors = select_fdr_significant_factors(train_df, schema, target)
        assert "Config" not in {f.kind for f in factors}, f"{target}: Config factor should never pass FDR"


def test_step1_d1_observed_count(primary_factors):
    y3_factor = primary_factors["Y3"]
    assert y3_factor.feature == "Step1_D1"
    assert y3_factor.n_observed == 479


def test_step18_r1_eps2_beats_pearson_r2(primary_factors):
    factor = primary_factors["Y5"]
    assert factor.feature == "Step18_R1"
    assert factor.pearson_r is not None
    assert factor.eps2 > factor.pearson_r**2
