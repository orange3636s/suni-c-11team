"""Golden-value regression tests for the Pareto correlation-factor module.

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
from src.analysis.screening.selector import select_pareto_factors_all_targets

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

GOLDEN_TABLE = {
    "Y1": {"feature": "Step28_R1", "eps2": 0.192, "cumulative_pct": 100.0, "shape": "u_shape", "center": 57.4},
    "Y2": {"feature": "Step16_R1", "eps2": 0.159, "cumulative_pct": 92.6, "shape": "u_shape", "center": 58.1},
    "Y3": {"feature": "Step1_D1", "eps2": 0.660, "cumulative_pct": 100.0, "shape": "monotonic_increasing", "center": None},
    "Y4": {"feature": "Step24_R1", "eps2": 0.073, "cumulative_pct": 100.0, "shape": "u_shape", "center": 56.9},
    "Y5": {"feature": "Step18_R1", "eps2": 0.287, "cumulative_pct": 100.0, "shape": "u_shape", "center": 56.1},
}

EPS2_TOLERANCE = 0.01
CENTER_TOLERANCE = 0.5

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def results(train_df: pd.DataFrame):
    schema = parse_schema(train_df)
    return select_pareto_factors_all_targets(train_df, schema)


@pytest.mark.parametrize("target", list(GOLDEN_TABLE))
def test_golden_selected_factor(results, target):
    expected = GOLDEN_TABLE[target]
    result = results[target]
    assert not result.no_significant_factor
    assert len(result.factors) == 1
    factor = result.factors[0]

    assert factor.feature == expected["feature"]
    assert factor.eps2 == pytest.approx(expected["eps2"], abs=EPS2_TOLERANCE)
    assert factor.cumulative_pct == pytest.approx(expected["cumulative_pct"], abs=0.5)
    assert factor.relation_shape == expected["shape"]
    if expected["center"] is None:
        assert factor.optimal_center is None
    else:
        assert factor.optimal_center == pytest.approx(expected["center"], abs=CENTER_TOLERANCE)


def test_no_config_factor_passes_fdr(results):
    for target, result in results.items():
        selected_kinds = {f.kind for f in result.factors}
        assert "Config" not in selected_kinds, f"{target}: Config factor should never pass FDR"


def test_y2_step24_r1_excluded_from_cutoff(results):
    y2 = results["Y2"]
    selected_features = {f.feature for f in y2.factors}
    assert "Step24_R1" not in selected_features

    reference_feature = next(
        (f for f in y2.reference_only if f.feature == "Step24_R1"),
        None,
    )
    assert reference_feature is not None
    assert reference_feature.eps2 == pytest.approx(0.013, abs=EPS2_TOLERANCE)


def test_step1_d1_observed_count(results):
    y3_factor = results["Y3"].factors[0]
    assert y3_factor.feature == "Step1_D1"
    assert y3_factor.n_observed == 479


def test_step18_r1_eps2_beats_pearson_r2(results):
    factor = results["Y5"].factors[0]
    assert factor.feature == "Step18_R1"
    assert factor.pearson_r is not None
    assert factor.eps2 > factor.pearson_r**2
