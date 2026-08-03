"""Golden-value regression tests for src/ml/pipeline.py.

Skips gracefully when data/raw/train.CSV or test.CSV are absent -- both are
gitignored raw data, not committed like data/bundled/. Run locally with the
real files in place to verify against the reference values in the module
spec (train fit -> test evaluation, tolerance ±0.03).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.screening.schema import parse_schema
from src.ml.pipeline import FAIL_RATE_TARGETS, FINAL_YIELD_COLUMN, train_and_evaluate

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"
TEST_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "test.CSV"

GOLDEN_METRICS = {
    "Y1": {"r2": 0.070, "mae": 1.154},
    "Y2": {"r2": 0.098, "mae": 1.954},
    "Y3": {"r2": 0.652, "mae": 0.955},
    "Y4": {"r2": 0.027, "mae": 1.049},
    "Y5": {"r2": 0.186, "mae": 0.309},
    "Y": {"r2": 0.274, "mae": 2.709},
}
GOLDEN_FACTORS = {
    "Y1": "Step28_R1",
    "Y2": "Step16_R1",
    "Y3": "Step1_D1",
    "Y4": "Step24_R1",
    "Y5": "Step18_R1",
}
TOLERANCE = 0.03

pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists()),
    reason="data/raw/train.CSV and test.CSV are not tracked in git; place them locally to run golden checks.",
)


@pytest.fixture(scope="module")
def evaluation():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    schema = parse_schema(train_df)
    return train_and_evaluate(train_df, test_df, schema)


@pytest.mark.parametrize("target", [*FAIL_RATE_TARGETS, FINAL_YIELD_COLUMN])
def test_golden_metrics(evaluation, target):
    expected = GOLDEN_METRICS[target]
    actual = evaluation.metrics[target]
    assert actual["r2"] == pytest.approx(expected["r2"], abs=TOLERANCE)
    assert actual["mae"] == pytest.approx(expected["mae"], abs=TOLERANCE)


@pytest.mark.parametrize("target", FAIL_RATE_TARGETS)
def test_golden_selected_factor(evaluation, target):
    result = evaluation.target_results[target]
    assert not result.no_factor_available
    assert result.factors[0].feature == GOLDEN_FACTORS[target]


def test_each_target_uses_exactly_one_factor(evaluation):
    """The model always uses the single strongest-by-eps2 factor per
    target now -- Y2's second FDR-significant factor (Step24_R1) still
    feeds the alarm engine (select_fdr_significant_factors) but is no
    longer a training feature; removing it changed Y2's R²/MAE by less
    than the golden tolerance above.
    """
    for target in FAIL_RATE_TARGETS:
        assert len(evaluation.target_results[target].factors) == 1


def test_no_config_factor_used_by_any_target(evaluation):
    for result in evaluation.target_results.values():
        assert all(factor.kind != "Config" for factor in result.factors)
