"""Golden-value regression tests for src/analysis/control_range.py.

Skips gracefully when data/raw/train.CSV or test.CSV are absent. Run
locally with the real files in place to verify against the reference
table (SPC control limits computed from X's own distribution, applied
to test.CSV).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import select_pareto_factors_all_targets

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"
TEST_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "test.CSV"

# mean, std, Q1, Q3, IQR*1.5 bounds -- all computed from X alone (never Y).
GOLDEN_STATS = {
    "Y1": {"feature": "Step28_R1", "mean": 59.9, "std": 4.44, "q1": 57.4, "q3": 62.4, "iqr_lo": 49.9, "iqr_hi": 69.9, "s3_lo": 46.6, "s3_hi": 73.2, "s6_lo": 33.3, "s6_hi": 86.5},
    "Y2": {"feature": "Step16_R1", "mean": 60.3, "std": 3.77, "q1": 58.2, "q3": 62.5, "iqr_lo": 51.8, "iqr_hi": 68.8, "s3_lo": 49.0, "s3_hi": 71.6, "s6_lo": 37.7, "s6_hi": 82.9},
    "Y3": {"feature": "Step1_D1", "mean": 9.7, "std": 3.05, "q1": 8.0, "q3": 12.0, "iqr_lo": 2.0, "iqr_hi": 18.0, "s3_lo": 0.6, "s3_hi": 18.9, "s6_lo": -8.6, "s6_hi": 28.0},
    "Y4": {"feature": "Step24_R1", "mean": 58.8, "std": 4.73, "q1": 56.1, "q3": 61.5, "iqr_lo": 48.1, "iqr_hi": 69.5, "s3_lo": 44.6, "s3_hi": 73.0, "s6_lo": 30.4, "s6_hi": 87.1},
    "Y5": {"feature": "Step18_R1", "mean": 58.0, "std": 4.83, "q1": 55.4, "q3": 60.6, "iqr_lo": 47.6, "iqr_hi": 68.4, "s3_lo": 43.5, "s3_hi": 72.5, "s6_lo": 29.0, "s6_hi": 87.0},
}
# The IQR*1.5 bound actually used for alarms (one-sided for Y3's
# monotonic_increasing Step1_D1 -- only the upper/iqr_hi side).
GOLDEN_ALARM_BOUNDS = {
    "Y1": {"lower": 49.9, "upper": 69.9},
    "Y2": {"lower": 51.8, "upper": 68.8},
    "Y3": {"lower": None, "upper": 18.0},
    "Y4": {"lower": 48.1, "upper": 69.5},
    "Y5": {"lower": 47.6, "upper": 68.4},
}
TOLERANCE = 0.3

pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists()),
    reason="data/raw/train.CSV and test.CSV are not tracked in git; place them locally to run golden checks.",
)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_CSV_PATH)


@pytest.fixture(scope="module")
def control_ranges(train_df):
    schema = parse_schema(train_df)
    results = select_pareto_factors_all_targets(train_df, schema)
    return {
        target: compute_control_range(train_df, results[target].factors[0])
        for target in GOLDEN_STATS
    }


@pytest.fixture(scope="module")
def alarms_by_target(control_ranges, test_df):
    return {target: evaluate_alarms(test_df, cr) for target, cr in control_ranges.items()}


@pytest.mark.parametrize("target", list(GOLDEN_STATS))
def test_golden_spc_stats(control_ranges, target):
    expected = GOLDEN_STATS[target]
    cr = control_ranges[target]
    assert cr.feature == expected["feature"]
    assert cr.mean == pytest.approx(expected["mean"], abs=TOLERANCE)
    assert cr.std == pytest.approx(expected["std"], abs=0.05)
    assert cr.q1 == pytest.approx(expected["q1"], abs=TOLERANCE)
    assert cr.q3 == pytest.approx(expected["q3"], abs=TOLERANCE)

    lines = {line.key: line for line in cr.reference_lines}
    assert lines["iqr_lo"].value == pytest.approx(expected["iqr_lo"], abs=TOLERANCE)
    assert lines["iqr_hi"].value == pytest.approx(expected["iqr_hi"], abs=TOLERANCE)
    assert lines["s3_lo"].value == pytest.approx(expected["s3_lo"], abs=TOLERANCE)
    assert lines["s3_hi"].value == pytest.approx(expected["s3_hi"], abs=TOLERANCE)
    assert lines["s6_lo"].value == pytest.approx(expected["s6_lo"], abs=TOLERANCE)
    assert lines["s6_hi"].value == pytest.approx(expected["s6_hi"], abs=TOLERANCE)


@pytest.mark.parametrize("target", list(GOLDEN_ALARM_BOUNDS))
def test_golden_alarm_bounds_are_iqr15(control_ranges, target):
    """The alarm-relevant lower/upper bound is IQR*1.5, not raw Q1/Q3 and
    not +-3sigma/+-6sigma -- those are reference-only.
    """
    expected = GOLDEN_ALARM_BOUNDS[target]
    cr = control_ranges[target]
    if expected["lower"] is None:
        assert cr.lower is None
    else:
        assert cr.lower == pytest.approx(expected["lower"], abs=TOLERANCE)
    assert cr.upper == pytest.approx(expected["upper"], abs=TOLERANCE)
    assert cr.fallback_applied is False


def test_six_sigma_never_drawable_for_any_golden_factor(control_ranges):
    """Empirically, +-6sigma falls outside the observed [min, max] for
    every one of the 5 golden factors in train.CSV -- axes must never
    stretch to include it.
    """
    for target, cr in control_ranges.items():
        lines = {line.key: line for line in cr.reference_lines}
        assert not lines["s6_lo"].drawable, f"{target}: s6_lo unexpectedly drawable"
        assert not lines["s6_hi"].drawable, f"{target}: s6_hi unexpectedly drawable"


def test_three_sigma_drawable_except_step1_d1_lower_edge_case(control_ranges):
    """+-3sigma is drawable for every golden factor except one documented
    edge case: Step1_D1's lower -3sigma (~0.58) sits just below its
    observed minimum (3.0) -- correctly excluded by the same "must be
    within observed [min, max]" rule applied uniformly to every line,
    not specially carved out for +-6sigma.
    """
    for target, cr in control_ranges.items():
        lines = {line.key: line for line in cr.reference_lines}
        assert lines["s3_hi"].drawable, f"{target}: s3_hi unexpectedly not drawable"
        if target == "Y3":
            assert not lines["s3_lo"].drawable
        else:
            assert lines["s3_lo"].drawable, f"{target}: s3_lo unexpectedly not drawable"


def test_monotonic_factor_only_alarms_on_worse_side(control_ranges):
    """Step1_D1 (defect count, monotonic_increasing) alarms only on the
    upper/iqr_hi side; the lower/iqr_lo side is a reference line only.
    """
    cr = control_ranges["Y3"]
    lines = {line.key: line for line in cr.reference_lines}
    assert lines["iqr_hi"].alarm_relevant is True
    assert lines["iqr_lo"].alarm_relevant is False
    assert cr.lower is None
    assert cr.upper is not None


def test_golden_wafer_status_summary(control_ranges, alarms_by_target, test_df):
    verdicts = summarize_wafer_status(
        test_df, list(control_ranges.values()), {cr.feature: alarms_by_target[t] for t, cr in control_ranges.items()}
    )
    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]

    assert len(alarm_ids) == 19
    assert len(alarm_ids) + len(normal_ids) + len(unmeasured_ids) == len(test_df)

    indexed = test_df.set_index("Lot_Wafer_ID")
    alarm_yield_avg = indexed.loc[alarm_ids, "Y"].mean()
    no_alarm_yield_avg = indexed.loc[normal_ids + unmeasured_ids, "Y"].mean()
    assert alarm_yield_avg == pytest.approx(83.15, abs=0.1)
    assert no_alarm_yield_avg == pytest.approx(89.29, abs=0.1)
    assert (alarm_yield_avg - no_alarm_yield_avg) == pytest.approx(-6.14, abs=0.1)
