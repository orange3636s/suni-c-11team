"""Golden-value regression tests for src/analysis/control_range.py.

Skips gracefully when data/raw/train.CSV or test.CSV are absent. Run
locally with the real files in place to verify against the 3-3 reference
table (train-derived normal ranges applied to test.CSV).
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

GOLDEN_RANGES = {
    "Y1": {"feature": "Step28_R1", "y_q1": 1.80, "y_q3": 3.90, "lower": 42.9, "upper": 68.6, "band_in_ratio": 0.517, "fallback": False},
    "Y2": {"feature": "Step16_R1", "y_q1": 2.30, "y_q3": 5.90, "lower": 51.4, "upper": 67.8, "band_in_ratio": 0.508, "fallback": False},
    "Y3": {"feature": "Step1_D1", "y_q1": 5.70, "y_q3": 8.70, "lower": None, "upper": 14.0, "band_in_ratio": 0.524, "fallback": False},
    "Y4": {"feature": "Step24_R1", "y_q1": 1.90, "y_q3": 3.70, "lower": 44.5, "upper": 67.5, "band_in_ratio": 0.513, "fallback": True},
    "Y5": {"feature": "Step18_R1", "y_q1": 0.40, "y_q3": 1.00, "lower": 46.8, "upper": 66.2, "band_in_ratio": 0.558, "fallback": False},
}
GOLDEN_ALARMS = {
    "Y1": {"count": 2, "observed": 155, "alarm_avg": 6.45, "normal_avg": 2.94},
    "Y2": {"count": 7, "observed": 145, "alarm_avg": 11.61, "normal_avg": 4.29},
    "Y3": {"count": 11, "observed": 71, "alarm_avg": 11.15, "normal_avg": 7.39},
    "Y4": {"count": 4, "observed": 155, "alarm_avg": 4.12, "normal_avg": 2.86},
    "Y5": {"count": 8, "observed": 147, "alarm_avg": 2.02, "normal_avg": 0.77},
}
TOLERANCE = 0.5

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
        for target in GOLDEN_RANGES
    }


@pytest.fixture(scope="module")
def alarms_by_target(control_ranges, test_df):
    return {target: evaluate_alarms(test_df, cr) for target, cr in control_ranges.items()}


@pytest.mark.parametrize("target", list(GOLDEN_RANGES))
def test_golden_control_range(control_ranges, target):
    expected = GOLDEN_RANGES[target]
    cr = control_ranges[target]
    assert cr.feature == expected["feature"]
    assert cr.y_q1 == pytest.approx(expected["y_q1"], abs=0.05)
    assert cr.y_q3 == pytest.approx(expected["y_q3"], abs=0.05)
    if expected["lower"] is None:
        assert cr.lower is None
    else:
        assert cr.lower == pytest.approx(expected["lower"], abs=TOLERANCE)
    assert cr.upper == pytest.approx(expected["upper"], abs=TOLERANCE)
    assert cr.fallback_applied == expected["fallback"]
    assert cr.band_in_ratio == pytest.approx(expected["band_in_ratio"], abs=0.01)


@pytest.mark.parametrize("target", list(GOLDEN_ALARMS))
def test_golden_alarm_counts_and_averages(alarms_by_target, test_df, control_ranges, target):
    expected = GOLDEN_ALARMS[target]
    alarms = alarms_by_target[target]
    cr = control_ranges[target]
    observed = pd.to_numeric(test_df[cr.feature], errors="coerce").notna().sum()

    assert len(alarms) == expected["count"]
    assert observed == expected["observed"]

    alarm_values = [a.actual_y for a in alarms if a.actual_y is not None]
    assert (sum(alarm_values) / len(alarm_values)) == pytest.approx(expected["alarm_avg"], abs=0.02)

    alarmed_ids = {a.lot_wafer_id for a in alarms}
    observed_mask = pd.to_numeric(test_df[cr.feature], errors="coerce").notna()
    normal_mask = observed_mask & ~test_df["Lot_Wafer_ID"].isin(alarmed_ids)
    normal_avg = pd.to_numeric(test_df.loc[normal_mask, target], errors="coerce").mean()
    assert normal_avg == pytest.approx(expected["normal_avg"], abs=0.02)


def test_golden_wafer_status_summary(control_ranges, alarms_by_target, test_df):
    verdicts = summarize_wafer_status(
        test_df, list(control_ranges.values()), {cr.feature: alarms_by_target[t] for t, cr in control_ranges.items()}
    )
    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]

    assert len(alarm_ids) == 30
    assert len(unmeasured_ids) == 489
    assert len(alarm_ids) + len(normal_ids) + len(unmeasured_ids) == len(test_df)

    indexed = test_df.set_index("Lot_Wafer_ID")
    alarm_yield_avg = indexed.loc[alarm_ids, "Y"].mean()
    no_alarm_yield_avg = indexed.loc[normal_ids + unmeasured_ids, "Y"].mean()
    assert alarm_yield_avg == pytest.approx(83.11, abs=0.05)
    assert no_alarm_yield_avg == pytest.approx(89.36, abs=0.05)

    lot_alarm_counts: dict[str, int] = {}
    for v in verdicts:
        if v.status == "alarm":
            lot_alarm_counts[v.lot_id] = lot_alarm_counts.get(v.lot_id, 0) + 1
    assert sum(1 for count in lot_alarm_counts.values() if count >= 2) == 8
    assert max(lot_alarm_counts.values()) == 3
