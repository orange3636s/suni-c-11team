"""Golden-value regression test for src/analysis/report.py.

Skips gracefully when data/raw/train.CSV or test.CSV are absent -- see
test_control_range_golden.py for the same convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.report import build_analysis_report

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"
TEST_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "test.CSV"

pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists()),
    reason="원본 train/test CSV가 없어 골든 검증을 건너뜁니다.",
)

EXPECTED_TOP_FACTOR = {
    "Y1": "Step28_R1",
    "Y2": "Step16_R1",
    "Y3": "Step1_D1",
    "Y4": "Step24_R1",
    "Y5": "Step18_R1",
}


@pytest.fixture(scope="module")
def report() -> dict:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eval_df = pd.read_csv(TEST_CSV_PATH)
    return build_analysis_report(
        train_df,
        eval_df,
        train_dataset_id="train",
        eval_dataset_id="test",
        train_meta={"original_filename": "train.CSV", "row_count": len(train_df), "lot_min": "L001", "lot_max": "L400", "lot_count": 400},
        eval_meta={"original_filename": "test.CSV", "row_count": len(eval_df), "lot_min": "L401", "lot_max": "L440", "lot_count": 40},
        app_version="test",
        generated_at="2026-01-01T00:00:00+09:00",
    )


def test_summary_matches_golden_counts(report):
    summary = report["summary"]
    assert summary["targets_analyzed"] == 5
    assert summary["factors_included"] == 5
    assert summary["excluded_low_significance"] == 83
    assert summary["alarm_wafers"] == 19
    assert summary["normal_wafers"] == 492
    assert summary["undecidable_wafers"] == 489
    assert summary["yield_gap_pp"] == pytest.approx(-6.14, abs=0.01)


def test_one_top_factor_per_target(report):
    for target_entry in report["targets"]:
        target = target_entry["target"]
        features = [f["feature"] for f in target_entry["factors"]]
        assert features == [EXPECTED_TOP_FACTOR[target]]


def test_grade_matches_p_value_tier(report):
    for target_entry in report["targets"]:
        for factor in target_entry["factors"]:
            assert factor["p_value"] < 0.05
            assert factor["grade"] in ("강함", "보통")


def test_alarms_include_every_record_and_sort_by_severity_desc(report):
    alarms = report["alarms"]
    assert len(alarms) >= report["summary"]["alarm_wafers"]
    assert len({a["lot_wafer_id"] for a in alarms}) == report["summary"]["alarm_wafers"]
    rank = {"high": 0, "medium": 1, "low": 2}
    ranks = [rank[a["severity"]] for a in alarms]
    assert ranks == sorted(ranks)


def test_no_field_omitted_when_alarms_present(report):
    assert "alarm_wafers" in report["summary"]
    assert isinstance(report["alarms"], list)


def test_floats_rounded_to_four_decimals(report):
    serialized = json.dumps(report)
    for target_entry in report["targets"]:
        for factor in target_entry["factors"]:
            text = repr(factor["adj_r2"])
            assert len(text.split(".")[-1]) <= 4, f"adj_r2 not rounded: {text}"
    assert serialized  # sanity: whole report is JSON-serializable as-is


def test_limitations_present(report):
    assert len(report["limitations"]) >= 3


def test_relation_interpretation_matches_shape(report):
    for target_entry in report["targets"]:
        for factor in target_entry["factors"]:
            shape = factor["relation"]["shape"]
            interpretation = factor["relation"]["interpretation"]
            if shape == "u_shape":
                assert "양방향" in interpretation
            assert interpretation
