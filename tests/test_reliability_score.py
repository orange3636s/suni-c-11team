"""Tests for src/analysis/reliability_score.py (RC-4/RC-4b) -- verifies the
0~100 confidence score formula against the 작업 지시서's reference table
(모든 값은 지시서에 명시된 실측 검증 표를 그대로 사용한다) and the
direction/shade cell-color rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.reliability_score import (
    PrimaryFactor,
    cell_color,
    cell_direction,
    compute_reliability_scores,
    shade_bucket,
)

# 지시서 RC-4 실측 검증 표의 1위 인자 기여율 (train.CSV 기준).
CONTRIBUTIONS = {"Y1": 68.2, "Y2": 64.7, "Y3": 88.2, "Y4": 48.8, "Y5": 82.5}


def _factors(measured_features: set[str] | None = None) -> dict[str, PrimaryFactor]:
    return {
        target: PrimaryFactor(feature=f"factor_{target}", contribution_pct=pct, relation_shape="monotonic_increasing", optimal_center=None)
        for target, pct in CONTRIBUTIONS.items()
    }


def _row(measured_targets: set[str], measured_factors: set[str]) -> dict[str, float]:
    row: dict[str, float] = {}
    for target in CONTRIBUTIONS:
        row[target] = 10.0 if target in measured_targets else np.nan
        row[f"factor_{target}"] = 50.0 if target in measured_factors else np.nan
    return row


def test_all_measured_scores_100():
    factors = _factors()
    df = pd.DataFrame([_row({"Y1", "Y2", "Y3", "Y4", "Y5"}, set())])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 100


def test_two_measured_rest_predicted_with_factors_scores_80():
    factors = _factors()
    df = pd.DataFrame([_row({"Y2", "Y3"}, {"Y1", "Y4", "Y5"})])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 80


def test_one_measured_rest_predicted_with_factors_scores_73():
    factors = _factors()
    df = pd.DataFrame([_row({"Y3"}, {"Y1", "Y2", "Y4", "Y5"})])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 73


def test_all_predicted_all_factors_measured_scores_70():
    factors = _factors()
    df = pd.DataFrame([_row(set(), {"Y1", "Y2", "Y3", "Y4", "Y5"})])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 70


def test_all_predicted_two_factors_measured_scores_31():
    factors = _factors()
    # Y2(64.7%) + Y3(88.2%) = 152.9% -> round(1.529 * 20) = 31.
    df = pd.DataFrame([_row(set(), {"Y2", "Y3"})])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 31


def test_all_predicted_one_factor_measured_scores_18():
    factors = _factors()
    df = pd.DataFrame([_row(set(), {"Y3"})])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 18


def test_all_predicted_no_factors_measured_scores_0():
    factors = _factors()
    df = pd.DataFrame([_row(set(), set())])
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 0


def test_missing_target_column_treated_as_unmeasured():
    factors = _factors()
    df = pd.DataFrame([{f"factor_{t}": 50.0 for t in CONTRIBUTIONS}])  # no Y columns at all
    scores = compute_reliability_scores(df, factors)
    assert scores.iloc[0] == 70  # same as "all predicted, all factors measured"


# -- RC-4b: 셀 방향/농도 --------------------------------------------------


def test_direction_u_shape_right_of_vertex_is_red():
    factor = PrimaryFactor(feature="f", contribution_pct=50.0, relation_shape="u_shape", optimal_center=55.8)
    assert cell_direction(factor, 73.2) == "red"


def test_direction_u_shape_left_of_vertex_is_blue():
    factor = PrimaryFactor(feature="f", contribution_pct=50.0, relation_shape="u_shape", optimal_center=55.8)
    assert cell_direction(factor, 30.0) == "blue"


def test_direction_monotonic_increasing_is_always_red():
    factor = PrimaryFactor(feature="f", contribution_pct=50.0, relation_shape="monotonic_increasing", optimal_center=None)
    assert cell_direction(factor, 1.0) == "red"
    assert cell_direction(factor, 999.0) == "red"


def test_direction_monotonic_decreasing_is_always_blue():
    factor = PrimaryFactor(feature="f", contribution_pct=50.0, relation_shape="monotonic_decreasing", optimal_center=None)
    assert cell_direction(factor, 1.0) == "blue"


def test_direction_unclear_shape_has_no_direction():
    factor = PrimaryFactor(feature="f", contribution_pct=50.0, relation_shape="unclear", optimal_center=None)
    assert cell_direction(factor, 42.0) is None


def test_shade_buckets_match_thresholds():
    assert shade_bucket(88.2) == "dark"
    assert shade_bucket(64.7) == "dark"
    assert shade_bucket(48.8) == "medium"
    assert shade_bucket(20.0) == "medium"
    assert shade_bucket(19.9) == "light"
    assert shade_bucket(5.0) == "light"
    assert shade_bucket(4.0) == "gray"
    assert shade_bucket(4.0) == "gray"  # Y1 두 번째 인자급 (전부 회색 구간 사례)


def test_cell_color_measured_value_has_no_direction_or_shade_bucket():
    factors = {"Y1": PrimaryFactor(feature="f1", contribution_pct=68.2, relation_shape="monotonic_increasing", optimal_center=None)}
    result = cell_color("Y1", 73.2, is_measured=True, primary_factors=factors)
    assert result["direction"] is None
    assert result["shade"] == "measured"


def test_cell_color_predicted_value_gets_direction_and_shade():
    factors = {"Y1": PrimaryFactor(feature="f1", contribution_pct=68.2, relation_shape="monotonic_increasing", optimal_center=None)}
    result = cell_color("Y1", 73.2, is_measured=False, primary_factors=factors)
    assert result["direction"] == "red"
    assert result["shade"] == "dark"
    assert result["contribution_pct"] == 68.2
