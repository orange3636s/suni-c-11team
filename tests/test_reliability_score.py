"""Tests for src/analysis/reliability_score.py (RC-4b) -- verifies the
direction/shade cell-color rules against the 작업 지시서's reference table
(모든 값은 지시서에 명시된 실측 검증 표를 그대로 사용한다).
"""

from __future__ import annotations

from types import SimpleNamespace

from src.analysis.reliability_score import cell_color, cell_direction, shade_bucket

# -- RC-4b: 셀 방향/농도 --------------------------------------------------


def test_direction_u_shape_right_of_vertex_is_red():
    factor = SimpleNamespace(feature="f", contribution_pct=50.0, relation_shape="u_shape", optimal_center=55.8)
    assert cell_direction(factor, 73.2) == "red"


def test_direction_u_shape_left_of_vertex_is_blue():
    factor = SimpleNamespace(feature="f", contribution_pct=50.0, relation_shape="u_shape", optimal_center=55.8)
    assert cell_direction(factor, 30.0) == "blue"


def test_direction_monotonic_increasing_is_always_red():
    factor = SimpleNamespace(feature="f", contribution_pct=50.0, relation_shape="monotonic_increasing", optimal_center=None)
    assert cell_direction(factor, 1.0) == "red"
    assert cell_direction(factor, 999.0) == "red"


def test_direction_monotonic_decreasing_is_always_blue():
    factor = SimpleNamespace(feature="f", contribution_pct=50.0, relation_shape="monotonic_decreasing", optimal_center=None)
    assert cell_direction(factor, 1.0) == "blue"


def test_direction_unclear_shape_has_no_direction():
    factor = SimpleNamespace(feature="f", contribution_pct=50.0, relation_shape="unclear", optimal_center=None)
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
    factors = {"Y1": SimpleNamespace(feature="f1", contribution_pct=68.2, relation_shape="monotonic_increasing", optimal_center=None)}
    result = cell_color("Y1", 73.2, is_measured=True, primary_factors=factors)
    assert result["direction"] is None
    assert result["shade"] == "measured"


def test_cell_color_predicted_value_gets_direction_and_shade():
    factors = {"Y1": SimpleNamespace(feature="f1", contribution_pct=68.2, relation_shape="monotonic_increasing", optimal_center=None)}
    result = cell_color("Y1", 73.2, is_measured=False, primary_factors=factors)
    assert result["direction"] == "red"
    assert result["shade"] == "dark"
    assert result["contribution_pct"] == 68.2
