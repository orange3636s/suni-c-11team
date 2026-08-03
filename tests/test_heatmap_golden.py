"""Golden-value regression tests for the factor x target correlation heatmap.

Mirrors test_screening_golden.py: skips gracefully when data/raw/train.CSV
(gitignored raw data) is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.screening.heatmap import build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import select_pareto_factors_all_targets

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)

TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"]

SPEARMAN_TOP8 = {
    "Step1_D1": [-0.02, 0.04, 0.84, -0.01, 0.04],
    "Step28_R1": [0.34, 0.05, 0.03, 0.01, -0.02],
    "Step18_R1": [-0.00, -0.03, -0.02, 0.00, 0.32],
    "Step16_R1": [-0.08, 0.28, -0.03, 0.01, 0.05],
    "Step24_R1": [0.00, 0.02, -0.00, 0.17, 0.05],
    "Step14_D1": [-0.03, -0.05, 0.08, -0.11, 0.09],
    "Step30_D1": [0.02, -0.03, -0.10, 0.02, -0.02],
    "Step11_D1": [-0.03, 0.08, -0.09, -0.03, -0.02],
}

EPS2_TOP5 = {
    "Step1_D1": [0.000, 0.009, 0.660, 0.000, 0.000],
    "Step28_R1": [0.192, 0.003, 0.000, 0.002, 0.001],
    "Step18_R1": [0.001, 0.000, 0.000, 0.001, 0.287],
    "Step16_R1": [0.007, 0.159, 0.000, 0.000, 0.000],
    "Step24_R1": [0.000, 0.013, 0.000, 0.073, 0.000],
}

SELECTED_FEATURES = {"Step28_R1", "Step16_R1", "Step1_D1", "Step24_R1", "Step18_R1"}

RHO_TOLERANCE = 0.01
EPS2_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def schema(train_df: pd.DataFrame):
    return parse_schema(train_df)


@pytest.fixture(scope="module")
def spearman_heatmap(train_df, schema):
    return build_heatmap(train_df, schema, metric="spearman")


@pytest.fixture(scope="module")
def eps2_heatmap(train_df, schema):
    return build_heatmap(train_df, schema, metric="eps2")


def test_excludes_config_entirely(spearman_heatmap, schema):
    assert set(spearman_heatmap.features).isdisjoint(schema.config_cols)
    assert spearman_heatmap.excluded_configs == len(schema.config_cols) == 30


def test_row_and_target_shape(spearman_heatmap, schema):
    assert len(spearman_heatmap.features) == len(schema.r_cols) + len(schema.d_cols) == 58
    assert spearman_heatmap.targets == TARGETS


def test_spearman_top8_matches_golden(spearman_heatmap):
    top8 = spearman_heatmap.features[:8]
    assert top8 == list(SPEARMAN_TOP8)
    for row_index, feature in enumerate(top8):
        expected = SPEARMAN_TOP8[feature]
        actual = spearman_heatmap.values[row_index]
        for col_index in range(5):
            assert actual[col_index] == pytest.approx(expected[col_index], abs=RHO_TOLERANCE)


def test_eps2_top5_matches_golden(eps2_heatmap):
    top5 = eps2_heatmap.features[:5]
    assert top5 == list(EPS2_TOP5)
    for row_index, feature in enumerate(top5):
        expected = EPS2_TOP5[feature]
        actual = eps2_heatmap.values[row_index]
        for col_index in range(5):
            assert actual[col_index] == pytest.approx(expected[col_index], abs=EPS2_TOLERANCE)


def test_exactly_five_cells_pass_rho_threshold_and_match_selection(spearman_heatmap):
    hits = []
    for row_index, feature in enumerate(spearman_heatmap.features):
        for col_index, target in enumerate(spearman_heatmap.targets):
            value = spearman_heatmap.values[row_index][col_index]
            if value is not None and abs(value) >= 0.15:
                hits.append((feature, target, value))
    assert len(hits) == 5
    assert {feature for feature, _, _ in hits} == SELECTED_FEATURES


def test_max_abs_rho_outside_selection_is_0_112(spearman_heatmap):
    best = max(
        abs(value)
        for row_index, feature in enumerate(spearman_heatmap.features)
        if feature not in SELECTED_FEATURES
        for value in spearman_heatmap.values[row_index]
        if value is not None
    )
    assert best == pytest.approx(0.112, abs=0.01)


def test_eps2_mode_never_disagrees_with_pareto_selection(train_df, schema, eps2_heatmap):
    """The heatmap is a browse tool, not a selection tool -- but it must
    never contradict the actual §3-1 selection output for this dataset.
    """
    results = select_pareto_factors_all_targets(train_df, schema)
    selected_by_target = {
        target: result.factors[0].feature
        for target, result in results.items()
        if result.factors
    }
    for target, expected_feature in selected_by_target.items():
        col_index = eps2_heatmap.targets.index(target)
        best_row = max(
            range(len(eps2_heatmap.features)),
            key=lambda i: (eps2_heatmap.values[i][col_index] or 0.0),
        )
        assert eps2_heatmap.features[best_row] == expected_feature


def test_significant_cells_use_shared_fdr_family(eps2_heatmap):
    """Border-worthy (FDR-passed) cells must come from the same q-values the
    selector produces -- q must be present whenever significant is True.
    """
    for row_index in range(len(eps2_heatmap.features)):
        for col_index in range(len(eps2_heatmap.targets)):
            if eps2_heatmap.significant[row_index][col_index]:
                assert eps2_heatmap.q[row_index][col_index] is not None
                assert eps2_heatmap.q[row_index][col_index] < 0.05


def test_low_n_cells_are_masked_null(spearman_heatmap):
    for row in spearman_heatmap.n:
        for n in row:
            assert n >= 0
    for row_index, n_row in enumerate(spearman_heatmap.n):
        for col_index, n in enumerate(n_row):
            if n < 30:
                assert spearman_heatmap.values[row_index][col_index] is None
