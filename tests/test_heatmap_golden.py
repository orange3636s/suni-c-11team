"""Golden-value regression tests for the factor x target correlation heatmap.

Mirrors test_screening_golden.py: skips gracefully when data/raw/train.CSV
(gitignored raw data) is absent.

TC-4: `build_heatmap` no longer takes a `metric` toggle -- every cell reports
both eps2 (`values`, always the displayed number/intensity) and rho (`rho`,
color direction only). Values here were regenerated after TC-5 switched
`eps2_numeric`'s bin count from a flat 8 to a Sturges-rule auto count
(`suggest_bin_count`) -- `rho` itself is unaffected by binning, so those
golden numbers are unchanged from before TC-4/TC-5.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.screening.heatmap import build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import select_primary_factor

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)

TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"]

# Row order is now eps2-descending (TC-4's server-default sort), not
# rho-descending -- so the top-8 membership/order differs from the old
# spearman-sorted golden table.
EPS2_TOP8 = {
    "Step1_D1": [0.000, 0.014, 0.687, 0.000, 0.000],
    "Step18_R1": [0.001, 0.006, 0.002, 0.000, 0.327],
    "Step28_R1": [0.209, 0.007, 0.000, 0.000, 0.001],
    "Step16_R1": [0.004, 0.196, 0.000, 0.000, 0.000],
    "Step24_R1": [0.000, 0.007, 0.000, 0.081, 0.000],
    "Step27_D1": [0.008, 0.000, 0.003, 0.000, 0.020],
    "Step13_D1": [0.016, 0.000, 0.012, 0.002, 0.000],
    "Step21_D1": [0.000, 0.005, 0.000, 0.016, 0.000],
}

# rho is unaffected by the eps2 bin-count change -- same values the old
# spearman-metric golden table had, just re-keyed by feature name instead of
# row position (row order is eps2-based now).
RHO_BY_FEATURE = {
    "Step1_D1": [-0.02, 0.04, 0.84, -0.01, 0.04],
    "Step28_R1": [0.34, 0.05, 0.03, 0.01, -0.02],
    "Step18_R1": [-0.00, -0.03, -0.02, 0.00, 0.32],
    "Step16_R1": [-0.08, 0.28, -0.03, 0.01, 0.05],
    "Step24_R1": [0.00, 0.02, -0.00, 0.17, 0.05],
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
def heatmap(train_df, schema):
    return build_heatmap(train_df, schema)


def test_excludes_config_entirely(heatmap, schema):
    assert set(heatmap.features).isdisjoint(schema.config_cols)
    assert heatmap.excluded_configs == len(schema.config_cols) == 30


def test_row_and_target_shape(heatmap, schema):
    assert len(heatmap.features) == len(schema.r_cols) + len(schema.d_cols) == 58
    assert heatmap.targets == TARGETS


def test_eps2_top8_matches_golden(heatmap):
    top8 = heatmap.features[:8]
    assert top8 == list(EPS2_TOP8)
    for row_index, feature in enumerate(top8):
        expected = EPS2_TOP8[feature]
        actual = heatmap.values[row_index]
        for col_index in range(5):
            assert actual[col_index] == pytest.approx(expected[col_index], abs=EPS2_TOLERANCE)


def test_rho_matches_golden_by_feature(heatmap):
    for feature, expected in RHO_BY_FEATURE.items():
        row_index = heatmap.features.index(feature)
        actual = heatmap.rho[row_index]
        for col_index in range(5):
            assert actual[col_index] == pytest.approx(expected[col_index], abs=RHO_TOLERANCE)


def test_categorical_rho_is_always_none(train_df, schema):
    from src.analysis.screening.heatmap import build_categorical_heatmap

    categorical = build_categorical_heatmap(train_df, schema)
    assert all(v is None for row in categorical.rho for v in row)


def test_exactly_five_cells_pass_rho_threshold_and_match_selection(heatmap):
    hits = []
    for row_index, feature in enumerate(heatmap.features):
        for col_index in range(len(heatmap.targets)):
            value = heatmap.rho[row_index][col_index]
            if value is not None and abs(value) >= 0.15:
                hits.append((feature, value))
    assert len(hits) == 5
    assert {feature for feature, _ in hits} == SELECTED_FEATURES


def test_max_abs_rho_outside_selection_is_0_112(heatmap):
    best = max(
        abs(value)
        for row_index, feature in enumerate(heatmap.features)
        if feature not in SELECTED_FEATURES
        for value in heatmap.rho[row_index]
        if value is not None
    )
    assert best == pytest.approx(0.112, abs=0.01)


def test_eps2_never_disagrees_with_pareto_selection(train_df, schema, heatmap):
    """The heatmap is a browse tool, not a selection tool -- but it must
    never contradict the actual §3-1 selection output for this dataset.
    """
    selected_by_target = {}
    for target in schema.target_cols:
        factor = select_primary_factor(train_df, schema, target)
        if factor is not None:
            selected_by_target[target] = factor.feature
    for target, expected_feature in selected_by_target.items():
        col_index = heatmap.targets.index(target)
        best_row = max(
            range(len(heatmap.features)),
            key=lambda i: (heatmap.values[i][col_index] or 0.0),
        )
        assert heatmap.features[best_row] == expected_feature


def test_significant_cells_use_shared_fdr_family(heatmap):
    """Border-worthy (FDR-passed) cells must come from the same q-values the
    selector produces -- q must be present whenever significant is True.
    """
    for row_index in range(len(heatmap.features)):
        for col_index in range(len(heatmap.targets)):
            if heatmap.significant[row_index][col_index]:
                assert heatmap.q[row_index][col_index] is not None
                assert heatmap.q[row_index][col_index] < 0.05


def test_low_n_cells_are_masked_null(heatmap):
    for row in heatmap.n:
        for n in row:
            assert n >= 0
    for row_index, n_row in enumerate(heatmap.n):
        for col_index, n in enumerate(n_row):
            if n < 30:
                assert heatmap.values[row_index][col_index] is None
                assert heatmap.rho[row_index][col_index] is None
