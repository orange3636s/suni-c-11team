"""Golden-value regression tests for the factor x target correlation heatmap.

Mirrors test_screening_golden.py: skips gracefully when data/raw/train.CSV
(gitignored raw data) is absent.

`build_heatmap` reports one number per cell: Adjusted R2 (`values`), the
identical number the scatter chart and the Pareto bar show for the same
factor/target pair, plus the polynomial `degree`/`shape`/`optimal_center`
grids that carry direction. The signed-correlation (`rho`) grid was removed
-- most core factors are U-shaped, so a whole-range Spearman sign reads
backwards for roughly half the sample; the two rho-threshold tests that
used to live here were rewritten against the equivalent Adjusted R2
behaviour (see their docstrings).
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

# Row order is Adjusted-R2-descending (the server default sort). Values
# regenerated for the epsilon-squared -> Adjusted R2 swap.
ADJ_R2_TOP8 = {
    "Step1_D1": [0.000, 0.002, 0.708, 0.000, 0.000],
    "Step18_R1": [0.004, 0.000, 0.000, 0.000, 0.371],
    "Step16_R1": [0.003, 0.345, 0.000, 0.000, 0.001],
    "Step28_R1": [0.234, 0.003, 0.000, 0.000, 0.000],
    "Step24_R1": [0.000, 0.000, 0.000, 0.094, 0.000],
    "Step14_D1": [0.000, 0.002, 0.001, 0.012, 0.015],
    "Step30_D1": [0.000, 0.000, 0.007, 0.000, 0.000],
    "Step11_D1": [0.000, 0.002, 0.005, 0.000, 0.000],
}

# The polynomial degree each of those cells was fit at -- degree 2 shows up
# exactly where the factor/target pair is the U-shaped driver the Pareto
# chart also picks (Step18_R1->Y5, Step16_R1->Y2, Step28_R1->Y1,
# Step24_R1->Y4); Step1_D1->Y3 is monotonic, hence degree 1 everywhere.
DEGREE_BY_FEATURE = {
    "Step1_D1": [1, 1, 1, 1, 1],
    "Step18_R1": [2, 1, 1, 1, 2],
    "Step16_R1": [1, 2, 1, 1, 1],
    "Step28_R1": [2, 1, 1, 1, 1],
    "Step24_R1": [1, 1, 1, 2, 1],
}

SELECTED_FEATURES = {"Step28_R1", "Step16_R1", "Step1_D1", "Step24_R1", "Step18_R1"}

# The effect-size floor `confidence_tier` uses for the "moderate" grade
# (grade_thresholds.yaml::min_eps2_moderate -- the YAML key kept its
# historical name; the value fed to it is Adjusted R2).
MODERATE_ADJ_R2_FLOOR = 0.05

ADJ_R2_TOLERANCE = 0.01


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


def test_adj_r2_top8_matches_golden(heatmap):
    top8 = heatmap.features[:8]
    assert top8 == list(ADJ_R2_TOP8)
    for row_index, feature in enumerate(top8):
        expected = ADJ_R2_TOP8[feature]
        actual = heatmap.values[row_index]
        for col_index in range(5):
            assert actual[col_index] == pytest.approx(expected[col_index], abs=ADJ_R2_TOLERANCE)


def test_degree_matches_golden_by_feature(heatmap):
    """Replaces the old per-feature rho table. Direction is no longer a
    scalar sign; the closest per-cell equivalent the heatmap still carries
    is which polynomial degree the fit settled on.
    """
    for feature, expected in DEGREE_BY_FEATURE.items():
        row_index = heatmap.features.index(feature)
        actual = heatmap.degree[row_index]
        for col_index in range(5):
            assert actual[col_index] == expected[col_index]


def test_exactly_five_cells_clear_the_effect_size_floor_and_match_selection(heatmap):
    """Replaces `test_exactly_five_cells_pass_rho_threshold_...`: the old
    test asserted exactly 5 cells cleared |rho| >= 0.15 and that they were
    the 5 selected factors. With rho gone, the equivalent statement is that
    exactly 5 cells clear the effect-size floor the confidence tiering
    itself uses -- and they are still the same 5 factor/target pairs the
    Pareto selection picks.
    """
    hits = [
        (feature, heatmap.values[row_index][col_index])
        for row_index, feature in enumerate(heatmap.features)
        for col_index in range(len(heatmap.targets))
        if (heatmap.values[row_index][col_index] or 0.0) >= MODERATE_ADJ_R2_FLOOR
    ]
    assert len(hits) == 5
    assert {feature for feature, _ in hits} == SELECTED_FEATURES


def test_max_adj_r2_outside_selection_is_far_below_the_floor(heatmap):
    """Replaces `test_max_abs_rho_outside_selection_is_0_112`: same intent
    (nothing outside the selected 5 comes close to the threshold), restated
    on Adjusted R2. The gap is much wider than it was on rho -- Adjusted R2
    removes the parameter-count bias that used to lift the also-rans.
    """
    best = max(
        value
        for row_index, feature in enumerate(heatmap.features)
        if feature not in SELECTED_FEATURES
        for value in heatmap.values[row_index]
        if value is not None
    )
    assert best == pytest.approx(0.015, abs=0.01)
    assert best < MODERATE_ADJ_R2_FLOOR


def test_adj_r2_never_disagrees_with_pareto_selection(train_df, schema, heatmap):
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
                assert heatmap.degree[row_index][col_index] is None
                assert heatmap.shape[row_index][col_index] is None
                assert heatmap.optimal_center[row_index][col_index] is None
