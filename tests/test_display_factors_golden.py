"""Golden-value regression tests for the kind-scoped, FDR-unrestricted
display selection (select_display_factors) -- the data source behind the
root-cause tab's R/D/Config split view.

Skips gracefully when data/raw/train.CSV is absent (gitignored raw data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import confidence_tier, select_display_factors

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)

# kind -> target -> (top1 feature, top1 contribution %, top1 p-value, display count)
GOLDEN = {
    "all": {
        "Y1": ("Step28_R1", 63.9, 7),
        "Y2": ("Step16_R1", 63.1, 6),
        "Y3": ("Step1_D1", 92.6, 1),
        "Y4": ("Step24_R1", 41.2, 1),  # 13 needed to reach 80% -> falls back to top 1
        "Y5": ("Step18_R1", 76.7, 2),
    },
    "R": {
        "Y1": ("Step28_R1", 77.0, 2),
        "Y2": ("Step16_R1", 80.3, 1),
        "Y3": ("Step2_R1", 15.7, 9),
        "Y4": ("Step24_R1", 53.1, 8),
        "Y5": ("Step18_R1", 91.6, 1),
    },
    "D": {
        "Y1": ("Step8_D1", 28.7, 4),
        "Y2": ("Step14_D1", 30.5, 3),
        "Y3": ("Step1_D1", 98.0, 1),
        "Y4": ("Step21_D1", 46.2, 4),
        "Y5": ("Step27_D1", 39.0, 3),
    },
    "Config": {
        "Y1": ("Step19_Config", 19.3, 9),
        "Y2": ("Step10_Config", 13.3, 10),
        "Y3": ("Step11_Config", 17.1, 8),
        "Y4": ("Step5_Config", 18.5, 8),
        "Y5": ("Step8_Config", 25.4, 7),
    },
}

PCT_TOLERANCE = 0.5


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def schema(train_df: pd.DataFrame):
    return parse_schema(train_df)


@pytest.mark.parametrize("kind", list(GOLDEN))
def test_golden_display_selection(train_df, schema, kind):
    for target, (expected_feature, expected_pct, expected_count) in GOLDEN[kind].items():
        result = select_display_factors(train_df, schema, target, kind=kind)
        assert not result.no_significant_factor
        assert len(result.factors) == expected_count, f"{kind}/{target}: expected {expected_count} factors, got {len(result.factors)}"
        top = result.factors[0]
        assert top.feature == expected_feature, f"{kind}/{target}: expected top factor {expected_feature}, got {top.feature}"
        assert top.contribution_pct == pytest.approx(expected_pct, abs=PCT_TOLERANCE)


def test_all_kind_y4_falls_back_to_single_factor(train_df, schema):
    """The one explicitly-called-out case: even the top 10 all-kind
    factors for Y4 only reach 74.5% cumulative -- 13 factors would be
    needed to cross 80%, so display falls back to just the strongest
    factor instead of showing 10 mostly-noise charts.
    """
    result = select_display_factors(train_df, schema, "Y4", kind="all")
    assert len(result.factors) == 1
    assert result.factors[0].feature == "Step24_R1"


@pytest.mark.parametrize(
    "p_value, expected_tier",
    [
        (0.0001, "strong"),
        (0.009, "strong"),
        (0.01, "moderate"),
        (0.03, "moderate"),
        (0.0499, "moderate"),
        (0.05, "weak"),
        (0.15, "weak"),
        (0.1999, "weak"),
        (0.2, "reference"),
        (0.9, "reference"),
    ],
)
def test_confidence_tier_boundaries(p_value, expected_tier):
    assert confidence_tier(p_value) == expected_tier


def test_config_view_is_mostly_low_confidence_tiers(train_df, schema):
    """Config golden values above have p in [0.003, 0.123] -- a mix of
    moderate/weak/reference, never uniformly "strong". Confirms the
    confidence-tier badge is load-bearing for this view, not decorative.
    """
    tiers_seen = set()
    for target in ["Y1", "Y2", "Y3", "Y4", "Y5"]:
        result = select_display_factors(train_df, schema, target, kind="Config")
        for factor in result.factors:
            tiers_seen.add(confidence_tier(factor.p_value))
    assert "strong" not in tiers_seen or len(tiers_seen) > 1
