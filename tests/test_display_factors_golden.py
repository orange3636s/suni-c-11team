"""Golden-value regression tests for the fixed top-5 Pareto display
selection (select_top_factors) -- the single shared data source behind
both the training tab's and the root-cause tab's Pareto chart, now that
the R/D/Config split view has been removed.

Skips gracefully when data/raw/train.CSV is absent (gitignored raw data).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import (
    confidence_tier,
    select_top_factors,
)

TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"

pytestmark = pytest.mark.skipif(
    not TRAIN_CSV_PATH.exists(),
    reason="data/raw/train.CSV is not tracked in git; place the real dataset there to run golden checks.",
)

## TC-5: regenerated after `eps2_numeric` switched from a flat 8-bin qcut to
## a Sturges-rule auto bin count (`suggest_bin_count`). Every factor's eps2
## shifted, so the ranking pool -- not just each #1 factor -- reshuffled;
## most #2-#5 slots changed feature identity, not just their percentage.

# target -> ordered (feature, contribution_pct) for the fixed top 5, full
# R+D+Config pool, denominated by that same full pool.
GOLDEN_TOP5 = {
    "Y1": [
        ("Step28_R1", 62.5),
        ("Step13_D1", 4.9),
        ("Step6_R1", 2.9),
        ("Step20_R1", 2.5),
        ("Step19_R1", 2.4),
    ],
    "Y2": [
        ("Step16_R1", 64.1),
        ("Step1_D1", 4.6),
        ("Step8_D1", 4.4),
        ("Step14_D1", 3.0),
        ("Step24_R1", 2.2),
    ],
    "Y3": [
        ("Step1_D1", 88.5),
        ("Step13_D1", 1.5),
        ("Step11_D1", 1.4),
        ("Step26_R1", 0.9),
        ("Step22_R3", 0.9),
    ],
    "Y4": [
        ("Step24_R1", 42.6),
        ("Step21_D1", 8.1),
        ("Step22_R1", 6.6),
        ("Step14_R2", 4.3),
        ("Step7_R2", 3.4),
    ],
    "Y5": [
        ("Step18_R1", 79.4),
        ("Step27_D1", 4.8),
        ("Step22_D1", 3.2),
        ("Step2_R1", 3.1),
        ("Step14_D1", 3.0),
    ],
}

# Whether the fixed top-5's cumulative contribution reaches 80%.
GOLDEN_CUM5 = {"Y1": (75.1, False), "Y2": (78.3, False), "Y3": (93.2, True), "Y4": (65.0, False), "Y5": (93.5, True)}

# Confidence-tier composition across the fixed top 5 (강함/보통/약함/참고).
# Updated for the eps2-gated grade (spec §5-2): p-value alone no longer
# decides the tier, so most of a target's #2-#5 factors (real, but with
# eps2 well under 2% explained) now read "참고" instead of "강함"/"보통".
GOLDEN_TIER_COUNTS = {
    "Y1": {"strong": 1, "moderate": 0, "weak": 0, "reference": 4},
    "Y2": {"strong": 1, "moderate": 0, "weak": 0, "reference": 4},
    "Y3": {"strong": 1, "moderate": 0, "weak": 0, "reference": 4},
    "Y4": {"strong": 0, "moderate": 1, "weak": 0, "reference": 4},
    "Y5": {"strong": 1, "moderate": 0, "weak": 0, "reference": 4},
}

PCT_TOLERANCE = 0.5


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV_PATH)


@pytest.fixture(scope="module")
def schema(train_df: pd.DataFrame):
    return parse_schema(train_df)


@pytest.mark.parametrize("target", list(GOLDEN_TOP5))
def test_golden_top5_always_exactly_five(train_df, schema, target):
    """The display count is fixed at 5 regardless of target -- no more
    per-target-varying cumulative-80%-cutoff count.
    """
    factors = select_top_factors(train_df, schema, target, limit=5)
    assert len(factors) == 5
    for factor, (expected_feature, expected_pct) in zip(factors, GOLDEN_TOP5[target]):
        assert factor.feature == expected_feature, f"{target}: expected {expected_feature}, got {factor.feature}"
        assert factor.contribution_pct == pytest.approx(expected_pct, abs=PCT_TOLERANCE)


@pytest.mark.parametrize("target", list(GOLDEN_CUM5))
def test_golden_cumulative_five_and_80pct_reach(train_df, schema, target):
    expected_cum5, expected_reached = GOLDEN_CUM5[target]
    factors = select_top_factors(train_df, schema, target, limit=5)
    cum5 = factors[-1].cumulative_pct
    assert cum5 == pytest.approx(expected_cum5, abs=PCT_TOLERANCE)
    assert (cum5 >= 80.0) == expected_reached


@pytest.mark.parametrize("target", list(GOLDEN_TIER_COUNTS))
def test_golden_tier_composition(train_df, schema, target):
    factors = select_top_factors(train_df, schema, target, limit=5)
    counts = Counter(confidence_tier(f.eps2, f.p_value) for f in factors)
    expected = GOLDEN_TIER_COUNTS[target]
    for tier, expected_count in expected.items():
        assert counts.get(tier, 0) == expected_count, f"{target}/{tier}: expected {expected_count}, got {counts.get(tier, 0)}"


def test_display_count_never_varies_with_significance(train_df, schema):
    """Every target gets exactly 5 factors regardless of how many pass
    FDR -- the old cumulative-80%-cutoff (which made counts vary 1~7 per
    target) is gone."""
    counts = {target: len(select_top_factors(train_df, schema, target, limit=5)) for target in schema.target_cols}
    assert set(counts.values()) == {5}


def test_limit_shrinks_for_small_pools():
    """Defensive code path: an uploaded dataset with fewer than 5
    candidate factors gets only as many as exist, never padded."""
    import numpy as np

    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"L{i}" for i in range(n)],
            "Step1_R1": rng.normal(size=n),
            "Step2_R1": rng.normal(size=n),
            "Y1": rng.normal(size=n),
        }
    )
    schema = parse_schema(df)
    factors = select_top_factors(df, schema, "Y1", limit=5)
    assert len(factors) == 2
