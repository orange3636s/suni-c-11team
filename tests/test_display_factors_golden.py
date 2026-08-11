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

## Regenerated for the epsilon-squared -> Adjusted R² swap (`adj_r2_numeric`
## now scores numeric factors off the same degree-1/degree-2 polynomial fit
## the scatter curve uses). Every factor's score shifted, so the ranking
## pool -- not just each #1 factor -- reshuffled; most #2-#5 slots changed
## feature identity, not just their percentage. Each target's #1 factor is
## unchanged, which is the invariant the swap was specified to preserve.

# target -> ordered (feature, contribution_pct) for the fixed top 5, full
# R+D+Config pool, denominated by that same full pool.
GOLDEN_TOP5 = {
    "Y1": [
        ("Step28_R1", 86.2),
        ("Step27_D1", 2.0),
        ("Step10_R1", 1.9),
        ("Step18_R1", 1.4),
        ("Step16_R1", 1.1),
    ],
    "Y2": [
        ("Step16_R1", 87.6),
        ("Step8_R1", 1.3),
        ("Step28_R1", 0.9),
        # Config can now place inside a target's top 5 (it never did under
        # epsilon-squared). The Pareto chart renders it fine; only the
        # yield-prediction fallback ladder filters Config out, since that
        # one needs a numeric value/curve/window per factor.
        ("Step10_Config", 0.7),
        ("Step6_D1", 0.7),
    ],
    "Y3": [
        ("Step1_D1", 95.7),
        ("Step30_D1", 0.9),
        ("Step11_D1", 0.7),
        ("Step13_R1", 0.3),
        ("Step24_R2", 0.2),
    ],
    "Y4": [
        ("Step24_R1", 65.5),
        ("Step14_D1", 8.5),
        ("Step21_D1", 3.4),
        ("Step14_R2", 2.3),
        ("Step13_R2", 1.9),
    ],
    "Y5": [
        ("Step18_R1", 90.9),
        ("Step14_D1", 3.7),
        ("Step14_R1", 0.8),
        ("Step2_R1", 0.6),
        ("Step6_R2", 0.6),
    ],
}

# Whether the fixed top-5's cumulative contribution reaches 80%. Under
# Adjusted R² every target now clears 80% within 5 factors -- the
# parameter-count bias correction shrank the also-ran factors' scores (and
# hence the full-pool denominator) far more than the #1 factor's.
GOLDEN_CUM5 = {"Y1": (92.5, True), "Y2": (91.2, True), "Y3": (97.9, True), "Y4": (81.5, True), "Y5": (96.6, True)}

# Confidence-tier composition across the fixed top 5 (강함/보통/약함/참고).
# Updated for the effect-size-gated grade (spec §5-2): p-value alone no
# longer decides the tier, so most of a target's #2-#5 factors (real, but
# with Adjusted R² well under 2% explained) read "참고" rather than
# "강함"/"보통". Unchanged by the epsilon-squared -> Adjusted R² swap.
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
    counts = Counter(confidence_tier(f.adj_r2, f.p_value) for f in factors)
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
