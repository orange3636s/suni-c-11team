"""Variable-schema tests for the Pareto module's parser/selector.

These synthesize small DataFrames with non-standard shapes (fewer steps,
a different R-count per step, a subset of targets, pure noise) to confirm
schema.py and selector.py never assume the 30-step / Y1..Y5 layout of
train.CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import confidence_tier, select_primary_factor


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_schema_with_fifteen_steps_only():
    rng = _rng(1)
    n = 400
    data = {"Lot_Wafer_ID": [f"L{i}" for i in range(n)], "Lot_ID": [f"L{i // 20}" for i in range(n)]}
    for step in range(1, 16):
        data[f"Step{step}_Config"] = rng.choice(["A", "B", "C"], size=n)
        data[f"Step{step}_R1"] = rng.normal(size=n)
    data["Y1"] = rng.normal(size=n)
    df = pd.DataFrame(data)

    schema = parse_schema(df)
    assert schema.max_step == 15
    assert schema.steps_present == list(range(1, 16))
    assert len(schema.r_cols) == 15
    assert len(schema.config_cols) == 15
    assert schema.unmapped == []


def test_schema_with_five_r_channels_on_one_step():
    rng = _rng(2)
    n = 300
    data = {"Lot_Wafer_ID": [f"L{i}" for i in range(n)]}
    for m in range(1, 6):
        data[f"Step3_R{m}"] = rng.normal(size=n)
    data["Step1_R1"] = rng.normal(size=n)
    data["Y1"] = rng.normal(size=n)
    df = pd.DataFrame(data)

    schema = parse_schema(df)
    step3_r_cols = [c for c in schema.r_cols if c.startswith("Step3_R")]
    assert len(step3_r_cols) == 5
    assert set(step3_r_cols) == {f"Step3_R{m}" for m in range(1, 6)}
    assert schema.unmapped == []


def test_schema_with_only_y1_to_y3():
    n = 200
    df = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"L{i}" for i in range(n)],
            "Step1_R1": np.random.default_rng(3).normal(size=n),
            "Y1": np.random.default_rng(4).normal(size=n),
            "Y2": np.random.default_rng(5).normal(size=n),
            "Y3": np.random.default_rng(6).normal(size=n),
        }
    )
    schema = parse_schema(df)
    assert schema.target_cols == ["Y1", "Y2", "Y3"]
    assert "Y4" not in schema.target_cols
    assert "Y5" not in schema.target_cols


def test_random_noise_still_returns_a_primary_factor_with_low_confidence():
    """The display-facing selector no longer hides a factor for failing
    significance -- pure noise (Y independent of every candidate) still
    returns the single strongest-by-eps2 factor, just tagged with a
    weak/reference confidence tier instead of vanishing as "no significant
    factor". This is the "유의한 인자 없음 폐기" policy: only an empty
    candidate pool (every factor below its own min-n gate) returns None.
    """
    rng = _rng(42)
    n = 500
    data = {"Lot_Wafer_ID": [f"L{i}" for i in range(n)]}
    for step in range(1, 6):
        data[f"Step{step}_R1"] = rng.normal(size=n)
        data[f"Step{step}_Config"] = rng.choice(["A", "B", "C", "D"], size=n)
    data["Y1"] = rng.normal(size=n)  # independent of every factor above
    df = pd.DataFrame(data)

    schema = parse_schema(df)
    factor = select_primary_factor(df, schema, "Y1")

    assert factor is not None
    assert confidence_tier(factor.eps2, factor.p_value) in ("weak", "reference")


def test_all_candidates_below_min_n_returns_none():
    """The one remaining "분석 불가" condition: every candidate factor's
    observed sample is too small to score at all (below its own min-n
    gate), not merely insignificant.
    """
    rng = _rng(7)
    n = 15  # below both DEFAULT_MIN_N_NUMERIC (100) and _CATEGORICAL (20)
    data = {
        "Lot_Wafer_ID": [f"L{i}" for i in range(n)],
        "Step1_R1": rng.normal(size=n),
        "Step1_Config": rng.choice(["A", "B"], size=n),
        "Y1": rng.normal(size=n),
    }
    df = pd.DataFrame(data)

    schema = parse_schema(df)
    factor = select_primary_factor(df, schema, "Y1")

    assert factor is None
