"""Tests for api.routes.monitoring._is_config_significant -- C-3: the
monitoring treemap must only color a step's tiles when that step's Config
column actually explains final-yield (Y) variance at FDR<0.05 (same rule
as the categorical heatmap). Coloring on raw mean differences alone
highlighted noise-level gaps (<1pp) as if they were real signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.routes.monitoring import _is_config_significant
from src.analysis.screening.schema import Schema


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = ["A", "B", "C"]
    offsets = {"A": 0.0, "B": 5.0, "C": 10.0}
    rows = []
    for _ in range(50):
        for g in groups:
            rows.append(
                {
                    # Strong, real per-group effect on Y.
                    "Step1_Config": g,
                    # Pure noise -- independent of Y.
                    "Step2_Config": rng.choice(groups),
                    "Y": offsets[g] + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_real_effect_is_significant():
    df = _synthetic_df()
    schema = Schema(config_cols=["Step1_Config", "Step2_Config"])
    assert _is_config_significant(df, schema, "Step1_Config") is True


def test_noise_column_is_not_significant():
    df = _synthetic_df()
    schema = Schema(config_cols=["Step1_Config", "Step2_Config"])
    assert _is_config_significant(df, schema, "Step2_Config") is False


def test_column_missing_from_schema_is_not_significant():
    df = _synthetic_df()
    schema = Schema(config_cols=["Step1_Config"])
    assert _is_config_significant(df, schema, "Step2_Config") is False


def test_real_bundled_train_dataset_has_no_significant_config_step():
    """Self-consistency check backing C-3's premise: on the actual bundled
    dataset, every step's Config column fails FDR against final yield --
    so the treemap should render fully neutral-colored today."""
    from pathlib import Path

    import pytest
    from src.analysis.screening.schema import parse_schema

    path = Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV"
    if not path.exists():
        pytest.skip("data/bundled/train.CSV not present")
    df = pd.read_csv(path)
    schema = parse_schema(df)
    results = {col: _is_config_significant(df, schema, col) for col in schema.config_cols}
    assert not any(results.values()), f"expected no FDR-significant config steps, got: {results}"
