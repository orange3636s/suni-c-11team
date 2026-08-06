"""Tests for src/analysis/target_fallback.py -- 인자 선정 실패 폴백
(spec: 알람 판정 GBDT 전환 §D)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.target_fallback import select_analysis_targets


def test_per_target_when_majority_valid():
    n = 100
    df = pd.DataFrame({f"Y{i}": np.arange(n, dtype=float) for i in range(1, 6)})
    result = select_analysis_targets(df)
    assert result.tier == "per_target"
    assert result.targets == ["Y1", "Y2", "Y3", "Y4", "Y5"]
    assert result.message is None


def test_falls_back_to_final_yield_when_y1_y5_mostly_missing():
    n = 100
    data = {f"Y{i}": [np.nan] * n for i in range(1, 6)}
    data["Y"] = np.arange(n, dtype=float)
    df = pd.DataFrame(data)
    result = select_analysis_targets(df)
    assert result.tier == "final_yield_only"
    assert result.targets == ["Y"]
    assert result.message is not None
    assert "Y1~Y5" in result.message


def test_unanalyzable_when_everything_missing():
    n = 100
    data = {f"Y{i}": [np.nan] * n for i in range(1, 6)}
    data["Y"] = [np.nan] * n
    df = pd.DataFrame(data)
    result = select_analysis_targets(df)
    assert result.tier == "unanalyzable"
    assert result.targets == []
    assert "분석할 수 없습니다" in result.message


def test_partial_targets_only_includes_majority_valid_ones():
    n = 100
    data = {
        "Y1": np.arange(n, dtype=float),
        "Y2": [np.nan] * n,  # entirely missing
        "Y3": [*np.arange(60, dtype=float), *([np.nan] * 40)],  # 60% valid
    }
    df = pd.DataFrame(data)
    result = select_analysis_targets(df)
    assert result.tier == "per_target"
    assert set(result.targets) == {"Y1", "Y3"}
