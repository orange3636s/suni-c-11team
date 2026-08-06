from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.preprocessing_compare import ADOPTED_MODE, compute_preprocessing_comparison


def _synthetic_df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lots = [f"L{(i // 10) + 1:03d}" for i in range(n)]
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    y = 100 - 0.5 * x1 + 0.3 * x2 + rng.normal(0, 2, n)
    df = pd.DataFrame({"Lot_ID": lots, "Step1_D1": x1, "Step2_R1": x2, "Y": y})
    # sprinkle some missingness
    df.loc[df.sample(frac=0.1, random_state=seed).index, "Step2_R1"] = np.nan
    return df


def test_returns_three_modes_with_b_adopted():
    df = _synthetic_df()
    result = compute_preprocessing_comparison(
        df, ["Step1_D1", "Step2_R1"], dataset_id="synthetic", dataset_label="synthetic.csv"
    )
    assert result is not None
    modes = {r.mode for r in result.results}
    assert modes == {"A", "B", "C"}
    adopted = [r for r in result.results if r.adopted]
    assert len(adopted) == 1
    assert adopted[0].mode == ADOPTED_MODE


def test_winner_note_only_when_winner_is_not_adopted():
    df = _synthetic_df()
    result = compute_preprocessing_comparison(
        df, ["Step1_D1", "Step2_R1"], dataset_id="synthetic", dataset_label="synthetic.csv"
    )
    assert result is not None
    if result.winner == ADOPTED_MODE:
        assert result.winner_note is None
    else:
        assert result.winner_note is not None
        assert ADOPTED_MODE in result.winner_note


def test_holdout_note_mentions_dataset_label_and_method():
    df = _synthetic_df()
    result = compute_preprocessing_comparison(
        df, ["Step1_D1", "Step2_R1"], dataset_id="synthetic", dataset_label="mentorship_final.csv"
    )
    assert result is not None
    assert "mentorship_final.csv" in result.holdout_note
    assert "LOT 70/30" in result.holdout_note


def test_returns_none_when_target_missing():
    df = _synthetic_df().drop(columns=["Y"])
    result = compute_preprocessing_comparison(df, ["Step1_D1"], dataset_id="x", dataset_label="x.csv")
    assert result is None


def test_returns_none_when_no_features():
    df = _synthetic_df()
    result = compute_preprocessing_comparison(df, [], dataset_id="x", dataset_label="x.csv")
    assert result is None


def test_different_datasets_can_have_different_winners():
    """spec §E-2: train.CSV에서는 B가 1위, mentorship_final에서는 C가 1위로
    나온다 -- 데이터셋마다 순위가 다를 수 있다는 전제 자체를 검증한다
    (동일한 합성 데이터로 두 값이 항상 같지는 않다는 것만 확인; 실제
    데이터셋별 순위는 실측 데이터를 쓰는 통합 검증에서 별도로 확인한다).
    """
    df_a = _synthetic_df(seed=1)
    df_b = _synthetic_df(seed=2)
    result_a = compute_preprocessing_comparison(df_a, ["Step1_D1", "Step2_R1"], dataset_id="a", dataset_label="a.csv")
    result_b = compute_preprocessing_comparison(df_b, ["Step1_D1", "Step2_R1"], dataset_id="b", dataset_label="b.csv")
    assert result_a is not None and result_b is not None
    # Both compute independently -- just confirm r2 values aren't hardcoded
    # constants (i.e. they actually vary with the input data).
    assert {r.mode: r.r2 for r in result_a.results} != {r.mode: r.r2 for r in result_b.results}
