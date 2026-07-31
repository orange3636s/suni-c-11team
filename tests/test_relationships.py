import numpy as np
import pandas as pd
import pytest

from src.analytics.relationships import (
    analyze_relationships,
    calculate_pareto,
    eta_squared,
    pair_association,
)


def sample_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    response = np.linspace(0, 10, rows)
    defect = response * 1.8 + rng.normal(0, 0.2, rows)
    equipment = np.where(np.arange(rows) % 2 == 0, "EQ_A", "EQ_B")
    equipment_shift = np.where(equipment == "EQ_A", 2.0, -2.0)
    defect = defect + equipment_shift
    target = 98 - defect * 0.7 + rng.normal(0, 0.2, rows)
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"L{i:03}" for i in range(rows)],
            "Step1_R1": response,
            "Step1_D1": defect,
            "Step1_EQ": equipment,
            "Step2_R1": rng.normal(size=rows),
            "Step2_D1": rng.normal(size=rows),
            "Y": target,
        }
    )


def test_pair_association_handles_missing_and_direction() -> None:
    result = pair_association(
        pd.Series([1, 2, 3, np.nan]),
        pd.Series([2, 4, 6, 8]),
    )
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["valid_count"] == 3
    assert result["excluded_count"] == 1
    assert result["direction"] == "positive"


def test_eta_squared_handles_categorical_equipment() -> None:
    result = eta_squared(
        pd.Series(["A", "A", "B", "B", None]),
        pd.Series([1, 1, 4, 4, 9]),
    )
    assert result["eta_squared"] == pytest.approx(1.0)
    assert result["category_count"] == 2
    assert result["excluded_count"] == 1


def test_pareto_finds_minimum_count_and_zero_total() -> None:
    rows = [
        {"feature": "a", "score": 6.0, "group": "R"},
        {"feature": "b", "score": 3.0, "group": "D"},
        {"feature": "c", "score": 1.0, "group": "EQ"},
    ]
    result = calculate_pareto(rows, score_field="score")
    assert result["required_feature_count"] == 2
    assert result["cumulative_contribution"] == pytest.approx(0.9)
    zero = calculate_pareto(
        [{"feature": "a", "score": 0.0, "group": "R"}],
        score_field="score",
    )
    assert zero["required_feature_count"] == 0
    assert zero["cumulative_contribution"] == 0


def test_analysis_returns_group_rankings_and_top_n() -> None:
    result = analyze_relationships(sample_frame(), top_n=1)
    rankings = result["rankings"]["correlation"]
    assert len(rankings["overall"]) == 1
    assert len(rankings["R"]) == 1
    assert len(rankings["D"]) == 1
    assert len(rankings["EQ"]) == 1
    assert rankings["EQ"][0]["ranking_basis"] == "Eta squared vs target"
    assert rankings["EQ"][0]["signed_association"] is None


def test_analysis_builds_relationship_path_and_confidence() -> None:
    frame = sample_frame()
    shap = [
        {
            "feature": "Step1_D1",
            "parameter_type": "D",
            "mean_abs_shap": 2.0,
        }
    ]
    result = analyze_relationships(
        frame,
        top_n=10,
        shap_importance=shap,
    )
    path = next(row for row in result["relationship_paths"] if row["step"] == 1)
    assert path["r_d"]["pearson"] > 0.8
    assert path["d_y"]["pearson"] < -0.8
    assert path["eq_d"]["eta_squared"] is not None
    assert 0 <= path["path_score"] <= 1
    assert path["confidence"] == "sufficient"
    assert path["eq_vs_d"]


def test_small_sample_is_insufficient_and_missing_values_are_counted() -> None:
    frame = sample_frame(8)
    frame.loc[:2, "Step1_R1"] = np.nan
    result = analyze_relationships(frame, top_n=5)
    path = next(row for row in result["relationship_paths"] if row["step"] == 1)
    assert path["confidence"] == "insufficient"
    assert path["missing_rate"] > 0


@pytest.mark.parametrize(
    ("frame", "method", "message"),
    [
        (pd.DataFrame(), "pearson", "목표 컬럼"),
        (pd.DataFrame({"Y": []}), "pearson", "빈 데이터"),
        (pd.DataFrame({"Y": [1]}), "kendall", "correlation_method"),
    ],
)
def test_invalid_analysis_inputs(
    frame: pd.DataFrame,
    method: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_relationships(frame, correlation_method=method)
