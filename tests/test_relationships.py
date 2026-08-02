import numpy as np
import pandas as pd
import pytest

from src.analytics.relationships import (
    analyze_relationships,
    calculate_pareto,
    categorical_association,
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
    assert result["pearson_p_value"] < 0.01
    assert result["effect_size"] == pytest.approx(1.0)


def test_eta_squared_handles_categorical_equipment() -> None:
    result = eta_squared(
        pd.Series(["A", "A", "B", "B", None]),
        pd.Series([1, 1, 4, 4, 9]),
    )
    assert result["eta_squared"] == pytest.approx(1.0)
    assert result["category_count"] == 2
    assert result["excluded_count"] == 1
    assert result["p_value"] is not None


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
    assert rankings["all"] == rankings["overall"]
    assert rankings["r"] == rankings["R"]
    assert rankings["d"] == rankings["D"]
    assert rankings["equipment"] == rankings["EQ"]
    assert rankings["eq"] == rankings["equipment"]
    assert rankings["missing"] == rankings["measurement"]
    assert set(("config", "model", "chamber", "observed", "indicator")) <= set(rankings)
    assert rankings["EQ"][0]["ranking_basis"] == "Eta squared vs target"
    assert rankings["overall"][0]["p_value"] is not None
    assert rankings["overall"][0]["fdr_p_value"] is not None
    assert rankings["overall"][0]["effect_size"] is not None
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
    assert path["r_vs_y"]
    assert path["eq_vs_y"]
    assert {
        "median",
        "q1",
        "q3",
        "whisker_min",
        "whisker_max",
        "outliers",
        "outlier_count",
        "count",
    }.issubset(path["eq_vs_y"][0])


def test_statistics_include_numeric_and_categorical_tests() -> None:
    frame = sample_frame()
    frame["Step1_Config"] = np.where(
        np.arange(len(frame)) % 2 == 0,
        "Step1_Model1_EQA_CH1",
        "Step1_Model2_EQB_CH2",
    )
    result = analyze_relationships(frame, top_n=10)
    statistics = result["statistics"]
    assert {"pearson", "spearman", "anova", "welch_anova", "kruskal", "fdr"} <= set(
        statistics["methods"]
    )
    assert {row["relation"] for row in statistics["numeric"]} >= {
        "R vs D", "R vs Y", "D vs Y"
    }
    config = next(
        row for row in statistics["categorical"]
        if row["feature"] == "Step1_Config"
    )
    assert config["valid_count"] == len(frame)
    assert config["coverage"] == pytest.approx(1.0)
    assert config["anova"]["p_value"] is not None
    assert config["welch_anova"]["p_value"] is not None
    assert config["kruskal"]["p_value"] is not None
    assert "fdr_p_value" in config["welch_anova"]
    assert config["category_summary"] == config["boxplot_data"]
    assert sum(item["count"] for item in config["category_summary"]) == len(frame)
    assert statistics["scatter_data"] == [
        point
        for row in statistics["numeric"]
        for point in row["scatter_data"]
    ]
    assert statistics["boxplot_data"] == [
        summary
        for row in statistics["categorical"]
        for summary in row["boxplot_data"]
    ]
    assert statistics["categorical_relationships"] == statistics["categorical"]


def test_target_statistics_include_every_r_and_d_feature_without_filtering() -> None:
    frame = sample_frame(400)
    result = analyze_relationships(frame, top_n=1)
    target_rows = [
        row for row in result["statistics"]["numeric"]
        if row["target"] == "Y" and row["group"] in {"R", "D"}
    ]

    assert {row["feature"] for row in target_rows} == {
        "Step1_R1", "Step2_R1", "Step1_D1", "Step2_D1",
    }
    weak = next(row for row in target_rows if row["feature"] == "Step2_R1")
    assert abs(weak["pearson"]) < 0.2
    for row in target_rows:
        assert row["pearson"] is not None
        assert row["spearman"] is not None
        assert row["pearson_p_value"] is not None
        assert row["spearman_p_value"] is not None
        assert row["pearson_fdr_p_value"] is not None
        assert row["spearman_fdr_p_value"] is not None
        assert row["effect_size"] is not None
        assert row["valid_count"] == len(frame)
        assert row["coverage"] == pytest.approx(1.0)
        assert len(row["scatter_data"]) == 150
        assert row["scatter_sampled"] is True


def test_raw_config_statistics_take_priority_and_include_boxplot_summary() -> None:
    frame = sample_frame(180)
    frame["Step1_Config"] = np.where(
        np.arange(len(frame)) % 2 == 0,
        "Step1_Model1_EQA_CH1",
        "Step1_Model2_EQB_CH2",
    )
    frame["Step2_Config"] = np.where(
        np.arange(len(frame)) % 3 == 0,
        "Step2_Model1_EQA_CH1",
        "Step2_Model2_EQC_CH3",
    )

    result = analyze_relationships(frame, top_n=10)
    categorical = result["statistics"]["categorical"]

    assert {row["feature"] for row in categorical} == {
        "Step1_Config", "Step2_Config",
    }
    assert all(row["source_type"] == "raw_config" for row in categorical)
    assert all(row["group"] == "Config" for row in categorical)
    assert all(row["category_summary"] for row in categorical)
    assert all(
        "fdr_p_value" in row["anova"]
        and "fdr_p_value" in row["welch_anova"]
        and "fdr_p_value" in row["kruskal"]
        for row in categorical
    )
    assert {
        row["feature"] for row in result["rankings"]["correlation"]["config"]
    } == {"Step1_Config", "Step2_Config"}


def test_legacy_eq_is_used_as_config_when_raw_config_is_absent() -> None:
    result = analyze_relationships(sample_frame(), top_n=10)
    categorical = result["statistics"]["categorical"]

    assert [row["feature"] for row in categorical] == ["Step1_EQ"]
    assert categorical[0]["source_type"] == "legacy_eq"
    assert categorical[0]["category_summary"]
    assert result["rankings"]["correlation"]["config"][0]["feature"] == "Step1_EQ"


def test_config_shap_aggregates_children_by_step_and_preserves_sources() -> None:
    shap = [
        {
            "feature": "Step1_Model_Model1",
            "parameter_type": "Model",
            "mean_abs_shap": 1.0,
            "direction": "yield_down",
        },
        {
            "feature": "Step1_Equipment_EQA",
            "parameter_type": "Equipment",
            "mean_abs_shap": 2.0,
            "direction": "yield_down",
        },
        {
            "feature": "Step1_Chamber_CH1",
            "parameter_type": "Chamber",
            "mean_abs_shap": 0.5,
            "direction": "yield_up",
        },
        {
            "feature": "Step2_EQ_EQB",
            "parameter_type": "EQ",
            "mean_abs_shap": 4.0,
            "direction": "yield_down",
        },
    ]

    result = analyze_relationships(
        sample_frame(),
        top_n=10,
        shap_importance=shap,
    )
    config = result["rankings"]["shap"]["config"]
    by_feature = {row["feature"]: row for row in config}

    assert by_feature["Step1_Config"]["score"] == pytest.approx(3.5)
    assert by_feature["Step1_Config"]["source_features"] == [
        "Step1_Equipment_EQA",
        "Step1_Model_Model1",
        "Step1_Chamber_CH1",
    ]
    assert by_feature["Step1_Config"]["source_feature_count"] == 3
    assert by_feature["Step2_Config"]["score"] == pytest.approx(4.0)
    assert result["rankings"]["shap"]["model"]
    assert result["rankings"]["shap"]["equipment"]
    assert result["rankings"]["shap"]["chamber"]


def test_empty_feature_groups_return_safe_empty_arrays() -> None:
    result = analyze_relationships(pd.DataFrame({"Y": [1.0, 2.0, 3.0]}))

    assert result["statistics"]["numeric"] == []
    assert result["statistics"]["categorical"] == []
    assert result["statistics"]["scatter_data"] == []
    assert result["statistics"]["boxplot_data"] == []
    assert result["statistics"]["categorical_relationships"] == []
    assert result["rankings"]["shap"]["all"] == []
    assert result["rankings"]["shap"]["config"] == []
    assert result["rankings"]["correlation"]["all"] == []
    assert result["rankings"]["correlation"]["config"] == []


def test_categorical_statistics_return_empty_result_for_insufficient_sample() -> None:
    result = categorical_association(
        pd.Series(["A", "B", None]),
        pd.Series([1.0, 2.0, 3.0]),
    )
    assert result["valid_count"] == 2
    assert result["anova"]["p_value"] is None
    assert result["welch_anova"]["p_value"] is None
    assert result["kruskal"]["p_value"] is None


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
