from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.analytics.lot_analysis import build_lot_cause_analysis
from src.ml.explainability import (
    ExplainResult,
    beneficial_values,
    harmful_values,
)
from src.ml.inference import PredictionResult


def _prediction(
    rows: list[dict[str, Any]],
    *,
    target: str = "Y",
) -> PredictionResult:
    values = [float(row[f"predicted_{target}"]) for row in rows]
    return PredictionResult(
        model_id=f"model-{target}",
        target=target,
        model_name="TestModel",
        identifier_column="Lot_Wafer_ID",
        predictions=rows,
        total_rows=len(rows),
        average_prediction=sum(values) / len(values),
        normal_count=sum(row.get("risk_level") == "normal" for row in rows),
        warning_count=sum(row.get("risk_level") == "warning" for row in rows),
        danger_count=sum(row.get("risk_level") == "danger" for row in rows),
    )


def _explanation(
    local_contributions: list[dict[str, Any]],
    *,
    target: str = "Y",
    total_rows: int | None = None,
) -> ExplainResult:
    resolved_total = total_rows if total_rows is not None else len(local_contributions)
    return ExplainResult(
        model_id=f"model-{target}",
        target=target,
        model_name="TestModel",
        total_rows=resolved_total,
        analyzed_rows=len(local_contributions),
        sampling_used=len(local_contributions) < resolved_total,
        sampling_strategy="test_fixture",
        explanation_method="shap_tree",
        is_fallback=False,
        global_importance=[],
        step_summary=[],
        parameter_type_summary=[],
        equipment_summary=[],
        identifier_column="Lot_Wafer_ID",
        wafer_explanations=[],
        model_quality_warnings=[],
        warnings=[],
        local_contributions=local_contributions,
    )


def _contribution(
    feature: str,
    shap_value: float,
    *,
    target: str = "Y",
    step: str = "Step1",
    parameter_type: str = "R",
) -> dict[str, Any]:
    values = np.asarray([[shap_value]], dtype=float)
    return {
        "feature": feature,
        "value": 1.0,
        "shap_value": shap_value,
        "harmful_contribution": float(harmful_values(values, target)[0, 0]),
        "beneficial_contribution": float(beneficial_values(values, target)[0, 0]),
        "step": step,
        "parameter_type": parameter_type,
    }


def _local(
    identifier: str,
    contributions: list[dict[str, Any]],
    **identifiers: Any,
) -> dict[str, Any]:
    return {
        "identifier": identifier,
        "prediction": 90.0,
        "risk_level": "normal",
        "contributions": contributions,
        **identifiers,
    }


def _lots(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["lot_id"]: row for row in result["lots"]}


def test_lot_id_resolution_prefers_explicit_ids_and_supports_real_fallbacks() -> None:
    rows = [
        {
            "Lot_Wafer_ID": "WRONGW01",
            "Lot_ID": "EXPLICIT",
            "lot_id": "API_IGNORED",
            "predicted_Y": 95.0,
            "risk_level": "normal",
        },
        {
            "Lot_Wafer_ID": "APIW09",
            "lot_id": "API",
            "predicted_Y": 94.0,
            "risk_level": "normal",
        },
        {
            "Lot_Wafer_ID": "L001W10",
            "predicted_Y": 93.0,
            "risk_level": "warning",
        },
        {
            "Lot_Wafer_ID": "LOT01_WF01",
            "predicted_Y": 92.0,
            "risk_level": "danger",
        },
        {
            "Lot_Wafer_ID": "malformed",
            "predicted_Y": 91.0,
            "risk_level": "normal",
        },
    ]

    result = build_lot_cause_analysis(_prediction(rows), _explanation([]))
    lots = _lots(result)

    assert set(lots) == {"EXPLICIT", "API", "L001", "LOT01"}
    assert "WRONG" not in lots
    assert "API_IGNORED" not in lots
    assert result["excluded_row_count"] == 1
    assert result["total_lot_count"] == 4
    assert all(lot["wafer_count"] == 1 for lot in lots.values())
    assert lots["L001"]["wafer_list"][0]["identifier"] == "L001W10"
    assert lots["LOT01"]["wafer_list"][0]["identifier"] == "LOT01_WF01"


def test_lot_rankings_are_scoped_and_report_signed_absolute_and_coverage() -> None:
    rows = [
        {"Lot_Wafer_ID": "A_W01", "Lot_ID": "A", "predicted_Y": 90.0, "risk_level": "danger"},
        {"Lot_Wafer_ID": "A_W02", "Lot_ID": "A", "predicted_Y": 92.0, "risk_level": "warning"},
        {"Lot_Wafer_ID": "A_W03", "Lot_ID": "A", "predicted_Y": 94.0, "risk_level": "normal"},
        {"Lot_Wafer_ID": "B_W01", "Lot_ID": "B", "predicted_Y": 96.0, "risk_level": "normal"},
    ]
    locals_ = [
        _local("A_W01", [_contribution("Step1_R1", -2.0)], lot_id="A"),
        _local("A_W02", [_contribution("Step1_R1", -4.0)], lot_id="A"),
        _local("B_W01", [_contribution("Step1_R1", 5.0)], lot_id="B"),
    ]

    result = build_lot_cause_analysis(
        _prediction(rows),
        _explanation(locals_, total_rows=len(rows)),
    )
    lots = _lots(result)
    lot_a = lots["A"]
    lot_b = lots["B"]
    a_feature = lot_a["feature_importance"]["r"][0]
    b_feature = lot_b["feature_importance"]["r"][0]

    assert lot_a["analyzed_wafer_count"] == 2
    assert lot_a["shap_coverage"] == pytest.approx(2 / 3)
    assert a_feature["mean_signed_shap"] == pytest.approx(-3.0)
    assert a_feature["mean_abs_shap"] == pytest.approx(3.0)
    assert a_feature["adverse_contribution"] == pytest.approx(3.0)
    assert a_feature["improvement_contribution"] == pytest.approx(0.0)
    assert a_feature["sample_count"] == 2
    assert a_feature["coverage"] == pytest.approx(2 / 3)

    assert b_feature["mean_signed_shap"] == pytest.approx(5.0)
    assert b_feature["mean_abs_shap"] == pytest.approx(5.0)
    assert b_feature["adverse_contribution"] == pytest.approx(0.0)
    assert b_feature["improvement_contribution"] == pytest.approx(5.0)
    assert b_feature != a_feature


def test_config_category_is_visible_but_low_sample_is_not_official() -> None:
    rows = [
        {"Lot_Wafer_ID": "CFG_W01", "Lot_ID": "CFG", "predicted_Y": 91.0, "risk_level": "normal"},
    ]
    children = [
        _contribution("Step2_Model_Model3", -1.0, step="Step2", parameter_type="Model"),
        _contribution("Step2_Equipment_EQC", 2.0, step="Step2", parameter_type="Equipment"),
        _contribution("Step2_Chamber_CH2", -3.0, step="Step2", parameter_type="Chamber"),
        _contribution("Step2_EQ_EQC", 4.0, step="Step2", parameter_type="EQ"),
    ]
    explanation = _explanation([_local("CFG_W01", children, lot_id="CFG")])

    result = build_lot_cause_analysis(_prediction(rows), explanation)
    lot = _lots(result)["CFG"]
    config_rows = lot["config_categories"]

    assert len(config_rows) == 1
    assert lot["feature_importance"]["config"] == []
    assert lot["pareto"]["config"] == []
    assert lot["low_sample_config"] == config_rows
    config = config_rows[0]
    assert config["feature"] == "Step2_Config::1.0"
    assert config["group"] == "Config"
    assert config["mean_signed_shap"] == pytest.approx(2.0)
    assert config["mean_abs_shap"] == pytest.approx(10.0)
    assert config["adverse_contribution"] == pytest.approx(4.0)
    assert config["improvement_contribution"] == pytest.approx(6.0)
    assert config["sample_count"] == 1
    assert config["coverage"] == pytest.approx(1.0)
    assert set(config["source_features"]) == {
        "Step2_Model_Model3",
        "Step2_Equipment_EQC",
        "Step2_Chamber_CH2",
        "Step2_EQ_EQC",
    }


def test_zero_adverse_pareto_does_not_invent_contribution_shares() -> None:
    rows = [
        {"Lot_Wafer_ID": "GOOD_W01", "Lot_ID": "GOOD", "predicted_Y": 99.0, "risk_level": "normal"},
    ]
    explanation = _explanation([
        _local(
            "GOOD_W01",
            [
                _contribution("Step1_R1", 2.0),
                _contribution("Step1_D1", 1.0, parameter_type="D"),
            ],
            lot_id="GOOD",
        ),
    ])

    result = build_lot_cause_analysis(_prediction(rows), explanation)
    pareto = _lots(result)["GOOD"]["pareto"]["all"]

    assert pareto
    assert all(row["adverse_contribution"] == 0.0 for row in pareto)
    assert all(row["share"] == 0.0 for row in pareto)
    assert all(row["cumulative_share"] == 0.0 for row in pareto)


@pytest.mark.parametrize(
    ("target", "signed_shap"),
    [("Y", -2.0), ("Y1", 2.0), ("Y6", 2.0)],
)
def test_lot_aggregation_preserves_target_aware_harmful_values(
    target: str,
    signed_shap: float,
) -> None:
    rows = [
        {
            "Lot_Wafer_ID": f"{target}_W01",
            "Lot_ID": target,
            f"predicted_{target}": 10.0,
            "risk_level": "danger",
        },
    ]
    contribution = _contribution(
        "Step1_R1",
        signed_shap,
        target=target,
    )
    explanation = _explanation(
        [_local(f"{target}_W01", [contribution], lot_id=target)],
        target=target,
    )

    result = build_lot_cause_analysis(
        _prediction(rows, target=target),
        explanation,
    )
    feature = _lots(result)[target]["feature_importance"]["r"][0]

    assert result["target"] == target
    assert contribution["harmful_contribution"] == pytest.approx(2.0)
    assert contribution["beneficial_contribution"] == pytest.approx(0.0)
    assert feature["mean_signed_shap"] == pytest.approx(signed_shap)
    assert feature["mean_abs_shap"] == pytest.approx(2.0)
    assert feature["adverse_contribution"] == pytest.approx(2.0)
    assert feature["improvement_contribution"] == pytest.approx(0.0)


def test_collapsed_config_keeps_child_absolute_shap_sum() -> None:
    rows = [
        {
            "Lot_Wafer_ID": "CFG_W01",
            "Lot_ID": "CFG",
            "predicted_Y": 91.0,
            "risk_level": "normal",
        },
    ]
    collapsed = _contribution(
        "Step2_Model",
        2.0,
        step="Step2",
        parameter_type="Model",
    )
    collapsed.update(
        {
            "absolute_shap": 10.0,
            "harmful_contribution": 4.0,
            "beneficial_contribution": 6.0,
        }
    )

    result = build_lot_cause_analysis(
        _prediction(rows),
        _explanation([_local("CFG_W01", [collapsed], lot_id="CFG")]),
    )
    config = _lots(result)["CFG"]["config_categories"][0]

    assert config["mean_signed_shap"] == pytest.approx(2.0)
    assert config["mean_abs_shap"] == pytest.approx(10.0)
    assert config["adverse_contribution"] == pytest.approx(4.0)
    assert config["improvement_contribution"] == pytest.approx(6.0)


def test_top_causes_use_adverse_contribution_not_absolute_importance() -> None:
    rows = [
        {
            "Lot_Wafer_ID": "CAUSE_W01",
            "Lot_ID": "CAUSE",
            "predicted_Y": 90.0,
            "risk_level": "warning",
        },
    ]
    contributions = [
        _contribution("Step1_R1", 10.0),
        _contribution("Step1_D1", -2.0, parameter_type="D"),
        _contribution(
            "Step1_Model_A",
            8.0,
            parameter_type="Model",
        ),
        _contribution(
            "Step1_Equipment_B",
            -1.0,
            parameter_type="Equipment",
        ),
    ]

    result = build_lot_cause_analysis(
        _prediction(rows),
        _explanation([_local("CAUSE_W01", contributions, lot_id="CAUSE")]),
    )
    lot = _lots(result)["CAUSE"]

    assert lot["feature_importance"]["all"][0]["feature"] == "Step1_R1"
    assert lot["top_causes"]["feature"] == "Step1_D1"
    assert lot["top_causes"]["config"] is None
    assert lot["low_sample_config"][0]["sample_count"] == 1
    assert lot["wafer_list"][0]["top_feature"] == "Step1_D1"


def test_hybrid_confidence_and_failure_units_are_aggregated_separately() -> None:
    rows = [
        {
            "Lot_Wafer_ID": "HYB_W01",
            "Lot_ID": "HYB",
            "predicted_Y": 82.0,
            "risk_level": "danger",
            "critical_probability": 0.2,
            "warning_probability": 0.7,
            "failure_rates": {"Y1": 0.2, "Y2": 0.5},
            "fail_bit_counts": {"Y6": 100.0, "Y7": 90.0},
        },
        {
            "Lot_Wafer_ID": "HYB_W02",
            "Lot_ID": "HYB",
            "predicted_Y": 88.0,
            "risk_level": "warning",
            "critical_probability": 0.1,
            "warning_probability": 0.2,
            "failure_rates": {"Y1": 0.4, "Y2": 0.6},
            "fail_bit_counts": {"Y6": 80.0, "Y7": 120.0},
        },
    ]

    result = build_lot_cause_analysis(_prediction(rows), _explanation([]))
    lot = _lots(result)["HYB"]

    assert lot["average_confidence"] == pytest.approx(0.65)
    assert lot["failure_rate_averages"] == pytest.approx(
        {"Y1": 0.3, "Y2": 0.55}
    )
    assert lot["fail_bit_count_averages"] == pytest.approx(
        {"Y6": 90.0, "Y7": 105.0}
    )
    assert lot["top_failure_rate_target"] == "Y2"
    assert lot["top_failure_rate_average"] == pytest.approx(0.55)
    assert lot["top_fail_bit_count_target"] == "Y7"
    assert lot["top_fail_bit_count_average"] == pytest.approx(105.0)
    assert lot["top_failure_target"] == "Y2"
    assert lot["top_causes"]["failure_rate_target"] == "Y2"
    assert lot["top_causes"]["fail_bit_count_target"] == "Y7"
    assert lot["risk_extreme_predicted_value"] == pytest.approx(82.0)
    assert lot["risk_extreme_direction"] == "minimum"


def test_non_y_lot_uses_value_semantics_and_maximum_risk_extreme() -> None:
    rows = [
        {
            "Lot_Wafer_ID": "COUNT_W01",
            "Lot_ID": "COUNT",
            "predicted_Y6": 5.0,
            "risk_level": "normal",
        },
        {
            "Lot_Wafer_ID": "COUNT_W02",
            "Lot_ID": "COUNT",
            "predicted_Y6": 9.0,
            "risk_level": "danger",
        },
    ]

    result = build_lot_cause_analysis(
        _prediction(rows, target="Y6"),
        _explanation([], target="Y6"),
    )
    lot = _lots(result)["COUNT"]

    assert lot["average_predicted_value"] == pytest.approx(7.0)
    assert lot["average_predicted_yield"] is None
    assert lot["minimum_predicted_value"] == pytest.approx(5.0)
    assert lot["maximum_predicted_value"] == pytest.approx(9.0)
    assert lot["risk_extreme_predicted_value"] == pytest.approx(9.0)
    assert lot["risk_extreme_direction"] == "maximum"
    assert lot["top_failure_target"] == "Y6"
