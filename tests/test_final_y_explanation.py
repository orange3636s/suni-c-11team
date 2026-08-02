from src.ml.explainability import ExplainResult, compose_final_y_explanation


def _result(target: str, value: float) -> ExplainResult:
    contribution = {"feature": "Step1_R1", "value": 1.0, "shap_value": value, "absolute_shap": abs(value), "harmful_contribution": max(value, 0.0), "beneficial_contribution": max(-value, 0.0), "step": "Step1", "parameter_type": "R", "direction": "defect_up"}
    return ExplainResult(model_id="m", target=target, model_name=target, total_rows=1, analyzed_rows=1, sampling_used=False, sampling_strategy="all", explanation_method="test", is_fallback=False, global_importance=[{"feature": "Step1_R1", "step": "Step1", "parameter_type": "R", "parameter_name": "R1", "mean_abs_shap": abs(value), "mean_harmful_contribution": max(value, 0.0), "direction": "defect_up"}], step_summary=[], parameter_type_summary=[], equipment_summary=[], identifier_column="Lot_Wafer_ID", wafer_explanations=[], model_quality_warnings=[], warnings=[], local_contributions=[{"identifier": "L1_W1", "lot_id": "L1", "contributions": [contribution]}])


def test_final_y_composition_reverses_failure_rate_sign() -> None:
    result = compose_final_y_explanation([_result(f"Y{i}", 1.0) for i in range(1, 6)], top_n=10)
    row = result.global_importance[0]
    assert result.target == "Y"
    assert row["mean_harmful_contribution"] == 5.0
    assert result.local_contributions[0]["contributions"][0]["shap_value"] == -5.0


def test_final_y_requires_all_five_component_explanations() -> None:
    try:
        compose_final_y_explanation([_result("Y1", 1.0)], top_n=10)
    except Exception as error:
        assert "Y1~Y5" in str(error)
    else:
        raise AssertionError("missing component models must reject final-y analysis")
