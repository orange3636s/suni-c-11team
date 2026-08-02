import asyncio
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import api.routes.data as data_routes
from src.ml.dataset import prepare_dataset
from src.ml.explainability import (
    ShapComputation,
    _collapse_local_contributions,
    _global_summaries,
    _sampling_indices,
    explain_dataframe,
    harmful_values,
    parse_feature_name,
)
from src.ml.inference import LoadedPredictionModel
from src.ml.model_io import save_model_bundle
from src.ml.training import _build_preprocessor
from src.preprocessing import preprocess_dataframe


@pytest.fixture(scope="module")
def explain_data() -> tuple[pd.DataFrame, object]:
    path = Path(__file__).parent / "fixtures" / "training_sample.csv"
    raw = pd.read_csv(path)
    processed, _ = preprocess_dataframe(raw)
    return raw, prepare_dataset(processed)


def _loaded_model(
    dataset,
    estimator,
    *,
    target: str = "Y",
    name: str = "test-model",
) -> LoadedPredictionModel:
    pipeline = Pipeline(
        [
            ("features", _build_preprocessor(dataset)),
            ("model", estimator),
        ]
    )
    pipeline.fit(dataset.features, dataset.target)
    return LoadedPredictionModel(
        model_id=name,
        model=pipeline,
        metadata={
            "target": target,
            "model_name": estimator.__class__.__name__,
            "feature_columns": dataset.feature_columns,
            "metrics": {
                "validation": {"r2": 0.7, "rmse": 0.2, "mae": 0.1},
                "test": {"r2": 0.65, "rmse": 0.3, "mae": 0.2},
            },
        },
    )


@pytest.mark.parametrize(
    ("estimator", "expected_method"),
    [
        (RandomForestRegressor(n_estimators=8, random_state=42), "tree"),
        (Ridge(alpha=1.0), "linear"),
    ],
)
def test_real_shap_explainers_succeed(
    explain_data,
    estimator,
    expected_method,
) -> None:
    raw, dataset = explain_data
    loaded = _loaded_model(dataset, estimator)

    result = explain_dataframe(
        raw,
        loaded,
        max_rows=6,
        top_n=5,
        per_wafer_top_n=3,
    )

    assert result.is_fallback is False
    assert expected_method in result.explanation_method
    assert result.global_importance
    assert len(result.wafer_explanations) == 6
    assert result.identifier_column == "Lot_Wafer_ID"


def test_feature_order_and_aggregations_are_preserved(explain_data) -> None:
    raw, dataset = explain_data
    loaded = _loaded_model(dataset, Ridge())

    result = explain_dataframe(raw, loaded, max_rows=5, top_n=20)

    returned_features = {
        item["feature"] for item in result.global_importance
    }
    expected_features = {
        *dataset.feature_columns,
        *{
            f"{feature}_outlier"
            for feature in dataset.numeric_columns
            if not feature.endswith("_missing")
        },
    }
    assert returned_features.issubset(expected_features)
    assert {item["step"] for item in result.step_summary} >= {"Step1"}
    assert {
        item["parameter_type"]
        for item in result.parameter_type_summary
    } >= {"R", "D", "EQ"}


def test_target_direction_rules() -> None:
    values = np.array([[-2.0, 0.0, 3.0]])

    assert harmful_values(values, "Y").tolist() == [[2.0, 0.0, 0.0]]
    assert harmful_values(values, "Y1").tolist() == [[0.0, 0.0, 3.0]]


@pytest.mark.parametrize(
    ("target", "expected_direction"),
    [("Y", "yield_down"), ("Y1", "defect_down")],
)
def test_global_one_hot_collapse_recomputes_target_direction(
    target: str,
    expected_direction: str,
) -> None:
    computation = ShapComputation(
        values=np.array([[3.0, -5.0]]),
        base_values=np.array([0.0]),
        feature_values=np.array([[1.0, 0.0]]),
        feature_names=["Step1_Model_A", "Step1_Model_B"],
        explanation_method="test",
        is_fallback=False,
    )

    global_importance, _, parameter_summary, _, _ = _global_summaries(
        computation,
        target,
        top_n=10,
    )

    assert len(global_importance) == 1
    assert global_importance[0]["feature"] == "Step1_Model"
    assert global_importance[0]["mean_abs_shap"] == pytest.approx(8.0)
    assert global_importance[0]["direction"] == expected_direction
    config_summary = next(
        item for item in parameter_summary
        if item["parameter_type"] == "Config"
    )
    assert config_summary["mean_abs_shap"] == pytest.approx(8.0)


def test_config_summary_includes_legacy_eq() -> None:
    computation = ShapComputation(
        values=np.array([[2.5]]),
        base_values=np.array([0.0]),
        feature_values=np.array([[1.0]]),
        feature_names=["Step2_EQ_TOOL_A"],
        explanation_method="test",
        is_fallback=False,
    )

    _, _, parameter_summary, _, _ = _global_summaries(
        computation,
        "Y",
        top_n=10,
    )

    eq_summary = next(
        item for item in parameter_summary
        if item["parameter_type"] == "EQ"
    )
    config_summary = next(
        item for item in parameter_summary
        if item["parameter_type"] == "Config"
    )
    assert config_summary["mean_abs_shap"] == pytest.approx(
        eq_summary["mean_abs_shap"]
    )
    assert config_summary["harmful_contribution"] == pytest.approx(
        eq_summary["harmful_contribution"]
    )


@pytest.mark.parametrize(
    ("target", "expected_direction"),
    [("Y", "yield_down"), ("Y1", "defect_down")],
)
def test_local_one_hot_collapse_preserves_absolute_shap_and_direction(
    target: str,
    expected_direction: str,
) -> None:
    rows = [
        {
            "feature": "Step1_Model",
            "value": 1.0,
            "shap_value": 3.0,
            "absolute_shap": 3.0,
            "harmful_contribution": 0.0 if target == "Y" else 3.0,
            "beneficial_contribution": 3.0 if target == "Y" else 0.0,
            "step": "Step1",
            "parameter_type": "Model",
            "direction": "yield_up" if target == "Y" else "defect_up",
        },
        {
            "feature": "Step1_Model",
            "value": 0.0,
            "shap_value": -5.0,
            "absolute_shap": 5.0,
            "harmful_contribution": 5.0 if target == "Y" else 0.0,
            "beneficial_contribution": 0.0 if target == "Y" else 5.0,
            "step": "Step1",
            "parameter_type": "Model",
            "direction": "yield_down" if target == "Y" else "defect_down",
        },
    ]

    collapsed = _collapse_local_contributions(rows, target)

    assert len(collapsed) == 1
    assert collapsed[0]["shap_value"] == pytest.approx(-2.0)
    assert collapsed[0]["absolute_shap"] == pytest.approx(8.0)
    assert collapsed[0]["direction"] == expected_direction
    assert collapsed[0]["value"] is None


def test_feature_name_parser_handles_known_and_unknown() -> None:
    known = parse_feature_name("numeric__Step12_R3")
    observed = parse_feature_name("numeric__Step12_R3_observed")
    indicator = parse_feature_name("numeric__Step12_D1_indicator")
    outlier = parse_feature_name("numeric__Step12_R2_outlier")
    unknown = parse_feature_name("temperature")

    assert known["step"] == "Step12"
    assert known["parameter_type"] == "R"
    assert known["parameter_name"] == "3"
    assert observed["parameter_type"] == "Measurement Pattern"
    assert observed["original_feature_name"] == "Step12_R3_observed"
    assert indicator["parameter_type"] == "Measurement Pattern"
    assert outlier["parameter_type"] == "Measurement Pattern"
    assert unknown["step"] == "unknown"
    assert unknown["parameter_type"] == "unknown"


def test_risk_priority_sampling() -> None:
    indices, strategy = _sampling_indices(
        [
            {"risk_level": "normal"},
            {"risk_level": "danger"},
            {"risk_level": "warning"},
            {"risk_level": "danger"},
        ],
        3,
    )

    assert indices == [1, 3, 2]
    assert strategy == "danger_warning_priority"


def test_max_rows_and_local_top_n_are_enforced(explain_data) -> None:
    raw, dataset = explain_data
    loaded = _loaded_model(dataset, Ridge())

    result = explain_dataframe(
        raw,
        loaded,
        max_rows=3,
        per_wafer_top_n=2,
    )

    assert result.analyzed_rows == 3
    assert result.sampling_used is True
    assert all(
        len(item["top_negative_contributors"]) <= 2
        and len(item["top_positive_contributors"]) <= 2
        for item in result.wafer_explanations
    )


def test_low_model_quality_is_not_hidden(explain_data) -> None:
    raw, dataset = explain_data
    loaded = _loaded_model(dataset, Ridge())
    loaded.metadata["metrics"]["test"]["r2"] = -0.2

    result = explain_dataframe(raw, loaded, max_rows=2)

    assert any(
        "Test R²가 낮아" in warning
        for warning in result.model_quality_warnings
    )


def test_explain_response_and_download_are_serializable(
    explain_data,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, dataset = explain_data
    loaded = _loaded_model(dataset, Ridge(), name="api-ridge")
    temporary_root = Path(__file__).parent / ".tmp_explain_models"
    temporary_root.mkdir(exist_ok=True)
    model_dir = temporary_root / f"run_{uuid4().hex}"
    model_dir.mkdir()
    model_path, _, _ = save_model_bundle(
        loaded.model,
        target="Y",
        model_name="Ridge",
        feature_columns=dataset.feature_columns,
        metrics=loaded.metadata["metrics"],
        random_state=42,
        split_method="test",
        model_dir=model_dir,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)

    def upload() -> UploadFile:
        return UploadFile(
            file=BytesIO(raw.to_csv(index=False).encode("utf-8")),
            filename="analysis.csv",
        )

    try:
        response = asyncio.run(
            data_routes.explain_csv(
                upload(),
                model_id=model_path.stem,
                max_rows=5,
                top_n=5,
                per_wafer_top_n=3,
            )
        )
        download = asyncio.run(
            data_routes.download_explanation(
                upload(),
                model_id=model_path.stem,
                max_rows=5,
                top_n=5,
                per_wafer_top_n=3,
            )
        )
        serialized = json.loads(response.model_dump_json())
        assert serialized["global_importance"]
        assert serialized["wafer_explanations"]
        assert download.status_code == 200
        assert "mean_abs_shap" in download.body.decode("utf-8-sig")
    finally:
        for generated_file in model_dir.iterdir():
            generated_file.unlink()
        model_dir.rmdir()
        if not any(temporary_root.iterdir()):
            temporary_root.rmdir()
