import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from uuid import uuid4

from src.config_parser import parse_config_columns, parse_config_value
from src.data_validation import validate_dataframe
from src.ml.dataset import prepare_dataset, split_dataset
from src.ml.inference import DEFAULT_DANGER_THRESHOLD, DEFAULT_WARNING_THRESHOLD
from src.ml.inference import load_prediction_model, predict_dataframe
from src.ml.explainability import explain_dataframe
from src.ml.model_io import save_model_bundle
from src.ml.training import train_regression_models
from src.reporting.report_builder import build_report
from src.schema_compatibility import model_schema_status, schema_fingerprint


def v2_frame(lots: int = 10, wafers: int = 4) -> pd.DataFrame:
    rows = []
    for lot in range(lots):
        for wafer in range(wafers):
            loss = float((lot + wafer) % 5)
            rows.append(
                {
                    "Lot_Wafer_ID": f"L{lot:03d}_W{wafer + 1:02d}",
                    "Lot_ID": f"L{lot:03d}",
                    "Wafer_Slot": wafer + 1,
                    "Step1_Config": f"Step1_Model{lot % 3 + 1}_EQ{chr(65 + lot % 3)}_CH{wafer % 4 + 1}",
                    "Step1_R1": np.nan if wafer else float(lot),
                    "Step1_D1": 0.0 if wafer == 0 else np.nan,
                    "Y": 100.0 - loss,
                    "Y1": loss,
                    "Y2": 0.0,
                    "Y3": 0.0,
                    "Y4": 0.0,
                    "Y5": 0.0,
                    "Y6": 0,
                    "Y7": 0,
                    "Y8": 0,
                    "Y9": 0,
                    "Y10": 0,
                }
            )
    return pd.DataFrame(rows)


def test_config_parser_expands_semantics_and_checks_step() -> None:
    parsed = parse_config_value("Step1_Config", "Step1_Model3_EQC_CH2")
    assert parsed.as_dict() == {
        "step": 1,
        "model": "Model3",
        "equipment": "EQC",
        "chamber": "CH2",
    }
    with pytest.raises(ValueError, match="일치하지"):
        parse_config_value("Step1_Config", "Step2_Model3_EQC_CH2")


def test_config_parser_records_malformed_values_without_normal_category() -> None:
    frame = v2_frame(1, 2)
    frame.loc[1, "Step1_Config"] = "bad-config"
    parsed, report = parse_config_columns(frame)
    assert report["parse_error_count"] == 1
    assert pd.isna(parsed.loc[1, "Step1_Model"])
    assert pd.isna(parsed.loc[1, "Step1_Equipment"])


def test_v2_quality_uses_dynamic_coverage_and_zero_is_observed() -> None:
    frame = v2_frame(2, 4)
    result = validate_dataframe(frame, validation_mode="training")
    assert result["is_valid"] is True
    assert result["config_completeness_rate"] == pytest.approx(1.0)
    assert result["r_measurement_coverage"] == pytest.approx(0.25)
    assert result["d_measurement_coverage"] == pytest.approx(0.25)
    assert result["target_consistency_rate"] == pytest.approx(1.0)
    assert result["duplicate_wafer_count"] == 0


def test_v2_training_rejects_config_parse_error() -> None:
    frame = v2_frame(2, 4)
    frame.loc[0, "Step1_Config"] = "invalid"
    result = validate_dataframe(frame, validation_mode="training")
    assert result["is_valid"] is False
    assert result["config_parse_error_count"] == 1


def test_lot_id_group_split_is_disjoint_and_targets_are_excluded() -> None:
    dataset = prepare_dataset(v2_frame())
    split = split_dataset(dataset)
    train_lots = set(split.train_groups)
    validation_lots = set(split.validation_groups)
    test_lots = set(split.test_groups)
    assert train_lots.isdisjoint(validation_lots)
    assert train_lots.isdisjoint(test_lots)
    assert validation_lots.isdisjoint(test_lots)
    assert dataset.target_leakage_check["passed"] is True
    assert not set(["Y", *[f"Y{i}" for i in range(1, 11)]]) & set(dataset.feature_columns)
    assert "Lot_Wafer_ID" not in dataset.feature_columns
    assert "Lot_ID" not in dataset.feature_columns
    assert "Wafer_Slot" not in dataset.feature_columns


def test_schema_fingerprint_compatibility_and_new_threshold_defaults() -> None:
    raw_features = ["Step1_Config", "Step1_R1", "Step1_D1"]
    metadata = {
        "schema_version": "semicon_yield_v2",
        "schema_fingerprint": schema_fingerprint(raw_features),
        "config_parser_version": "2.0",
    }
    assert model_schema_status(metadata, raw_features) == "compatible"
    assert model_schema_status(metadata, ["Step1_Config", "Step1_R2"]) == "incompatible"
    assert model_schema_status({"feature_columns": raw_features}) == "legacy"
    assert DEFAULT_WARNING_THRESHOLD == 90.0
    assert DEFAULT_DANGER_THRESHOLD == 85.0


def test_v2_model_prediction_shap_and_report_smoke() -> None:
    temporary_root = Path(__file__).parent / ".tmp_models"
    temporary_root.mkdir(exist_ok=True)
    model_dir = temporary_root / f"v2_smoke_{uuid4().hex}"
    model_dir.mkdir()
    frame = v2_frame(12, 4)
    try:
        dataset = prepare_dataset(frame)
        split = split_dataset(dataset)
        training = train_regression_models(dataset, split)
        metrics = {name: value.as_dict() for name, value in training.metrics.items()}
        model_path, _, _ = save_model_bundle(
            training.best_model,
            target="Y",
            model_name=training.best_model_name,
            feature_columns=dataset.feature_columns,
            metrics=metrics,
            random_state=42,
            split_method=split.split_method,
            model_dir=model_dir,
            metadata_extensions={
                "schema_version": "semicon_yield_v2",
                "schema_fingerprint": schema_fingerprint(dataset.raw_feature_columns),
                "raw_feature_columns": dataset.raw_feature_columns,
                "config_parser_version": "2.0",
                "missing_indicator_used": True,
                "outlier_policy": "flag_only",
                "group_column": "Lot_ID",
            },
        )
        loaded = load_prediction_model(model_path.stem, model_dir)
        inference_frame = frame.drop(columns=["Y", *[f"Y{i}" for i in range(1, 11)]])
        prediction = predict_dataframe(inference_frame, loaded, max_rows=None)
        explanation = explain_dataframe(inference_frame, loaded, max_rows=20, top_n=10)
        report = build_report("v2_fixture.csv", loaded, prediction, explanation)
        assert prediction.total_rows == len(frame)
        assert prediction.evaluation is None
        assert explanation.global_importance
        assert report["report_id"].startswith("report_")
        assert any("Schema Version" in note for note in report["methodology_notes"])
    finally:
        for generated in model_dir.iterdir():
            generated.unlink()
        model_dir.rmdir()
        if not any(temporary_root.iterdir()):
            temporary_root.rmdir()
