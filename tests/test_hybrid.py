from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile

import api.routes.data as data_routes
from src.ml.hybrid import (
    COUNT_TARGETS,
    FAIL_RATE_TARGETS,
    PIPELINE_VERSION,
    TARGET_MODEL_ARTIFACTS,
    _selection_reason,
    _should_run_random_forest,
    detect_auto_schema,
    normalized_failure_rates,
    save_hybrid_bundle,
    train_hybrid_multi_y,
)
from src.ml.inference import (
    InferenceInputError,
    list_prediction_models,
    load_prediction_model,
    load_prediction_model_target,
    predict_dataframe,
)


@pytest.fixture(scope="module")
def hybrid_dataframe() -> pd.DataFrame:
    random = np.random.default_rng(2026)
    rows = 100
    response = random.normal(size=rows)
    rates = np.column_stack([
        np.clip(1.2 + (index + 1) * 0.25 * response + random.normal(0, 0.08, rows), 0, None)
        for index in range(5)
    ])
    frame = pd.DataFrame({
        "Lot_Wafer_ID": [f"LOT{index // 5:02d}_WF{index % 5 + 1:02d}" for index in range(rows)],
        "Lot_ID": [f"LOT{index // 5:02d}" for index in range(rows)],
        "Y": np.clip(100.0 - rates.sum(axis=1), 0, 100),
        "Step1_R1": response,
        "Step1_D1": np.where(np.arange(rows) % 7 == 0, np.abs(response), 0.0),
        "Step1_Config": [f"Step1_Model{index % 3}_EQ{index % 4}_CH{index % 2}" for index in range(rows)],
    })
    for index, target in enumerate(FAIL_RATE_TARGETS):
        frame[target] = rates[:, index]
    for index, target in enumerate(COUNT_TARGETS, 1):
        frame[target] = np.clip(30 + index * 5 + response * index * 3, 0, None)
    return frame


@pytest.fixture
def hybrid_model_dir():
    root = Path(__file__).parent / ".tmp_hybrid_models"
    root.mkdir(exist_ok=True)
    output = root / f"run_{uuid4().hex}"
    output.mkdir()
    yield output
    for generated_path in output.iterdir():
        if generated_path.is_dir():
            for generated_file in generated_path.iterdir():
                generated_file.unlink()
            generated_path.rmdir()
        else:
            generated_path.unlink()
    output.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def test_final_y_uses_nonnegative_unscaled_failure_rates() -> None:
    rates = np.array([[-1, 2, 3, 4, 5], [80, 30, 10, 0, 0]], dtype=float)
    nonnegative, derived, overflow_count = normalized_failure_rates(rates)
    assert nonnegative[0].tolist() == pytest.approx([0, 2, 3, 4, 5])
    assert nonnegative[1].sum() == pytest.approx(120.0)
    assert derived.tolist() == pytest.approx([86.0, 0.0])
    assert overflow_count == 1


def test_y1_y5_only_bundle_split_save_reload_and_analysis(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
) -> None:
    trained = train_hybrid_multi_y(hybrid_dataframe)
    assert trained.metadata["pipeline_version"] == PIPELINE_VERSION
    assert trained.metadata["available_targets"] == FAIL_RATE_TARGETS
    assert set(trained.bundle.target_models) == set(FAIL_RATE_TARGETS)
    assert not set(COUNT_TARGETS) & set(trained.bundle.target_models)
    assert not hasattr(trained.bundle, "direct_model")
    assert trained.metadata["dataset_split"] == {
        "train": 0.7, "validation": 0.15, "test": 0.15,
    }
    assignments = trained.metadata["split_metadata"]["lot_assignments"]
    split_sets = [set(assignments[name]) for name in ("train", "validation", "test")]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    assert trained.metadata["split_metadata"]["lot_overlap_count"] == 0

    model_id = "AUTO_Y1_Y5_TEST"
    trained.metadata["model_id"] = model_id
    save_hybrid_bundle(trained, hybrid_model_dir, model_id)
    model_root = hybrid_model_dir / model_id
    assert all((model_root / name).is_file() for name in TARGET_MODEL_ARTIFACTS.values())
    assert len(TARGET_MODEL_ARTIFACTS) == 5
    assert not any(target in path.name for target in COUNT_TARGETS for path in model_root.iterdir())

    models, warnings = list_prediction_models(hybrid_model_dir)
    assert warnings == []
    assert models[0]["available_targets"] == FAIL_RATE_TARGETS
    loaded = load_prediction_model(model_id, hybrid_model_dir)
    prediction = predict_dataframe(hybrid_dataframe, loaded, max_rows=None)
    first = prediction.predictions[0]
    assert not {"direct_y", "derived_y", "hybrid_y", "direct_Y", "derived_Y"} & set(first)
    assert set(first["failure_rates"]) == set(FAIL_RATE_TARGETS)
    assert set(first["fail_bit_counts"]) == set(COUNT_TARGETS)
    assert first["predicted_Y"] == pytest.approx(
        np.clip(100 - sum(first["failure_rates"].values()), 0, 100)
    )
    assert 0 <= first["predicted_Y"] <= 100
    with pytest.raises(InferenceInputError):
        load_prediction_model_target(model_id, "Y", hybrid_model_dir)
    with pytest.raises(InferenceInputError):
        load_prediction_model_target(model_id, "Y6", hybrid_model_dir)

    y1_model = load_prediction_model_target(model_id, "Y1", hybrid_model_dir)
    assert y1_model is not None


def test_train_api_forces_automatic_contract(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_routes, "MODEL_DIR", hybrid_model_dir)
    upload = UploadFile(
        file=BytesIO(hybrid_dataframe.to_csv(index=False).encode("utf-8")),
        filename="automatic.csv",
    )
    response = asyncio.run(data_routes.train_model(upload))
    assert response.model_type == "HistGradientBoostingRegressor"
    assert response.split.train_rows + response.split.validation_rows + response.split.test_rows == len(hybrid_dataframe)
    assert response.target == "Y"


def test_random_forest_is_skipped_when_hgbr_beats_baseline() -> None:
    run_rf, reason = _should_run_random_forest(
        object(),
        {"rmse": 4.0},
        np.array([1.0, 2.0, 3.0]),
        {"rmse": 10.0},
    )
    assert run_rf is False
    assert reason is None


def test_random_forest_runs_when_hgbr_improvement_is_insufficient() -> None:
    run_rf, reason = _should_run_random_forest(
        object(),
        {"rmse": 9.8},
        np.array([1.0, 2.0, 3.0]),
        {"rmse": 10.0},
    )
    assert run_rf is True
    assert reason == "hgbr_baseline_improvement_below_5_percent"


def test_one_percent_rmse_tie_selects_smaller_model() -> None:
    selected, reason = _selection_reason(
        {
            "rmse": 10.0,
            "model_file_size": 100,
            "validation_inference_seconds": 0.2,
        },
        {
            "rmse": 10.05,
            "model_file_size": 1000,
            "validation_inference_seconds": 0.1,
        },
    )
    assert selected == "HistGradientBoostingRegressor"
    assert reason == "validation_rmse_within_1_percent_smaller_model"


def test_auto_schema_whole_config_and_unknown_frequency_resilience(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
) -> None:
    schema = detect_auto_schema(hybrid_dataframe)
    assert schema["config_columns"] == ["Step1_Config"]
    assert not any("Model" in name or "Equipment" in name or "Chamber" in name for name in schema["feature_columns"])
    trained = train_hybrid_multi_y(hybrid_dataframe)
    model_id = "AUTO_RESILIENCE"
    trained.metadata["model_id"] = model_id
    save_hybrid_bundle(trained, hybrid_model_dir, model_id)
    loaded = load_prediction_model(model_id, hybrid_model_dir)
    inference = hybrid_dataframe.drop(columns=["Step1_D1"]).copy()
    inference.loc[0, "Step1_Config"] = "UNSEEN_WHOLE_CONFIG"
    inference["Step99_R1"] = 123.0
    result = predict_dataframe(inference, loaded, max_rows=None)
    assert len(result.predictions) == len(inference)
    assert result.preprocessing_summary["missing_input_features"] == ["Step1_D1"]
    assert result.preprocessing_summary["ignored_extra_features"] == ["Step99_R1"]
    assert result.preprocessing_summary["unknown_config_count"] >= 1
