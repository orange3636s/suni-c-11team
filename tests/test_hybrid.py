from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import asyncio
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile

import api.routes.data as data_routes

from src.analytics.lot_analysis import build_lot_cause_analysis
from src.ml.explainability import explain_dataframe
from src.ml.hybrid import (
    COUNT_TARGETS,
    FAIL_RATE_TARGETS,
    normalized_failure_rates,
    save_hybrid_bundle,
    train_hybrid_multi_y,
)
from src.ml.inference import (
    list_prediction_models,
    load_prediction_model,
    load_prediction_model_target,
    predict_dataframe,
)
from src.schema_compatibility import schema_fingerprint


@pytest.fixture(scope="module")
def hybrid_dataframe() -> pd.DataFrame:
    random = np.random.default_rng(2026)
    rows = 80
    response = random.normal(size=rows)
    rates = np.column_stack([
        np.clip(1.2 + (index + 1) * 0.25 * response + random.normal(0, 0.08, rows), 0, None)
        for index in range(5)
    ])
    frame = pd.DataFrame({
        "Lot_Wafer_ID": [f"LOT{index // 4:02d}_WF{index % 4 + 1:02d}" for index in range(rows)],
        "Y": 100.0 - rates.sum(axis=1),
        "Step1_R1": response,
        "Step1_D1": random.normal(size=rows),
        "Step1_EQ": ["EQ_A" if index % 2 else "EQ_B" for index in range(rows)],
    })
    for index, target in enumerate(FAIL_RATE_TARGETS):
        frame[target] = rates[:, index]
    for index, target in enumerate(COUNT_TARGETS, 1):
        frame[target] = np.clip(30 + index * 5 + response * index * 3 + random.normal(0, 2, rows), 0, None)
    return frame


@pytest.fixture
def hybrid_model_dir():
    root = Path(__file__).parent / ".tmp_hybrid_models"
    root.mkdir(exist_ok=True)
    output = root / f"run_{uuid4().hex}"
    output.mkdir()
    yield output
    for bundle_dir in output.iterdir():
        for generated_file in bundle_dir.iterdir():
            generated_file.unlink()
        bundle_dir.rmdir()
    output.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def test_derived_y_clips_and_normalizes_without_count_targets() -> None:
    rates = np.array([[-1, 2, 3, 4, 5], [80, 30, 10, 0, 0]], dtype=float)
    normalized, derived, count = normalized_failure_rates(rates)

    assert normalized[0, 0] == 0
    assert normalized[1].sum() == pytest.approx(100.0)
    assert derived.tolist() == pytest.approx([86.0, 0.0])
    assert count == 1


def test_hybrid_oof_bundle_prediction_and_target_explanation(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = train_hybrid_multi_y(hybrid_dataframe, oof_folds=3)
    assignments = trained.metadata["oof_group_assignments"]
    assert assignments
    assert all(
        set(fold["train_groups"]).isdisjoint(fold["holdout_groups"])
        for fold in assignments
    )
    assert set(trained.bundle.target_models) == set(FAIL_RATE_TARGETS + COUNT_TARGETS)
    assert trained.metadata["selected_final_output"] in {"direct", "derived", "hybrid"}
    assert set(trained.metadata["final_y_metrics"]) == {"direct", "derived", "hybrid"}
    protocol = trained.metadata["cv_protocol"]
    assert protocol["name"] == "nested_group_kfold"
    assert protocol["outer_folds"] == 5
    assert protocol["inner_folds"] == 3
    assert protocol["seed"] == 42
    assert trained.metadata["fallback_used"] is False
    assert trained.metadata["outlier_strategy"] in {"flag_only", "iqr", "model_specific"}
    assert trained.metadata["preprocessing_summary"]["model_outlier_strategies"]
    assert len(protocol["fold_metrics"]) == 5
    assert all(
        set(fold["train_groups"]).isdisjoint(fold["holdout_groups"])
        for fold in protocol["outer_group_assignments"]
    )

    model_id = "HYBRID_MULTI_Y_TEST"
    trained.metadata.update({
        "model_id": model_id,
        "created_at": "2026-08-01T00:00:00+09:00",
        "schema_version": "semicon_yield_v2",
        "schema_fingerprint": schema_fingerprint(["Step1_R1", "Step1_D1", "Step1_EQ"]),
        "raw_feature_columns": ["Step1_R1", "Step1_D1", "Step1_EQ"],
    })
    save_hybrid_bundle(trained, hybrid_model_dir, model_id)
    assert (hybrid_model_dir / model_id / "oof_predictions.json.gz").is_file()
    assert (hybrid_model_dir / model_id / "fold_assignments.json.gz").is_file()
    models, warnings = list_prediction_models(hybrid_model_dir)
    assert warnings == []
    assert len(models) == 1
    assert models[0]["model_type"] == "hybrid_multi_y"
    assert models[0]["available_targets"] == [
        "Y",
        *FAIL_RATE_TARGETS,
        *COUNT_TARGETS,
    ]

    loaded = load_prediction_model(model_id, hybrid_model_dir)
    prediction = predict_dataframe(hybrid_dataframe, loaded, max_rows=None)
    first = prediction.predictions[0]
    assert set(first["failure_rates"]) == set(FAIL_RATE_TARGETS)
    assert set(first["fail_bit_counts"]) == set(COUNT_TARGETS)
    assert first["derived_y"] == pytest.approx(100 - sum(first["failure_rates"].values()))

    y6_model = load_prediction_model_target(model_id, "Y6", hybrid_model_dir)
    y6_prediction = predict_dataframe(hybrid_dataframe, y6_model, max_rows=None)
    monkeypatch.setattr(data_routes, "MODEL_DIR", hybrid_model_dir)
    multi_y, multi_y_warnings = data_routes._collect_multi_y_predictions(
        hybrid_dataframe, y6_model, y6_prediction
    )
    assert multi_y_warnings == []
    assert set(multi_y["failure_rates"]) == set(FAIL_RATE_TARGETS)
    assert set(multi_y["fail_bit_counts"]) == set(COUNT_TARGETS)
    assert multi_y["ensemble_y"] is not None
    selected_first = y6_prediction.predictions[0]
    assert set(selected_first["failure_rates"]) == set(FAIL_RATE_TARGETS)
    assert set(selected_first["fail_bit_counts"]) == set(COUNT_TARGETS)
    assert selected_first["critical_probability"] == pytest.approx(
        first["critical_probability"]
    )
    assert selected_first["warning_probability"] == pytest.approx(
        first["warning_probability"]
    )
    explanation = explain_dataframe(hybrid_dataframe, y6_model, max_rows=8, top_n=5)
    assert explanation.target == "Y6"
    assert explanation.global_importance
    lot_analysis = build_lot_cause_analysis(y6_prediction, explanation)
    assert lot_analysis["lots"]
    assert all(
        lot["average_confidence"] is not None
        for lot in lot_analysis["lots"]
    )
    assert all(
        lot["top_failure_rate_target"] in FAIL_RATE_TARGETS
        and lot["top_fail_bit_count_target"] in COUNT_TARGETS
        for lot in lot_analysis["lots"]
    )


def test_train_and_predict_api_use_one_hybrid_bundle_without_target_selection(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_routes, "MODEL_DIR", hybrid_model_dir)

    def upload(filename: str) -> UploadFile:
        return UploadFile(
            file=BytesIO(hybrid_dataframe.to_csv(index=False).encode("utf-8")),
            filename=filename,
        )

    trained = asyncio.run(
        data_routes.train_model(
            upload("hybrid-training.csv"),
            target=None,
            train_ratio=64,
            validation_ratio=16,
            test_ratio=20,
            missing_indicator=True,
            compare_missingness=False,
        )
    )
    assert trained.model_type == "hybrid_multi_y"
    assert trained.model_id
    assert trained.selected_final_output in {"direct", "derived", "hybrid"}
    assert len(list(hybrid_model_dir.iterdir())) == 1

    prediction = asyncio.run(
        data_routes.predict_csv(
            upload("hybrid-prediction.csv"),
            model_id=trained.model_id,
            warning_threshold=90,
            danger_threshold=85,
        )
    )
    first = prediction.predictions[0]
    assert set(first["failure_rates"]) == set(FAIL_RATE_TARGETS)
    assert set(first["fail_bit_counts"]) == set(COUNT_TARGETS)
    assert prediction.preprocessing
