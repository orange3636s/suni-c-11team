from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
import shutil
from pathlib import Path
from uuid import uuid4
from sklearn.dummy import DummyRegressor

import src.ml.inference as inference_module
from src.ml.dataset import prepare_dataset
from src.ml.hybrid import (
    ModelArtifactRef,
    ModelStagingDirectory,
    TRAINING_TARGET_ORDER,
    AutoFeaturePreprocessor,
)
from src.ml.inference import get_prediction_model_detail, list_prediction_models
from src.ml.memory_usage import memory_snapshot
from src.ml.model_io import save_model_bundle


@pytest.fixture
def memory_tmp_path():
    parent = (Path(__file__).parent / ".ml_memory_cases").resolve()
    parent.mkdir(exist_ok=True)
    root = (parent / uuid4().hex).resolve()
    if root.parent != parent:
        raise RuntimeError("테스트 임시 경로가 허용된 부모를 벗어났습니다.")
    root.mkdir()
    try:
        yield root
    finally:
        if root.is_dir() and root.parent == parent:
            shutil.rmtree(root)
        try:
            parent.rmdir()
        except OSError:
            pass


def _legacy_frame(rows: int = 18) -> pd.DataFrame:
    values = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"LOT{index // 3:02d}_WF{index % 3:02d}" for index in range(rows)],
            "Y": 90.0 + values,
            "Step1_R1": values.astype(np.float64),
            "Step1_D1": (values * 2).astype(np.float64),
            "Step1_EQ": ["EQ_A" if index % 2 else "EQ_B" for index in range(rows)],
        }
    )


def test_prepared_numeric_features_and_target_use_float32() -> None:
    prepared = prepare_dataset(_legacy_frame())

    assert prepared.target.dtype == np.float32
    assert prepared.features["Step1_R1"].dtype == np.float32
    assert prepared.features["Step1_D1"].dtype == np.float32


def test_categorical_preprocessor_is_frequency_encoded_float32() -> None:
    features = pd.DataFrame({
        "Step1_R1": np.arange(10, dtype=float),
        "Step1_Config": ["A"] * 7 + ["B"] * 3,
    })
    preprocessor = AutoFeaturePreprocessor(
        response_columns=["Step1_R1"],
        defect_columns=[],
        categorical_columns=["Step1_Config"],
    ).fit(features)
    unknown = features.iloc[:2].copy()
    unknown["Step1_Config"] = "NEVER_SEEN"
    transformed = preprocessor.transform(unknown)

    assert preprocessor.frequency_mappings_["Step1_Config"] == {"A": 0.7, "B": 0.3}
    assert transformed[:, 1].tolist() == [0.0, 0.0]
    assert transformed.dtype == np.float32
    assert transformed.shape[1] == 2


def test_training_target_order_is_y1_through_y5() -> None:
    # ND: this used to also cover src.ml.hybrid._splitter (the fixed
    # stratified-group-holdout split for the now-removed
    # train_hybrid_multi_y HGBR/RandomForest comparison path) -- that
    # function is gone with it, so only the still-live constant remains
    # covered here.
    assert TRAINING_TARGET_ORDER == [f"Y{index}" for index in range(1, 6)]


def test_disk_backed_target_model_releases_staging() -> None:
    staging = ModelStagingDirectory()
    artifact = staging.path / "target_Y1.joblib"
    estimator = DummyRegressor(strategy="mean").fit(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    joblib.dump(estimator, artifact, compress=3)
    reference = ModelArtifactRef(str(artifact))

    loaded = reference.load()
    assert loaded.predict(np.asarray([[2.0]], dtype=np.float32))[0] == pytest.approx(3.0)

    staging_path = staging.path
    del loaded
    staging.cleanup()
    assert not staging_path.exists()


def test_next_training_cleans_only_allowlisted_stale_staging(
    memory_tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(memory_tmp_path)
    root = memory_tmp_path / ".ml-training-staging"
    stale = root / ("a" * 32)
    stale.mkdir(parents=True)
    (stale / "target_Y1.joblib").write_bytes(b"interrupted")
    unsafe = root / ("b" * 32)
    unsafe.mkdir()
    (unsafe / "keep.csv").write_text("preserve", encoding="utf-8")

    current = ModelStagingDirectory()

    assert not stale.exists()
    assert unsafe.is_dir()
    assert (unsafe / "keep.csv").is_file()
    current.cleanup()


def test_model_discovery_never_deserializes_joblib(
    memory_tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = DummyRegressor(strategy="mean").fit(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([2.0, 4.0], dtype=np.float32),
    )
    _, _, metadata = save_model_bundle(
        estimator,
        target="Y",
        model_name="DummyRegressor",
        feature_columns=["Step1_R1"],
        metrics={
            name: {"r2": 0.0, "rmse": 1.0, "mae": 1.0}
            for name in ("train", "validation", "test")
        },
        random_state=42,
        split_method="group",
        model_dir=memory_tmp_path,
    )
    monkeypatch.setattr(
        inference_module,
        "load_model",
        lambda path: (_ for _ in ()).throw(
            AssertionError(f"모델 검색 중 역직렬화했습니다: {path}")
        ),
    )

    models, warnings = list_prediction_models(memory_tmp_path)
    detail = get_prediction_model_detail(metadata["model_id"], memory_tmp_path)

    assert warnings == []
    assert models[0]["available"] is True
    assert detail["available"] is True


def test_memory_snapshot_is_optional_and_numeric() -> None:
    snapshot = memory_snapshot()

    assert set(snapshot) == {"rss_mb", "max_rss_mb"}
    assert all(value is None or value > 0 for value in snapshot.values())
