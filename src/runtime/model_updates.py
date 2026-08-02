"""Background training for the one active direct-Y model."""
from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np

from src.ml.dataset import prepare_dataset, split_dataset
from src.ml.inference import load_prediction_model, predict_dataframe
from src.ml.model_io import save_model_bundle
from src.ml.training import train_regression_models
from src.runtime.cumulative_data import CumulativeDataStore
from src.runtime.operation_coordinator import OperationCoordinator, operation_coordinator
from src.runtime.store import RuntimeStore


class ModelUpdateManager:
    """Serialise direct-Y retraining without candidate comparison or promotion."""

    def __init__(self, *, store: RuntimeStore, data: CumulativeDataStore,
                 model_dir: str | Path,
                 coordinator: OperationCoordinator = operation_coordinator) -> None:
        self.store, self.data, self.model_dir, self.coordinator = store, data, Path(model_dir), coordinator
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-update")
        self._lock, self._futures = Lock(), set()

    def submit(self) -> dict[str, Any]:
        self.coordinator.reserve_job("training")
        job_id = f"model_update_{uuid4().hex}"
        self.store.create_training_job(job_id, source_filename="cumulative-dataset")
        future = self.executor.submit(self._run, job_id)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard)
        active = self.store.active_model()
        return {"job_id": job_id, "status": "queued", "active_model_id": active.get("active_model_id") if active else None}

    def _discard(self, future: Any) -> None:
        with self._lock:
            self._futures.discard(future)

    def _run(self, job_id: str) -> None:
        try:
            self.store.start_training_job(job_id)
            def progress(stage: str, value: int) -> None:
                self.store.update_training_job(job_id, stage=stage, progress=value)
            progress("Checking labelled data", 5)
            frame, sample = self.data.training_frame()
            dataset = prepare_dataset(frame, target="Y", add_missing_indicators=False)
            split = split_dataset(dataset)
            progress("Training direct Y model", 40)
            trained = train_regression_models(dataset, split)
            metrics = {name: item.as_dict() for name, item in trained.metrics.items()}
            progress("Saving and smoke-testing model", 82)
            _, _, metadata = save_model_bundle(
                trained.best_model, target="Y", model_name=trained.best_model_name,
                feature_columns=dataset.feature_columns, metrics=metrics,
                random_state=42, split_method=split.split_method,
                dataset_rows={"train": len(split.x_train), "validation": len(split.x_validation), "test": len(split.x_test)},
                metadata_extensions={"target_leakage_check": dataset.target_leakage_check}, model_dir=self.model_dir,
            )
            model_id = str(metadata["model_id"])
            loaded = load_prediction_model(model_id, self.model_dir)
            smoke = predict_dataframe(frame.head(min(3, len(frame))), loaded, max_rows=None)
            if not smoke.predictions or not np.isfinite([row["predicted_Y"] for row in smoke.predictions]).all():
                raise ValueError("Direct Y model smoke test failed")
            progress("Replacing active model", 97)
            version = int(self.data.status().get("dataset_version_number") or 0)
            self.store.promote_model(model_id=model_id, pipeline_version="direct_y_v1", dataset_version=version,
                                     metadata={"model_id": model_id, "target": "Y", "test_metrics": metrics.get("test")})
            self.store.mark_analysis_snapshot_stale("active_model_changed")
            self.store.complete_training_job(job_id, result={"active_model_id": model_id, "target": "Y", "test_metrics": metrics.get("test"), "sample": sample})
        except Exception as exc:
            self.store.fail_training_job(job_id, str(exc))
        finally:
            gc.collect()
            self.coordinator.release_job("training")
