from __future__ import annotations

import gc
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from src.ml.hybrid import PIPELINE_VERSION, save_hybrid_bundle, train_hybrid_multi_y
from src.ml.inference import load_prediction_model, predict_dataframe
from src.runtime.cumulative_data import CumulativeDataStore
from src.runtime.operation_coordinator import OperationCoordinator, operation_coordinator
from src.runtime.store import RuntimeStore


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual): return {"rmse": None, "r2": None, "mae": None, "critical_recall": None}
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    r2 = None if len(actual) < 2 or np.var(actual) == 0 else float(1 - np.sum((actual-predicted)**2) / np.sum((actual-np.mean(actual))**2))
    critical = actual < 85
    recall = float(np.mean(predicted[critical] < 85)) if critical.any() else 1.0
    return {"rmse": rmse, "r2": r2, "mae": float(np.mean(np.abs(actual-predicted))), "critical_recall": recall}


class ModelUpdateManager:
    def __init__(self, *, store: RuntimeStore, data: CumulativeDataStore, model_dir: str | Path, coordinator: OperationCoordinator = operation_coordinator) -> None:
        self.store, self.data, self.model_dir, self.coordinator = store, data, Path(model_dir), coordinator
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-update")
        self._lock, self._futures = Lock(), set()

    def submit(self) -> dict[str, Any]:
        self.coordinator.reserve_job("training")
        job_id = f"model_update_{uuid4().hex}"
        self.store.create_training_job(job_id, source_filename="cumulative-dataset")
        future = self.executor.submit(self._run, job_id)
        with self._lock: self._futures.add(future)
        future.add_done_callback(lambda item: self._discard(item))
        active = self.store.active_model()
        return {"job_id": job_id, "status": "queued", "active_model_id": active.get("active_model_id") if active else None, "dataset_version": self.data.status().get("dataset_version")}

    def _discard(self, future: Any) -> None:
        with self._lock: self._futures.discard(future)

    def _run(self, job_id: str) -> None:
        candidate_dir: Path | None = None
        try:
            self.store.start_training_job(job_id)
            def progress(stage: str, value: int) -> None: self.store.update_training_job(job_id, stage=stage, progress=value)
            progress("누적 Label 데이터 확인", 5)
            frame, sample = self.data.training_frame()
            if len(frame) < 20: raise ValueError("학습 가능한 누적 Label 데이터가 부족합니다.")
            progress("Y1~Y5 후보 모델 순차 학습", 20)
            trained = train_hybrid_multi_y(frame, progress_callback=progress)
            candidate_id = f"candidate_{uuid4().hex}"
            trained.metadata["training_sample"] = sample
            version = self.data.status()["dataset_version_number"]
            trained.metadata["dataset_version"] = version
            progress("후보 모델 저장 및 Load 검증", 82)
            candidate_dir, _ = save_hybrid_bundle(trained, self.model_dir, candidate_id)
            candidate = load_prediction_model(candidate_id, self.model_dir)
            # Smoke test detects incomplete bundles before any pointer update.
            smoke = predict_dataframe(frame.head(min(3, len(frame))), candidate, max_rows=None)
            if not smoke.predictions or any(not np.isfinite(float(row["predicted_Y"])) for row in smoke.predictions): raise ValueError("후보 모델 Prediction Smoke Test 실패")
            progress("Champion 비교", 90)
            candidate_result = predict_dataframe(frame, candidate, max_rows=None)
            actual = pd.to_numeric(frame.get("Y"), errors="coerce").to_numpy(dtype=float)
            candidate_y = np.asarray([row["predicted_Y"] for row in candidate_result.predictions], dtype=float)
            candidate_metrics = _metrics(actual, candidate_y)
            active = self.store.active_model()
            promote, reason = self._should_promote(active, frame, candidate_metrics, candidate_id)
            if not promote:
                shutil.rmtree(candidate_dir)
                self.store.complete_training_job(job_id, result={"promotion_result": "rejected", "reason": reason, "candidate_metrics": candidate_metrics, "sample": sample})
                return
            progress("활성 모델 승격", 97)
            summary = {"model_id": candidate_id, "final_y_test": trained.metadata.get("final_y_metrics", {}).get("derived", {}).get("test"), "selected_models": trained.metadata.get("selected_models"), "candidate_metrics": candidate_metrics}
            self.store.promote_model(model_id=candidate_id, pipeline_version=PIPELINE_VERSION, dataset_version=version, metadata=summary)
            self.store.mark_analysis_snapshot_stale("active_model_changed")
            self.store.complete_training_job(job_id, result={"promotion_result": "promoted", "active_model_id": candidate_id, "candidate_metrics": candidate_metrics, "sample": sample})
        except Exception as exc:
            if candidate_dir and candidate_dir.exists():
                shutil.rmtree(candidate_dir, ignore_errors=True)
            self.store.fail_training_job(job_id, str(exc))
        finally:
            gc.collect(); self.coordinator.release_job("training")

    def _should_promote(self, active: dict[str, Any] | None, frame: pd.DataFrame, candidate_metrics: dict[str, float | None], candidate_id: str) -> tuple[bool, str]:
        if active is None: return True, "initial_champion"
        champion = load_prediction_model(str(active["active_model_id"]), self.model_dir)
        result = predict_dataframe(frame, champion, max_rows=None)
        actual = pd.to_numeric(frame.get("Y"), errors="coerce").to_numpy(dtype=float)
        old = _metrics(actual, np.asarray([row["predicted_Y"] for row in result.predictions], dtype=float))
        new_rmse, old_rmse = candidate_metrics["rmse"], old["rmse"]
        if new_rmse is None or old_rmse is None: return False, "final_y_metric_unavailable"
        if candidate_metrics["critical_recall"] is not None and old["critical_recall"] is not None and candidate_metrics["critical_recall"] < old["critical_recall"] - .05: return False, "critical_recall_declined"
        if candidate_metrics.get("r2") is not None and old.get("r2") is not None and candidate_metrics["r2"] < old["r2"] - .02: return False, "final_y_r2_declined"
        if new_rmse <= old_rmse * .99: return True, "rmse_improved_1_percent"
        # Within 1%, only a smaller candidate is allowed.  This is deterministic and avoids timing noise.
        if new_rmse <= old_rmse * 1.01:
            new_size = sum(item.stat().st_size for item in (self.model_dir/candidate_id).glob("*") if item.is_file())
            old_size = sum(item.stat().st_size for item in (self.model_dir/str(active["active_model_id"])).glob("*") if item.is_file())
            if new_size < old_size: return True, "comparable_rmse_smaller_model"
        return False, "champion_not_improved"
