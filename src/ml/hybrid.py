from __future__ import annotations

import gc
import gzip
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import shutil
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


PIPELINE_VERSION = "auto_y1_y5_frequency_v2"
FAIL_RATE_TARGETS = [f"Y{index}" for index in range(1, 6)]
COUNT_TARGETS = [f"Y{index}" for index in range(6, 11)]
# Only Y1~Y5 have fitted estimators. Y and Y6~Y10 remain analysis fields.
TARGETS = list(FAIL_RATE_TARGETS)
TRAINING_TARGET_ORDER = list(FAIL_RATE_TARGETS)
TARGET_MODEL_ARTIFACTS = {
    target: f"target_{target}.joblib" for target in TRAINING_TARGET_ORDER
}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "-"}
logger = logging.getLogger(__name__)
_STAGING_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ACTIVE_STAGING_PATHS: set[Path] = set()
_STAGING_LOCK = Lock()
ProgressCallback = Callable[[str, int], None]


def normalized_failure_rates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the production final-y rule without changing valid fail rates."""
    rates = np.maximum(np.asarray(values, dtype=np.float32), np.float32(0.0))
    totals = rates.sum(axis=1)
    overflow = totals > 100.0
    derived = np.clip(
        np.float32(100.0) - totals,
        np.float32(0.0),
        np.float32(100.0),
    ).astype(np.float32, copy=False)
    return rates, derived, int(overflow.sum())


def _normalize_category(value: object) -> object:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text.lower() in MISSING_STRINGS else text


class AutoFeaturePreprocessor(TransformerMixin, BaseEstimator):
    """Train-only imputation, R/D clipping, and Config frequency encoding."""

    def __init__(
        self,
        response_columns: list[str],
        defect_columns: list[str],
        categorical_columns: list[str],
        r_lower_quantile: float = 0.01,
        r_upper_quantile: float = 0.99,
        d_upper_quantile: float = 0.999,
        unknown_frequency: float = 0.0,
    ) -> None:
        self.response_columns = response_columns
        self.defect_columns = defect_columns
        self.categorical_columns = categorical_columns
        self.r_lower_quantile = r_lower_quantile
        self.r_upper_quantile = r_upper_quantile
        self.d_upper_quantile = d_upper_quantile
        self.unknown_frequency = unknown_frequency

    @staticmethod
    def _frame(x: Any) -> pd.DataFrame:
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def fit(self, x: Any, y: Any = None) -> "AutoFeaturePreprocessor":
        del y
        frame = self._frame(x)
        self.response_columns_ = []
        self.defect_columns_ = []
        self.numeric_medians_: dict[str, float] = {}
        self.r_bounds_: dict[str, tuple[float, float] | None] = {}
        self.d_upper_bounds_: dict[str, float | None] = {}
        self.d_zero_ratios_: dict[str, float] = {}
        self.all_missing_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        train_missing_count = 0

        for group, configured in (
            ("R", self.response_columns),
            ("D", self.defect_columns),
        ):
            for column in configured:
                values = self._numeric(frame, column)
                observed = values.dropna()
                if observed.empty:
                    self.all_missing_columns_.append(column)
                    continue
                if observed.nunique() <= 1:
                    self.constant_columns_.append(column)
                    continue
                train_missing_count += int(values.isna().sum())
                self.numeric_medians_[column] = float(observed.median())
                if group == "R":
                    self.response_columns_.append(column)
                    bounds: tuple[float, float] | None = None
                    if len(observed) >= 10 and observed.nunique() > 2:
                        lower, upper = observed.quantile(
                            [self.r_lower_quantile, self.r_upper_quantile]
                        ).to_numpy(dtype=float)
                        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                            bounds = (float(lower), float(upper))
                    self.r_bounds_[column] = bounds
                else:
                    self.defect_columns_.append(column)
                    self.d_zero_ratios_[column] = float((observed == 0).mean())
                    upper_bound: float | None = None
                    if len(observed) >= 10 and observed.nunique() > 2:
                        upper = float(observed.quantile(self.d_upper_quantile))
                        maximum = float(observed.max())
                        if np.isfinite(upper) and upper < maximum and not np.isclose(upper, maximum):
                            upper_bound = upper
                    self.d_upper_bounds_[column] = upper_bound

        self.categorical_columns_ = []
        self.config_modes_: dict[str, str] = {}
        self.frequency_mappings_: dict[str, dict[str, float]] = {}
        config_missing_count = 0
        for column in self.categorical_columns:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category)
            observed = normalized.dropna().astype(str)
            if observed.empty:
                self.all_missing_columns_.append(column)
                continue
            mode = str(observed.mode(dropna=True).iloc[0])
            filled = normalized.fillna(mode).astype(str)
            config_missing_count += int(normalized.isna().sum())
            frequency = filled.value_counts(dropna=False) / max(len(filled), 1)
            self.categorical_columns_.append(column)
            self.config_modes_[column] = mode
            self.frequency_mappings_[column] = {
                str(category): float(value) for category, value in frequency.items()
            }

        self.feature_names_out_ = [
            *self.response_columns_,
            *self.defect_columns_,
            *self.categorical_columns_,
        ]
        if not self.feature_names_out_:
            raise ValueError("Train 데이터에 학습 가능한 R, D, Config Feature가 없습니다.")
        self.summary_ = {
            "r_column_count": len(self.response_columns_),
            "d_column_count": len(self.defect_columns_),
            "config_column_count": len(self.categorical_columns_),
            "numeric_feature_count": len(self.response_columns_) + len(self.defect_columns_),
            "categorical_column_count": len(self.categorical_columns_),
            "total_category_count": int(sum(len(v) for v in self.frequency_mappings_.values())),
            "numeric_missing_fill_count_train": train_missing_count,
            "config_missing_fill_count_train": config_missing_count,
            "removed_all_missing_columns": list(self.all_missing_columns_),
            "removed_constant_columns": list(self.constant_columns_),
            "r_clipped_column_count": int(sum(v is not None for v in self.r_bounds_.values())),
            "d_upper_clipped_column_count": int(sum(v is not None for v in self.d_upper_bounds_.values())),
            "r_clipping_quantiles": [self.r_lower_quantile, self.r_upper_quantile],
            "d_upper_clipping_quantile": self.d_upper_quantile,
            "r_bounds": self.r_bounds_,
            "d_upper_bounds": self.d_upper_bounds_,
            "d_zero_ratios": self.d_zero_ratios_,
            "config_frequency_mappings": self.frequency_mappings_,
            "unknown_frequency": self.unknown_frequency,
            "config_encoding": "frequency",
            "config_strings_decomposed": False,
        }
        return self

    def transform(self, x: Any) -> np.ndarray:
        frame = self._frame(x)
        output = np.empty((len(frame), len(self.feature_names_out_)), dtype=np.float32)
        position = 0
        for column in self.response_columns_:
            values = self._numeric(frame, column).fillna(self.numeric_medians_[column])
            bounds = self.r_bounds_[column]
            if bounds is not None:
                values = values.clip(lower=bounds[0], upper=bounds[1])
            output[:, position] = values.to_numpy(dtype=np.float32)
            position += 1
        for column in self.defect_columns_:
            values = self._numeric(frame, column).fillna(self.numeric_medians_[column])
            upper = self.d_upper_bounds_[column]
            if upper is not None:
                values = values.clip(upper=upper)
            output[:, position] = values.to_numpy(dtype=np.float32)
            position += 1
        for column in self.categorical_columns_:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category).fillna(self.config_modes_[column]).astype(str)
            mapping = self.frequency_mappings_[column]
            output[:, position] = normalized.map(mapping).fillna(self.unknown_frequency).to_numpy(dtype=np.float32)
            position += 1
        return output

    def audit(self, x: Any) -> dict[str, Any]:
        frame = self._frame(x)
        r_adjusted = 0
        d_adjusted = 0
        unknown_by_column: dict[str, int] = {}
        for column in self.response_columns_:
            values = self._numeric(frame, column)
            bounds = self.r_bounds_[column]
            if bounds is not None:
                r_adjusted += int(((values < bounds[0]) | (values > bounds[1])).sum())
        for column in self.defect_columns_:
            values = self._numeric(frame, column)
            upper = self.d_upper_bounds_[column]
            if upper is not None:
                d_adjusted += int((values > upper).sum())
        for column in self.categorical_columns_:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category).fillna(self.config_modes_[column]).astype(str)
            unknown_by_column[column] = int((~normalized.isin(self.frequency_mappings_[column])).sum())
        return {
            "r_adjusted_value_count": r_adjusted,
            "d_adjusted_value_count": d_adjusted,
            "unknown_config_count": int(sum(unknown_by_column.values())),
            "unknown_config_by_column": unknown_by_column,
        }

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.feature_names_out_, dtype=object)



@dataclass
class ModelArtifactRef:
    artifact_path: str
    artifact_root: str | None = field(default=None, repr=False)

    def attach(self, root: str | Path) -> None:
        self.artifact_root = str(Path(root).resolve())

    def resolved_path(self) -> Path:
        configured = Path(self.artifact_path)
        if configured.is_absolute():
            candidate = configured.resolve()
        else:
            if self.artifact_root is None:
                raise FileNotFoundError("모델 Artifact 기준 경로가 설정되지 않았습니다.")
            root = Path(self.artifact_root).resolve()
            candidate = (root / configured).resolve()
            if candidate.parent != root:
                raise ValueError("모델 Artifact 경로가 Bundle 외부를 가리킵니다.")
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError("모델 Target Artifact를 찾을 수 없습니다.")
        return candidate

    def load(self) -> Any:
        return joblib.load(self.resolved_path())

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = self.load()
        try:
            return np.asarray(model.predict(features), dtype=np.float32)
        finally:
            del model
            gc.collect()


def _is_junction(path: Path) -> bool:
    check = getattr(path, "is_junction", None)
    return bool(check()) if callable(check) else False


def _cleanup_stale_staging(root: Path) -> None:
    if not root.is_dir() or root.is_symlink() or _is_junction(root):
        return
    allowed = set(TARGET_MODEL_ARTIFACTS.values())
    for candidate in root.iterdir():
        if (
            candidate in _ACTIVE_STAGING_PATHS
            or not _STAGING_NAME_PATTERN.fullmatch(candidate.name)
            or candidate.is_symlink()
            or _is_junction(candidate)
            or not candidate.is_dir()
        ):
            continue
        entries = list(candidate.iterdir())
        if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):
            logger.warning("안전하지 않은 ML staging 경로를 보존합니다: %s", candidate.name)
            continue
        shutil.rmtree(candidate)


class ModelStagingDirectory:
    def __init__(self) -> None:
        workspace = Path.cwd().resolve()
        self.root = workspace / ".ml-training-staging"
        if self.root.exists() and (self.root.is_symlink() or _is_junction(self.root) or not self.root.is_dir()):
            raise ValueError("ML staging root가 안전하지 않습니다.")
        self.root.mkdir(exist_ok=True)
        with _STAGING_LOCK:
            _cleanup_stale_staging(self.root)
            self.path = self.root / uuid4().hex
            self.path.mkdir(exist_ok=False)
            _ACTIVE_STAGING_PATHS.add(self.path)
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        resolved = self.path.resolve()
        if resolved.parent != self.root or resolved.is_symlink():
            raise ValueError("모델 임시 저장 경로가 안전하지 않습니다.")
        with _STAGING_LOCK:
            if resolved.is_dir():
                entries = list(resolved.iterdir())
                allowed = set(TARGET_MODEL_ARTIFACTS.values())
                if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):
                    raise ValueError("모델 임시 저장 경로에 허용되지 않은 파일이 있습니다.")
                shutil.rmtree(resolved)
            _ACTIVE_STAGING_PATHS.discard(self.path)
            try:
                self.root.rmdir()
            except OSError:
                pass
            self._cleaned = True

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


@dataclass
class HybridMultiYBundle:
    feature_columns: list[str]
    target_models: dict[str, Any]
    selected_final_output: str = "derived"
    warning_threshold: float = 90.0
    critical_threshold: float = 85.0
    pipeline_version: str = PIPELINE_VERSION

    def attach_artifact_root(self, root: str | Path) -> None:
        for stored in self.target_models.values():
            if isinstance(stored, ModelArtifactRef):
                stored.attach(root)

    def model_for_target(self, target: str) -> Any:
        stored = self.target_models.get(target)
        return stored.load() if isinstance(stored, ModelArtifactRef) else stored

    def predict_components(self, features: pd.DataFrame) -> dict[str, Any]:
        predictions: dict[str, np.ndarray] = {}
        for target in FAIL_RATE_TARGETS:
            stored = self.target_models.get(target)
            if stored is None:
                raise ValueError(f"{target} 모델 Artifact가 없습니다.")
            model = stored.load() if isinstance(stored, ModelArtifactRef) else stored
            try:
                values = np.asarray(model.predict(features), dtype=np.float32)
            finally:
                if isinstance(stored, ModelArtifactRef):
                    del model
                    gc.collect()
            predictions[target] = np.maximum(values, 0.0).astype(np.float32, copy=False)
        rates, derived, overflow_count = normalized_failure_rates(
            np.column_stack([predictions[target] for target in FAIL_RATE_TARGETS])
        )
        for position, target in enumerate(FAIL_RATE_TARGETS):
            predictions[target] = rates[:, position]
        critical_probability = 1.0 / (1.0 + np.exp((derived - self.critical_threshold) / 3.0))
        warning_probability = 1.0 / (1.0 + np.exp((derived - self.warning_threshold) / 3.0))
        return {
            "selected": derived,
            "targets": predictions,
            "critical_probability": critical_probability,
            "warning_probability": warning_probability,
            "normalization_count": 0,
            "sum_over_100_count": overflow_count,
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_components(features)["selected"]


@dataclass
class HybridTrainingResult:
    bundle: HybridMultiYBundle
    metadata: dict[str, Any]
    warnings: list[str]
    oof_predictions: dict[str, np.ndarray] | None = None
    _staging_directory: ModelStagingDirectory | None = field(default=None, repr=False)




def _write_oof_predictions(path: Path, predictions: dict[str, np.ndarray] | None) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                str(target): np.asarray(values, dtype=np.float32).tolist()
                for target, values in (predictions or {}).items()
            },
            handle,
            ensure_ascii=False,
        )


def save_hybrid_bundle(
    result: HybridTrainingResult,
    model_dir: str | Path,
    model_id: str,
) -> tuple[Path, Path]:
    bundle_dir = Path(model_dir) / model_id
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = bundle_dir / "bundle.joblib"
    metadata_path = bundle_dir / "metadata.json"
    predictions_path = bundle_dir / "oof_predictions.json.gz"
    assignments_path = bundle_dir / "fold_assignments.json.gz"
    target_paths = {
        target: bundle_dir / filename for target, filename in TARGET_MODEL_ARTIFACTS.items()
    }
    try:
        persistent: dict[str, ModelArtifactRef] = {}
        for target in TRAINING_TARGET_ORDER:
            stored = result.bundle.target_models[target]
            destination = target_paths[target]
            if isinstance(stored, ModelArtifactRef):
                shutil.copy2(stored.resolved_path(), destination)
            else:
                joblib.dump(stored, destination, compress=3)
            persistent[target] = ModelArtifactRef(TARGET_MODEL_ARTIFACTS[target])
        bundle = replace(result.bundle, target_models=persistent)
        joblib.dump(bundle, bundle_path, compress=3)
        bundle.attach_artifact_root(bundle_dir)
        metadata_path.write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_oof_predictions(predictions_path, result.oof_predictions)
        with gzip.open(assignments_path, "wt", encoding="utf-8") as handle:
            json.dump(
                result.metadata.get("split_metadata", {}).get("lot_assignments", {}),
                handle,
                ensure_ascii=False,
            )
    except Exception:
        for generated in [bundle_path, metadata_path, predictions_path, assignments_path, *target_paths.values()]:
            if generated.is_file():
                generated.unlink()
        try:
            bundle_dir.rmdir()
        except OSError:
            pass
        raise
    result.bundle = bundle
    result.oof_predictions = None
    if result._staging_directory is not None:
        result._staging_directory.cleanup()
        result._staging_directory = None
    gc.collect()
    return bundle_path, metadata_path
