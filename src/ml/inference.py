from __future__ import annotations

import importlib.util
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema, validate_dataframe
from src.ml.evaluation import RegressionMetrics, evaluate_regression
from src.ml.model_io import (
    DEFAULT_MODEL_DIR,
    load_metadata,
    load_model,
)
from src.preprocessing import preprocess_dataframe
from src.schema_compatibility import model_schema_status


logger = logging.getLogger(__name__)

DEFAULT_WARNING_THRESHOLD = 90.0
DEFAULT_DANGER_THRESHOLD = 85.0
MAX_PREDICTION_ROWS = 5000
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MODEL_DELETE_STAGING_DIR = ".deleting"
HYBRID_TARGET_MODEL_FILES = tuple(
    f"target_{target}.joblib"
    for target in ["Y", *[f"Y{index}" for index in range(1, 11)]]
)
HYBRID_BUNDLE_FILES = (
    "bundle.joblib",
    "metadata.json",
    "oof_predictions.json.gz",
    "fold_assignments.json.gz",
    *HYBRID_TARGET_MODEL_FILES,
)
_MODEL_DELETE_LOCK = Lock()
REQUIRED_METADATA_FIELDS = {
    "target",
    "model_name",
    "feature_columns",
    "created_at",
    "metrics",
    "split_method",
}


class InferenceInputError(ValueError):
    """사용자가 입력 데이터나 모델 선택을 수정할 수 있는 오류."""


class InvalidModelIdError(InferenceInputError):
    """허용된 model_id 형식을 벗어난 요청."""


class ModelNotFoundError(InferenceInputError):
    """삭제 또는 조회할 저장 모델이 존재하지 않는 요청."""


class ModelLoadError(RuntimeError):
    """모델 파일 접근 또는 역직렬화 오류."""


class ModelDeletionError(ModelLoadError):
    """모델 격리 또는 파일 정리 중 발생한 오류."""

    def __init__(
        self,
        message: str,
        *,
        failed_files: list[str] | None = None,
        deleted_files: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_files = failed_files or []
        self.deleted_files = deleted_files or []


@dataclass
class LoadedPredictionModel:
    model_id: str
    model: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DeleteModelResult:
    model_id: str
    deleted_files: list[str]
    missing_files: list[str]
    metadata_deleted: bool
    bundle_deleted: bool


@dataclass
class PredictionResult:
    model_id: str
    target: str
    model_name: str
    identifier_column: str
    predictions: list[dict[str, Any]]
    total_rows: int
    average_prediction: float
    normal_count: int
    warning_count: int
    danger_count: int
    evaluation: RegressionMetrics | None = None
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False
    preprocessing_summary: dict[str, Any] = field(default_factory=dict)


def _metadata_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _validate_model_id(model_id: str) -> None:
    if (
        not model_id
        or not MODEL_ID_PATTERN.fullmatch(model_id)
        or ".." in model_id
        or "/" in model_id
        or "\\" in model_id
    ):
        raise InvalidModelIdError("유효하지 않은 모델 ID입니다.")


def _model_paths(
    model_id: str,
    model_dir: str | Path,
) -> tuple[Path, Path]:
    _validate_model_id(model_id)
    root = Path(model_dir).resolve()
    bundle_dir = (root / model_id).resolve()
    if bundle_dir.parent == root and bundle_dir.is_dir():
        return bundle_dir / "bundle.joblib", bundle_dir / "metadata.json"
    model_path = (root / f"{model_id}.joblib").resolve()
    metadata_path = (root / f"{model_id}.json").resolve()
    if model_path.parent != root or metadata_path.parent != root:
        raise InvalidModelIdError("유효하지 않은 모델 ID입니다.")
    return model_path, metadata_path


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing_fields = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    version = metadata.get("sklearn_version") or metadata.get(
        "scikit_learn_version"
    )
    if missing_fields or not version:
        missing = [*missing_fields]
        if not version:
            missing.append("sklearn_version")
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다. 누락 항목: "
            + ", ".join(missing)
        )
    if not isinstance(metadata["feature_columns"], list) or not all(
        isinstance(column, str)
        for column in metadata["feature_columns"]
    ):
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다: feature_columns"
        )
    metrics = metadata.get("metrics")
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    if not isinstance(test_metrics, dict) or not {
        "r2",
        "rmse",
        "mae",
    }.issubset(test_metrics):
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다: metrics.test"
        )


def load_prediction_model(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> LoadedPredictionModel:
    model_path, metadata_path = _model_paths(model_id, model_dir)
    if not metadata_path.is_file() or not model_path.is_file():
        raise InferenceInputError(
            "존재하지 않거나 사용할 수 없는 모델 ID입니다."
        )
    try:
        metadata = load_metadata(metadata_path)
    except (OSError, ValueError) as exc:
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다."
        ) from exc
    _validate_metadata(metadata)
    try:
        model = load_model(model_path)
    except Exception as exc:
        raise ModelLoadError("모델 파일을 불러오지 못했습니다.") from exc
    if not callable(getattr(model, "predict", None)):
        raise ModelLoadError("예측 가능한 모델 파일이 아닙니다.")
    attach_artifact_root = getattr(model, "attach_artifact_root", None)
    if callable(attach_artifact_root):
        attach_artifact_root(model_path.parent)
    return LoadedPredictionModel(
        model_id=model_id,
        model=model,
        metadata=metadata,
    )


def _missing_dependency_name(error: BaseException) -> str | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            name = getattr(current, "name", None)
            if isinstance(name, str) and name.strip():
                return name.strip().split(".", 1)[0]
        current = current.__cause__ or current.__context__
    return None


def _model_availability(
    model_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        _validate_metadata(metadata)
    except InferenceInputError as exc:
        return {
            "available": False,
            "loadable": False,
            "compatibility_status": "invalid_metadata",
            "incompatibility_reason": str(exc),
        }
    schema_status = model_schema_status(metadata)
    if not model_path.is_file():
        return {
            "available": False,
            "loadable": False,
            "compatibility_status": "model_file_missing",
            "incompatibility_reason": "모델 파일이 존재하지 않습니다.",
        }
    target_artifacts = metadata.get("target_model_artifacts")
    if isinstance(target_artifacts, dict):
        root = model_path.parent.resolve()
        for filename in target_artifacts.values():
            if not isinstance(filename, str):
                return {
                    "available": False,
                    "loadable": False,
                    "compatibility_status": "invalid_metadata",
                    "incompatibility_reason": "모델 Target Artifact 정보가 올바르지 않습니다.",
                }
            candidate = (root / filename).resolve()
            if (
                candidate.parent != root
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                return {
                    "available": False,
                    "loadable": False,
                    "compatibility_status": "model_file_missing",
                    "incompatibility_reason": "모델 Target Artifact가 존재하지 않습니다.",
                }
    dependency_markers = {
        "xgboost": ("xgboost", "xgbregressor", "xgb"),
        "catboost": ("catboost", "catboostregressor"),
    }
    declared_models = [
        metadata.get("model_name"),
        metadata.get("model_type"),
        metadata.get("bundle_type"),
        *(
            metadata.get("base_model_names", [])
            if isinstance(metadata.get("base_model_names"), list)
            else []
        ),
        model_path.stem,
        model_path.parent.name,
    ]
    declaration = " ".join(str(value).lower() for value in declared_models if value)
    for dependency, markers in dependency_markers.items():
        if any(marker in declaration for marker in markers) and importlib.util.find_spec(dependency) is None:
            return {
                "available": False,
                "loadable": False,
                "compatibility_status": "dependency_missing",
                "incompatibility_reason": f"{dependency}가 설치되어 있지 않습니다.",
            }

    # Listing/detail endpoints deliberately do not deserialize model files.
    # A tiny signature read catches obvious truncation/text files while keeping
    # the expensive compatibility check on the actual prediction path.
    try:
        with model_path.open("rb") as handle:
            header = handle.read(8)
    except OSError:
        return {
            "available": False,
            "loadable": False,
            "compatibility_status": "load_error",
            "incompatibility_reason": "모델 파일을 불러올 수 없습니다.",
        }
    known_headers = (
        b"\x80",  # pickle/joblib without compression
        b"\x78",  # zlib
        b"\x1f\x8b",  # gzip
        b"BZh",  # bzip2
        b"\xfd7zXZ\x00",  # xz
        b"\x04\x22\x4d\x18",  # lz4 frame
    )
    if not header or not any(header.startswith(prefix) for prefix in known_headers):
        return {
            "available": False,
            "loadable": False,
            "compatibility_status": "load_error",
            "incompatibility_reason": "모델 파일을 불러올 수 없습니다.",
        }
    if schema_status == "incompatible":
        return {
            "available": False,
            "loadable": True,
            "compatibility_status": "schema_incompatible",
            "incompatibility_reason": "현재 데이터 스키마와 호환되지 않는 모델입니다.",
        }
    return {
        "available": True,
        "loadable": True,
        "compatibility_status": schema_status,
        "incompatibility_reason": None,
    }


def _delete_staging_paths(root: Path, model_id: str) -> tuple[Path, Path]:
    staging_root = root / MODEL_DELETE_STAGING_DIR
    staging_root.mkdir(exist_ok=True)
    resolved_staging_root = staging_root.resolve()
    if (
        resolved_staging_root != staging_root
        or resolved_staging_root.parent != root
        or not resolved_staging_root.is_dir()
    ):
        raise ModelDeletionError("모델 삭제 격리 경로가 안전하지 않습니다.")
    staging_root = resolved_staging_root
    expected_staged_model_dir = staging_root / model_id
    staged_model_dir = expected_staged_model_dir.resolve()
    if (
        staged_model_dir != expected_staged_model_dir
        or staged_model_dir.parent != staging_root
    ):
        raise ModelDeletionError("모델별 삭제 격리 경로가 안전하지 않습니다.")
    return staging_root, staged_model_dir


def _validated_staged_files(
    staged_model_dir: Path,
    model_id: str,
) -> tuple[list[Path], str]:
    entries = list(staged_model_dir.iterdir())
    flat_names = {f"{model_id}.joblib", f"{model_id}.json"}
    entry_names = {entry.name for entry in entries}
    if entry_names.issubset(flat_names):
        layout = "flat"
        allowed_names = flat_names
    elif entry_names.issubset(set(HYBRID_BUNDLE_FILES)):
        layout = "hybrid"
        allowed_names = set(HYBRID_BUNDLE_FILES)
    else:
        layout = "unknown"
        allowed_names = flat_names | set(HYBRID_BUNDLE_FILES)
    unsafe = sorted(
        entry.name
        for entry in entries
        if entry.name not in allowed_names or entry.is_symlink() or not entry.is_file()
    )
    if layout == "unknown" or unsafe:
        names = sorted(entry_names) if layout == "unknown" else unsafe
        raise ModelDeletionError(
            "격리된 모델에 삭제할 수 없는 파일이 있습니다: " + ", ".join(names),
            failed_files=names,
        )
    return entries, layout


def _cleanup_staged_model(
    root: Path,
    staging_root: Path,
    staged_model_dir: Path,
    model_id: str,
) -> list[str]:
    entries, layout = _validated_staged_files(staged_model_dir, model_id)
    original_names = [
        (
            f"{model_id}/{entry.name}"
            if layout == "hybrid"
            else entry.name
        )
        for entry in entries
    ]
    deletion_order = {
        "oof_predictions.json.gz": 0,
        "fold_assignments.json.gz": 1,
        "bundle.joblib": 2,
        "metadata.json": 3,
    }
    deleted: list[str] = []
    failed: list[str] = []
    for entry, original_name in sorted(
        zip(entries, original_names),
        key=lambda item: (deletion_order.get(item[0].name, 0), item[0].name),
    ):
        try:
            entry.unlink()
            deleted.append(original_name)
        except FileNotFoundError:
            continue
        except OSError:
            failed.append(original_name)
    if failed:
        raise ModelDeletionError(
            "모델 파일 일부를 삭제하지 못했습니다: " + ", ".join(failed),
            failed_files=failed,
            deleted_files=deleted,
        )
    try:
        staged_model_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        relative_stage = staged_model_dir.relative_to(root).as_posix()
        raise ModelDeletionError(
            f"모델 삭제 격리 폴더를 정리하지 못했습니다: {relative_stage}",
            failed_files=[relative_stage],
            deleted_files=deleted,
        ) from exc
    try:
        staging_root.rmdir()
    except OSError:
        # 다른 모델 삭제 격리 폴더가 있으면 공용 staging root는 유지한다.
        pass
    return deleted


def _delete_prediction_model_locked(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[str]:
    """모델 Artifact만 격리 후 삭제하며 Runtime History는 변경하지 않는다."""
    root = Path(model_dir).resolve()
    _validate_model_id(model_id)
    unresolved_bundle_dir = root / model_id
    if unresolved_bundle_dir.is_symlink() or (
        unresolved_bundle_dir.exists()
        and unresolved_bundle_dir.resolve() != unresolved_bundle_dir
    ):
        raise ModelDeletionError(
            "심볼릭 링크 또는 Junction인 모델 Bundle은 삭제할 수 없습니다.",
            failed_files=[f"{model_id}/"],
        )
    model_path, metadata_path = _model_paths(model_id, root)
    staging_root_path = root / MODEL_DELETE_STAGING_DIR
    staged_model_path = staging_root_path / model_id

    bundle_dir = model_path.parent
    is_hybrid = bundle_dir != root
    flat_candidates = (
        (root / f"{model_id}.joblib").resolve(),
        (root / f"{model_id}.json").resolve(),
    )
    unresolved_flat_candidates = (
        root / f"{model_id}.joblib",
        root / f"{model_id}.json",
    )
    unsafe_flat_links = sorted(
        path.name for path in unresolved_flat_candidates if path.is_symlink()
    )
    if unsafe_flat_links:
        raise ModelDeletionError(
            "심볼릭 링크인 모델 파일은 삭제할 수 없습니다: "
            + ", ".join(unsafe_flat_links),
            failed_files=unsafe_flat_links,
        )
    flat_existing = [path for path in flat_candidates if path.is_file()]
    if is_hybrid and flat_existing:
        raise ModelDeletionError(
            "동일한 model_id가 Flat 파일과 Bundle 폴더에 함께 존재합니다."
        )

    if is_hybrid:
        entries = list(bundle_dir.iterdir())
        unsafe = sorted(
            entry.name
            for entry in entries
            if (
                entry.name not in HYBRID_BUNDLE_FILES
                or entry.is_symlink()
                or not entry.is_file()
            )
        )
        if unsafe:
            raise ModelDeletionError(
                "알 수 없는 Bundle 파일이 있어 안전하게 삭제할 수 없습니다: "
                + ", ".join(unsafe),
                failed_files=unsafe,
            )
        active_files = entries
    else:
        active_files = [
            path for path in (model_path, metadata_path) if path.is_file()
        ]

    if staged_model_path.exists():
        staging_root, staged_model_dir = _delete_staging_paths(root, model_id)
        staged_entries, staged_layout = _validated_staged_files(
            staged_model_dir,
            model_id,
        )
        if active_files and not staged_entries:
            staged_model_dir.rmdir()
            try:
                staging_root.rmdir()
            except OSError:
                pass
        elif active_files and not is_hybrid and staged_layout == "flat":
            conflicts = sorted(
                entry.name
                for entry in staged_entries
                if (root / entry.name).exists()
            )
            if conflicts:
                raise ModelDeletionError(
                    "활성 모델 파일과 이전 삭제 격리 파일이 중복됩니다: "
                    + ", ".join(conflicts),
                    failed_files=conflicts,
                )
            restored: list[str] = []
            try:
                for entry in staged_entries:
                    destination = root / entry.name
                    entry.replace(destination)
                    restored.append(entry.name)
                staged_model_dir.rmdir()
                try:
                    staging_root.rmdir()
                except OSError:
                    pass
            except OSError as exc:
                raise ModelDeletionError(
                    "이전 삭제 실패 파일을 활성 모델 위치로 복구하지 못했습니다.",
                    failed_files=[
                        entry.name
                        for entry in staged_entries
                        if entry.name not in restored
                    ],
                ) from exc
            active_files = [
                path for path in (model_path, metadata_path) if path.is_file()
            ]
        elif active_files:
            raise ModelDeletionError(
                "활성 모델 파일과 이전 삭제 격리 파일이 함께 존재합니다."
            )
        elif not staged_entries:
            staged_model_dir.rmdir()
            try:
                staging_root.rmdir()
            except OSError:
                pass
            raise ModelNotFoundError("존재하지 않거나 이미 삭제된 모델 ID입니다.")
        else:
            return _cleanup_staged_model(
                root,
                staging_root,
                staged_model_dir,
                model_id,
            )
    if not active_files:
        raise ModelNotFoundError("존재하지 않거나 이미 삭제된 모델 ID입니다.")

    staging_root, staged_model_dir = _delete_staging_paths(root, model_id)
    if is_hybrid:
        try:
            bundle_dir.replace(staged_model_dir)
        except OSError as exc:
            try:
                staging_root.rmdir()
            except OSError:
                pass
            raise ModelDeletionError(
                "모델 Bundle을 삭제 격리 경로로 이동하지 못했습니다.",
                failed_files=[f"{model_id}/"],
            ) from exc
    else:
        staged_model_dir.mkdir()
        moved: list[tuple[Path, Path]] = []
        try:
            for source in active_files:
                destination = staged_model_dir / source.name
                source.replace(destination)
                moved.append((source, destination))
        except OSError as exc:
            rollback_failed: list[str] = []
            for source, destination in reversed(moved):
                try:
                    destination.replace(source)
                except OSError:
                    rollback_failed.append(source.name)
            try:
                staged_model_dir.rmdir()
            except OSError:
                pass
            try:
                staging_root.rmdir()
            except OSError:
                pass
            message = "모델 파일을 삭제 격리 경로로 이동하지 못했습니다."
            if rollback_failed:
                message += " 원위치 복구 실패: " + ", ".join(rollback_failed)
            raise ModelDeletionError(
                message,
                failed_files=[source.name, *rollback_failed],
            ) from exc

    return _cleanup_staged_model(
        root,
        staging_root,
        staged_model_dir,
        model_id,
    )


def _delete_model_result(model_id: str, deleted_files: list[str]) -> DeleteModelResult:
    hybrid_layout = any(name.startswith(f"{model_id}/") for name in deleted_files)
    if hybrid_layout:
        deleted_names = {
            name.split("/", 1)[1]
            for name in deleted_files
            if "/" in name
        }
        core_files = HYBRID_BUNDLE_FILES[:4]
        expected_names = (
            HYBRID_BUNDLE_FILES
            if deleted_names.intersection(HYBRID_TARGET_MODEL_FILES)
            else core_files
        )
        expected_files = [f"{model_id}/{name}" for name in expected_names]
    else:
        expected_files = [f"{model_id}.joblib", f"{model_id}.json"]
    deleted_set = set(deleted_files)
    metadata_names = {f"{model_id}.json", f"{model_id}/metadata.json"}
    bundle_names = {f"{model_id}.joblib", f"{model_id}/bundle.joblib"}
    return DeleteModelResult(
        model_id=model_id,
        deleted_files=deleted_files,
        missing_files=[name for name in expected_files if name not in deleted_set],
        metadata_deleted=bool(metadata_names & deleted_set),
        bundle_deleted=bool(bundle_names & deleted_set),
    )


def delete_model_bundle(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> DeleteModelResult:
    with _MODEL_DELETE_LOCK:
        deleted_files = _delete_prediction_model_locked(model_id, model_dir)
        return _delete_model_result(model_id, deleted_files)


def delete_prediction_model(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[str]:
    """이전 호출자 호환성을 위해 삭제 파일 목록만 반환한다."""
    return delete_model_bundle(model_id, model_dir).deleted_files


def load_prediction_model_target(
    model_id: str,
    target: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> LoadedPredictionModel:
    loaded = load_prediction_model(model_id, model_dir)
    if loaded.metadata.get("model_type") != "hybrid_multi_y":
        if loaded.metadata.get("target") != target:
            raise InferenceInputError("선택한 Legacy 모델에는 요청한 Target 서브모델이 없습니다.")
        return loaded
    model_for_target = getattr(loaded.model, "model_for_target", None)
    if target == "Y":
        raise InferenceInputError(
            "신규 Bundle은 Direct Y 모델을 저장하지 않습니다. Y1~Y5 중 하나를 선택해 주세요."
        )
    if callable(model_for_target):
        selected_model = model_for_target(target)
    else:
        selected_model = loaded.model.target_models.get(target)
    if selected_model is None:
        raise InferenceInputError(f"Hybrid Bundle에 {target} 서브모델이 없습니다.")
    metadata = {
        **loaded.metadata,
        "target": target,
        "model_name": f"Hybrid Multi-Y · {target}",
        "metrics": (
            loaded.metadata.get("final_y_metrics", {}).get("direct", {})
            if target == "Y"
            else loaded.metadata.get("target_metrics", {}).get(target, {})
        ),
    }
    return LoadedPredictionModel(model_id=model_id, model=selected_model, metadata=metadata)


def _metadata_available_targets(metadata: dict[str, Any]) -> list[str]:
    configured = metadata.get("available_targets")
    configured_targets = list(
        dict.fromkeys(
            str(value).strip()
            for value in configured
            if str(value).strip()
        )
    ) if isinstance(configured, list) else []
    if configured_targets:
        return configured_targets
    target_metrics = metadata.get("target_metrics")
    if isinstance(target_metrics, dict) and target_metrics:
        candidates = list(target_metrics)
    else:
        target = metadata.get("target")
        candidates = [target] if target is not None else []
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in candidates
            if str(value).strip()
        )
    )


def _compact_cv_summary(value: Any) -> dict[str, Any]:
    """Keep model discovery payloads small; fold assignments live on disk."""
    if not isinstance(value, dict):
        return {}
    scalar_fields = (
        "name",
        "group_column",
        "outer_folds",
        "inner_folds",
        "seed",
        "selection_target",
    )
    result = {
        field: value.get(field)
        for field in scalar_fields
        if field in value
    }
    metric_summary = value.get("metric_summary")
    if isinstance(metric_summary, dict):
        result["metric_summary"] = metric_summary
    fold_metrics = value.get("fold_metrics")
    if isinstance(fold_metrics, list):
        result["fold_metrics"] = fold_metrics[:3]
    return result


def list_prediction_models(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(model_dir)
    if not root.exists():
        return [], []
    models: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        metadata_paths = [*root.glob("*.json"), *root.glob("*/metadata.json")]
    except OSError as exc:
        raise ModelLoadError("모델 폴더를 읽지 못했습니다.") from exc

    for metadata_path in metadata_paths:
        is_bundle = metadata_path.name == "metadata.json" and metadata_path.parent != root
        model_id = metadata_path.parent.name if is_bundle else metadata_path.stem
        model_path = metadata_path.parent / "bundle.joblib" if is_bundle else root / f"{model_id}.joblib"
        try:
            metadata = load_metadata(metadata_path)
            _validate_metadata(metadata)
            availability = _model_availability(model_path, metadata)
            test_metrics = metadata.get("metrics", {}).get("test", {})
            models.append(
                {
                    "model_id": model_id,
                    "target": str(metadata["target"]),
                    "model_name": str(metadata["model_name"]),
                    "created_at": str(metadata["created_at"]),
                    "test_metrics": {
                        "r2": test_metrics.get("r2"),
                        "rmse": test_metrics.get("rmse"),
                        "mae": test_metrics.get("mae"),
                    },
                    "feature_count": len(metadata["feature_columns"]),
                    "compatibility": model_schema_status(metadata),
                    "schema_version": metadata.get("schema_version"),
                    "model_type": metadata.get("model_type"),
                    "bundle_type": metadata.get("bundle_type"),
                    "selected_final_output": metadata.get("selected_final_output"),
                    "cv_summary": _compact_cv_summary(
                        metadata.get("cv_summary") or metadata.get("cv_protocol")
                    ),
                    "available_targets": _metadata_available_targets(metadata),
                    **availability,
                }
            )
            if not availability["available"]:
                warnings.append(
                    f"{model_id}: {availability['incompatibility_reason']}"
                )
        except Exception as exc:
            logger.warning(
                "모델 메타데이터 제외: %s",
                model_id,
                exc_info=exc,
            )
            warnings.append(
                f"{model_id}: 유효하지 않은 모델 메타데이터입니다."
            )
    models.sort(key=lambda item: str(item["created_at"]), reverse=True)
    return models, warnings


def get_prediction_model_detail(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    model_path, metadata_path = _model_paths(model_id, model_dir)
    if not metadata_path.is_file():
        raise InferenceInputError("존재하지 않는 모델 ID입니다.")
    try:
        metadata = load_metadata(metadata_path)
    except (OSError, ValueError) as exc:
        raise InferenceInputError(
            "유효하지 않은 모델 메타데이터입니다."
        ) from exc

    def normalized_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def normalized_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def metric_number(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    feature_names = [str(value) for value in normalized_list(metadata.get("feature_columns"))]
    metrics_source = normalized_dict(metadata.get("metrics"))
    for split_name, alias in (
        ("train", "train_metrics"),
        ("validation", "validation_metrics"),
        ("test", "test_metrics"),
    ):
        if not isinstance(metrics_source.get(split_name), dict):
            alias_value = metadata.get(alias)
            if isinstance(alias_value, dict):
                metrics_source[split_name] = alias_value
    metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        split_metrics = normalized_dict(metrics_source.get(split_name))
        if split_metrics:
            metrics[split_name] = {
                key: metric_number(split_metrics.get(key))
                for key in ("r2", "rmse", "mse", "mae")
            }

    dataset_split = {
        str(key): float(value)
        for key, value in normalized_dict(metadata.get("dataset_split")).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }
    dataset_rows = {
        str(key): int(value)
        for key, value in normalized_dict(metadata.get("dataset_rows")).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }
    row_aliases = {"train": "train_size", "validation": "validation_size", "test": "test_size"}
    for split_name, alias in row_aliases.items():
        alias_value = metadata.get(alias)
        if split_name not in dataset_rows and isinstance(alias_value, (int, float)) and not isinstance(alias_value, bool) and math.isfinite(alias_value):
            dataset_rows[split_name] = int(alias_value)
    preprocessing_config = normalized_dict(metadata.get("preprocessing_config"))
    target_ensemble_configs = metadata.get("target_ensemble_configs")
    if not isinstance(target_ensemble_configs, dict):
        target_ensemble_configs = normalized_dict(metadata.get("ensemble_config"))
    target_metrics = normalized_dict(metadata.get("target_metrics"))
    available_targets = _metadata_available_targets(metadata)
    availability = _model_availability(model_path, metadata)
    return {
        "model_id": model_id,
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "model_version": metadata.get("model_version"),
        "created_at": metadata.get("created_at"),
        "target": metadata.get("target"),
        "feature_count": metadata.get("feature_count", len(feature_names)),
        "feature_names": feature_names,
        "dataset_split": dataset_split,
        "dataset_rows": dataset_rows,
        "metrics": metrics,
        "random_seed": metadata.get("random_state"),
        "split_method": metadata.get("split_method"),
        "preprocessing_version": metadata.get("preprocessing_version"),
        "preprocessing_config": preprocessing_config,
        "training_time_seconds": metadata.get("training_time_seconds"),
        "source_filename": metadata.get("source_filename"),
        "model_file": metadata.get("model_file", model_path.name),
        "metadata_file": metadata.get(
            "metadata_file",
            metadata_path.name,
        ),
        "storage_status": (
            "available" if model_path.is_file() else "model_file_missing"
        ),
        "champion": metadata.get("champion"),
        "sklearn_version": (
            metadata.get("sklearn_version")
            or metadata.get("scikit_learn_version")
        ),
        "compatibility": model_schema_status(metadata),
        "schema_version": metadata.get("schema_version"),
        "schema_fingerprint": metadata.get("schema_fingerprint"),
        "config_parser_version": metadata.get("config_parser_version"),
        "missing_indicator_used": metadata.get("missing_indicator_used"),
        "outlier_policy": metadata.get("outlier_policy"),
        "group_column": metadata.get("group_column"),
        "target_leakage_check": normalized_dict(metadata.get("target_leakage_check")),
        "ensemble_enabled": metadata.get("ensemble_enabled"),
        "ensemble_mode": metadata.get("ensemble_mode"),
        "ensemble_method": metadata.get("ensemble_method"),
        "target_ensemble_configs": target_ensemble_configs,
        "target_metrics": target_metrics,
        "outer_fold_metrics": normalized_list(metadata.get("outer_fold_metrics")),
        "inner_fold_metrics": normalized_list(metadata.get("inner_fold_metrics")),
        "available_targets": available_targets,
        "cv_summary": _compact_cv_summary(
            metadata.get("cv_summary") or metadata.get("cv_protocol")
        ),
        "ensemble_weights": normalized_dict(metadata.get("ensemble_weights")),
        "hybrid_summary": normalized_dict(metadata.get("hybrid_summary")),
        "risk_metrics": normalized_dict(metadata.get("risk_metrics")),
        "preprocessing_summary": normalized_dict(metadata.get("preprocessing_summary")),
        "training_config": normalized_dict(metadata.get("training_config")),
        "model_agreement_summary": normalized_dict(metadata.get("model_agreement_summary")),
        "production_ensemble_retrained": metadata.get("production_ensemble_retrained"),
        **availability,
    }


def prepare_inference_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    *,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()
    if duplicate_columns:
        raise InferenceInputError(
            "예측 데이터에 중복된 컬럼명이 있습니다: "
            + ", ".join(dict.fromkeys(duplicate_columns))
        )
    ordered_features = list(dict.fromkeys(feature_columns))
    missing_features = [
        column for column in ordered_features
        if column not in dataframe.columns
    ]
    if missing_features and not allow_missing:
        raise InferenceInputError(
            "예측에 필요한 feature가 누락되었습니다: "
            + ", ".join(missing_features)
        )

    schema = load_data_schema()
    detected = detect_feature_columns(list(dataframe.columns), schema)
    detected_features = set(
        [
            *detected["r_columns"],
            *detected["d_columns"],
            *detected["eq_columns"],
        ]
    )
    extra_features = [
        column
        for column in dataframe.columns
        if column in detected_features and column not in ordered_features
    ]
    warnings: list[str] = []
    if extra_features:
        warnings.append(
            "학습에 사용하지 않은 feature를 제외했습니다: "
            + ", ".join(extra_features)
        )
    if allow_missing:
        features = dataframe.reindex(columns=ordered_features)
    else:
        features = dataframe.loc[:, ordered_features]
    # Copy only the selected model features, never the complete upload frame.
    features = features.copy()
    if features.empty:
        raise InferenceInputError("유효한 예측 행이 없습니다.")
    return features, warnings


def _identifier_values(
    dataframe: pd.DataFrame,
) -> tuple[str, pd.Series]:
    schema = load_data_schema()
    id_column = schema["id_column"]
    if id_column in dataframe.columns:
        return id_column, dataframe[id_column].astype("string")
    row_ids = pd.Series(
        [
            f"ROW_{index:06d}"
            for index in range(1, len(dataframe) + 1)
        ],
        index=dataframe.index,
        dtype="string",
    )
    return "row_id", row_ids


def _identifier_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _wafer_slot(value: Any) -> int | None:
    text = _identifier_text(value)
    if text is None:
        return None
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return int(numeric)
    match = re.search(r"(?:WAFER|WF|W)?[_-]?(\d+)$", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_lot_wafer_id(value: Any) -> tuple[str | None, str | None, int | None]:
    text = _identifier_text(value)
    if text is None:
        return None, None, None
    match = re.match(
        r"^(?P<lot>.+?)[_-]?(?:WAFER|WF|W)[_-]?(?P<slot>\d+)$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None, None
    lot = match.group("lot").rstrip("_-") or None
    slot_text = match.group("slot")
    slot = int(slot_text)
    wafer = f"W{slot:0{max(2, len(slot_text))}d}"
    return lot, wafer, slot


def _canonical_identifiers(
    source_row: pd.Series,
    identifier: Any,
    identifier_column: str,
) -> dict[str, Any]:
    combined = (
        _identifier_text(source_row.get("Lot_Wafer_ID"))
        or _identifier_text(source_row.get("lot_wafer_id"))
        or (_identifier_text(identifier) if identifier_column != "row_id" else None)
    )
    parsed_lot, parsed_wafer, parsed_slot = _parse_lot_wafer_id(combined)
    lot = (
        _identifier_text(source_row.get("Lot_ID"))
        or _identifier_text(source_row.get("lot_id"))
        or parsed_lot
    )
    source_wafer = (
        _identifier_text(source_row.get("Wafer_ID"))
        or _identifier_text(source_row.get("wafer_id"))
    )
    source_slot = _wafer_slot(source_row.get("Wafer_Slot"))
    if source_slot is None:
        source_slot = _wafer_slot(source_row.get("wafer_slot"))
    slot = source_slot or _wafer_slot(source_wafer) or parsed_slot
    wafer = source_wafer or (f"W{slot:02d}" if slot is not None else None) or parsed_wafer
    if combined is None and lot is not None and wafer is not None:
        combined = f"{lot}_{wafer}"
    return {
        "Lot_Wafer_ID": combined,
        "Lot_ID": lot,
        "Wafer_ID": wafer,
        "Wafer_Slot": slot,
    }


def _finite_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def risk_class_confidence(
    critical_probability: float,
    warning_probability: float,
) -> float:
    """Return confidence across critical, warning-only, and normal classes."""
    critical = min(max(float(critical_probability), 0.0), 1.0)
    warning = min(max(float(warning_probability), 0.0), 1.0)
    warning_only = max(warning - critical, 0.0)
    normal = max(1.0 - warning, 0.0)
    return max(critical, warning_only, normal)


def _risk_level(
    prediction: float,
    warning_threshold: float,
    danger_threshold: float,
) -> str:
    if prediction >= warning_threshold:
        return "normal"
    if prediction >= danger_threshold:
        return "warning"
    return "danger"


def predict_dataframe(
    dataframe: pd.DataFrame,
    loaded: LoadedPredictionModel,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
    max_rows: int | None = MAX_PREDICTION_ROWS,
) -> PredictionResult:
    if (
        not math.isfinite(warning_threshold)
        or not math.isfinite(danger_threshold)
        or warning_threshold <= danger_threshold
    ):
        raise InferenceInputError(
            "주의 기준값은 위험 기준값보다 커야 합니다."
        )
    validation = validate_dataframe(
        dataframe,
        validation_mode="inference",
    )
    if not validation["is_valid"]:
        raise InferenceInputError(
            "예측 데이터 검증에 실패했습니다: "
            + " ".join(validation["errors"])
        )
    is_auto_pipeline = str(loaded.metadata.get("pipeline_version") or "").startswith("auto_")
    if is_auto_pipeline:
        processed = dataframe
        preprocessing_report = {"warnings": []}
    else:
        processed, preprocessing_report = preprocess_dataframe(dataframe)
    schema = load_data_schema()
    detected_raw = detect_feature_columns(list(dataframe.columns), schema)
    raw_features = list(dict.fromkeys([
        *detected_raw["r_columns"],
        *detected_raw["d_columns"],
        *detected_raw["eq_columns"],
        *detected_raw.get("config_columns", []),
    ]))
    compatibility = model_schema_status(loaded.metadata, raw_features)
    if compatibility == "incompatible" and not is_auto_pipeline:
        raise InferenceInputError(
            "선택한 모델은 현재 데이터 스키마와 호환되지 않습니다. "
            "신규 데이터로 모델을 다시 학습해 주세요."
        )
    feature_columns = list(loaded.metadata["feature_columns"])
    features, feature_warnings = prepare_inference_features(
        dataframe if is_auto_pipeline else processed,
        feature_columns,
        allow_missing=is_auto_pipeline,
    )
    if len(features) == 0:
        raise InferenceInputError("유효한 예측 행이 없습니다.")

    hybrid_components: dict[str, Any] | None = None
    try:
        if callable(getattr(loaded.model, "predict_components", None)):
            hybrid_components = loaded.model.predict_components(features)
            raw_predictions = np.asarray(
                hybrid_components["selected"],
                dtype=np.float32,
            )
        else:
            raw_predictions = np.asarray(
                loaded.model.predict(features),
                dtype=np.float32,
            )
    except Exception as exc:
        raise ModelLoadError("모델 예측 실행에 실패했습니다.") from exc
    if raw_predictions.ndim != 1 or len(raw_predictions) != len(features):
        raise ModelLoadError("모델 예측 결과의 행 수가 올바르지 않습니다.")
    if not np.isfinite(raw_predictions).all():
        raise ModelLoadError("모델 예측 결과에 유효하지 않은 값이 있습니다.")

    target = str(loaded.metadata["target"])
    display_predictions = raw_predictions
    warnings = list(
        dict.fromkeys(
            [
                *([] if is_auto_pipeline else preprocessing_report["warnings"]),
                *([] if is_auto_pipeline else feature_warnings),
            ]
        )
    )
    if compatibility == "legacy":
        warnings.append(
            "이 모델은 schema fingerprint가 없는 Legacy 모델입니다. "
            "정확한 feature 일치가 확인된 범위에서만 예측했습니다."
        )
    if target == "Y":
        clipped = np.clip(display_predictions, 0.0, 100.0)
        clipped_count = int(np.count_nonzero(clipped != display_predictions))
        if clipped_count:
            warnings.append(
                f"표시 범위를 위해 {clipped_count}개 Y 예측값을 "
                "0~100으로 조정했습니다."
            )
        display_predictions = clipped
    else:
        warnings.append(
            f"{target} 목표 변수의 위험 등급 정책이 정의되지 않아 "
            "위험 등급을 계산하지 않았습니다."
        )

    identifier_column, identifiers = _identifier_values(dataframe)
    actual_values: pd.Series | None = None
    evaluation: RegressionMetrics | None = None
    if target in dataframe.columns:
        actual_values = pd.to_numeric(
            dataframe[target],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        evaluation_mask = actual_values.notna()
        if evaluation_mask.any():
            evaluation = evaluate_regression(
                actual_values.loc[evaluation_mask],
                raw_predictions[evaluation_mask.to_numpy()],
            )

    prediction_column = f"predicted_{target}"
    prediction_rows: list[dict[str, Any]] = []
    normal_count = 0
    warning_count = 0
    danger_count = 0
    for position, (index, identifier) in enumerate(identifiers.items()):
        prediction = float(display_predictions[position])
        identifier_value = None if pd.isna(identifier) else str(identifier)
        row: dict[str, Any] = {
            **_canonical_identifiers(
                dataframe.loc[index],
                identifier_value,
                identifier_column,
            ),
            identifier_column: identifier_value,
            prediction_column: prediction,
        }
        if hybrid_components is not None:
            failure_rates = {
                target_name: float(hybrid_components["targets"][target_name][position])
                for target_name in ("Y1", "Y2", "Y3", "Y4", "Y5")
            }
            fail_bit_counts = {
                target_name: value
                for target_name in ("Y6", "Y7", "Y8", "Y9", "Y10")
                if (
                    value := _finite_float(dataframe.loc[index, target_name])
                    if target_name in dataframe.columns
                    else None
                ) is not None
            }
            critical_probability = float(hybrid_components["critical_probability"][position])
            warning_probability = float(hybrid_components["warning_probability"][position])
            confidence_probability = risk_class_confidence(
                critical_probability,
                warning_probability,
            )
            row.update({
                "failure_rates": failure_rates,
                "fail_bit_counts": fail_bit_counts,
                "critical_probability": critical_probability,
                "warning_probability": warning_probability,
                "confidence": (
                    "high" if confidence_probability >= 0.8
                    else "medium" if confidence_probability >= 0.6
                    else "low"
                ),
                "warnings": [],
            })
            row.update(failure_rates)
            row.update(fail_bit_counts)
        if target == "Y":
            risk = _risk_level(
                prediction,
                warning_threshold,
                danger_threshold,
            )
            row["risk_level"] = risk
            if risk == "normal":
                normal_count += 1
            elif risk == "warning":
                warning_count += 1
            else:
                danger_count += 1
        else:
            row["risk_level"] = None

        if actual_values is not None:
            actual = _finite_float(actual_values.loc[index])
            row[f"actual_{target}"] = actual
            if actual is not None:
                residual = actual - float(raw_predictions[position])
                row["residual"] = residual
                row["absolute_error"] = abs(residual)
            else:
                row["residual"] = None
                row["absolute_error"] = None
        prediction_rows.append(row)

    truncated = (
        max_rows is not None and len(prediction_rows) > max_rows
    )
    if truncated:
        warnings.append(
            f"화면 응답은 최대 {max_rows}행까지만 표시합니다."
        )
        prediction_rows = prediction_rows[:max_rows]

    preprocessing_audit: dict[str, Any] = {}
    if is_auto_pipeline:
        audit_model = loaded.model
        model_for_target = getattr(audit_model, "model_for_target", None)
        if callable(model_for_target):
            audit_model = model_for_target("Y1")
        feature_step = getattr(audit_model, "named_steps", {}).get("features")
        audit = getattr(feature_step, "audit", None)
        if callable(audit):
            preprocessing_audit = audit(dataframe)

    return PredictionResult(
        model_id=loaded.model_id,
        target=target,
        model_name=str(loaded.metadata["model_name"]),
        identifier_column=identifier_column,
        predictions=prediction_rows,
        total_rows=len(dataframe),
        average_prediction=float(np.mean(display_predictions)),
        normal_count=normal_count,
        warning_count=warning_count,
        danger_count=danger_count,
        evaluation=evaluation,
        warnings=warnings,
        truncated=truncated,
        preprocessing_summary={
            "pipeline_version": loaded.metadata.get("pipeline_version"),
            "config_parser_version": loaded.metadata.get("config_parser_version"),
            "schema_version": loaded.metadata.get("schema_version"),
            "measurement_coverage": {
                "r": validation.get("r_measurement_coverage", 0.0),
                "d": validation.get("d_measurement_coverage", 0.0),
            },
            **_metadata_dict(loaded.metadata.get("preprocessing_summary")),
            "missing_handling": loaded.metadata.get("preprocessing_strategy"),
            "missing_indicator": loaded.metadata.get("missing_indicator_used"),
            "outlier_policy": loaded.metadata.get("outlier_policy"),
            "policy": _metadata_dict(loaded.metadata.get("preprocessing_config")),
            "missing_input_features": [
                column for column in feature_columns if column not in dataframe.columns
            ] if is_auto_pipeline else [],
            "ignored_extra_features": [
                column for column in raw_features if column not in feature_columns
            ] if is_auto_pipeline else [],
            **preprocessing_audit,
        },
    )
