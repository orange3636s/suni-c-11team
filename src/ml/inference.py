from __future__ import annotations

import importlib.util
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema
from src.ml.model_io import (
    DEFAULT_MODEL_DIR,
    load_metadata,
    load_model,
)
from src.schema_compatibility import model_schema_status


logger = logging.getLogger(__name__)

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


_PREDICTION_MODEL_CACHE_SIZE = 2


@lru_cache(maxsize=_PREDICTION_MODEL_CACHE_SIZE)
def _load_prediction_model_cached(
    model_id: str,
    model_path_str: str,
    metadata_path_str: str,
    model_mtime_ns: int,
    metadata_mtime_ns: int,
) -> LoadedPredictionModel:
    # mtime_ns 두 개는 캐시 키에만 쓰인다(_read_bundled_csv와 같은 패턴,
    # src/runtime/datasets.py) -- 재학습으로 같은 model_id 파일이
    # 덮어써져도 캐시가 stale 모델을 계속 돌려주지 않는다.
    del model_mtime_ns, metadata_mtime_ns
    model_path = Path(model_path_str)
    metadata_path = Path(metadata_path_str)
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


def load_prediction_model(
    model_id: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> LoadedPredictionModel:
    model_path, metadata_path = _model_paths(model_id, model_dir)
    if not metadata_path.is_file() or not model_path.is_file():
        raise InferenceInputError(
            "존재하지 않거나 사용할 수 없는 모델 ID입니다."
        )
    # target_hydration의 산점도 choke point(캐시 조회 전
    # 단계)가 요청마다 이 함수를 부른다 -- joblib.load + JSON 파싱을 매번
    # 다시 하지 않도록 파일이 안 바뀐 동안은(mtime 동일) 로드 결과를
    # 재사용한다.
    return _load_prediction_model_cached(
        model_id,
        str(model_path),
        str(metadata_path),
        model_path.stat().st_mtime_ns,
        metadata_path.stat().st_mtime_ns,
    )


def get_latest_model_metadata(store: Any) -> dict[str, Any] | None:
    active = store.active_model()
    if not active:
        return None
    metadata = dict(active.get("metadata") or {})
    return {
        **metadata,
        "model_id": active.get("active_model_id"),
        "version": active.get("pipeline_version"),
        "trained_at": active.get("promoted_at"),
        "target": "Y",
    }


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
    # model_id는 위에서 이미 `_validate_model_id`를 통과했으므로, 여기서
    # `_model_paths`가 여전히 `InvalidModelIdError`를 던질 수 있는 유일한
    # 경로는 "flat 파일이 심볼릭 링크를 통해 root 밖으로 resolve된다"는
    # 삭제 안전성 문제뿐이다 -- 잘못된 ID 형식과는 다른 오류이므로
    # 삭제 경로에서는 `ModelDeletionError`로 다시 던져 호출자가 구분하게
    # 한다.
    try:
        model_path, metadata_path = _model_paths(model_id, root)
    except InvalidModelIdError as exc:
        raise ModelDeletionError(str(exc)) from exc
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


# 보관 정책 -- 최근 이 개수만큼의 세트(+ 현재 챔피언)만 남긴다. 학습
# 라이브러리가 바뀌면 오래된 세트는 로드조차 안 되므로 무기한 쌓아둘
# 이유가 없다.
MODEL_RETENTION_KEEP = 3


def enforce_model_retention(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    keep: int = MODEL_RETENTION_KEEP,
    active_model_id: str | None = None,
) -> list[str]:
    """새 모델을 저장한 직후 호출한다 -- `created_at` 내림차순으로 최근
    `keep`개(기본 3)만 남기고 나머지를 지운다. `active_model_id`(현재
    챔피언)는 그 순위와 무관하게 항상 보존한다 -- 롤백 대상이 사라지면
    안 되기 때문이다. 삭제마다 로그를 남긴다."""
    models, _warnings = list_prediction_models(model_dir)
    keep_ids = {item["model_id"] for item in models[:keep]}
    if active_model_id:
        keep_ids.add(active_model_id)
    deleted: list[str] = []
    for item in models:
        model_id = item["model_id"]
        if model_id in keep_ids:
            continue
        try:
            delete_model_bundle(model_id, model_dir)
        except Exception:
            logger.exception("models: 보관 정책에 따른 삭제 실패: %s", model_id)
            continue
        deleted.append(model_id)
        logger.info("models: %s 삭제 (보관 %d개 초과)", model_id, keep)
    return deleted


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
        "target_metrics": target_metrics,
        "outer_fold_metrics": normalized_list(metadata.get("outer_fold_metrics")),
        "inner_fold_metrics": normalized_list(metadata.get("inner_fold_metrics")),
        "available_targets": available_targets,
        "cv_summary": _compact_cv_summary(
            metadata.get("cv_summary") or metadata.get("cv_protocol")
        ),
        "hybrid_summary": normalized_dict(metadata.get("hybrid_summary")),
        "risk_metrics": normalized_dict(metadata.get("risk_metrics")),
        "preprocessing_summary": normalized_dict(metadata.get("preprocessing_summary")),
        "training_config": normalized_dict(metadata.get("training_config")),
        "model_agreement_summary": normalized_dict(metadata.get("model_agreement_summary")),
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
