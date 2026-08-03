from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.ml.hybrid import PIPELINE_VERSION
from src.ml.inference import MODEL_ID_PATTERN, delete_model_bundle
from src.runtime.store import RuntimeStore


logger = logging.getLogger(__name__)
LEGACY_MODEL_MIGRATION_ID = "cleanup_legacy_models_v2"


def _metadata_candidates(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir() or root.is_symlink():
        return []
    candidates: list[tuple[str, Path]] = []
    for entry in root.iterdir():
        if entry.is_symlink():
            continue
        if entry.is_file() and entry.suffix == ".json":
            model_id = entry.stem
            if MODEL_ID_PATTERN.fullmatch(model_id):
                candidates.append((model_id, entry))
        elif entry.is_dir() and MODEL_ID_PATTERN.fullmatch(entry.name):
            metadata = entry / "metadata.json"
            if metadata.is_file() and not metadata.is_symlink():
                candidates.append((entry.name, metadata))
    return candidates


def _legacy_metadata(metadata: dict[str, Any]) -> bool:
    # Only former multi-target bundles are incompatible.  Direct Y pipelines
    # produced by the current app must survive startup untouched.
    target = str(metadata.get("target") or "")
    model_type = " ".join(
        str(metadata.get(key) or "")
        for key in ("model_type", "bundle_type", "model_name")
    ).lower()
    if target == "Y" and "multi" not in model_type and "hybrid" not in model_type:
        return False
    if "multi" in model_type or "hybrid" in model_type or target.startswith("Y"):
        return True
    pipeline = str(metadata.get("pipeline_version") or "").strip()
    if pipeline == PIPELINE_VERSION:
        return False
    algorithm = " ".join(
        str(metadata.get(key) or "")
        for key in ("algorithm", "model_name", "model_type", "bundle_type")
    ).lower()
    split = str(metadata.get("split_method") or metadata.get("cv_protocol") or "").lower()
    schema_version = str(metadata.get("metadata_schema_version") or metadata.get("schema_version") or "")
    return bool(
        pipeline != PIPELINE_VERSION
        or "xgboost" in algorithm
        or "xgb" in algorithm
        or "kfold" in split
        or "fold" in split
        or schema_version not in {"semicon_yield_v2", ""}
    )


def cleanup_legacy_models(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve()
    deleted: list[str] = []
    skipped: list[str] = []
    invalid: list[str] = []
    for model_id, metadata_path in _metadata_candidates(root):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
        except (OSError, ValueError, json.JSONDecodeError):
            invalid.append(model_id)
            continue
        if not _legacy_metadata(metadata):
            skipped.append(model_id)
            continue
        result = delete_model_bundle(model_id, root)
        if result.deleted_files:
            deleted.append(model_id)
    return {
        "deleted_model_ids": deleted,
        "deleted_model_count": len(deleted),
        "preserved_current_model_ids": skipped,
        "invalid_metadata_model_ids": invalid,
        "decision_basis": [
            "pipeline_version",
            "algorithm",
            "split_method",
            "metadata_schema_version",
        ],
    }


def _run_once(
    store: RuntimeStore,
    migration_id: str,
    action: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    current = store.migration_status(migration_id)
    if current is not None and current.get("status") == "completed":
        return {"migration_id": migration_id, "status": "already_completed", "details": current.get("details")}
    store.start_migration(migration_id)
    try:
        details = action()
        store.complete_migration(migration_id, details)
        return {"migration_id": migration_id, "status": "completed", "details": details}
    except Exception as exc:
        store.fail_migration(migration_id, str(exc))
        logger.exception("Startup Migration 실패: %s", migration_id)
        return {"migration_id": migration_id, "status": "failed", "error": str(exc)}


def run_startup_migrations(
    *,
    model_dir: str | Path,
    store: RuntimeStore,
) -> list[dict[str, Any]]:
    return [
        _run_once(
            store,
            LEGACY_MODEL_MIGRATION_ID,
            lambda: cleanup_legacy_models(model_dir),
        ),
    ]
