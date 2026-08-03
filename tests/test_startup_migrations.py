from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from src.ml.hybrid import PIPELINE_VERSION
from src.runtime.migrations import (
    LEGACY_MODEL_MIGRATION_ID,
    run_startup_migrations,
)
from src.runtime.store import RuntimeStore


@pytest.fixture
def migration_root() -> Path:
    parent = Path(__file__).parent / ".tmp_startup_migrations"
    root = parent / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def _flat_model(root: Path, model_id: str, pipeline_version: str) -> None:
    (root / f"{model_id}.joblib").write_bytes(b"test-artifact")
    (root / f"{model_id}.json").write_text(
        json.dumps(
            {
                "pipeline_version": pipeline_version,
                "model_type": "hybrid_multi_y",
                "schema_version": "semicon_yield_v2",
                "split_method": "fixed_lot_holdout_70_15_15",
            }
        ),
        encoding="utf-8",
    )


def test_startup_migrations_delete_only_legacy_models(
    migration_root: Path,
) -> None:
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    _flat_model(model_root, "legacy_model", "old_pipeline_v1")
    _flat_model(model_root, "current_model", PIPELINE_VERSION)

    results = run_startup_migrations(model_dir=model_root, store=store)

    assert [row["status"] for row in results] == ["completed"]
    assert not (model_root / "legacy_model.joblib").exists()
    assert not (model_root / "legacy_model.json").exists()
    assert not (model_root / "current_model.joblib").exists()
    assert not (model_root / "current_model.json").exists()
    assert store.migration_status(LEGACY_MODEL_MIGRATION_ID)["status"] == "completed"


def test_completed_startup_migrations_do_not_run_again(migration_root: Path) -> None:
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    first = run_startup_migrations(model_dir=model_root, store=store)
    assert [row["status"] for row in first] == ["completed"]

    _flat_model(model_root, "created_after_migration", "old_pipeline_v1")
    second = run_startup_migrations(model_dir=model_root, store=store)

    assert [row["status"] for row in second] == ["already_completed"]
    assert (model_root / "created_after_migration.joblib").is_file()
    assert (model_root / "created_after_migration.json").is_file()
