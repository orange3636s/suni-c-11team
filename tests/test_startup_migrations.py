from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from src.ml.hybrid import PIPELINE_VERSION
from src.runtime.migrations import (
    LEGACY_MODEL_MIGRATION_ID,
    REPORT_MIGRATION_ID,
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


def _analysis_with_report_fields(store: RuntimeStore) -> str:
    analysis_id = "analysis_migration_case"
    store.start_analysis(
        analysis_id=analysis_id,
        prediction_id=None,
        source_filename="migration.csv",
        model_id="current_model",
    )
    store.complete_analysis(
        analysis_id,
        metadata={
            "duration_ms": 1.0,
            "dataset_fingerprint": "migration",
            "model_name_snapshot": "Current",
            "model_version_snapshot": PIPELINE_VERSION,
            "model_type_snapshot": "hybrid_multi_y",
            "schema_version": "semicon_yield_v2",
            "row_count": 2,
            "lot_count": 1,
            "available_targets_json": '["Y1", "Y2", "Y3", "Y4", "Y5"]',
            "default_target": "Y1",
        },
        summary={"average_predicted_yield": 90.0},
        methodology={},
        artifact={
            "analysis_result": {
                "report": {"legacy": True},
                "nested": {"report_snapshot": "old", "keep": "yes"},
            },
            "keep": [1, 2, 3],
        },
        warnings=[],
    )
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE analysis_runs SET report_snapshot_available=1 WHERE analysis_id=?",
            (analysis_id,),
        )
    return analysis_id


def test_startup_migrations_delete_only_legacy_assets_and_preserve_history(
    migration_root: Path,
) -> None:
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    _flat_model(model_root, "legacy_model", "old_pipeline_v1")
    _flat_model(model_root, "current_model", PIPELINE_VERSION)
    analysis_id = _analysis_with_report_fields(store)
    report_root = artifact_root / "reports"
    report_root.mkdir()
    (report_root / "report_old.json").write_text("{}", encoding="utf-8")
    (report_root / "keep.txt").write_text("preserve", encoding="utf-8")

    results = run_startup_migrations(model_dir=model_root, store=store)

    assert [row["status"] for row in results] == ["completed", "completed"]
    assert not (model_root / "legacy_model.joblib").exists()
    assert not (model_root / "legacy_model.json").exists()
    assert not (model_root / "current_model.joblib").exists()
    assert not (model_root / "current_model.json").exists()
    assert not (report_root / "report_old.json").exists()
    assert (report_root / "keep.txt").read_text(encoding="utf-8") == "preserve"

    detail = store.get_analysis(analysis_id)
    assert detail is not None
    assert detail["artifact"]["keep"] == [1, 2, 3]
    assert detail["artifact"]["analysis_result"]["nested"] == {"keep": "yes"}
    assert "report" not in detail["artifact"]["analysis_result"]
    with closing(sqlite3.connect(store.path)) as connection:
        marker = connection.execute(
            "SELECT report_snapshot_available FROM analysis_runs WHERE analysis_id=?",
            (analysis_id,),
        ).fetchone()[0]
    assert marker == 0
    assert store.migration_status(LEGACY_MODEL_MIGRATION_ID)["status"] == "completed"
    assert store.migration_status(REPORT_MIGRATION_ID)["status"] == "completed"


def test_completed_startup_migrations_do_not_run_again(migration_root: Path) -> None:
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    first = run_startup_migrations(model_dir=model_root, store=store)
    assert [row["status"] for row in first] == ["completed", "completed"]

    _flat_model(model_root, "created_after_migration", "old_pipeline_v1")
    second = run_startup_migrations(model_dir=model_root, store=store)

    assert [row["status"] for row in second] == [
        "already_completed",
        "already_completed",
    ]
    assert (model_root / "created_after_migration.joblib").is_file()
    assert (model_root / "created_after_migration.json").is_file()
