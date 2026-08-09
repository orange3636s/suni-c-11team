from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from src.ml.hybrid import PIPELINE_VERSION
from src.runtime.migrations import (
    LEGACY_ALARM_DEFAULTS_MIGRATION_ID,
    LEGACY_MODEL_MIGRATION_ID,
    normalize_legacy_alarm_defaults,
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


def _flat_model(root: Path, model_id: str, pipeline_version: str, *, target: str = "Y") -> None:
    (root / f"{model_id}.joblib").write_bytes(b"test-artifact")
    (root / f"{model_id}.json").write_text(
        json.dumps(
            {
                "pipeline_version": pipeline_version,
                "target": target,
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
    """A-2 회귀: 현재 파이프라인 버전을 단 champion 번들(target="Y",
    model_type="hybrid_multi_y" -- src/ml/hybrid.py의 실제 메타데이터
    모양)은 살아남아야 한다. 이전에는 "hybrid"가 model_type에 있다는
    이유만으로 target/pipeline_version과 무관하게 legacy로 오판·삭제
    됐다(migration_registry가 비거나 볼륨이 바뀐 배포에서 기동 시
    챔피언 모델이 사라지는 치명 결함)."""
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    _flat_model(model_root, "legacy_model", "old_pipeline_v1")
    _flat_model(model_root, "current_model", PIPELINE_VERSION)

    results = run_startup_migrations(model_dir=model_root, store=store)

    # 지시서 JD-2①: 이 store에서 처음 도는 것이므로 legacy 모델 정리
    # 마이그레이션과 함께 alarms 기본값 정리 마이그레이션도 완료된다
    # (alarms 레코드가 없으니 "kept/deleted" 판단 없이 그냥 완료).
    assert [row["status"] for row in results] == ["completed", "completed"]
    assert not (model_root / "legacy_model.joblib").exists()
    assert not (model_root / "legacy_model.json").exists()
    assert (model_root / "current_model.joblib").exists()
    assert (model_root / "current_model.json").exists()
    assert store.migration_status(LEGACY_MODEL_MIGRATION_ID)["status"] == "completed"


def test_completed_startup_migrations_do_not_run_again(migration_root: Path) -> None:
    model_root = migration_root / "models"
    artifact_root = migration_root / "runtime"
    model_root.mkdir()
    store = RuntimeStore(artifact_root / "dashboard.db", artifact_root)
    first = run_startup_migrations(model_dir=model_root, store=store)
    assert [row["status"] for row in first] == ["completed", "completed"]

    _flat_model(model_root, "created_after_migration", "old_pipeline_v1")
    second = run_startup_migrations(model_dir=model_root, store=store)

    assert [row["status"] for row in second] == ["already_completed", "already_completed"]
    assert (model_root / "created_after_migration.joblib").is_file()
    assert (model_root / "created_after_migration.json").is_file()


def _runtime_store(migration_root: Path) -> RuntimeStore:
    artifact_root = migration_root / "runtime"
    return RuntimeStore(artifact_root / "dashboard.db", artifact_root)


def test_normalize_legacy_alarm_defaults_deletes_exact_legacy_match(migration_root: Path) -> None:
    """지시서 JD-2①: 기본값이 88.0/0.20으로 바뀌기 전에 저장된 정확히
    91.0/0.5인 레코드만 지운다."""
    store = _runtime_store(migration_root)
    store.set_app_state(
        "latest_alarms",
        {
            "schema_version": 1,
            "created_at": "2026-08-07T17:18:08.895535+00:00",
            "train_dataset": "train",
            "eval_dataset": "train",
            "payload": {"targetYield": 91, "sensitivity": 0.5},
        },
    )

    result = normalize_legacy_alarm_defaults(store)

    assert result["action"] == "deleted"
    assert store.get_app_state("latest_alarms") is None


def test_normalize_legacy_alarm_defaults_keeps_real_user_choice(migration_root: Path) -> None:
    """91.0/0.5와 다른 값은 사용자가 실제로 고른 것일 수 있어 보존한다."""
    store = _runtime_store(migration_root)
    store.set_app_state(
        "latest_alarms",
        {
            "schema_version": 1,
            "created_at": "2026-08-08T00:00:00+00:00",
            "train_dataset": "train",
            "eval_dataset": "test",
            "payload": {"targetYield": 93.0, "sensitivity": 0.5},
        },
    )

    result = normalize_legacy_alarm_defaults(store)

    assert result["action"] == "kept"
    assert store.get_app_state("latest_alarms") is not None


def test_normalize_legacy_alarm_defaults_noop_when_no_record(migration_root: Path) -> None:
    store = _runtime_store(migration_root)
    result = normalize_legacy_alarm_defaults(store)
    assert result["action"] == "none"


def test_normalize_legacy_alarm_defaults_migration_runs_once(migration_root: Path) -> None:
    model_root = migration_root / "models"
    model_root.mkdir()
    store = _runtime_store(migration_root)
    store.set_app_state(
        "latest_alarms",
        {
            "schema_version": 1,
            "created_at": "2026-08-07T17:18:08.895535+00:00",
            "train_dataset": "train",
            "eval_dataset": "train",
            "payload": {"targetYield": 91, "sensitivity": 0.5},
        },
    )

    first = run_startup_migrations(model_dir=model_root, store=store)
    alarm_row = next(row for row in first if row["migration_id"] == LEGACY_ALARM_DEFAULTS_MIGRATION_ID)
    assert alarm_row["status"] == "completed"
    assert alarm_row["details"]["action"] == "deleted"
    assert store.get_app_state("latest_alarms") is None

    # 사용자가 다시 저장해도(마이그레이션이 이미 완료로 기록됐으므로)
    # 두 번째 실행이 그 레코드를 건드리지 않는다 (하지 말 것: 두 번 실행).
    store.set_app_state(
        "latest_alarms",
        {
            "schema_version": 1,
            "created_at": "2026-08-09T00:00:00+00:00",
            "train_dataset": "train",
            "eval_dataset": "test",
            "payload": {"targetYield": 91, "sensitivity": 0.5},
        },
    )
    second = run_startup_migrations(model_dir=model_root, store=store)
    alarm_row_2 = next(row for row in second if row["migration_id"] == LEGACY_ALARM_DEFAULTS_MIGRATION_ID)
    assert alarm_row_2["status"] == "already_completed"
    assert store.get_app_state("latest_alarms") is not None
