from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from api.settings import settings


logger = logging.getLogger(__name__)
_lock = threading.RLock()


class RuntimeStore:
    def __init__(
        self,
        path: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.path = Path(path or settings.runtime_db_path)
        self.artifact_root = Path(
            artifact_root
            or (
                settings.runtime_artifact_dir
                if path is None
                else self.path.parent
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with _lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    model_id TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    row_count INTEGER,
                    status TEXT NOT NULL,
                    error_type TEXT,
                    critical_count INTEGER,
                    warning_count INTEGER,
                    schema_version TEXT,
                    filename TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_completed ON runs(completed_at DESC);
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    source_filename TEXT,
                    result_json TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_training_jobs_created
                ON training_jobs(created_at DESC);
                CREATE TABLE IF NOT EXISTS migration_registry (
                    migration_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    details_json TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS model_slots (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    active_model_id TEXT, pipeline_version TEXT, promoted_at TEXT,
                    dataset_version INTEGER, previous_model_id TEXT, status TEXT NOT NULL DEFAULT 'empty',
                    rollback_json TEXT NOT NULL DEFAULT '[]', active_metadata_json TEXT
                );
                INSERT OR IGNORE INTO model_slots(singleton,status) VALUES(1,'empty');
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    lot_min TEXT,
                    lot_max TEXT,
                    lot_count INTEGER,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    unmapped_columns_json TEXT NOT NULL DEFAULT '[]',
                    schema_diff_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_datasets_uploaded
                ON datasets(uploaded_at DESC);
                CREATE TABLE IF NOT EXISTS app_state (
                    state_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def active_model(self) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM model_slots WHERE singleton=1").fetchone()
        if row is None or not row["active_model_id"]:
            return None
        value = dict(row)
        value["rollback_model_ids"] = json.loads(value.pop("rollback_json") or "[]")
        value["metadata"] = json.loads(value.pop("active_metadata_json") or "{}")
        return value

    def promote_model(self, *, model_id: str, pipeline_version: str, dataset_version: int, metadata: dict[str, Any]) -> dict[str, Any]:
        """Atomically switch only the pointer; model files are never overwritten."""
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT active_model_id,rollback_json FROM model_slots WHERE singleton=1").fetchone()
            rollbacks = json.loads(current["rollback_json"] or "[]") if current else []
            previous = current["active_model_id"] if current else None
            if previous and previous != model_id:
                rollbacks = [previous, *[item for item in rollbacks if item != previous]][:2]
            connection.execute("""UPDATE model_slots SET active_model_id=?,pipeline_version=?,promoted_at=?,dataset_version=?,previous_model_id=?,status='active',rollback_json=?,active_metadata_json=? WHERE singleton=1""", (model_id, pipeline_version, now, dataset_version, previous, self._json(rollbacks), self._json(metadata)))
        return self.active_model() or {}

    def migration_status(self, migration_id: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM migration_registry WHERE migration_id=?",
                (migration_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        raw_details = result.pop("details_json", None)
        try:
            result["details"] = json.loads(raw_details) if raw_details else None
        except json.JSONDecodeError:
            result["details"] = None
        return result

    def start_migration(self, migration_id: str) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO migration_registry
                (migration_id,status,started_at,completed_at,details_json,error_message)
                VALUES (?, 'running', ?, NULL, NULL, NULL)
                ON CONFLICT(migration_id) DO UPDATE SET
                status='running',started_at=excluded.started_at,completed_at=NULL,
                details_json=NULL,error_message=NULL""",
                (migration_id, started_at),
            )

    def complete_migration(self, migration_id: str, details: dict[str, Any]) -> None:
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE migration_registry SET status='completed',completed_at=?,
                details_json=?,error_message=NULL WHERE migration_id=?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    self._json(details),
                    migration_id,
                ),
            )

    def fail_migration(self, migration_id: str, message: str) -> None:
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE migration_registry SET status='failed',completed_at=?,
                error_message=? WHERE migration_id=?""",
                (datetime.now(timezone.utc).isoformat(), str(message)[:1000], migration_id),
            )

    @staticmethod
    def _json(value: Any) -> str:
        def clean(item: Any) -> Any:
            if isinstance(item, dict):
                return {str(key): clean(entry) for key, entry in item.items()}
            if isinstance(item, (list, tuple)):
                return [clean(entry) for entry in item]
            if isinstance(item, float) and (item != item or item in {float("inf"), float("-inf")}):
                return None
            if hasattr(item, "item"):
                try:
                    return clean(item.item())
                except (TypeError, ValueError):
                    pass
            return item
        return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, default=str)

    def create_training_job(
        self,
        job_id: str,
        *,
        source_filename: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO training_jobs
                (job_id,created_at,status,stage,progress,source_filename)
                VALUES (?,?,?,?,?,?)""",
                (
                    job_id,
                    now,
                    "queued",
                    "학습 준비",
                    0,
                    Path(source_filename).name,
                ),
            )

    def start_training_job(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE training_jobs SET
                status='running',started_at=?,stage='데이터 검증',progress=5,
                error_message=NULL
                WHERE job_id=? AND status='queued'""",
                (now, job_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("학습 Job을 running 상태로 전환하지 못했습니다.")

    def update_training_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
    ) -> None:
        normalized_progress = max(0, min(int(progress), 99))
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE training_jobs SET stage=?,progress=?
                WHERE job_id=? AND status='running'""",
                (stage, normalized_progress, job_id),
            )

    def complete_training_job(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE training_jobs SET
                status='completed',completed_at=?,stage='학습 완료',progress=100,
                result_json=?,error_message=NULL
                WHERE job_id=? AND status='running'""",
                (now, self._json(result), job_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("학습 Job 완료 상태를 저장하지 못했습니다.")

    def fail_training_job(self, job_id: str, message: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE training_jobs SET
                status='failed',completed_at=?,stage='학습 실패',
                error_message=?,result_json=NULL
                WHERE job_id=? AND status IN ('queued','running')""",
                (now, str(message)[:1000], job_id),
            )

    def interrupt_training_jobs(self) -> list[str]:
        """Recover work that cannot survive a single-worker process restart."""
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT job_id FROM training_jobs
                WHERE status IN ('queued','running')"""
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            if job_ids:
                connection.execute(
                    """UPDATE training_jobs SET
                    status='interrupted',completed_at=?,stage='서버 재시작으로 중단',
                    error_message='서버가 재시작되어 학습이 중단되었습니다.',
                    result_json=NULL
                    WHERE status IN ('queued','running')""",
                    (now,),
                )
            return job_ids

    def get_training_job(self, job_id: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        raw_result = result.pop("result_json", None)
        try:
            result["result"] = json.loads(raw_result) if raw_result else None
        except (TypeError, json.JSONDecodeError):
            result["result"] = None
        return result

    def record_run(self, **values: Any) -> str:
        run_id = str(values.get("run_id") or f"run_{uuid4().hex}")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "run_id": run_id,
            "event_type": values["event_type"],
            "model_id": values.get("model_id"),
            "started_at": values.get("started_at") or now,
            "completed_at": values.get("completed_at") or now,
            "duration_ms": float(values.get("duration_ms") or 0.0),
            "row_count": values.get("row_count"),
            "status": values.get("status", "success"),
            "error_type": values.get("error_type"),
            "critical_count": values.get("critical_count"),
            "warning_count": values.get("warning_count"),
            "schema_version": values.get("schema_version"),
            "filename": Path(str(values["filename"])).name if values.get("filename") else None,
        }
        with _lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (:run_id,:event_type,:model_id,:started_at,:completed_at,:duration_ms,:row_count,:status,:error_type,:critical_count,:warning_count,:schema_version,:filename)",
                record,
            )
        return run_id

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY completed_at DESC LIMIT ?", (min(max(limit, 1), 500),)).fetchall()
        return [dict(row) for row in rows]

    def model_reference_counts(self, model_id: str) -> dict[str, int]:
        with _lock, self._connect() as connection:
            prediction_count = int(connection.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id = ? AND event_type IN ('predict','prediction')",
                (model_id,),
            ).fetchone()[0])
            analysis_count = int(connection.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id = ? AND event_type IN ('explain','analyze','report')",
                (model_id,),
            ).fetchone()[0])
        return {"prediction_history_count": prediction_count, "analysis_history_count": analysis_count}

    def create_dataset(self, **values: Any) -> None:
        record = {
            "dataset_id": values["dataset_id"],
            "original_filename": values["original_filename"],
            "stored_path": values["stored_path"],
            "uploaded_at": values.get("uploaded_at") or datetime.now(timezone.utc).isoformat(),
            "row_count": int(values["row_count"]),
            "column_count": int(values["column_count"]),
            "lot_min": values.get("lot_min"),
            "lot_max": values.get("lot_max"),
            "lot_count": values.get("lot_count"),
            "warnings_json": self._json(list(values.get("warnings") or [])),
            "unmapped_columns_json": self._json(list(values.get("unmapped_columns") or [])),
            "schema_diff_json": self._json(dict(values.get("schema_diff") or {})),
        }
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO datasets
                (dataset_id,original_filename,stored_path,uploaded_at,row_count,column_count,
                 lot_min,lot_max,lot_count,warnings_json,unmapped_columns_json,schema_diff_json)
                VALUES (:dataset_id,:original_filename,:stored_path,:uploaded_at,:row_count,:column_count,
                        :lot_min,:lot_max,:lot_count,:warnings_json,:unmapped_columns_json,:schema_diff_json)""",
                record,
            )

    @staticmethod
    def _decode_dataset_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["warnings"] = json.loads(result.pop("warnings_json") or "[]")
        result["unmapped_columns"] = json.loads(result.pop("unmapped_columns_json") or "[]")
        result["schema_diff"] = json.loads(result.pop("schema_diff_json") or "{}")
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC").fetchall()
        return [self._decode_dataset_row(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        return self._decode_dataset_row(row) if row is not None else None

    def delete_dataset(self, dataset_id: str) -> bool:
        with _lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM datasets WHERE dataset_id=?", (dataset_id,))
            return cursor.rowcount > 0

    # -- Generic key-value state (학습/원인 분석/사전 알람 "최근 결과 1개"
    # persistence -- one row per kind, overwritten on every fresh save, no
    # dedicated table per kind by design). --

    def set_app_state(self, state_key: str, value: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO app_state (state_key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
                (state_key, self._json(value), now),
            )

    def get_app_state(self, state_key: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM app_state WHERE state_key=?", (state_key,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return None

    def get_all_app_state(self, state_keys: list[str]) -> dict[str, dict[str, Any] | None]:
        if not state_keys:
            return {}
        with _lock, self._connect() as connection:
            placeholders = ",".join("?" for _ in state_keys)
            rows = connection.execute(
                f"SELECT state_key, value_json FROM app_state WHERE state_key IN ({placeholders})",
                state_keys,
            ).fetchall()
        found: dict[str, dict[str, Any] | None] = {}
        for row in rows:
            try:
                found[row["state_key"]] = json.loads(row["value_json"])
            except json.JSONDecodeError:
                found[row["state_key"]] = None
        return {key: found.get(key) for key in state_keys}

    def delete_app_state(self, state_key: str) -> bool:
        with _lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM app_state WHERE state_key=?", (state_key,))
            return cursor.rowcount > 0


def safe_runtime_call(method: str, **values: Any) -> Any:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        return getattr(RuntimeStore(), method)(**values)
    except Exception:
        logger.warning("Runtime dashboard 저장 실패: %s", method, exc_info=True)
        return None
