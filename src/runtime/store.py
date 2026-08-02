from __future__ import annotations

import json
import gzip
import logging
import os
import re
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
_HISTORY_ID = re.compile(r"^(prediction|analysis)_[A-Za-z0-9_-]+$")


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
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT,
                    lot_wafer_id TEXT NOT NULL,
                    lot_id TEXT,
                    predicted_y REAL,
                    direct_y REAL,
                    derived_y REAL,
                    critical_probability REAL,
                    warning_probability REAL,
                    risk_level TEXT NOT NULL,
                    confidence TEXT,
                    top_failure_target TEXT,
                    top_feature TEXT,
                    top_step TEXT,
                    top_equipment TEXT,
                    status TEXT NOT NULL DEFAULT 'New',
                    external_delivery_status TEXT NOT NULL DEFAULT 'Not Configured',
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    UNIQUE(analysis_id, lot_wafer_id, risk_level, model_id)
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    prediction_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    source_filename TEXT,
                    dataset_fingerprint TEXT,
                    model_id TEXT,
                    model_name_snapshot TEXT,
                    model_version_snapshot TEXT,
                    model_type_snapshot TEXT,
                    schema_version TEXT,
                    row_count INTEGER,
                    lot_count INTEGER,
                    warning_threshold REAL,
                    critical_threshold REAL,
                    final_strategy TEXT,
                    summary_json TEXT,
                    preprocessing_json TEXT,
                    artifact_path TEXT,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_prediction_runs_created
                ON prediction_runs(created_at DESC);
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    analysis_id TEXT PRIMARY KEY,
                    prediction_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    source_filename TEXT,
                    dataset_fingerprint TEXT,
                    model_id TEXT,
                    model_name_snapshot TEXT,
                    model_version_snapshot TEXT,
                    model_type_snapshot TEXT,
                    schema_version TEXT,
                    row_count INTEGER,
                    lot_count INTEGER,
                    available_targets_json TEXT,
                    default_target TEXT,
                    summary_json TEXT,
                    methodology_json TEXT,
                    artifact_path TEXT,
                    report_snapshot_available INTEGER NOT NULL DEFAULT 0,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_runs_created
                ON analysis_runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_analysis_prediction
                ON analysis_runs(prediction_id, created_at DESC);
                """
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

    def _artifact_path(self, history_id: str, kind: str) -> Path:
        if not _HISTORY_ID.fullmatch(history_id):
            raise ValueError("유효하지 않은 이력 ID입니다.")
        expected = "prediction" if kind == "predictions" else "analysis"
        if not history_id.startswith(f"{expected}_"):
            raise ValueError("이력 ID와 Artifact 유형이 일치하지 않습니다.")
        return self.artifact_root / kind / f"{history_id}.json.gz"

    def _write_artifact(self, history_id: str, kind: str, payload: dict[str, Any]) -> str:
        path = self._artifact_path(history_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with gzip.open(temporary, "wt", encoding="utf-8") as output:
                output.write(self._json(payload))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return str(path.resolve())

    @staticmethod
    def _decode_json_fields(row: dict[str, Any]) -> dict[str, Any]:
        decode_errors: list[str] = []
        for key in (
            "summary_json", "preprocessing_json", "available_targets_json",
            "methodology_json",
        ):
            if key in row:
                raw = row.pop(key)
                decoded_key = key.removesuffix("_json")
                try:
                    row[decoded_key] = json.loads(raw) if raw else None
                except (TypeError, json.JSONDecodeError):
                    row[decoded_key] = None
                    decode_errors.append(decoded_key)
        if decode_errors:
            row["metadata_decode_errors"] = decode_errors
        return row

    def _resolve_artifact(
        self,
        path: str | None,
        *,
        history_id: str | None = None,
        kind: str | None = None,
    ) -> Path | None:
        root = self.artifact_root.resolve()
        candidates: list[Path] = []
        if path:
            candidates.append(Path(path))
        if history_id and kind:
            try:
                canonical = self._artifact_path(history_id, kind)
            except ValueError:
                canonical = None
            if canonical is not None and all(canonical != item for item in candidates):
                candidates.append(canonical)
        for candidate in candidates:
            resolved = candidate.resolve()
            if root in resolved.parents and resolved.is_file():
                return resolved
        return None

    def _read_artifact_state(
        self,
        path: str | None,
        *,
        history_id: str | None = None,
        kind: str | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        resolved = self._resolve_artifact(
            path,
            history_id=history_id,
            kind=kind,
        )
        if resolved is None:
            return None, "missing"
        try:
            with gzip.open(resolved, "rt", encoding="utf-8") as source:
                value = json.load(source)
            if not isinstance(value, dict):
                logger.warning("Runtime Artifact root must be an object: %s", resolved)
                return None, "corrupted"
            return value, "available"
        except (OSError, json.JSONDecodeError):
            logger.warning("Runtime Artifact 읽기 실패: %s", resolved, exc_info=True)
            return None, "corrupted"

    def _read_artifact(self, path: str | None) -> dict[str, Any] | None:
        artifact, _ = self._read_artifact_state(path)
        return artifact

    def start_prediction(self, **values: Any) -> str:
        prediction_id = str(values.get("prediction_id") or f"prediction_{uuid4().hex}")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "prediction_id": prediction_id,
            "created_at": values.get("created_at") or now,
            "started_at": values.get("started_at") or now,
            "status": "running",
            "source_filename": Path(str(values.get("source_filename") or "")).name or None,
            "model_id": values.get("model_id"),
            "warning_threshold": values.get("warning_threshold"),
            "critical_threshold": values.get("critical_threshold"),
        }
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO prediction_runs
                (prediction_id,created_at,started_at,status,source_filename,model_id,warning_threshold,critical_threshold)
                VALUES (:prediction_id,:created_at,:started_at,:status,:source_filename,:model_id,:warning_threshold,:critical_threshold)""",
                record,
            )
        return prediction_id

    def complete_prediction(
        self,
        prediction_id: str,
        *,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        preprocessing: dict[str, Any] | None,
        artifact: dict[str, Any],
        warnings: list[str],
    ) -> bool:
        artifact_path = self._write_artifact(prediction_id, "predictions", artifact)
        now = datetime.now(timezone.utc).isoformat()
        values = {
            **metadata,
            "prediction_id": prediction_id,
            "completed_at": now,
            "duration_ms": metadata.get("duration_ms"),
            "status": "completed",
            "summary_json": self._json(summary),
            "preprocessing_json": self._json(preprocessing or {}),
            "artifact_path": artifact_path,
            "warning_count": len(warnings),
        }
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE prediction_runs SET
                completed_at=:completed_at,duration_ms=:duration_ms,status=:status,
                dataset_fingerprint=:dataset_fingerprint,model_name_snapshot=:model_name_snapshot,
                model_version_snapshot=:model_version_snapshot,model_type_snapshot=:model_type_snapshot,
                schema_version=:schema_version,row_count=:row_count,lot_count=:lot_count,
                final_strategy=:final_strategy,summary_json=:summary_json,
                preprocessing_json=:preprocessing_json,artifact_path=:artifact_path,
                warning_count=:warning_count,error_message=NULL
                WHERE prediction_id=:prediction_id""",
                values,
            )
        return True

    def fail_prediction(self, prediction_id: str, message: str) -> None:
        with _lock, self._connect() as connection:
            connection.execute(
                "UPDATE prediction_runs SET status='failed',completed_at=?,error_message=? WHERE prediction_id=?",
                (datetime.now(timezone.utc).isoformat(), message, prediction_id),
            )

    def start_analysis(self, **values: Any) -> str:
        analysis_id = str(values.get("analysis_id") or f"analysis_{uuid4().hex}")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "analysis_id": analysis_id,
            "prediction_id": values.get("prediction_id"),
            "created_at": values.get("created_at") or now,
            "started_at": values.get("started_at") or now,
            "status": "running",
            "source_filename": Path(str(values.get("source_filename") or "")).name or None,
            "model_id": values.get("model_id"),
        }
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO analysis_runs
                (analysis_id,prediction_id,created_at,started_at,status,source_filename,model_id)
                VALUES (:analysis_id,:prediction_id,:created_at,:started_at,:status,:source_filename,:model_id)""",
                record,
            )
        return analysis_id

    def complete_analysis(
        self,
        analysis_id: str,
        *,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        methodology: dict[str, Any],
        artifact: dict[str, Any],
        warnings: list[str],
    ) -> bool:
        artifact_path = self._write_artifact(analysis_id, "analyses", artifact)
        values = {
            **metadata,
            "analysis_id": analysis_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "summary_json": self._json(summary),
            "methodology_json": self._json(methodology),
            "artifact_path": artifact_path,
            "warning_count": len(warnings),
        }
        with _lock, self._connect() as connection:
            connection.execute(
                """UPDATE analysis_runs SET
                completed_at=:completed_at,duration_ms=:duration_ms,status=:status,
                dataset_fingerprint=:dataset_fingerprint,model_name_snapshot=:model_name_snapshot,
                model_version_snapshot=:model_version_snapshot,model_type_snapshot=:model_type_snapshot,
                schema_version=:schema_version,row_count=:row_count,lot_count=:lot_count,
                available_targets_json=:available_targets_json,default_target=:default_target,
                summary_json=:summary_json,methodology_json=:methodology_json,
                artifact_path=:artifact_path,report_snapshot_available=:report_snapshot_available,
                warning_count=:warning_count,error_message=NULL
                WHERE analysis_id=:analysis_id""",
                values,
            )
        return True

    def fail_analysis(self, analysis_id: str, message: str) -> None:
        with _lock, self._connect() as connection:
            connection.execute(
                "UPDATE analysis_runs SET status='failed',completed_at=?,error_message=? WHERE analysis_id=?",
                (datetime.now(timezone.utc).isoformat(), message, analysis_id),
            )

    def _list_history(self, table: str, id_column: str, filters: dict[str, Any]) -> dict[str, Any]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for key in ("model_id", "status"):
            if filters.get(key):
                clauses.append(f"{key} = :{key}")
                params[key] = filters[key]
        if filters.get("prediction_id") and table == "analysis_runs":
            clauses.append("prediction_id = :prediction_id")
            params["prediction_id"] = filters["prediction_id"]
        if filters.get("filename"):
            clauses.append("source_filename LIKE :filename")
            params["filename"] = f"%{filters['filename']}%"
        if filters.get("search"):
            clauses.append(
                f"({id_column} LIKE :search OR source_filename LIKE :search "
                "OR model_id LIKE :search OR model_name_snapshot LIKE :search)"
            )
            params["search"] = f"%{filters['search']}%"
        if filters.get("date_from"):
            clauses.append("created_at >= :date_from")
            params["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            clauses.append("created_at <= :date_to")
            params["date_to"] = filters["date_to"]
        if filters.get("target") and table == "analysis_runs":
            clauses.append("default_target = :target")
            params["target"] = filters["target"]
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "ASC" if filters.get("sort") == "oldest" else "DESC"
        limit = min(max(int(filters.get("limit", 50)), 1), 200)
        offset = max(int(filters.get("offset", 0)), 0)
        with _lock, self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM {table}{where}", params).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM {table}{where} "
                f"ORDER BY COALESCE(completed_at, created_at) {order}, created_at {order} "
                "LIMIT :limit OFFSET :offset",
                {**params, "limit": limit, "offset": offset},
            ).fetchall()
        items = [self._decode_json_fields(dict(row)) for row in rows]
        for item in items:
            kind = "analyses" if table == "analysis_runs" else "predictions"
            history_id = str(item.get(id_column) or "")
            item["artifact_available"] = self._resolve_artifact(
                item.get("artifact_path"),
                history_id=history_id,
                kind=kind,
            ) is not None
            item["model_name"] = item.get("model_name_snapshot")
            summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
            item["average_predicted_yield"] = summary.get("average_predicted_yield")
            item["critical_count"] = summary.get("critical_count")
            item["warning_wafer_count"] = summary.get("warning_count")
            item["top_failure_target"] = summary.get("top_failure_target")
            if item.get("status") == "completed" and not item["artifact_available"]:
                item["status"] = "artifact_missing"
            item.pop("artifact_path", None)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def list_predictions(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self._list_history("prediction_runs", "prediction_id", filters)

    def list_analyses(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self._list_history("analysis_runs", "analysis_id", filters)

    def _get_history(self, table: str, id_column: str, history_id: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE {id_column}=?", (history_id,)
            ).fetchone()
            if row is None:
                return None
            linked_count = 0
            source_deleted = False
            if table == "prediction_runs":
                linked_count = int(connection.execute(
                    "SELECT COUNT(*) FROM analysis_runs WHERE prediction_id=?", (history_id,)
                ).fetchone()[0])
            elif row["prediction_id"]:
                source_deleted = connection.execute(
                    "SELECT 1 FROM prediction_runs WHERE prediction_id=?", (row["prediction_id"],)
                ).fetchone() is None
        metadata = self._decode_json_fields(dict(row))
        kind = "analyses" if table == "analysis_runs" else "predictions"
        artifact, artifact_state = self._read_artifact_state(
            metadata.get("artifact_path"),
            history_id=history_id,
            kind=kind,
        )
        metadata["artifact_status"] = artifact_state
        metadata["artifact_available"] = artifact_state == "available"
        if artifact is None and metadata.get("status") in {"completed", "partial"}:
            metadata["status"] = (
                "artifact_corrupted" if artifact_state == "corrupted" else "artifact_missing"
            )
        metadata.pop("artifact_path", None)
        return {
            "metadata": metadata,
            "artifact": artifact,
            "linked_analysis_count": linked_count,
            "source_prediction_deleted": source_deleted,
        }

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        return self._get_history("prediction_runs", "prediction_id", prediction_id)

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        return self._get_history("analysis_runs", "analysis_id", analysis_id)

    def _delete_history(self, table: str, id_column: str, history_id: str) -> bool:
        with _lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT artifact_path FROM {table} WHERE {id_column}=?", (history_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute(f"DELETE FROM {table} WHERE {id_column}=?", (history_id,))
        if row["artifact_path"]:
            path = Path(row["artifact_path"]).resolve()
            if self.artifact_root.resolve() in path.parents and path.is_file():
                path.unlink()
        return True

    def delete_prediction(self, prediction_id: str) -> bool:
        return self._delete_history("prediction_runs", "prediction_id", prediction_id)

    def delete_analysis(self, analysis_id: str) -> bool:
        return self._delete_history("analysis_runs", "analysis_id", analysis_id)

    def latest_completed(self, kind: str) -> dict[str, Any] | None:
        table, id_column = (
            ("analysis_runs", "analysis_id") if kind == "analysis"
            else ("prediction_runs", "prediction_id")
        )
        with _lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT {id_column} FROM {table} WHERE status='completed' "
                "ORDER BY COALESCE(completed_at, created_at) DESC, created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        getter = self.get_analysis if kind == "analysis" else self.get_prediction
        return getter(row[id_column])

    def latest_analysis_for_overview(self) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT analysis_id FROM analysis_runs "
                "WHERE status='completed' "
                "ORDER BY COALESCE(completed_at, created_at) DESC, created_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT analysis_id FROM analysis_runs "
                    "WHERE status='partial' "
                    "ORDER BY COALESCE(completed_at, created_at) DESC, created_at DESC LIMIT 1"
                ).fetchone()
        return self.get_analysis(row["analysis_id"]) if row is not None else None

    def artifact_usage(self) -> dict[str, Any]:
        result: dict[str, Any] = {"total_bytes": 0, "prediction_count": 0, "analysis_count": 0}
        for kind, key in (("predictions", "prediction_count"), ("analyses", "analysis_count")):
            directory = self.artifact_root / kind
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json.gz"):
                try:
                    result["total_bytes"] += path.stat().st_size
                    result[key] += 1
                except OSError:
                    continue
        result["total_mb"] = round(result["total_bytes"] / (1024 * 1024), 3)
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

    def record_prediction_alerts(
        self,
        *,
        analysis_id: str,
        model_id: str,
        model_version: str | None,
        predictions: list[dict[str, Any]],
        identifier_column: str,
    ) -> int:
        created = 0
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            for row in predictions:
                risk = row.get("risk_level")
                if risk not in {"danger", "warning"}:
                    continue
                identifier = str(row.get(identifier_column) or "")
                lot_id = identifier.rsplit("_", 1)[0] if "_" in identifier else None
                failure_rates = row.get("failure_rates") if isinstance(row.get("failure_rates"), dict) else {}
                fail_counts = row.get("fail_bit_counts") if isinstance(row.get("fail_bit_counts"), dict) else {}
                combined = {**failure_rates, **fail_counts}
                top_failure = max(combined, key=combined.get) if combined else None
                values = {
                    "alert_id": f"alert_{uuid4().hex}", "created_at": now,
                    "analysis_id": analysis_id, "model_id": model_id,
                    "model_version": model_version, "lot_wafer_id": identifier,
                    "lot_id": lot_id, "predicted_y": row.get("predicted_Y"),
                    "direct_y": row.get("direct_y"), "derived_y": row.get("derived_y"),
                    "critical_probability": row.get("critical_probability"),
                    "warning_probability": row.get("warning_probability"),
                    "risk_level": risk, "confidence": row.get("confidence"),
                    "top_failure_target": top_failure, "top_feature": row.get("top_feature"),
                    "top_step": row.get("top_step"), "top_equipment": row.get("top_equipment"),
                }
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO alerts
                    (alert_id,created_at,analysis_id,model_id,model_version,lot_wafer_id,lot_id,predicted_y,direct_y,derived_y,critical_probability,warning_probability,risk_level,confidence,top_failure_target,top_feature,top_step,top_equipment)
                    VALUES (:alert_id,:created_at,:analysis_id,:model_id,:model_version,:lot_wafer_id,:lot_id,:predicted_y,:direct_y,:derived_y,:critical_probability,:warning_probability,:risk_level,:confidence,:top_failure_target,:top_feature,:top_step,:top_equipment)""",
                    values,
                )
                created += int(cursor.rowcount > 0)
        return created

    def list_alerts(self, filters: dict[str, Any]) -> dict[str, Any]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for key, column in {
            "risk_level": "risk_level", "status": "status", "model_id": "model_id", "lot_id": "lot_id",
        }.items():
            if filters.get(key):
                clauses.append(f"{column} = :{key}")
                params[key] = filters[key]
        if filters.get("wafer_id"):
            clauses.append("lot_wafer_id LIKE :wafer_id")
            params["wafer_id"] = f"%{filters['wafer_id']}%"
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "risk_level ASC, created_at DESC" if filters.get("sort") == "risk" else "created_at DESC"
        limit = min(max(int(filters.get("limit", 50)), 1), 200)
        offset = max(int(filters.get("offset", 0)), 0)
        with _lock, self._connect() as connection:
            total = connection.execute(f"SELECT COUNT(*) FROM alerts{where}", params).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM alerts{where} ORDER BY {order} LIMIT :limit OFFSET :offset",
                {**params, "limit": limit, "offset": offset},
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        with _lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def update_alert(self, alert_id: str, status: str) -> dict[str, Any] | None:
        if status not in {"New", "Acknowledged", "Resolved"}:
            raise ValueError("지원하지 않는 알람 상태입니다.")
        now = datetime.now(timezone.utc).isoformat()
        acknowledged = now if status in {"Acknowledged", "Resolved"} else None
        resolved = now if status == "Resolved" else None
        with _lock, self._connect() as connection:
            connection.execute(
                "UPDATE alerts SET status=?, acknowledged_at=COALESCE(acknowledged_at, ?), resolved_at=? WHERE alert_id=?",
                (status, acknowledged, resolved, alert_id),
            )
        return self.get_alert(alert_id)

    def alert_summary(self) -> dict[str, int]:
        with _lock, self._connect() as connection:
            rows = connection.execute("SELECT status, risk_level, external_delivery_status, COUNT(*) count FROM alerts GROUP BY status,risk_level,external_delivery_status").fetchall()
        summary = {"total": 0, "new_count": 0, "acknowledged_count": 0, "resolved_count": 0, "critical_count": 0, "warning_count": 0, "external_not_configured_count": 0}
        for row in rows:
            count = int(row["count"]); summary["total"] += count
            if row["status"] == "New": summary["new_count"] += count
            elif row["status"] == "Acknowledged": summary["acknowledged_count"] += count
            elif row["status"] == "Resolved": summary["resolved_count"] += count
            if row["risk_level"] == "danger": summary["critical_count"] += count
            elif row["risk_level"] == "warning": summary["warning_count"] += count
            if row["external_delivery_status"] == "Not Configured": summary["external_not_configured_count"] += count
        return summary

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY completed_at DESC LIMIT ?", (min(max(limit, 1), 500),)).fetchall()
        return [dict(row) for row in rows]

    def model_reference_counts(self, model_id: str) -> dict[str, int]:
        with _lock, self._connect() as connection:
            prediction_count = int(connection.execute(
                "SELECT COUNT(*) FROM prediction_runs WHERE model_id = ?",
                (model_id,),
            ).fetchone()[0])
            analysis_count = int(connection.execute(
                "SELECT COUNT(*) FROM analysis_runs WHERE model_id = ?",
                (model_id,),
            ).fetchone()[0])
            prediction_count += int(connection.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id = ? AND event_type IN ('predict','prediction')",
                (model_id,),
            ).fetchone()[0])
            analysis_count += int(connection.execute(
                "SELECT COUNT(*) FROM runs WHERE model_id = ? AND event_type IN ('explain','analyze','report')",
                (model_id,),
            ).fetchone()[0])
        return {"prediction_history_count": prediction_count, "analysis_history_count": analysis_count}


def safe_runtime_call(method: str, **values: Any) -> Any:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        return getattr(RuntimeStore(), method)(**values)
    except Exception:
        logger.warning("Runtime dashboard 저장 실패: %s", method, exc_info=True)
        return None
