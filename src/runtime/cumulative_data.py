"""Persistent, append-safe process data used by Champion--Challenger training.

The operational database is deliberately the source of truth for the compact
canonical rows.  It avoids loading or replacing an uploaded CSV wholesale and
keeps ingestion, delayed labels, and metadata updates in one SQLite
transaction.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
import pandas as pd

from src.ml.hybrid import FAIL_RATE_TARGETS, detect_auto_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _canon(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return the required strongest available stable Wafer identifier."""
    def value(*names: str) -> str:
        for name in names:
            if name in row and _canon(row[name]):
                return _canon(row[name])
        return ""
    combined = value("Lot_Wafer_ID", "lot_wafer_id")
    if combined:
        return "lot_wafer", combined
    lot, slot = value("Lot_ID", "lot_id"), value("Wafer_Slot", "wafer_slot")
    if lot and slot:
        return "lot_slot", f"{lot}\x1f{slot}"
    wafer = value("Wafer_ID", "wafer_id")
    if lot and wafer:
        return "lot_wafer_id", f"{lot}\x1f{wafer}"
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return "row_hash", hashlib.sha256(canonical.encode()).hexdigest()


class CumulativeDataStore:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
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
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS process_wafers (
              wafer_key TEXT PRIMARY KEY, key_type TEXT NOT NULL,
              lot_id TEXT, payload_json TEXT NOT NULL, feature_hash TEXT NOT NULL,
              label_hash TEXT, label_status TEXT NOT NULL,
              source_batch_id TEXT NOT NULL, ingested_at TEXT NOT NULL,
              process_timestamp TEXT, label_updated_at TEXT,
              conflict_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_process_wafers_status
              ON process_wafers(label_status, ingested_at);
            CREATE TABLE IF NOT EXISTS process_dataset_meta (
              singleton INTEGER PRIMARY KEY CHECK(singleton=1), dataset_version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO process_dataset_meta(singleton,dataset_version) VALUES(1,0);
            """)

    @staticmethod
    def _labels(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: payload.get(key) for key in ["Y", *[f"Y{i}" for i in range(1, 11)]] if payload.get(key) is not None}

    @staticmethod
    def _features(payload: dict[str, Any]) -> dict[str, Any]:
        schema = detect_auto_schema(pd.DataFrame([payload]))
        names = [*schema["feature_columns"], *schema["identifier_columns"]]
        return {key: payload.get(key) for key in names if key in payload and payload.get(key) is not None}

    @staticmethod
    def _hash(value: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

    def ingest(self, dataframe: pd.DataFrame, *, source_batch_id: str | None = None) -> dict[str, Any]:
        batch_id = source_batch_id or f"batch_{uuid4().hex}"
        counts = {"inserted_rows": 0, "updated_label_rows": 0, "duplicate_rows": 0, "conflict_rows": 0, "rejected_rows": 0}
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = False
            for _, source in dataframe.iterrows():
                payload = {str(key): _clean(value) for key, value in source.to_dict().items()}
                key_type, wafer_key = _row_key(payload)
                labels = self._labels(payload)
                features = self._features(payload)
                feature_hash, label_hash = self._hash(features), self._hash(labels)
                existing = db.execute("SELECT * FROM process_wafers WHERE wafer_key=?", (wafer_key,)).fetchone()
                if existing is None:
                    status = "labeled" if all(payload.get(target) is not None for target in FAIL_RATE_TARGETS) else "pending_label"
                    db.execute("""INSERT INTO process_wafers
                      (wafer_key,key_type,lot_id,payload_json,feature_hash,label_hash,label_status,source_batch_id,ingested_at,process_timestamp,label_updated_at,conflict_reason)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)""", (wafer_key, key_type, _canon(payload.get("Lot_ID") or payload.get("lot_id")) or None, json.dumps(payload, ensure_ascii=False, default=str), feature_hash, label_hash, status, batch_id, now, payload.get("process_timestamp"), now if labels else None))
                    counts["inserted_rows"] += 1; changed = True; continue
                old = dict(existing); old_payload = json.loads(old["payload_json"])
                if old["feature_hash"] != feature_hash and features:
                    db.execute("UPDATE process_wafers SET label_status='conflict', conflict_reason='feature_conflict' WHERE wafer_key=?", (wafer_key,))
                    counts["conflict_rows"] += 1; changed = True; continue
                old_labels = self._labels(old_payload)
                if labels and old_labels and old["label_hash"] != label_hash:
                    db.execute("UPDATE process_wafers SET label_status='conflict', conflict_reason='label_conflict' WHERE wafer_key=?", (wafer_key,))
                    counts["conflict_rows"] += 1; changed = True; continue
                if labels and not old_labels:
                    merged = {**old_payload, **{key: value for key, value in labels.items() if value is not None}}
                    status = "labeled" if all(merged.get(target) is not None for target in FAIL_RATE_TARGETS) else "pending_label"
                    db.execute("UPDATE process_wafers SET payload_json=?,label_hash=?,label_status=?,label_updated_at=?,source_batch_id=? WHERE wafer_key=?", (json.dumps(merged, ensure_ascii=False, default=str), self._hash(self._labels(merged)), status, now, batch_id, wafer_key))
                    counts["updated_label_rows"] += 1; changed = True; continue
                counts["duplicate_rows"] += 1
            if changed:
                db.execute("UPDATE process_dataset_meta SET dataset_version=dataset_version+1 WHERE singleton=1")
            version = int(db.execute("SELECT dataset_version FROM process_dataset_meta WHERE singleton=1").fetchone()[0])
        status = self.status()
        return {"success": True, "source_batch_id": batch_id, "dataset_version": f"dataset_{version}", "received_rows": int(len(dataframe)), **counts, **status}

    def status(self, *, active_dataset_version: int | None = None, active_promoted_at: str | None = None) -> dict[str, Any]:
        with self._connect() as db:
            rows = db.execute("SELECT label_status,lot_id,ingested_at,label_updated_at FROM process_wafers").fetchall()
            version = int(db.execute("SELECT dataset_version FROM process_dataset_meta WHERE singleton=1").fetchone()[0])
        labeled = [row for row in rows if row["label_status"] == "labeled"]
        since = [row for row in labeled if not active_promoted_at or (row["label_updated_at"] or row["ingested_at"]) > active_promoted_at]
        return {"dataset_version": f"dataset_{version}", "dataset_version_number": version, "total_rows": len(rows), "total_lots": len({row["lot_id"] for row in rows if row["lot_id"]}), "labeled_rows": len(labeled), "pending_label_rows": sum(row["label_status"] == "pending_label" for row in rows), "conflict_rows": sum(row["label_status"] == "conflict" for row in rows), "new_labeled_rows_since_active_model": len(since), "new_lots_since_active_model": len({row["lot_id"] for row in since if row["lot_id"]}), "active_model_dataset_version": f"dataset_{active_dataset_version}" if active_dataset_version is not None else None}

    def training_frame(self, maximum_rows: int = 20_000) -> tuple[pd.DataFrame, dict[str, int]]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json,ingested_at FROM process_wafers WHERE label_status='labeled' ORDER BY ingested_at DESC").fetchall()
        recent = rows[:15_000]
        # Preserve older potentially severe examples without duplicating recent rows.
        older = rows[15_000:]
        rare = []
        for row in older:
            value = json.loads(row["payload_json"]); y = value.get("Y")
            if (isinstance(y, (int, float)) and y < 85) or any((value.get(t) or 0) > 5 for t in FAIL_RATE_TARGETS):
                rare.append(row)
            if len(rare) >= 5_000: break
        chosen = [*recent, *rare][:maximum_rows]
        return pd.DataFrame([json.loads(row["payload_json"]) for row in reversed(chosen)]), {"recent_sample_count": len(recent), "rare_sample_count": len(rare), "critical_sample_count": len(rare), "total_training_rows": len(chosen), "total_training_lots": len({json.loads(row["payload_json"]).get("Lot_ID") for row in chosen})}
