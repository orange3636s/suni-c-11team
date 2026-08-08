from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from api.settings import settings


logger = logging.getLogger(__name__)
_lock = threading.RLock()

# 자동 수집 파이프라인 §2-1: 홀드아웃 R²가 이만큼 이내로 떨어지는 건
# 노이즈로 보고 승격을 막지 않는다. 이보다 크게 떨어지면 게이트 미달.
PROMOTION_TOLERANCE = 0.005

# J-3: 자동 갱신 스냅샷 스키마 버전 -- 필드 모양이 호환되지 않게 바뀌면
# 올린다. 복원 시 이 값이 다르면 옛 스냅샷을 쓰지 않는다(그대로 쓰면
# 백엔드 로직이 바뀐 뒤에도 옛 스냅샷이 새 화면을 덮어쓴다).
REFRESH_SNAPSHOT_SCHEMA_VERSION = 3
REFRESH_SNAPSHOT_STATE_KEY = "automation:refresh_snapshot"

# W-2/W-6: 첫 기동 스냅샷 부트스트랩 -- 단일 실행 잠금과 진행 상태를
# app_state 테이블에 얹는다(전용 테이블을 새로 만들지 않는다). 잠금은
# 프로세스가 죽어 release가 호출되지 못한 경우를 대비해 일정 시간이
# 지나면 다른 프로세스가 가져갈 수 있게 한다(영구 데드락 방지).
BOOTSTRAP_LOCK_STATE_KEY = "automation:bootstrap_lock"
BOOTSTRAP_STATUS_STATE_KEY = "automation:bootstrap_status"
BOOTSTRAP_LOCK_STALE_SECONDS = 3600


def _favorite_dedupe_key(snapshot: dict[str, Any]) -> str:
    """D-1: 같은 (dataset, target, feature, viewType)는 같은 즐겨찾기로
    본다 -- viewType을 빼면 Box 뷰로 저장하려는 클릭이 기존 Scatter
    즐겨찾기와 같은 키로 잡혀 그것을 지워버린다(프런트가 dedupe 판단에
    쓰는 키와 반드시 같아야 한다, root-cause/page.tsx의 favoriteKeyOf)."""
    return "::".join(str(snapshot.get(field, "")) for field in ("dataset", "target", "feature", "viewType"))


class RuntimeStore:
    # E-1: 매 요청마다 새 RuntimeStore(...)를 만드는 라우트가 30여 곳이다
    # (Depends 기반 싱글턴으로 전부 바꾸는 건 이 배치 범위를 넘는 리팩터라
    # 하지 않는다) -- 진짜 비용은 인스턴스 생성 자체가 아니라
    # `_initialize()`가 매번 CREATE TABLE 9개 + ALTER TABLE 점검을 전역
    # `_lock` 아래 다시 실행해, 모든 설정 조회가 학습 잡의 쓰기와 경합하는
    # 것이다. DB 파일 경로별로 프로세스당 한 번만 실행되도록 막는다 --
    # 인스턴스는 여전히 가볍게 매번 새로 만들어지지만 DDL은 반복되지 않는다.
    _initialized_paths: set[str] = set()

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
        resolved = str(self.path.resolve())
        if resolved not in RuntimeStore._initialized_paths:
            with _lock:
                if resolved not in RuntimeStore._initialized_paths:
                    self._initialize()
                    RuntimeStore._initialized_paths.add(resolved)

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
                CREATE TABLE IF NOT EXISTS notify_sent_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    wafer_id TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    sent_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notify_sent_log_lookup
                ON notify_sent_log(dataset_id, wafer_id, sent_at DESC);
                CREATE TABLE IF NOT EXISTS promotion_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    candidate_model_id TEXT NOT NULL,
                    promoted INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    candidate_metric REAL,
                    active_metric REAL,
                    previous_model_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_promotion_events_created
                ON promotion_events(created_at DESC);
                CREATE TABLE IF NOT EXISTS favorites (
                    favorite_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_favorites_created
                ON favorites(created_at DESC);
                CREATE TABLE IF NOT EXISTS refresh_dispatch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    new_alarm_count INTEGER NOT NULL,
                    blocked_reason TEXT,
                    summarized INTEGER NOT NULL DEFAULT 0,
                    channels_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_refresh_dispatch_log_created
                ON refresh_dispatch_log(created_at DESC);
                """
            )
            # D-7: notify_sent_log은 원래 채널 구분 없이 (dataset, wafer,
            # grade)만 기록했다 -- 한 채널만 성공해도 발송 완료로 찍혀서
            # 실패한 채널은 24시간 동안 재시도되지 않았다. CREATE TABLE IF
            # NOT EXISTS는 이미 만들어진 테이블에 컬럼을 추가하지 않으므로
            # 여기서 직접 확인 후 ALTER TABLE한다.
            existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(notify_sent_log)")}
            if "channel" not in existing_columns:
                connection.execute("ALTER TABLE notify_sent_log ADD COLUMN channel TEXT NOT NULL DEFAULT ''")

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

    def promote_if_better(
        self,
        *,
        model_id: str,
        pipeline_version: str,
        dataset_version: int,
        metadata: dict[str, Any],
        metric_path: tuple[str, ...] = ("metrics", "test", "r2"),
        higher_is_better: bool = True,
        tolerance: float = PROMOTION_TOLERANCE,
    ) -> dict[str, Any]:
        """승격 게이트 (지시서 I-4, 자동 수집 파이프라인 §2-1) -- 후보
        모델이 현재 활성 모델의 `metric_path` 지표(기본: 최종 수율 테스트
        R², 높을수록 좋음)보다 `tolerance` 이상 나쁘지 않을 때만
        `promote_model`을 호출한다. 작은 노이즈로 승격이 막히지 않도록
        완전히 같거나 나은 경우만 요구하지 않고 `tolerance`만큼의 하락은
        허용한다. 활성 모델이 없거나 지표를 비교할 수 없으면(둘 중
        하나라도 None) 비교 불가로 보고 승격시킨다 -- 게이트가 첫 학습
        자체를 막아서는 안 된다.

        수동 학습("수동 학습 실행")과 자동 재학습이 이 메서드 하나를
        공유하므로 두 경로 모두 같은 게이트를 통과한다. 매 학습이 자신의
        85/15 분할에서 계산한 테스트 지표를 비교하는 것이라 완전히
        고정된 단일 홀드아웃은 아니지만, 동일한 random_state로 분할하므로
        데이터셋이 그대로면 분할도 그대로다.
        """
        def _dig(source: dict[str, Any] | None) -> float | None:
            node: Any = source
            for key in metric_path:
                if not isinstance(node, dict):
                    return None
                node = node.get(key)
            return float(node) if isinstance(node, (int, float)) else None

        metric_name = metric_path[-1]
        active = self.active_model()
        candidate_metric = _dig(metadata)
        active_metadata = active.get("metadata") if active else None
        active_metric = _dig(active_metadata)

        if active is None:
            promoted, reason = True, "최초 모델 -- 게이트 없이 승격"
        elif candidate_metric is None or active_metric is None:
            promoted, reason = True, f"지표 비교 불가({metric_name} 없음) -- 비교할 수 없어 승격"
        else:
            regressed = (
                candidate_metric < active_metric - tolerance
                if higher_is_better
                else candidate_metric > active_metric + tolerance
            )
            if not regressed:
                promoted, reason = True, "게이트 통과"
            else:
                promoted, reason = (
                    False,
                    f"홀드아웃 {metric_name} 저하 ({active_metric:.4f} → {candidate_metric:.4f})",
                )

        self.record_promotion_event(
            candidate_model_id=model_id,
            promoted=promoted,
            reason=reason,
            candidate_metric=candidate_metric,
            active_metric=active_metric,
            previous_model_id=(active or {}).get("active_model_id"),
        )
        if promoted:
            return self.promote_model(
                model_id=model_id,
                pipeline_version=pipeline_version,
                dataset_version=dataset_version,
                metadata=metadata,
            )
        return active or {}

    def record_promotion_event(
        self,
        *,
        candidate_model_id: str,
        promoted: bool,
        reason: str,
        candidate_metric: float | None,
        active_metric: float | None,
        previous_model_id: str | None,
    ) -> None:
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO promotion_events
                (created_at,candidate_model_id,promoted,reason,candidate_metric,active_metric,previous_model_id)
                VALUES (?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    candidate_model_id,
                    1 if promoted else 0,
                    str(reason)[:500],
                    candidate_metric,
                    active_metric,
                    previous_model_id,
                ),
            )

    def list_promotion_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM promotion_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_favorite(self, favorite_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        """D-1: (dataset, target, feature, viewType)가 같은 즐겨찾기가 이미
        있으면 새로 만들지 않고 그 레코드를 그대로 돌려준다 -- 프런트의
        더블클릭 가드(ref 기반)는 같은 브라우저 탭 안에서만 막는다. 두 탭
        에서 거의 동시에 누르거나 요청이 늦게 도착하는 경우까지 막으려면
        서버도 유니크를 보장해야 한다. `_lock`으로 조회-후-삽입을 원자적
        으로 만든다.
        """
        dedupe_key = _favorite_dedupe_key(snapshot)
        created_at = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            rows = connection.execute("SELECT favorite_id,created_at,snapshot_json FROM favorites").fetchall()
            for row in rows:
                if _favorite_dedupe_key(json.loads(row["snapshot_json"])) == dedupe_key:
                    return {
                        "favorite_id": row["favorite_id"],
                        "created_at": row["created_at"],
                        "snapshot": json.loads(row["snapshot_json"]),
                    }
            connection.execute(
                "INSERT INTO favorites (favorite_id,created_at,snapshot_json) VALUES (?,?,?)",
                (favorite_id, created_at, self._json(snapshot)),
            )
        return {"favorite_id": favorite_id, "created_at": created_at, "snapshot": snapshot}

    def list_favorites(self) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT favorite_id,created_at,snapshot_json FROM favorites ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            result.append({
                "favorite_id": row["favorite_id"],
                "created_at": row["created_at"],
                "snapshot": json.loads(row["snapshot_json"]),
            })
        return result

    def delete_favorite(self, favorite_id: str) -> bool:
        with _lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM favorites WHERE favorite_id=?", (favorite_id,))
        return cursor.rowcount > 0

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

    def has_corrupted_app_state(self, state_keys: list[str]) -> bool:
        """D-2: true if any of these keys holds a value that failed to
        JSON-decode -- `get_all_app_state` silently maps that to the same
        `None` a never-saved key returns, so a caller that needs to
        distinguish "restore failed" from "nothing to restore" (spec:
        복원 실패와 DB 손상이 조용히 '결과 없음'과 같아 보이면 안 된다)
        must check separately here rather than inferring it from `None`.
        """
        if not state_keys:
            return False
        with _lock, self._connect() as connection:
            placeholders = ",".join("?" for _ in state_keys)
            rows = connection.execute(
                f"SELECT value_json FROM app_state WHERE state_key IN ({placeholders})",
                state_keys,
            ).fetchall()
        for row in rows:
            try:
                json.loads(row["value_json"])
            except json.JSONDecodeError:
                return True
        return False

    def delete_app_state(self, state_key: str) -> bool:
        with _lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM app_state WHERE state_key=?", (state_key,))
            return cursor.rowcount > 0

    # -- 알림 발송 이력 (spec 알림 연동 §C-7: 동일 (dataset, wafer, grade) 조합은
    # 24시간 내 재발송하지 않는다) --

    def recent_notifications(self, dataset_id: str, since_iso: str, *, channel: str) -> list[dict[str, Any]]:
        """D-7: channel별로 조회한다 -- 채널을 구분하지 않으면 한 채널만
        성공해도 다른(실패한) 채널까지 24시간 동안 "이미 발송됨"으로
        보여 재시도되지 않는다."""
        with _lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT wafer_id, grade, sent_at FROM notify_sent_log WHERE dataset_id=? AND channel=? AND sent_at>=?",
                (dataset_id, channel, since_iso),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_notifications_sent(self, dataset_id: str, entries: list[tuple[str, str]], *, channel: str) -> None:
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.executemany(
                "INSERT INTO notify_sent_log (dataset_id, wafer_id, grade, sent_at, channel) VALUES (?,?,?,?,?)",
                [(dataset_id, wafer_id, grade, now, channel) for wafer_id, grade in entries],
            )

    def purge_old_notification_log(self, *, older_than_iso: str) -> int:
        """H-3②: notify_sent_log는 24시간 재발송 방지 조회(recent_notifications)
        용도라 그보다 훨씬 오래된 행은 볼 일이 없다 -- 지우지 않으면 이
        테이블이 무한히 커진다. 발송 잡과는 별도 스케줄 id로 도는 주기
        정리 잡이 호출한다."""
        with _lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM notify_sent_log WHERE sent_at < ?", (older_than_iso,)
            )
            return cursor.rowcount

    def notifications_sent_since(self, since_iso: str) -> int:
        """J-5: 시간당 발송 예산 확인용 -- 채널·데이터셋 구분 없이 최근
        발송 건수를 센다(과도한 발송 자체를 막는 전역 안전판이다)."""
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM notify_sent_log WHERE sent_at >= ?", (since_iso,)
            ).fetchone()
        return int(row["n"]) if row else 0

    # -- 자동 갱신 파이프라인 스냅샷 (J-3) --------------------------------

    def save_refresh_snapshot(self, snapshot: dict[str, Any]) -> None:
        """model/analysis/alarms/monitoring 네 블록을 하나의 JSON
        문서로 묶어 단일 UPSERT로 저장한다 -- 네 번 나눠 쓰면 중간에
        실패했을 때 화면마다 다른 시점의 데이터가 섞인다(예: 알람은 새
        목표 기준인데 모니터링은 옛 기준인 상태가 실제로 있었다). 하나의
        SQL 문 = 하나의 트랜잭션이므로 이 방식 자체로 원자성이 보장된다.
        """
        record = {**snapshot, "schema_version": REFRESH_SNAPSHOT_SCHEMA_VERSION}
        now = datetime.now(timezone.utc).isoformat()
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO app_state (state_key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
                (REFRESH_SNAPSHOT_STATE_KEY, self._json(record), now),
            )

    def get_refresh_snapshot_status(self) -> dict[str, Any]:
        """`schema_version`이 다르면 복원하지 않는다 -- 백엔드 로직이
        바뀐 뒤 옛 스냅샷이 새 화면을 조용히 덮어쓰는 사고를 막는다.
        "저장된 적 없음"과 "저장은 됐는데 버전이 옛날 것이라 못 씀"을
        구분해야 안내 문구가 달라지므로 `stale_version`을 따로 둔다."""
        raw = self.get_app_state(REFRESH_SNAPSHOT_STATE_KEY)
        if raw is None:
            return {"snapshot": None, "stale_version": False}
        if raw.get("schema_version") != REFRESH_SNAPSHOT_SCHEMA_VERSION:
            return {"snapshot": None, "stale_version": True}
        return {"snapshot": raw, "stale_version": False}

    def get_refresh_snapshot_meta(self) -> dict[str, Any] | None:
        """프런트의 가벼운 폴링 엔드포인트용 -- `created_at`만 반환하고
        본문 전체는 꺼내지 않는다."""
        status = self.get_refresh_snapshot_status()
        if status["snapshot"] is None:
            return None
        return {"created_at": status["snapshot"].get("created_at")}

    def has_valid_snapshot(self) -> bool:
        """W-2: 스키마 버전이 맞는 스냅샷이 이미 있으면 부트스트랩을
        건너뛴다는 판단에 쓰는 헬퍼."""
        status = self.get_refresh_snapshot_status()
        return status["snapshot"] is not None and not status["stale_version"]

    # -- 첫 기동 스냅샷 부트스트랩 단일 실행 잠금 (W-2) -------------------

    def acquire_bootstrap_lock(self) -> bool:
        """app_state를 잠금으로 재사용한다(전용 잠금 테이블을 새로 만들지
        않는다) -- 행이 없으면 INSERT로 잡고, 있어도 `updated_at`이
        `BOOTSTRAP_LOCK_STALE_SECONDS`보다 오래됐으면(부트스트랩 도중
        프로세스가 죽어 release가 호출되지 못한 경우) 훔쳐올 수 있다.
        SQLite UPSERT의 `DO UPDATE ... WHERE` 절이 이 비교와 쓰기를
        하나의 원자적 문장으로 묶어준다."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        stale_before_iso = (now - timedelta(seconds=BOOTSTRAP_LOCK_STALE_SECONDS)).isoformat()
        payload = self._json({"acquired_at": now_iso})
        with _lock, self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO app_state (state_key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                WHERE app_state.updated_at < ?""",
                (BOOTSTRAP_LOCK_STATE_KEY, payload, now_iso, stale_before_iso),
            )
            return cursor.rowcount > 0

    def release_bootstrap_lock(self) -> None:
        self.delete_app_state(BOOTSTRAP_LOCK_STATE_KEY)

    def set_bootstrap_status(
        self, status: str, stage: str | None, *, error: str | None = None
    ) -> None:
        """W-4: 프런트가 `/api/state/snapshot/meta`로 읽어 진행 배너에
        쓴다. 실제 학습 진행률(0~99%)은 이미 `training_jobs.progress`가
        갖고 있으므로 여기서는 다시 만들지 않고, 큰 단계 이름(stage)만
        남긴다 -- 없으면 프런트는 '첫 분석 진행 중'만 보여주고 가짜
        진행률을 만들지 않는다."""
        self.set_app_state(
            BOOTSTRAP_STATUS_STATE_KEY,
            {
                "status": status,
                "stage": stage,
                "error": error,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def get_bootstrap_status(self) -> dict[str, Any] | None:
        return self.get_app_state(BOOTSTRAP_STATUS_STATE_KEY)

    def latest_training_job(self) -> dict[str, Any] | None:
        """부트스트랩이 자신이 유발한 학습 Job의 진행 상태(stage)를
        읽어오는 용도 -- job_id를 따로 들고 다니지 않아도 되도록 가장
        최근 Job 하나만 본다(단일 워커 배포라 동시에 여러 Job이 쌓이지
        않는다, operation_coordinator가 그 자체를 이미 보장)."""
        with _lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None

    # -- 자동 갱신 알림 발송 기록 (J-5) -----------------------------------

    def record_refresh_dispatch(
        self,
        *,
        new_alarm_count: int,
        blocked_reason: str | None,
        summarized: bool,
        channels: dict[str, Any],
    ) -> None:
        """차단된 경우에도 기록한다 -- "왜 안 보냈는지"가 알림 기록
        화면에서 보여야 한다."""
        with _lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO refresh_dispatch_log
                (created_at, new_alarm_count, blocked_reason, summarized, channels_json)
                VALUES (?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    new_alarm_count,
                    blocked_reason,
                    1 if summarized else 0,
                    self._json(channels),
                ),
            )

    def list_refresh_dispatch_log(self, limit: int = 20) -> list[dict[str, Any]]:
        with _lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM refresh_dispatch_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            value = dict(row)
            value["summarized"] = bool(value["summarized"])
            value["channels"] = json.loads(value.pop("channels_json") or "{}")
            results.append(value)
        return results


def safe_runtime_call(method: str, **values: Any) -> Any:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        return getattr(RuntimeStore(), method)(**values)
    except Exception:
        logger.warning("Runtime dashboard 저장 실패: %s", method, exc_info=True)
        return None
