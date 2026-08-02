from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from src.runtime.operation_coordinator import (
    OperationCoordinator,
    operation_coordinator,
)
from src.runtime.store import RuntimeStore


logger = logging.getLogger(__name__)
_JOB_ID = re.compile(r"^train_[0-9a-f]{32}$")
_WINDOWS_REPARSE_POINT = 0x400

ProgressCallback = Callable[[str, int], None]
TrainingRunner = Callable[[ProgressCallback], dict[str, Any]]


def new_training_job_id() -> str:
    return f"train_{uuid4().hex}"


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & _WINDOWS_REPARSE_POINT)
    except OSError:
        return False


class TrainingJobManager:
    """Run one reserved training job outside the request event loop.

    The uploaded CSV is the only temporary file managed here.  Model bundles
    remain owned by the existing training code and job responses contain only
    a compact summary persisted in SQLite.
    """

    def __init__(
        self,
        *,
        store: RuntimeStore,
        input_root: str | Path,
        coordinator: OperationCoordinator = operation_coordinator,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.store = store
        self.input_root = Path(input_root)
        self.coordinator = coordinator
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="model-training",
        )
        self._futures: set[Future[None]] = set()
        self._futures_lock = Lock()

    def allocate_input_path(self, job_id: str) -> Path:
        root = self._validated_root(create=True)
        directory = self._job_directory(job_id, root)
        if directory.exists() and _is_link_or_reparse(directory):
            raise RuntimeError("학습 Job 임시 경로가 안전하지 않습니다.")
        directory.mkdir(parents=False, exist_ok=False)
        return directory / "input.csv"

    def submit(
        self,
        *,
        job_id: str,
        source_filename: str,
        input_path: Path,
        runner: TrainingRunner,
    ) -> None:
        reserved = False
        created = False
        try:
            self.coordinator.reserve_job("training")
            reserved = True
            self.store.create_training_job(
                job_id,
                source_filename=source_filename,
            )
            created = True
            future = self._executor.submit(
                self._run,
                job_id,
                input_path,
                runner,
            )
            with self._futures_lock:
                self._futures.add(future)
            future.add_done_callback(self._discard_future)
        except Exception:
            if created:
                self.store.fail_training_job(
                    job_id,
                    "학습 Job을 실행 대기열에 등록하지 못했습니다.",
                )
            if reserved:
                self.coordinator.release_job("training")
            self.cleanup_input(job_id)
            raise

    def get(self, job_id: str) -> dict[str, Any] | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        row = self.store.get_training_job(job_id)
        if row is None:
            return None
        row["elapsed_seconds"] = self._elapsed_seconds(row)
        return row

    def recover_interrupted(self) -> int:
        job_ids = self.store.interrupt_training_jobs()
        stale_ids = set(job_ids)
        if self.input_root.exists():
            try:
                root = self._validated_root(create=False)
                stale_ids.update(
                    entry.name
                    for entry in root.iterdir()
                    if entry.is_dir() and _JOB_ID.fullmatch(entry.name)
                )
            except (OSError, RuntimeError):
                logger.exception("시작 시 학습 Job 임시 경로 검사 실패")
        for job_id in stale_ids:
            self.cleanup_input(job_id)
        if job_ids:
            logger.warning(
                "서버 시작 시 중단된 학습 Job %d개를 복구했습니다.",
                len(job_ids),
            )
        return len(job_ids)

    def cleanup_input(self, job_id: str) -> None:
        if not _JOB_ID.fullmatch(job_id):
            logger.error("잘못된 학습 Job ID의 임시 파일 정리를 거부했습니다.")
            return
        try:
            root = self._validated_root(create=False)
        except RuntimeError:
            logger.exception("학습 Job 임시 Root 검증 실패")
            return
        directory = self._job_directory(job_id, root)
        if not directory.exists():
            return
        if _is_link_or_reparse(directory):
            logger.error("링크 또는 Junction 학습 Job 경로 정리를 거부했습니다: %s", job_id)
            return
        for filename in ("input.csv", "input.csv.tmp"):
            candidate = directory / filename
            if candidate.exists() and not _is_link_or_reparse(candidate):
                try:
                    candidate.unlink()
                except OSError:
                    logger.warning("학습 Job 임시 파일 정리 실패: %s", job_id, exc_info=True)
        try:
            directory.rmdir()
        except OSError:
            logger.warning(
                "학습 Job 임시 디렉터리가 비어 있지 않아 보존했습니다: %s",
                job_id,
            )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run(
        self,
        job_id: str,
        input_path: Path,
        runner: TrainingRunner,
    ) -> None:
        try:
            self.store.start_training_job(job_id)

            def progress(stage: str, value: int) -> None:
                self.store.update_training_job(
                    job_id,
                    stage=str(stage)[:200],
                    progress=value,
                )

            result = runner(progress)
            self.store.complete_training_job(job_id, result=result)
        except Exception as exc:
            logger.exception("비동기 학습 Job 실패: job_id=%s", job_id)
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict):
                message = str(detail.get("message") or "학습 요청을 처리하지 못했습니다.")
            elif isinstance(detail, str):
                message = detail
            else:
                message = "모델 학습 중 서버 오류가 발생했습니다."
            try:
                self.store.fail_training_job(job_id, message)
            except Exception:
                logger.exception("학습 Job 실패 상태 저장 실패: job_id=%s", job_id)
        finally:
            self.cleanup_input(job_id)
            self.coordinator.release_job("training")

    def _discard_future(self, future: Future[None]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _validated_root(self, *, create: bool) -> Path:
        unresolved = self.input_root.expanduser().absolute()
        if create:
            unresolved.mkdir(parents=True, exist_ok=True)
        resolved = unresolved.resolve()
        if resolved == resolved.parent:
            raise RuntimeError("볼륨 Root는 학습 Job 임시 경로가 될 수 없습니다.")
        if not resolved.is_dir() or _is_link_or_reparse(unresolved):
            raise RuntimeError("학습 Job 임시 Root가 안전한 디렉터리가 아닙니다.")
        return resolved

    @staticmethod
    def _job_directory(job_id: str, root: Path) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise RuntimeError("유효하지 않은 학습 Job ID입니다.")
        directory = root / job_id
        if directory.parent != root:
            raise RuntimeError("학습 Job 경로가 허용 Root를 벗어났습니다.")
        return directory

    @staticmethod
    def _elapsed_seconds(row: dict[str, Any]) -> float:
        raw_start = row.get("started_at") or row.get("created_at")
        raw_end = row.get("completed_at")
        try:
            started = datetime.fromisoformat(str(raw_start))
            ended = (
                datetime.fromisoformat(str(raw_end))
                if raw_end
                else datetime.now(timezone.utc)
            )
            return max((ended - started).total_seconds(), 0.0)
        except (TypeError, ValueError):
            return 0.0
