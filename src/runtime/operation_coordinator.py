from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Literal


OperationKind = Literal["training", "prediction", "analysis"]
HEAVY_JOB_MESSAGE = "현재 다른 분석 작업이 실행 중입니다."


class ActiveOperationError(RuntimeError):
    """Raised when two protected (memory-heavy) operations would overlap."""


class OperationCoordinator:
    """Single-process gate for every memory-heavy operation (training/
    prediction/analysis) -- only one may run at a time.

    Deployment is intentionally limited to one Uvicorn worker, so keeping the
    gate process-local avoids another polling thread or a database lock being
    held for the full duration of model training.  ``reserve_job`` exists for
    background jobs: accepting a job and reserving its heavy-work slot must be
    one atomic decision, before the HTTP request returns.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[OperationKind, int] = {
            "training": 0,
            "prediction": 0,
            "analysis": 0,
        }

    @contextmanager
    def job(self, kind: OperationKind) -> Iterator[None]:
        self.reserve_job(kind)
        try:
            yield
        finally:
            self.release_job(kind)

    def reserve_job(self, kind: OperationKind) -> None:
        with self._lock:
            if any(self._active.values()):
                raise ActiveOperationError(HEAVY_JOB_MESSAGE)
            self._active[kind] += 1

    def release_job(self, kind: OperationKind) -> None:
        with self._lock:
            self._active[kind] = max(self._active[kind] - 1, 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._active)


operation_coordinator = OperationCoordinator()
