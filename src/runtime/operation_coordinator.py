from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Literal


OperationKind = Literal["training", "prediction", "analysis"]
ACTIVE_JOB_MESSAGE = (
    "현재 실행 중인 작업이 있습니다. 작업 완료 후 다시 시도해 주세요."
)


class ActiveOperationError(RuntimeError):
    """Raised when a reset and a protected operation would overlap."""


class OperationCoordinator:
    """Process-local gate that closes the reset/job start race."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[OperationKind, int] = {
            "training": 0,
            "prediction": 0,
            "analysis": 0,
        }
        self._reset_active = False

    @contextmanager
    def job(self, kind: OperationKind) -> Iterator[None]:
        with self._lock:
            if self._reset_active:
                raise ActiveOperationError(ACTIVE_JOB_MESSAGE)
            self._active[kind] += 1
        try:
            yield
        finally:
            with self._lock:
                self._active[kind] = max(self._active[kind] - 1, 0)

    @contextmanager
    def exclusive_reset(self) -> Iterator[None]:
        with self._lock:
            if self._reset_active or any(self._active.values()):
                raise ActiveOperationError(ACTIVE_JOB_MESSAGE)
            self._reset_active = True
        try:
            yield
        finally:
            with self._lock:
                self._reset_active = False

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {**self._active, "reset_active": self._reset_active}


operation_coordinator = OperationCoordinator()
