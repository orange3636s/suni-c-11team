from __future__ import annotations

from collections import defaultdict, deque
import ipaddress
import logging
from threading import Lock
import time

from fastapi import APIRouter, Body, HTTPException, Request, status

from api.schemas.admin import (
    HistoryResetRequest,
    HistoryResetResponse,
    HistoryResetSummary,
)
from api.settings import settings
from src.runtime.history_reset import HistoryResetError, HistoryResetService
from src.runtime.operation_coordinator import (
    ACTIVE_JOB_MESSAGE,
    ActiveOperationError,
)
from src.runtime.store import RuntimeStore


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["history-reset"])
RESET_CONFIRMATION = "RESET_ALL_HISTORY"
RATE_LIMIT_WINDOW_SECONDS = 10 * 60
RATE_LIMIT_MAX_REQUESTS = 3
RATE_LIMIT_MESSAGE = "초기화 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."


class ResetRateLimiter:
    """Single-replica TTL limiter keyed by normalized client IP."""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - RATE_LIMIT_WINDOW_SECONDS
        with self._lock:
            entries = self._requests[key]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= RATE_LIMIT_MAX_REQUESTS:
                return False
            entries.append(timestamp)
            return True


_RESET_LIMITER = ResetRateLimiter()


def _normalized_ip(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().strip('"')
    if candidate.startswith("[") and "]" in candidate:
        candidate = candidate[1 : candidate.index("]")]
    elif candidate.count(":") == 1 and "." in candidate:
        host, _, port = candidate.rpartition(":")
        if port.isdigit():
            candidate = host
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        return None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("forwarded")
    if forwarded:
        for part in forwarded.split(",")[0].split(";"):
            name, separator, value = part.partition("=")
            if separator and name.strip().lower() == "for":
                normalized = _normalized_ip(value)
                if normalized:
                    return normalized
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        for value in forwarded_for.split(","):
            normalized = _normalized_ip(value)
            if normalized:
                return normalized
    client = request.client.host if request.client is not None else None
    return _normalized_ip(client) or "unknown-client"


def get_history_reset_service() -> HistoryResetService:
    return HistoryResetService(
        model_dir=settings.model_dir,
        store=RuntimeStore(
            settings.runtime_db_path,
            settings.runtime_artifact_dir,
        ),
    )


@router.get(
    "/history/reset/summary",
    response_model=HistoryResetSummary,
)
def get_history_reset_summary() -> HistoryResetSummary:
    try:
        return HistoryResetSummary(**get_history_reset_service().summary())
    except Exception as exc:
        logger.exception("이력 초기화 Summary 조회 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 정보를 조회하지 못했습니다.",
        ) from exc


@router.post(
    "/history/reset",
    response_model=HistoryResetResponse,
    responses={
        400: {"description": "Invalid confirmation"},
        409: {"description": "Active operation"},
        429: {"description": "Too many requests"},
        500: {"description": "Reset failed"},
    },
)
def reset_history(
    request: Request,
    payload: HistoryResetRequest | None = Body(default=None),
) -> HistoryResetResponse:
    if not _RESET_LIMITER.allow(_client_ip(request)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=RATE_LIMIT_MESSAGE,
        )
    if payload is None or payload.confirmation != RESET_CONFIRMATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="초기화 확인값이 올바르지 않습니다.",
        )
    try:
        return HistoryResetResponse(**get_history_reset_service().reset())
    except ActiveOperationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ACTIVE_JOB_MESSAGE,
        ) from exc
    except HistoryResetError as exc:
        logger.exception("이력 초기화 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 중 서버 오류가 발생했습니다.",
        ) from exc
    except Exception as exc:
        logger.exception("이력 초기화 중 예기치 않은 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 중 서버 오류가 발생했습니다.",
        ) from exc
