from __future__ import annotations

import logging
import secrets

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    status,
)

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
router = APIRouter(prefix="/api/admin", tags=["admin-history-reset"])
RESET_CONFIRMATION = "RESET_ALL_HISTORY"


def _require_admin_reset_secret(
    provided: str | None = Header(
        default=None,
        alias="X-Admin-Reset-Secret",
    ),
) -> None:
    configured = settings.admin_reset_secret
    if (
        not configured
        or provided is None
        or not secrets.compare_digest(provided, configured)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="초기화 권한이 없습니다.",
        )


def get_history_reset_service() -> HistoryResetService:
    return HistoryResetService(
        model_dir=settings.model_dir,
        store=RuntimeStore(
            settings.runtime_db_path,
            settings.runtime_artifact_dir,
        ),
    )


@router.get(
    "/history/summary",
    response_model=HistoryResetSummary,
    responses={403: {"description": "Forbidden"}},
)
def get_admin_history_summary(
    _: None = Depends(_require_admin_reset_secret),
) -> HistoryResetSummary:
    try:
        return HistoryResetSummary(
            **get_history_reset_service().summary()
        )
    except Exception as exc:
        logger.exception("관리자 이력 Summary 조회 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 정보를 조회하지 못했습니다.",
        ) from exc


@router.delete(
    "/history",
    response_model=HistoryResetResponse,
    responses={
        400: {"description": "Invalid confirmation"},
        403: {"description": "Forbidden"},
        409: {"description": "Active operation"},
        500: {"description": "Reset failed"},
    },
)
def delete_admin_history(
    payload: HistoryResetRequest | None = Body(default=None),
    _: None = Depends(_require_admin_reset_secret),
) -> HistoryResetResponse:
    if payload is None or payload.confirmation != RESET_CONFIRMATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="초기화 확인값이 올바르지 않습니다.",
        )
    try:
        return HistoryResetResponse(
            **get_history_reset_service().reset()
        )
    except ActiveOperationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ACTIVE_JOB_MESSAGE,
        ) from exc
    except HistoryResetError as exc:
        logger.exception("관리자 이력 초기화 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 중 서버 오류가 발생했습니다.",
        ) from exc
    except Exception as exc:
        logger.exception("관리자 이력 초기화 중 예기치 않은 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="이력 초기화 중 서버 오류가 발생했습니다.",
        ) from exc
