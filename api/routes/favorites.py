from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from api.schemas.favorites import (
    FavoriteCreateRequest,
    FavoriteDeleteResponse,
    FavoriteListResponse,
    FavoriteResponse,
)
from api.settings import settings
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/favorites", tags=["favorites"])


def _store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


@router.post("", response_model=FavoriteResponse, status_code=status.HTTP_201_CREATED)
def create_favorite(body: FavoriteCreateRequest) -> dict:
    favorite_id = f"fav_{uuid4().hex}"
    try:
        return _store().create_favorite(favorite_id, body.snapshot)
    except Exception as exc:
        logger.exception("즐겨찾기 저장 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="즐겨찾기를 저장하지 못했습니다.",
        ) from exc


@router.get("", response_model=FavoriteListResponse)
def list_favorites() -> dict:
    # 최신순 -- RuntimeStore.list_favorites가 이미
    # created_at DESC로 정렬해 내려준다.
    return {"items": _store().list_favorites()}


@router.delete("/{favorite_id}", response_model=FavoriteDeleteResponse)
def delete_favorite(favorite_id: str) -> dict:
    deleted = _store().delete_favorite(favorite_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="즐겨찾기를 찾을 수 없습니다.")
    return {"deleted": True}
