from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FavoriteCreateRequest(BaseModel):
    # 자유 형식 스냅샷(지시서 J-2) -- dataset/target/feature/view_type/
    # color_by/method/저장 시각을 프론트가 실어 보낸다. 점 데이터는 절대
    # 포함하지 않는다(J-2: "점 데이터는 저장하지 말 것").
    snapshot: dict[str, Any]


class FavoriteResponse(BaseModel):
    favorite_id: str
    created_at: str
    snapshot: dict[str, Any]


class FavoriteListResponse(BaseModel):
    items: list[FavoriteResponse]


class FavoriteDeleteResponse(BaseModel):
    deleted: bool
