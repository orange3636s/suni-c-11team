"""Tests for RuntimeStore.create_favorite's dedupe -- D-1: a duplicate
rapid double-click (or two near-simultaneous requests) for the same
(dataset, target, feature, viewType) must not create a second record
that the client can never delete, and saving a different viewType for
the same factor must not collide with an existing one.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.runtime.store import RuntimeStore


def _store(tmp_path: Path) -> RuntimeStore:
    return RuntimeStore(tmp_path / "dashboard.db")


def _snapshot(**overrides) -> dict:
    base = {"dataset": "train", "target": "Y1", "feature": "Step1_R1", "viewType": "scatter", "isConfig": False}
    return {**base, **overrides}


def test_duplicate_create_returns_existing_record_not_a_new_one(tmp_path: Path):
    store = _store(tmp_path)
    first = store.create_favorite(f"fav_{uuid4().hex}", _snapshot())
    second = store.create_favorite(f"fav_{uuid4().hex}", _snapshot())

    assert second["favorite_id"] == first["favorite_id"]
    assert len(store.list_favorites()) == 1


def test_different_view_type_creates_a_separate_favorite(tmp_path: Path):
    """저장하려는 뷰가 다르면(예: Box) 기존 Scatter 즐겨찾기를 지우거나
    합치지 않고 별도로 만든다."""
    store = _store(tmp_path)
    scatter = store.create_favorite(f"fav_{uuid4().hex}", _snapshot(viewType="scatter"))
    box = store.create_favorite(f"fav_{uuid4().hex}", _snapshot(viewType="box"))

    assert scatter["favorite_id"] != box["favorite_id"]
    items = store.list_favorites()
    assert len(items) == 2
    assert {item["favorite_id"] for item in items} == {scatter["favorite_id"], box["favorite_id"]}


def test_different_feature_is_not_deduped(tmp_path: Path):
    store = _store(tmp_path)
    a = store.create_favorite(f"fav_{uuid4().hex}", _snapshot(feature="Step1_R1"))
    b = store.create_favorite(f"fav_{uuid4().hex}", _snapshot(feature="Step2_R1"))
    assert a["favorite_id"] != b["favorite_id"]
    assert len(store.list_favorites()) == 2
