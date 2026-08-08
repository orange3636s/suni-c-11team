"""J-1: SQL 연결 상태 판단 -- 드라이버/쿼리/접속 정보 중 하나라도 빠지면
"SQL 모드 시도 대상"조차 아니어야 한다(곧장 폴백)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from uuid import uuid4

import pytest

from api.settings import settings as real_settings
from src.automation import sql_source
from src.runtime.app_state import save_state
from src.runtime.store import RuntimeStore


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_not_configured_without_driver_or_query(monkeypatch: pytest.MonkeyPatch) -> None:
    store, path = _store()
    try:
        monkeypatch.setattr(sql_source, "settings", dataclasses.replace(real_settings, db_driver=None, auto_ingest_query=None))
        assert sql_source.is_sql_configured(store) is False
    finally:
        _cleanup(path)


def test_not_configured_without_stored_connection_info(monkeypatch: pytest.MonkeyPatch) -> None:
    store, path = _store()
    try:
        monkeypatch.setattr(
            sql_source, "settings",
            dataclasses.replace(real_settings, db_driver="postgresql+psycopg2", auto_ingest_query="SELECT 1"),
        )
        # state/training이 저장된 적 없으면 host/port/db/user도 없다.
        assert sql_source.is_sql_configured(store) is False
    finally:
        _cleanup(path)


def test_configured_when_driver_query_and_connection_info_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    store, path = _store()
    try:
        monkeypatch.setattr(
            sql_source, "settings",
            dataclasses.replace(real_settings, db_driver="postgresql+psycopg2", auto_ingest_query="SELECT 1"),
        )
        save_state(
            store, "training",
            dataset={"dataset": "training-settings"},
            payload={"sqlHost": "db.internal", "sqlPort": "5432", "sqlDb": "fab", "sqlUser": "svc"},
        )
        assert sql_source.is_sql_configured(store) is True
    finally:
        _cleanup(path)


def test_not_configured_when_connection_info_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    store, path = _store()
    try:
        monkeypatch.setattr(
            sql_source, "settings",
            dataclasses.replace(real_settings, db_driver="postgresql+psycopg2", auto_ingest_query="SELECT 1"),
        )
        save_state(
            store, "training",
            dataset={"dataset": "training-settings"},
            payload={"sqlHost": "db.internal", "sqlPort": "", "sqlDb": "fab", "sqlUser": "svc"},
        )
        assert sql_source.is_sql_configured(store) is False
    finally:
        _cleanup(path)


def test_fetch_incremental_returns_none_on_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """실제 DB 없이도, 잘못된 드라이버 접두사로 접속 시도가 실패하면
    예외를 던지지 않고 None을 반환해야 한다(스케줄러를 죽이면 안 된다)."""
    store, path = _store()
    try:
        monkeypatch.setattr(
            sql_source, "settings",
            dataclasses.replace(
                real_settings,
                db_driver="postgresql+psycopg2",
                auto_ingest_query="SELECT 1",
                db_password="x",
            ),
        )
        save_state(
            store, "training",
            dataset={"dataset": "training-settings"},
            payload={"sqlHost": "nonexistent.invalid", "sqlPort": "5432", "sqlDb": "fab", "sqlUser": "svc"},
        )
        result = sql_source.fetch_incremental(store)
        assert result is None
    finally:
        _cleanup(path)
