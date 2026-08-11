"""SD그룹: 자동화(주기 SQL 수율 예측 발송)는 모니터링·트리맵·원인분석을
계산하지 않고 스냅샷도 건드리지 않는다 -- 수율 예측만 계산해 알림만
보낸다. 모델이 없거나 배치가 분석 데이터 모양이 아니면 건너뛰고 이력에
사유를 남긴다(SD-3/SD-2)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.automation import yield_dispatch
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore


def _store_and_registry(tmp_path: Path) -> tuple[RuntimeStore, DatasetRegistry]:
    store = RuntimeStore(tmp_path / f"dashboard_{uuid4().hex}.db")
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    return store, registry


def _wire(monkeypatch: pytest.MonkeyPatch, store: RuntimeStore, registry: DatasetRegistry) -> None:
    monkeypatch.setattr(yield_dispatch, "_runtime_store", lambda: store)
    monkeypatch.setattr(yield_dispatch, "_dataset_registry", lambda s: registry)


def test_disabled_automation_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=False)

    called = {"fetch": 0}
    monkeypatch.setattr(
        yield_dispatch.sql_source, "is_sql_configured", lambda s: (called.__setitem__("fetch", called["fetch"] + 1) or True)
    )

    yield_dispatch.run_automation_yield_dispatch_job()

    assert called["fetch"] == 0
    assert store.list_notify_history() == []


def test_sql_not_configured_skips_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=True)
    monkeypatch.setattr(yield_dispatch.sql_source, "is_sql_configured", lambda s: False)

    yield_dispatch.run_automation_yield_dispatch_job()

    assert store.list_notify_history() == []


def test_non_eval_shaped_batch_is_skipped_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=True)
    monkeypatch.setattr(yield_dispatch.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(
        yield_dispatch.sql_source, "fetch_incremental", lambda s: pd.DataFrame({"Notes": ["a", "b"], "Y": [1.0, 2.0]})
    )

    yield_dispatch.run_automation_yield_dispatch_job()

    items = store.list_notify_history()
    assert len(items) == 1
    assert items[0]["status"] == "skipped"
    assert "Step" in items[0]["skip_reason"]


def test_missing_model_is_skipped_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=True)
    monkeypatch.setattr(yield_dispatch.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(
        yield_dispatch.sql_source,
        "fetch_incremental",
        lambda s: pd.DataFrame({"Step1_R1": [1.0, 2.0], "Lot_Wafer_ID": ["L1_W1", "L1_W2"]}),
    )
    assert store.active_model() is None

    yield_dispatch.run_automation_yield_dispatch_job()

    items = store.list_notify_history()
    assert len(items) == 1
    assert items[0]["status"] == "skipped"
    assert items[0]["skip_reason"] == "학습된 모델 없음"

    automation = store.get_automation_settings()
    assert automation["lastRunStatus"] == "skipped"


def test_empty_batch_is_a_silent_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=True)
    monkeypatch.setattr(yield_dispatch.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(yield_dispatch.sql_source, "fetch_incremental", lambda s: pd.DataFrame())

    yield_dispatch.run_automation_yield_dispatch_job()

    assert store.list_notify_history() == []


def test_automation_never_writes_a_refresh_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SD그룹 "하지 말 것": 자동화가 화면 스냅샷을 갱신하게 하지 마라."""
    store, registry = _store_and_registry(tmp_path)
    _wire(monkeypatch, store, registry)
    store.save_automation_settings(enabled=True)
    monkeypatch.setattr(yield_dispatch.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(
        yield_dispatch.sql_source,
        "fetch_incremental",
        lambda s: pd.DataFrame({"Step1_R1": [1.0, 2.0], "Lot_Wafer_ID": ["L1_W1", "L1_W2"]}),
    )
    assert store.active_model() is None

    yield_dispatch.run_automation_yield_dispatch_job()

    assert store.get_refresh_snapshot_status()["snapshot"] is None
