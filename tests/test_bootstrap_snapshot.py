"""W-2/W-6: 첫 기동 스냅샷 부트스트랩 -- 단일 실행 잠금, 진행 상태
기록, 그리고 lifespan에서 호출하는 오케스트레이션(`api.main._run_bootstrap`
/ `_bootstrap_snapshot_background`)이 유효 스냅샷·챔피언 유무에 따라
재학습을 건너뛰거나 기다리는지 확인한다."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from src.runtime.store import BOOTSTRAP_LOCK_STALE_SECONDS, RuntimeStore


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_has_valid_snapshot_false_until_saved() -> None:
    store, path = _store()
    try:
        assert store.has_valid_snapshot() is False
        store.save_refresh_snapshot({"created_at": "2026-01-01T00:00:00+00:00", "source": {}})
        assert store.has_valid_snapshot() is True
    finally:
        _cleanup(path)


def test_bootstrap_lock_is_single_flight() -> None:
    store, path = _store()
    try:
        assert store.acquire_bootstrap_lock() is True
        # 아직 보유 중이므로 두 번째 시도는 실패해야 한다 (동시 접속/
        # 워커 재시작 겹침에서 부트스트랩이 두 번 돌지 않게 하는 핵심).
        assert store.acquire_bootstrap_lock() is False
        store.release_bootstrap_lock()
        assert store.acquire_bootstrap_lock() is True
    finally:
        _cleanup(path)


def test_stale_bootstrap_lock_can_be_stolen() -> None:
    """부트스트랩 도중 프로세스가 죽어 release가 호출되지 못하면, 잠금이
    영원히 남아 다음 배포도 부트스트랩을 영영 건너뛰게 된다 -- 오래된
    잠금은 다른 프로세스가 가져갈 수 있어야 한다."""
    store, path = _store()
    try:
        assert store.acquire_bootstrap_lock() is True
        stale_iso = "2000-01-01T00:00:00+00:00"
        with store._connect() as connection:
            connection.execute(
                "UPDATE app_state SET updated_at = ? WHERE state_key = ?",
                (stale_iso, "automation:bootstrap_lock"),
            )
        assert store.acquire_bootstrap_lock() is True
    finally:
        _cleanup(path)


def test_bootstrap_status_round_trip() -> None:
    store, path = _store()
    try:
        assert store.get_bootstrap_status() is None
        store.set_bootstrap_status("running", "학습 중")
        status = store.get_bootstrap_status()
        assert status["status"] == "running"
        assert status["stage"] == "학습 중"
        assert status["error"] is None

        store.set_bootstrap_status("failed", None, error="자동 학습이 실패했습니다.")
        status = store.get_bootstrap_status()
        assert status["status"] == "failed"
        assert status["error"] == "자동 학습이 실패했습니다."
    finally:
        _cleanup(path)


def test_latest_training_job_returns_most_recent() -> None:
    store, path = _store()
    try:
        assert store.latest_training_job() is None
        store.create_training_job("train_a" + "0" * 27, source_filename="a.csv")
        store.create_training_job("train_b" + "0" * 27, source_filename="b.csv")
        latest = store.latest_training_job()
        assert latest is not None
        assert latest["job_id"] == "train_b" + "0" * 27
    finally:
        _cleanup(path)


class _FakeRuntimeStore:
    """`api.main._run_bootstrap`이 오직 이 표면(active_model/
    set_bootstrap_status/latest_training_job/has_valid_snapshot)만
    쓰는지 검증하기 위한 최소 더블. 실제 RuntimeStore 대신 씀으로써
    학습·분석 파이프라인 전체를 매 테스트마다 돌리지 않아도 된다."""

    def __init__(self, *, champion_exists: bool, snapshot_after_pipeline: bool) -> None:
        self._champion_exists = champion_exists
        self._snapshot_after_pipeline = snapshot_after_pipeline
        self.statuses: list[tuple[str, str | None]] = []
        self.pipeline_calls = 0

    def active_model(self):
        return {"active_model_id": "m1"} if self._champion_exists else None

    def set_bootstrap_status(self, status, stage, *, error=None):
        self.statuses.append((status, stage))
        if error is not None:
            self.statuses.append(("error", error))

    def has_valid_snapshot(self) -> bool:
        # 파이프라인이 최소 한 번 불린 뒤에만 스냅샷이 생긴 것으로 본다.
        return self.pipeline_calls > 0 and self._snapshot_after_pipeline

    def latest_training_job(self):
        return None


def test_run_bootstrap_skips_retrain_when_champion_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    store = _FakeRuntimeStore(champion_exists=True, snapshot_after_pipeline=True)

    def tracked_pipeline() -> None:
        store.pipeline_calls += 1

    monkeypatch.setattr(main_module, "run_refresh_pipeline", tracked_pipeline)

    asyncio.run(main_module._run_bootstrap(store))

    # 챔피언이 이미 있으므로 파이프라인은 "평가 · 원인분석"용으로 단
    # 한 번만 호출돼야 한다(학습 대기를 위한 추가 호출이 없어야 한다).
    assert store.pipeline_calls == 1
    assert store.statuses[-1] == ("done", None)


def test_run_bootstrap_records_failure_when_pipeline_never_produces_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    def fake_pipeline() -> None:
        return None

    monkeypatch.setattr(main_module, "run_refresh_pipeline", fake_pipeline)
    store = _FakeRuntimeStore(champion_exists=True, snapshot_after_pipeline=False)

    asyncio.run(main_module._run_bootstrap(store))

    assert store.statuses[-2] == ("failed", None)
    assert store.statuses[-1][0] == "error"


def test_bootstrap_background_skips_when_snapshot_already_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    class _AlreadyValidStore:
        def has_valid_snapshot(self) -> bool:
            return True

        def acquire_bootstrap_lock(self) -> bool:
            raise AssertionError("valid snapshot을 스킵하지 않고 잠금을 시도했습니다.")

    monkeypatch.setattr(main_module, "_bootstrap_runtime_store", lambda: _AlreadyValidStore())
    asyncio.run(main_module._bootstrap_snapshot_background())


def test_bootstrap_background_skips_when_lock_not_acquired(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    calls = {"run_bootstrap": 0}

    class _LockedStore:
        def has_valid_snapshot(self) -> bool:
            return False

        def acquire_bootstrap_lock(self) -> bool:
            return False

    async def fake_run_bootstrap(store):
        calls["run_bootstrap"] += 1

    monkeypatch.setattr(main_module, "_bootstrap_runtime_store", lambda: _LockedStore())
    monkeypatch.setattr(main_module, "_run_bootstrap", fake_run_bootstrap)
    asyncio.run(main_module._bootstrap_snapshot_background())

    assert calls["run_bootstrap"] == 0
