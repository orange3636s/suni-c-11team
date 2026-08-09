"""AF/AG: 수동 최신화(모니터링 버튼)·업로드 연동이 공유하는 단일 진입점
(`run_refresh_pipeline`)의 동시 실행 방지 락과, 그 상태를 읽는
`/api/state/snapshot/meta`의 `refresh_running` 필드를 확인한다."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from src.automation import refresh


def test_is_refresh_running_reflects_lock_state() -> None:
    assert refresh.is_refresh_running() is False
    assert refresh._refresh_lock.acquire(blocking=False) is True
    try:
        assert refresh.is_refresh_running() is True
    finally:
        refresh._refresh_lock.release()
    assert refresh.is_refresh_running() is False


def test_concurrent_run_refresh_pipeline_calls_do_not_overlap(monkeypatch) -> None:
    """두 스레드가 동시에 run_refresh_pipeline을 부르면, 먼저 든 쪽만
    실제로 파이프라인을 실행하고 나머지는 조용히 건너뛴다."""
    started = threading.Event()
    release = threading.Event()
    call_count = 0
    lock = threading.Lock()

    def fake_inner(store) -> None:
        nonlocal call_count
        with lock:
            call_count += 1
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(refresh, "_run_refresh_pipeline_inner", fake_inner)

    t1 = threading.Thread(target=refresh.run_refresh_pipeline)
    t1.start()
    assert started.wait(timeout=5)

    # 첫 번째 호출이 아직 락을 쥔 채로 두 번째 호출은 즉시 반환해야 한다.
    t2_finished = threading.Event()

    def call_again():
        refresh.run_refresh_pipeline()
        t2_finished.set()

    t2 = threading.Thread(target=call_again)
    t0 = time.monotonic()
    t2.start()
    assert t2_finished.wait(timeout=5)
    assert time.monotonic() - t0 < 2  # 락을 기다리지 않고 즉시 반환(건너뜀)

    release.set()
    t1.join(timeout=5)
    assert call_count == 1


def test_trigger_refresh_endpoint_returns_409_when_already_running(monkeypatch) -> None:
    from api.main import app

    monkeypatch.setattr("api.routes.state.is_refresh_running", lambda: True)
    with TestClient(app) as client:
        response = client.post("/api/state/refresh")
    assert response.status_code == 409


def test_trigger_refresh_endpoint_accepts_and_schedules_when_idle(monkeypatch) -> None:
    from api.main import app

    called = {"n": 0}

    def fake_pipeline() -> None:
        called["n"] += 1

    monkeypatch.setattr("api.routes.state.is_refresh_running", lambda: False)
    monkeypatch.setattr("api.routes.state.run_refresh_pipeline", fake_pipeline)
    with TestClient(app) as client:
        response = client.post("/api/state/refresh")
    assert response.status_code == 200
    assert response.json() == {"triggered": True}
    # BackgroundTasks run after the response within the same TestClient call.
    assert called["n"] == 1
