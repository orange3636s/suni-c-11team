from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.runtime.store import RuntimeStore, safe_runtime_call


def _store() -> tuple[RuntimeStore, Path]:
    root = Path(__file__).parent / ".tmp_runtime"
    root.mkdir(exist_ok=True)
    path = root / f"dashboard_{uuid4().hex}.db"
    return RuntimeStore(path), path


def _cleanup(path: Path) -> None:
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def test_run_history_records_latency_and_error_count_inputs() -> None:
    store, path = _store()
    try:
        store.record_run(event_type="predict", duration_ms=12.5, status="success", row_count=20)
        store.record_run(event_type="report", duration_ms=7.5, status="failed", error_type="RuntimeError")
        runs = store.list_runs()
        assert len(runs) == 2
        assert {run["status"] for run in runs} == {"success", "failed"}
        assert sum(run["duration_ms"] for run in runs) == 20.0
    finally:
        _cleanup(path)


def test_runtime_store_failure_isolated_from_main_request(monkeypatch) -> None:
    monkeypatch.setattr(RuntimeStore, "record_run", lambda self, **values: (_ for _ in ()).throw(OSError("db unavailable")))
    assert safe_runtime_call("record_run", event_type="predict") is None
