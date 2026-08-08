"""E-1: RuntimeStore._initialize() (9 CREATE TABLE statements + an ALTER
TABLE check, all under the process-wide `_lock`) must run at most once
per database file path per process -- every API route currently builds a
fresh RuntimeStore(...) per request, so without this guard every request
re-runs the full DDL script and contends with the global lock against
real writes (e.g. a training job).
"""

from __future__ import annotations

from pathlib import Path

from src.runtime.store import RuntimeStore


def test_initialize_runs_once_per_path(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []
    original = RuntimeStore._initialize

    def counting_initialize(self: RuntimeStore) -> None:
        calls.append(self.path)
        original(self)

    monkeypatch.setattr(RuntimeStore, "_initialize", counting_initialize)

    path = tmp_path / "dashboard.db"
    RuntimeStore(path)
    RuntimeStore(path)
    RuntimeStore(path)

    assert len(calls) == 1


def test_initialize_runs_again_for_a_different_path(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []
    original = RuntimeStore._initialize

    def counting_initialize(self: RuntimeStore) -> None:
        calls.append(self.path)
        original(self)

    monkeypatch.setattr(RuntimeStore, "_initialize", counting_initialize)

    RuntimeStore(tmp_path / "a.db")
    RuntimeStore(tmp_path / "b.db")

    assert len(calls) == 2


def test_second_store_for_same_path_is_still_fully_usable(tmp_path: Path) -> None:
    """The skip-reinit optimization must not leave a second instance
    unable to read/write -- it shares the same on-disk schema."""
    path = tmp_path / "dashboard.db"
    first = RuntimeStore(path)
    first.set_app_state("latest_training", {"dataset": "train"})

    second = RuntimeStore(path)
    assert second.get_app_state("latest_training") == {"dataset": "train"}
