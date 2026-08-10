"""Tests for src/automation/ingest.py -- the file-based auto-ingest
pipeline (자동 수집 파이프라인 1단계). Focuses on the pure filesystem
logic (new_csv_files/move_to) and run_auto_ingest_job's dispatch/best-
effort behavior; the actual dataset-registration/training/analysis
calls are exercised elsewhere (test_ml_training.py's has_target_column
tests, test_promotion_gate.py) and are monkeypatched here so this file
doesn't need a full RuntimeStore/DatasetRegistry stack.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.automation import ingest


@pytest.fixture()
def watch_dir(tmp_path: Path) -> Path:
    return tmp_path / "incoming"


def test_new_csv_files_lists_only_top_level_csv(watch_dir: Path) -> None:
    watch_dir.mkdir()
    (watch_dir / "a.csv").write_text("x")
    (watch_dir / "b.CSV".lower()).write_text("x")
    (watch_dir / "note.txt").write_text("x")
    (watch_dir / ingest.PROCESSED_SUBDIR).mkdir()
    (watch_dir / ingest.PROCESSED_SUBDIR / "old.csv").write_text("x")

    files = ingest.new_csv_files(watch_dir)

    assert [f.name for f in files] == ["a.csv", "b.csv"]


def test_new_csv_files_missing_directory_returns_empty() -> None:
    assert ingest.new_csv_files(Path("/does/not/exist")) == []


def test_move_to_moves_file_into_subdir(watch_dir: Path) -> None:
    watch_dir.mkdir()
    source = watch_dir / "a.csv"
    source.write_text("x")

    destination = ingest.move_to(source, ingest.PROCESSED_SUBDIR)

    assert not source.exists()
    assert destination.exists()
    assert destination.parent.name == ingest.PROCESSED_SUBDIR


def test_move_to_avoids_overwriting_same_name(watch_dir: Path) -> None:
    watch_dir.mkdir()
    processed = watch_dir / ingest.PROCESSED_SUBDIR
    processed.mkdir()
    (processed / "a.csv").write_text("already here")
    source = watch_dir / "a.csv"
    source.write_text("new content")

    destination = ingest.move_to(source, ingest.PROCESSED_SUBDIR)

    assert destination.name != "a.csv"
    assert destination.read_text() == "new content"
    assert (processed / "a.csv").read_text() == "already here"


def test_run_auto_ingest_job_skips_when_disabled(monkeypatch: pytest.MonkeyPatch, watch_dir: Path) -> None:
    watch_dir.mkdir()
    (watch_dir / "a.csv").write_text("x")
    monkeypatch.setattr(ingest, "settings", SimpleNamespace(auto_ingest_enabled=False, auto_ingest_dir=str(watch_dir)))
    called = []
    monkeypatch.setattr(ingest, "_process_incoming_csv", lambda path: called.append(path))

    ingest.run_auto_ingest_job()

    assert called == []
    assert (watch_dir / "a.csv").exists()  # untouched


def test_run_auto_ingest_job_moves_successful_file_to_processed(
    monkeypatch: pytest.MonkeyPatch, watch_dir: Path,
) -> None:
    watch_dir.mkdir()
    (watch_dir / "good.csv").write_text("Y\n1\n")
    monkeypatch.setattr(ingest, "settings", SimpleNamespace(auto_ingest_enabled=True, auto_ingest_dir=str(watch_dir)))
    monkeypatch.setattr(ingest, "_process_incoming_csv", lambda path: None)

    ingest.run_auto_ingest_job()

    assert not (watch_dir / "good.csv").exists()
    assert (watch_dir / ingest.PROCESSED_SUBDIR / "good.csv").exists()


def test_run_auto_ingest_job_moves_broken_file_to_failed_without_raising(
    monkeypatch: pytest.MonkeyPatch, watch_dir: Path,
) -> None:
    watch_dir.mkdir()
    (watch_dir / "broken.csv").write_text("not,a,valid,training,file")
    monkeypatch.setattr(ingest, "settings", SimpleNamespace(auto_ingest_enabled=True, auto_ingest_dir=str(watch_dir)))

    def _boom(path: Path) -> None:
        raise ValueError("simulated failure")

    monkeypatch.setattr(ingest, "_process_incoming_csv", _boom)

    ingest.run_auto_ingest_job()  # must not raise

    assert not (watch_dir / "broken.csv").exists()
    assert (watch_dir / ingest.FAILED_SUBDIR / "broken.csv").exists()


def test_process_incoming_csv_always_ingests_as_eval_regardless_of_target_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """RB-3: 자동 수집은 더 이상 학습을 트리거하지 않는다 -- Y 컬럼이
    있든 없든 항상 평가(분석) 데이터셋으로만 등록된다."""
    eval_calls = []
    monkeypatch.setattr(
        ingest, "_ingest_eval_csv",
        lambda store, registry, filename, content: eval_calls.append(filename),
    )
    monkeypatch.setattr(ingest, "_runtime_store", lambda: SimpleNamespace())
    monkeypatch.setattr(ingest, "_dataset_registry", lambda store: SimpleNamespace())

    with_y = tmp_path / "with_y.csv"
    with_y.write_text("Lot_Wafer_ID,Y\nL1W1,90\n")
    ingest._process_incoming_csv(with_y)

    without_y = tmp_path / "without_y.csv"
    without_y.write_text("Lot_Wafer_ID,Y1\nL1W1,3\n")
    ingest._process_incoming_csv(without_y)

    assert eval_calls == ["with_y.csv", "without_y.csv"]


def test_refresh_analysis_snapshot_preserves_existing_state_when_all_targets_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """B-6 회귀: 전 타깃 Pareto 계산이 실패하면(예: 스키마가 맞지 않는
    평가 파일) 기존 정상 스냅샷을 빈 payload로 덮어쓰면 안 된다."""
    from src.runtime.app_state import get_latest_state, save_state
    from src.runtime.store import RuntimeStore

    store = RuntimeStore(tmp_path / "dashboard.db")
    save_state(
        store, "analysis", dataset={"dataset": "train"},
        payload={"activeTarget": "Y1", "paretoByTarget": {"Y1": {"kept": True}}, "measurementExpansion": None},
    )
    save_state(
        store, "alarms", dataset={"train_dataset": "train", "eval_dataset": "test"},
        payload={"targetYield": 85.0, "sensitivity": 0.5},
    )

    monkeypatch.setattr(
        "api.routes.analysis._pareto_payload",
        lambda dataset_id, target, top_n: (_ for _ in ()).throw(ValueError("broken schema")),
    )

    ingest._refresh_analysis_snapshot(store, "broken-eval-id")

    latest = get_latest_state(store)
    assert latest["analysis"]["payload"]["paretoByTarget"] == {"Y1": {"kept": True}}
    assert latest["alarms"]["eval_dataset"] == "test"
