"""AG: 업로드로 활성화된 수동 평가 데이터셋 -- 자동 갱신(SQL/폴백)을
덮어쓰지 않고, 학습을 자동으로 걸지 않고, 발송을 중단하는지 확인한다."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.automation import refresh
from src.runtime.datasets import DatasetRegistry, validate_dataset
from src.runtime.store import RuntimeStore


def _store_and_registry(tmp_path: Path) -> tuple[RuntimeStore, DatasetRegistry]:
    store = RuntimeStore(tmp_path / "dashboard.db")
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    return store, registry


def test_manual_eval_override_round_trip(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "dashboard.db")
    assert store.get_manual_eval_override() is None
    store.set_manual_eval_override("abc123", "uploaded_0809.csv")
    override = store.get_manual_eval_override()
    assert override is not None
    assert override["dataset_id"] == "abc123"
    assert override["filename"] == "uploaded_0809.csv"
    assert store.clear_manual_eval_override() is True
    assert store.get_manual_eval_override() is None


def test_validate_dataset_allows_missing_target_as_warning() -> None:
    """AG-5: Y 계열이 없어도 차단하지 않는다 -- 평가 전용 업로드가
    통과해야 한다. Step 컬럼이 하나도 없는 파일만 차단 대상이다."""
    df = pd.DataFrame({"Step1_R1": [1.0, 2.0, 3.0], "Lot_Wafer_ID": ["L1_W1", "L1_W2", "L1_W3"]})
    result = validate_dataset(df)
    assert result.is_valid is True
    assert any("타깃" in w for w in result.warnings)


def test_validate_dataset_still_blocks_files_with_no_process_columns() -> None:
    df = pd.DataFrame({"Y1": [1.0, 2.0], "Notes": ["a", "b"]})
    result = validate_dataset(df)
    assert result.is_valid is False


def test_resolve_source_prefers_manual_override_over_fallback(tmp_path: Path) -> None:
    if not (Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV").exists():
        pytest.skip("data/bundled/train.CSV not present")
    store, registry = _store_and_registry(tmp_path)
    content = (registry.bundled_root / "test.CSV").read_bytes()
    upload_result = registry.upload("uploaded_manual.csv", content)
    assert upload_result["success"]
    store.set_manual_eval_override(upload_result["dataset_id"], "uploaded_manual.csv")

    errors: list[str] = []
    mode, train_dataset_id, eval_dataset_id, row_count = refresh._resolve_source(store, registry, errors)

    assert mode == "manual"
    assert eval_dataset_id == upload_result["dataset_id"]
    # 학습 대상은 건드리지 않는다 -- 직전 스냅샷이 없으면 폴백 train으로.
    assert train_dataset_id == refresh.FALLBACK_TRAIN_DATASET
    assert row_count > 0
    assert errors == []


def test_resolve_source_keeps_previous_train_dataset_in_manual_mode(tmp_path: Path) -> None:
    if not (Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV").exists():
        pytest.skip("data/bundled/train.CSV not present")
    store, registry = _store_and_registry(tmp_path)
    store.save_refresh_snapshot({
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": {"mode": "sql", "train_dataset": "sql_batch_1", "eval_dataset": "sql_batch_1", "row_count": 10},
    })
    content = (registry.bundled_root / "test.CSV").read_bytes()
    upload_result = registry.upload("uploaded_manual2.csv", content)
    store.set_manual_eval_override(upload_result["dataset_id"], "uploaded_manual2.csv")

    errors: list[str] = []
    mode, train_dataset_id, eval_dataset_id, _ = refresh._resolve_source(store, registry, errors)
    assert mode == "manual"
    # AG-2: 학습 대상은 업로드로 바뀌지 않는다 -- 직전 스냅샷의 것을 그대로.
    assert train_dataset_id == "sql_batch_1"
    assert eval_dataset_id == upload_result["dataset_id"]


def test_dispatch_is_called_in_manual_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """EB그룹: 이전(AG-3)에는 수동 모드에서 발송 자체를 건너뛰었지만,
    이제는 refresh_dispatch.dispatch_new_alarms를 정상 호출한다 -- 차단
    여부(신규 0건/10분 간격/게이트 등)는 그 함수 내부가 판단한다."""
    store, registry = _store_and_registry(tmp_path)
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": None, "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(
        refresh,
        "_analyze_and_score",
        lambda s, eid, e: (
            {"paretoByTarget": {}, "measurementExpansion": None},
            {"gate_passed": True, "target_yield": 85.0, "sensitivity": 0.2, "counts": {}, "items_top": [], "total": 0},
            {"predicted_yield": None, "gap": None, "gap_pareto": [], "treemap": None},
            [],
            "train",
        ),
    )
    dispatch_called = {"n": 0}

    class _FakeDispatch:
        @staticmethod
        def dispatch_new_alarms(*args, **kwargs):
            dispatch_called["n"] += 1

    # refresh.py는 함수 안에서 매번 `from src.automation import
    # refresh_dispatch`를 실행한다 -- `src.automation` 패키지가 이미
    # (다른 테스트 모듈이 먼저 real import했을 수 있어) `refresh_dispatch`
    # 속성을 캐싱하고 있으면, CPython의 fromlist 처리(`hasattr(module, x)`가
    # 참이면 sys.modules를 다시 보지 않는다)는 sys.modules만 바꿔서는
    # 우회되지 않는다 -- 패키지 객체 자체의 속성을 갈아끼워야 한다.
    import src.automation as automation_pkg

    monkeypatch.setattr(automation_pkg, "refresh_dispatch", _FakeDispatch)

    refresh.run_refresh_pipeline()

    assert dispatch_called["n"] == 1
    status = store.get_refresh_snapshot_status()
    assert status["snapshot"] is not None
    assert status["snapshot"]["source"]["mode"] == "manual"


def test_activate_and_deactivate_dataset_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from api.main import app
    import api.routes.state as state_routes

    # 실제 기본 런타임 DB를 건드리지 않도록 격리한다.
    isolated_store = RuntimeStore(tmp_path / "dashboard.db")
    monkeypatch.setattr(state_routes, "_store", lambda: isolated_store)

    calls = {"pipeline": 0}
    monkeypatch.setattr(state_routes, "run_refresh_pipeline", lambda: calls.__setitem__("pipeline", calls["pipeline"] + 1))
    monkeypatch.setattr(state_routes, "is_refresh_running", lambda: False)

    fake_summary = {"original_filename": "uploaded_0809.csv"}

    class _FakeRegistry:
        def get_summary(self, dataset_id):
            return fake_summary if dataset_id == "ds-1" else None

    monkeypatch.setattr(state_routes, "get_dataset_registry", lambda: _FakeRegistry())

    with TestClient(app) as client:
        missing = client.post("/api/state/activate-dataset", json={"dataset_id": "does-not-exist"})
        assert missing.status_code == 404

        ok = client.post("/api/state/activate-dataset", json={"dataset_id": "ds-1"})
        assert ok.status_code == 200
        assert ok.json() == {"activated": True, "dataset_id": "ds-1"}

        meta = client.get("/api/state/snapshot/meta").json()
        assert meta["manual_eval_override"]["dataset_id"] == "ds-1"
        assert meta["manual_eval_override"]["filename"] == "uploaded_0809.csv"

        revert = client.post("/api/state/deactivate-dataset")
        assert revert.status_code == 200
        assert revert.json()["deactivated"] is True

        meta2 = client.get("/api/state/snapshot/meta").json()
        assert meta2["manual_eval_override"] is None

    assert calls["pipeline"] == 2  # activate + deactivate 각 1회씩
