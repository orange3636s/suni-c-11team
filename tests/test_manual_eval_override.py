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
    content = (registry.bundled_root / "test_remove_y.CSV").read_bytes()
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
    content = (registry.bundled_root / "test_remove_y.CSV").read_bytes()
    upload_result = registry.upload("uploaded_manual2.csv", content)
    store.set_manual_eval_override(upload_result["dataset_id"], "uploaded_manual2.csv")

    errors: list[str] = []
    mode, train_dataset_id, eval_dataset_id, _ = refresh._resolve_source(store, registry, errors)
    assert mode == "manual"
    # AG-2: 학습 대상은 업로드로 바뀌지 않는다 -- 직전 스냅샷의 것을 그대로.
    assert train_dataset_id == "sql_batch_1"
    assert eval_dataset_id == upload_result["dataset_id"]


def test_manual_mode_saves_snapshot_without_dispatching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SC/SD그룹: "모델 분석"([분석 시작], `run_refresh_pipeline`)은
    수동(업로드) 모드를 포함해 어떤 소스에서도 알림을 보내지 않는다 --
    알림 발송은 전적으로 "알림·자동화 설정"(주기 자동화·매일 09:00/13:00
    잡)의 책임이다. 이 파이프라인은 네 화면 스냅샷을 저장할 뿐이다."""
    store, registry = _store_and_registry(tmp_path)
    monkeypatch.setattr(refresh, "_runtime_store", lambda: store)
    monkeypatch.setattr(refresh, "_resolve_source", lambda s, r, e: ("manual", "train", "test", 5))
    monkeypatch.setattr(refresh, "_current_model_meta", lambda s: {
        "champion_version": None, "trained_at": None, "promoted": None, "gate_reason": None, "skipped_reason": None,
    })
    monkeypatch.setattr(refresh, "get_latest_state", lambda s: {})
    monkeypatch.setattr(refresh, "_fmea_stage", lambda eid, e: (None, None))
    monkeypatch.setattr(refresh, "_action_priority_stage", lambda t, e: (None, None))
    monkeypatch.setattr(refresh, "_warmup_common_prerequisites", lambda eid: None)
    monkeypatch.setattr(refresh, "_pareto_stage", lambda eid, e: ({"Y1": {"items": []}}, None, []))

    class _FakeTable:
        candidates: list = []
        unmeasured_wafer_ids: list = []
        total_wafers = 0
        fallback_summary = type("_FB", (), {"rank_counts": {}, "none_count": 0, "total_combinations": 0})()

    monkeypatch.setattr(refresh, "_yield_prediction_stage", lambda r, t, eid, e: _FakeTable())

    # 알림 발송 파이프라인이 조금이라도 호출되면 즉시 실패시킨다 -- 이
    # 파이프라인의 책임 범위 밖이라는 것을 강하게 확인한다.
    monkeypatch.setattr(
        "src.notifications.yield_update_dispatch.dispatch_yield_update",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_refresh_pipeline은 알림을 보내면 안 된다")),
    )

    refresh.run_refresh_pipeline()

    status = store.get_refresh_snapshot_status()
    assert status["snapshot"] is not None
    assert status["snapshot"]["source"]["mode"] == "manual"
    assert status["snapshot"]["analysis"]["yieldPrediction"] is not None


def test_activate_and_deactivate_dataset_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SC-2/SC-3: 데이터 소스 등록과 분석 실행을 분리했다 -- 파일 선택·
    "데이터베이스에서 불러오기"는 활성 분석 데이터를 등록할 뿐, 4화면
    분석을 자동으로 돌리지 않는다. 사용자가 [분석 시작]을 눌러야
    `POST /api/state/refresh`가 파이프라인을 실행한다."""
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
        assert revert.json() == {"deactivated": True}

        meta2 = client.get("/api/state/snapshot/meta").json()
        assert meta2["manual_eval_override"] is None

    # 등록만 했을 뿐 분석 파이프라인은 한 번도 자동 실행되지 않았다.
    assert calls["pipeline"] == 0
