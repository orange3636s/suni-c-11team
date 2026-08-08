"""J-1/J-2: 리프레시 파이프라인의 재학습 생략 조건과 데이터 소스 해석을
단위 테스트한다. 실제 GBDT 적합·부트스트랩은 무거우므로, 학습 제출
경로는 `_run_persisted_training_job`을 모킹해 실제로 돌리지 않는다."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest

from src.automation import refresh
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore


def _store_and_registry(tmp_path: Path) -> tuple[RuntimeStore, DatasetRegistry]:
    store = RuntimeStore(tmp_path / "dashboard.db")
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    return store, registry


def test_maybe_retrain_skips_when_hash_unchanged_and_champion_exists(tmp_path: Path) -> None:
    if not (Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV").exists():
        pytest.skip("data/bundled/train.CSV not present")
    store, registry = _store_and_registry(tmp_path)
    content = (registry.bundled_root / "train.CSV").read_bytes()
    content_hash = refresh._content_hash(content)

    store.promote_if_better(
        model_id="m-existing", pipeline_version="v1", dataset_version=0,
        metadata={"model_id": "m-existing", "metrics": {"test": {"r2": 0.5}}},
    )
    store.set_app_state(refresh.DATA_HASH_STATE_KEY, {"value": content_hash})

    errors: list[str] = []
    result = refresh._maybe_retrain(store, registry, "fallback", errors)

    assert result["skipped_reason"] == "데이터 내용 변경 없음 -- 재학습 생략"
    assert errors == []


def test_maybe_retrain_submits_training_when_no_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not (Path(__file__).resolve().parents[1] / "data" / "bundled" / "train.CSV").exists():
        pytest.skip("data/bundled/train.CSV not present")
    store, registry = _store_and_registry(tmp_path)

    def _fake_training_job(input_path, filename, options, progress):
        return {"model_id": "fake-model", "status": "completed"}

    import api.routes.data as data_routes

    monkeypatch.setattr(data_routes, "_run_persisted_training_job", _fake_training_job)

    errors: list[str] = []
    result = refresh._maybe_retrain(store, registry, "fallback", errors)

    assert result["skipped_reason"] is None
    assert result.get("training_job_submitted")
    # 해시가 기록돼 있어야 다음 사이클(같은 데이터)에 재제출하지 않는다.
    stored = store.get_app_state(refresh.DATA_HASH_STATE_KEY)
    assert stored is not None

    # 백그라운드 학습 스레드가 operation_coordinator 예약을 풀 시간을 준다
    # (다음 테스트가 "다른 작업 실행 중"으로 오판하지 않도록).
    for _ in range(50):
        if not any(data_routes.operation_coordinator.snapshot().values()):
            break
        time.sleep(0.1)


def test_resolve_source_falls_back_when_sql_not_configured(tmp_path: Path) -> None:
    store, registry = _store_and_registry(tmp_path)
    errors: list[str] = []
    mode, train_id, eval_id, row_count = refresh._resolve_source(store, registry, errors)
    assert mode == "fallback"
    assert train_id == refresh.FALLBACK_TRAIN_DATASET
    assert eval_id == refresh.FALLBACK_EVAL_DATASET
