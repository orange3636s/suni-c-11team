"""Deleting an uploaded dataset must invalidate any saved 학습/원인 분석/
사전 알람 result that references it (spec: 학습·분석 결과 상태 유지 §3-5),
without touching results for other datasets, and without ever firing on
upload (uploads don't call delete()).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.runtime.app_state import save_state
from src.runtime.datasets import DatasetRegistry
from src.runtime.store import RuntimeStore


def _registry(tmp_path: Path) -> DatasetRegistry:
    store = RuntimeStore(tmp_path / "dashboard.db")
    upload_root = tmp_path / "uploads"
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir(parents=True, exist_ok=True)
    return DatasetRegistry(store=store, upload_root=upload_root, bundled_root=bundled_root)


def _seed_uploaded_dataset(registry: DatasetRegistry, dataset_id: str) -> None:
    stored_path = f"{dataset_id}.csv"
    (registry.upload_root / stored_path).write_text("Lot_Wafer_ID,Y\nA,1\n", encoding="utf-8")
    registry.store.create_dataset(
        dataset_id=dataset_id,
        original_filename="uploaded.csv",
        stored_path=stored_path,
        row_count=1,
        column_count=2,
    )


def test_deleting_dataset_invalidates_its_own_saved_state(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    dataset_id = f"upload-{uuid4().hex}"
    _seed_uploaded_dataset(registry, dataset_id)

    save_state(registry.store, "analysis", dataset={"dataset": dataset_id}, payload={})
    save_state(registry.store, "alarms", dataset={"train_dataset": dataset_id, "eval_dataset": "test"}, payload={})

    registry.delete(dataset_id)

    assert registry.store.get_app_state("latest_analysis") is None
    assert registry.store.get_app_state("latest_alarms") is None


def test_deleting_dataset_leaves_other_datasets_results_alone(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    target_id = f"upload-{uuid4().hex}"
    other_id = f"upload-{uuid4().hex}"
    _seed_uploaded_dataset(registry, target_id)
    _seed_uploaded_dataset(registry, other_id)

    save_state(registry.store, "training", dataset={"dataset": other_id}, payload={"kept": True})

    registry.delete(target_id)

    record = registry.store.get_app_state("latest_training")
    assert record is not None
    assert record["payload"] == {"kept": True}


def test_uploading_a_new_dataset_never_invalidates_existing_state(tmp_path: Path) -> None:
    """spec: "데이터셋을 새로 업로드해도 기존 결과를 지우지 마라" -- upload()
    never reaches invalidate_state_for_dataset at all (it's only wired
    into delete()), verified end-to-end via a real upload call against
    the actual bundled train.CSV (upload() validates new files against
    its schema, so a real bundled_root is needed here, unlike the other
    tests in this module).
    """
    bundled_root = Path(__file__).resolve().parents[1] / "data" / "bundled"
    if not (bundled_root / "train.CSV").exists():
        pytest.skip("data/bundled/train.CSV not present")
    store = RuntimeStore(tmp_path / "dashboard.db")
    registry = DatasetRegistry(store=store, upload_root=tmp_path / "uploads", bundled_root=bundled_root)
    save_state(store, "training", dataset={"dataset": "train"}, payload={"kept": True})

    train_bytes = (bundled_root / "train.CSV").read_bytes()
    result = registry.upload("reuploaded_train.csv", train_bytes)
    assert result.get("success") is True

    record = store.get_app_state("latest_training")
    assert record is not None
    assert record["payload"] == {"kept": True}


def test_deleting_dataset_clears_dataframe_cache_for_reused_path(tmp_path: Path) -> None:
    """H-3③: `get_dataframe`은 파일 경로로 lru_cache된다 -- 삭제 후 같은
    stored_path에 새 파일이 놓이면(업로드 파일명이 겹치는 경우) 캐시가
    옛 내용을 계속 돌려주면 안 된다."""
    registry = _registry(tmp_path)
    dataset_id = f"upload-{uuid4().hex}"
    _seed_uploaded_dataset(registry, dataset_id)

    first = registry.get_dataframe(dataset_id)
    assert first["Y"].iloc[0] == 1

    registry.delete(dataset_id)

    # 같은 stored_path에 다른 내용의 새 데이터셋을 등록한다.
    stored_path = f"{dataset_id}.csv"
    (registry.upload_root / stored_path).write_text("Lot_Wafer_ID,Y\nA,999\n", encoding="utf-8")
    registry.store.create_dataset(
        dataset_id=dataset_id,
        original_filename="uploaded.csv",
        stored_path=stored_path,
        row_count=1,
        column_count=2,
    )

    second = registry.get_dataframe(dataset_id)
    assert second["Y"].iloc[0] == 999
