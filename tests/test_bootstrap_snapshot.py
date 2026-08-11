"""W-2/W-6: 첫 기동 스냅샷 부트스트랩 -- 단일 실행 잠금, 진행 상태
기록, 그리고 lifespan에서 호출하는 오케스트레이션(`api.main._run_bootstrap`
/ `_bootstrap_snapshot_background`)이 유효 스냅샷·챔피언 유무에 따라
재학습을 건너뛰거나 기다리는지 확인한다."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest

from src.ml.inference import InferenceInputError
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
    set_bootstrap_status/latest_training_job/has_valid_snapshot/
    clear_active_model)만 쓰는지 검증하기 위한 최소 더블. 실제
    RuntimeStore 대신 씀으로써 학습·분석 파이프라인 전체를 매 테스트마다
    돌리지 않아도 된다.

    RA-B3: `active_model_id="m1"`은 실제 `models/`에 존재하지 않으므로
    `_has_usable_champion`이 실제 `load_prediction_model`을 시도하면
    항상 실패해 `clear_active_model`을 호출한다 -- 그래서 이 더블도
    그 호출을 흡수할 수 있어야 한다(champion_exists 플래그가 나타내는
    "레지스트리에 포인터가 있다"는 사실과, 실제 로드 가능 여부는
    별개다: 이 더블은 후자를 항상 "불가능"으로 둔다는 뜻이고, 그 결과
    `_run_bootstrap`은 실제로 매번 `_train_bootstrap_champion`을 타
    실제 재학습을 수행한다 -- 이 파일의 기존 동작 그대로다)."""

    def __init__(self, *, champion_exists: bool, snapshot_after_pipeline: bool) -> None:
        self._champion_exists = champion_exists
        self._snapshot_after_pipeline = snapshot_after_pipeline
        self.statuses: list[tuple[str, str | None]] = []
        self.pipeline_calls = 0
        self.clear_active_model_calls: list[bool] = []

    def active_model(self):
        return {"active_model_id": "m1"} if self._champion_exists else None

    def clear_active_model(self, *, also_clear_previous: bool = False) -> None:
        self.clear_active_model_calls.append(also_clear_previous)
        self._champion_exists = False

    def set_bootstrap_status(self, status, stage, *, error=None, reason=None):
        self.statuses.append((status, stage))
        if error is not None:
            self.statuses.append(("error", error))
        if reason is not None:
            self.statuses.append(("reason", reason))

    def has_valid_snapshot(self) -> bool:
        # 파이프라인이 최소 한 번 불린 뒤에만 스냅샷이 생긴 것으로 본다.
        return self.pipeline_calls > 0 and self._snapshot_after_pipeline

    def latest_training_job(self):
        return None

    def get_manual_eval_override(self):
        return None

    def acquire_bootstrap_lock(self) -> bool:
        return True

    def release_bootstrap_lock(self) -> None:
        return None


def test_run_bootstrap_skips_retrain_when_champion_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    store = _FakeRuntimeStore(champion_exists=True, snapshot_after_pipeline=True)

    dispatch_flags: list[bool] = []

    def tracked_pipeline(*, dispatch: bool = True) -> None:
        store.pipeline_calls += 1
        dispatch_flags.append(dispatch)

    monkeypatch.setattr(main_module, "run_refresh_pipeline", tracked_pipeline)

    asyncio.run(main_module._run_bootstrap(store))

    # 챔피언이 이미 있으므로 파이프라인은 "평가 · 원인분석"용으로 단
    # 한 번만 호출돼야 한다(학습 대기를 위한 추가 호출이 없어야 한다).
    assert store.pipeline_calls == 1
    # WK-5: 콜드 스타트 결과로는 알림을 보내지 않는다.
    assert dispatch_flags == [False]
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
    # RA-B5: 일시적 실패(내장 데이터 누락이 아닌 경우)는 reason을 남기지
    # 않는다 -- 런타임 복구 훅이 다음 요청에서 다시 시도할 수 있어야
    # 하므로, 이 실패가 "영구적"이라는 신호를 오염시키면 안 된다.
    assert not any(entry[0] == "reason" for entry in store.statuses)


def test_run_bootstrap_sets_distinguishable_reason_when_bundled_train_data_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RA-B5: 내장 학습 데이터(data/bundled/train.CSV) 자체가 없어서 실패한
    경우는 재시도해도 소용없는 케이스다 -- 다른 실패와 구분되는 reason을
    남겨야 `BootstrapStatusBanner`가 구체적인(재시도 버튼 없는) 안내를
    보여줄 수 있다."""
    import api.main as main_module

    async def fake_train_bootstrap_champion() -> None:
        raise main_module.BundledTrainingDataMissingError("data/bundled/train.CSV")

    monkeypatch.setattr(main_module, "_train_bootstrap_champion", fake_train_bootstrap_champion)
    store = _FakeRuntimeStore(champion_exists=False, snapshot_after_pipeline=False)

    asyncio.run(main_module._run_bootstrap(store))

    assert ("failed", None) in store.statuses
    reason_entries = [entry for entry in store.statuses if entry[0] == "reason"]
    assert reason_entries == [("reason", main_module.BOOTSTRAP_FAILURE_REASON_DATA_MISSING)]


def test_bootstrap_background_skips_when_snapshot_already_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """RA-B1: 스냅샷도 유효하고 챔피언도 실제로 로드 가능할 때만
    건너뛴다 -- 둘 다 확인해야 한다는 게 이번 근본 원인 수정의 핵심."""
    import api.main as main_module

    class _AlreadyValidStore:
        def get_manual_eval_override(self):
            return None

        def has_valid_snapshot(self) -> bool:
            return True

        def acquire_bootstrap_lock(self) -> bool:
            raise AssertionError("valid snapshot + usable champion을 스킵하지 않고 잠금을 시도했습니다.")

    monkeypatch.setattr(main_module, "_bootstrap_runtime_store", lambda: _AlreadyValidStore())
    monkeypatch.setattr(main_module, "_has_usable_champion", lambda store: True)
    asyncio.run(main_module._bootstrap_snapshot_background())


def test_bootstrap_background_does_not_skip_when_snapshot_valid_but_champion_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RA-B1 근본 원인 재현: 스냅샷 행은 스키마상 유효해도(예: models/가
    비워진 뒤 재시작) 그 model_id를 실제로 로드할 수 없으면 부트스트랩을
    건너뛰면 안 된다. 옛 게이트(`has_valid_snapshot()`만 봄)는 여기서
    그대로 반환해 재학습이 영원히 트리거되지 않았다 -- 이 테스트가
    깨지면 그 회귀가 되살아난 것이다."""
    import api.main as main_module

    class _StaleChampionStore:
        def __init__(self) -> None:
            self.lock_acquired = False
            self.released = False

        def get_manual_eval_override(self):
            return None

        def has_valid_snapshot(self) -> bool:
            return True

        def acquire_bootstrap_lock(self) -> bool:
            self.lock_acquired = True
            return True

        def release_bootstrap_lock(self) -> None:
            self.released = True

    store = _StaleChampionStore()
    calls = {"run_bootstrap": 0}

    async def fake_run_bootstrap(passed_store):
        calls["run_bootstrap"] += 1
        assert passed_store is store

    monkeypatch.setattr(main_module, "_bootstrap_runtime_store", lambda: store)
    monkeypatch.setattr(main_module, "_has_usable_champion", lambda s: False)
    monkeypatch.setattr(main_module, "_run_bootstrap", fake_run_bootstrap)

    asyncio.run(main_module._bootstrap_snapshot_background())

    assert store.lock_acquired is True
    assert calls["run_bootstrap"] == 1
    assert store.released is True


def test_bootstrap_background_skips_when_manual_override_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """NE-6: 수동 override가 있으면 챔피언 상태와 무관하게(has_valid_snapshot
    조차 보지 않고) 곧장 반환한다 -- 사용자가 업로드한 평가 데이터셋을
    내장 데이터 부트스트랩이 덮어쓰지 않는다."""
    import api.main as main_module

    class _ManualOverrideStore:
        def get_manual_eval_override(self):
            return {"dataset_id": "d1", "filename": "f.csv"}

        def has_valid_snapshot(self) -> bool:
            raise AssertionError("manual override가 있으면 has_valid_snapshot을 보지 않고 반환해야 합니다.")

        def acquire_bootstrap_lock(self) -> bool:
            raise AssertionError("manual override를 스킵하지 않고 잠금을 시도했습니다.")

    monkeypatch.setattr(main_module, "_bootstrap_runtime_store", lambda: _ManualOverrideStore())
    asyncio.run(main_module._bootstrap_snapshot_background())


def test_bootstrap_background_skips_when_lock_not_acquired(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    calls = {"run_bootstrap": 0}

    class _LockedStore:
        def get_manual_eval_override(self):
            return None

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


# -- RA-B3: `_has_usable_champion`이 죽은 레지스트리 포인터를 정리하는지 --


class _RegistryStore:
    """`_has_usable_champion`/`ensure_usable_champion`이 등록정보 정리
    (RA-B3) 및 락(RA-B2/B4)과 상호작용하는 방식만 검증하기 위한 최소
    더블."""

    def __init__(self, *, active_model_id: str | None, previous_model_id: str | None = None) -> None:
        self._active_model_id = active_model_id
        self._previous_model_id = previous_model_id
        self.clear_calls: list[bool] = []
        self.lock_acquired_calls = 0
        self.lock_available = True
        self.released = False

    def active_model(self):
        if not self._active_model_id:
            return None
        return {
            "active_model_id": self._active_model_id,
            "previous_model_id": self._previous_model_id,
        }

    def clear_active_model(self, *, also_clear_previous: bool = False) -> None:
        self.clear_calls.append(also_clear_previous)
        self._active_model_id = None

    def acquire_bootstrap_lock(self) -> bool:
        self.lock_acquired_calls += 1
        if not self.lock_available:
            return False
        self.lock_available = False
        return True

    def release_bootstrap_lock(self) -> None:
        self.released = True

    def set_bootstrap_status(self, status, stage, *, error=None, reason=None):
        return None

    def has_valid_snapshot(self) -> bool:
        return False

    def latest_training_job(self):
        return None


def test_has_usable_champion_clears_stale_pointer_when_active_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.main as main_module

    def fake_load(model_id: str, model_dir):
        raise InferenceInputError("존재하지 않거나 사용할 수 없는 모델 ID입니다.")

    monkeypatch.setattr("src.ml.inference.load_prediction_model", fake_load)
    store = _RegistryStore(active_model_id="dead1")

    assert main_module._has_usable_champion(store) is False
    assert store.clear_calls == [False]


def test_has_usable_champion_clears_previous_too_when_also_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """previous_model_id(롤백 대상)도 파일이 없으면 함께 지운다."""
    import api.main as main_module

    def fake_load(model_id: str, model_dir):
        raise InferenceInputError("존재하지 않거나 사용할 수 없는 모델 ID입니다.")

    monkeypatch.setattr("src.ml.inference.load_prediction_model", fake_load)
    store = _RegistryStore(active_model_id="dead1", previous_model_id="dead0")

    assert main_module._has_usable_champion(store) is False
    assert store.clear_calls == [True]


def test_has_usable_champion_keeps_previous_when_still_loadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """previous_model_id는 아직 살아있는 롤백 후보이므로, 그 파일이
    실제로 로드 가능하면 지우지 않는다."""
    import api.main as main_module

    def fake_load(model_id: str, model_dir):
        if model_id == "alive0":
            return object()
        raise InferenceInputError("존재하지 않거나 사용할 수 없는 모델 ID입니다.")

    monkeypatch.setattr("src.ml.inference.load_prediction_model", fake_load)
    store = _RegistryStore(active_model_id="dead1", previous_model_id="alive0")

    assert main_module._has_usable_champion(store) is False
    assert store.clear_calls == [False]


# -- RA-B2/B4: `ensure_usable_champion` (런타임 409 사이트의 복구 훅) --


def test_ensure_usable_champion_fast_path_when_already_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main as main_module

    monkeypatch.setattr(main_module, "_has_usable_champion", lambda store: True)
    store = _RegistryStore(active_model_id="alive1")

    assert main_module.ensure_usable_champion(store) is True
    # 빠른 경로는 락조차 건드리지 않아야 한다 -- 매 요청마다 불릴 수
    # 있으므로 저렴해야 한다는 요구사항.
    assert store.lock_acquired_calls == 0


def test_ensure_usable_champion_spawns_background_recovery_when_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """복구가 필요하면 현재 요청을 블록하지 않고(별도 스레드) 부트스트랩과
    같은 `_run_bootstrap`을 스폰해야 한다."""
    import api.main as main_module

    monkeypatch.setattr(main_module, "_has_usable_champion", lambda store: False)
    recovery_started = threading.Event()

    async def fake_run_bootstrap(passed_store):
        recovery_started.set()

    monkeypatch.setattr(main_module, "_run_bootstrap", fake_run_bootstrap)
    store = _RegistryStore(active_model_id="dead1")

    result = main_module.ensure_usable_champion(store)

    assert result is False
    assert store.lock_acquired_calls == 1
    assert recovery_started.wait(timeout=2.0) is True
    deadline = time.monotonic() + 2.0
    while not store.released and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.released is True


def test_ensure_usable_champion_does_not_spawn_duplicate_when_lock_already_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이미 다른 복구(또는 기동 시 부트스트랩)가 락을 쥐고 있으면 새
    재학습을 스폰하지 않는다 -- 중복 학습을 막는다."""
    import api.main as main_module

    monkeypatch.setattr(main_module, "_has_usable_champion", lambda store: False)
    calls = {"run_bootstrap": 0}

    async def fake_run_bootstrap(passed_store):
        calls["run_bootstrap"] += 1

    monkeypatch.setattr(main_module, "_run_bootstrap", fake_run_bootstrap)
    store = _RegistryStore(active_model_id="dead1")
    store.lock_available = False

    result = main_module.ensure_usable_champion(store)

    assert result is False
    assert calls["run_bootstrap"] == 0
