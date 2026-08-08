"""Route-level test for POST /api/notify/dispatch (api/routes/notify.py) --
A-1: the existing tests/test_notify_dispatch.py only calls
`dispatch.dispatch_alarm_notifications` directly, which never exercises
`notify.dispatch_now`/`run_daily_dispatch_job` themselves. Both called
`_cached_reliability(train)` with a single argument against a 2-argument
signature `(dataset_id, eval_dataset_id)` -- a TypeError that only a call
through the actual route function (same convention as
test_state_endpoints.py: route functions called directly with `settings`
monkeypatched, not full TestClient/HTTP) would catch.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes import datasets as datasets_routes
from api.routes import notify as notify_routes
from api.schemas.notify import DispatchRequest
from src.notifications import settings_store

BUNDLED_ROOT = Path(__file__).resolve().parents[1] / "data" / "bundled"

pytestmark = pytest.mark.skipif(
    not (BUNDLED_ROOT / "train.CSV").exists() and not (BUNDLED_ROOT / "test.CSV").exists(),
    reason="data/bundled의 train/test.CSV가 없어 건너뜁니다.",
)


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime_db = tmp_path / "runtime" / f"dashboard_{uuid4().hex}.db"
    artifact_root = tmp_path / "runtime"
    test_settings = SimpleNamespace(
        runtime_db_path=runtime_db,
        runtime_artifact_dir=artifact_root,
        dataset_upload_dir=tmp_path / "uploads",
        bundled_dataset_dir=BUNDLED_ROOT,
    )
    # notify.py와 datasets.py 둘 다 `from api.settings import settings`로
    # 각자 모듈 스코프에 이름을 바인딩하므로 두 곳 다 갈아끼워야 같은
    # 격리된 RuntimeStore/bundled 경로를 본다.
    monkeypatch.setattr(notify_routes, "settings", test_settings)
    monkeypatch.setattr(datasets_routes, "settings", test_settings)
    return test_settings


def test_dispatch_now_route_does_not_crash_on_reliability_call(
    isolated_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A-1 회귀: 이 호출이 예전에는 `_cached_reliability(body.train_dataset)`
    처럼 인자 1개로 불려 TypeError를 던지며 즉시 발송 경로 전체가
    죽어 있었다."""
    store = notify_routes._store()
    settings_store.save_slack(store, webhook_url="https://hooks.slack.com/services/FAKE/FAKE/FAKE", channel="#eng-yield")

    sent = {"called": False}

    def _fake_send(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        sent["called"] = True
        return True, None

    monkeypatch.setattr(notify_routes.dispatch.senders, "send_slack_alarm", _fake_send)

    result = notify_routes.dispatch_now(DispatchRequest(train_dataset="train", eval_dataset="test", dashboard_url=None))

    assert isinstance(result, dict)
    assert "skipped" in result
