"""SC-2 "데이터베이스에서 불러오기" -- 등록만 하고 활성화하지 않는지,
서버 미설정/새 데이터 없음을 올바른 상태 코드로 알리는지 확인한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.runtime.store import RuntimeStore


def test_fetch_from_db_requires_sql_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from api.main import app
    import api.routes.state as state_routes

    isolated_store = RuntimeStore(tmp_path / "dashboard.db")
    monkeypatch.setattr(state_routes, "_store", lambda: isolated_store)
    monkeypatch.setattr(state_routes.sql_source, "is_sql_configured", lambda s: False)

    with TestClient(app) as client:
        response = client.post("/api/state/fetch-from-db")
        assert response.status_code == 400


def test_fetch_from_db_registers_without_activating(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from api.main import app
    import api.routes.state as state_routes

    isolated_store = RuntimeStore(tmp_path / "dashboard.db")
    monkeypatch.setattr(state_routes, "_store", lambda: isolated_store)
    monkeypatch.setattr(state_routes.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(
        state_routes.sql_source,
        "fetch_incremental",
        lambda s: pd.DataFrame({"Step1_R1": [1.0, 2.0], "Lot_Wafer_ID": ["L1_W1", "L1_W2"]}),
    )

    class _FakeRegistry:
        def upload(self, filename, content):
            return {"success": True, "dataset_id": "db-fetch-1", "row_count": 2, "column_count": 2}

    monkeypatch.setattr(state_routes, "get_dataset_registry", lambda: _FakeRegistry())

    with TestClient(app) as client:
        response = client.post("/api/state/fetch-from-db")
        assert response.status_code == 200
        body = response.json()
        assert body["dataset_id"] == "db-fetch-1"

        # 등록만 했을 뿐 활성화되지 않았다 -- override는 아직 없다.
        assert isolated_store.get_manual_eval_override() is None


def test_fetch_from_db_no_new_data_is_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from api.main import app
    import api.routes.state as state_routes

    isolated_store = RuntimeStore(tmp_path / "dashboard.db")
    monkeypatch.setattr(state_routes, "_store", lambda: isolated_store)
    monkeypatch.setattr(state_routes.sql_source, "is_sql_configured", lambda s: True)
    monkeypatch.setattr(state_routes.sql_source, "fetch_incremental", lambda s: pd.DataFrame())

    with TestClient(app) as client:
        response = client.post("/api/state/fetch-from-db")
        assert response.status_code == 404
