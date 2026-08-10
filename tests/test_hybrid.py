from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile

import api.routes.data as data_routes
from src.ml.hybrid import FAIL_RATE_TARGETS, normalized_failure_rates

COUNT_TARGETS = [f"Y{index}" for index in range(6, 11)]


@pytest.fixture
def hybrid_model_dir():
    root = Path(__file__).parent / ".tmp_hybrid_models"
    root.mkdir(exist_ok=True)
    output = root / f"run_{uuid4().hex}"
    output.mkdir()
    yield output
    for generated_path in output.iterdir():
        if generated_path.is_dir():
            for generated_file in generated_path.iterdir():
                generated_file.unlink()
            generated_path.rmdir()
        else:
            generated_path.unlink()
    output.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


@pytest.fixture(scope="module")
def hybrid_dataframe() -> pd.DataFrame:
    random = np.random.default_rng(2026)
    rows = 100
    response = random.normal(size=rows)
    rates = np.column_stack([
        np.clip(1.2 + (index + 1) * 0.25 * response + random.normal(0, 0.08, rows), 0, None)
        for index in range(5)
    ])
    frame = pd.DataFrame({
        "Lot_Wafer_ID": [f"LOT{index // 5:02d}_WF{index % 5 + 1:02d}" for index in range(rows)],
        "Lot_ID": [f"LOT{index // 5:02d}" for index in range(rows)],
        "Y": np.clip(100.0 - rates.sum(axis=1), 0, 100),
        "Step1_R1": response,
        "Step1_D1": np.where(np.arange(rows) % 7 == 0, np.abs(response), 0.0),
        "Step1_Config": [f"Step1_Model{index % 3}_EQ{index % 4}_CH{index % 2}" for index in range(rows)],
    })
    for index, target in enumerate(FAIL_RATE_TARGETS):
        frame[target] = rates[:, index]
    for index, target in enumerate(COUNT_TARGETS, 1):
        frame[target] = np.clip(30 + index * 5 + response * index * 3, 0, None)
    return frame


def test_final_y_uses_nonnegative_unscaled_failure_rates() -> None:
    rates = np.array([[-1, 2, 3, 4, 5], [80, 30, 10, 0, 0]], dtype=float)
    nonnegative, derived, overflow_count = normalized_failure_rates(rates)
    assert nonnegative[0].tolist() == pytest.approx([0, 2, 3, 4, 5])
    assert nonnegative[1].sum() == pytest.approx(120.0)
    assert derived.tolist() == pytest.approx([86.0, 0.0])
    assert overflow_count == 1


def test_train_api_forces_automatic_contract(
    hybrid_dataframe: pd.DataFrame,
    hybrid_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_routes, "MODEL_DIR", hybrid_model_dir)
    upload = UploadFile(
        file=BytesIO(hybrid_dataframe.to_csv(index=False).encode("utf-8")),
        filename="automatic.csv",
    )
    response = asyncio.run(data_routes.train_model(upload))
    # ND: LightGBM replaced HistGradientBoostingRegressor as the one
    # production model (src/ml/pipeline.py's module docstring has the
    # comparison that picked it).
    assert response.model_type == "LGBMRegressor"
    assert response.split.train_rows + response.split.validation_rows + response.split.test_rows == len(hybrid_dataframe)
    assert response.target == "Y"
