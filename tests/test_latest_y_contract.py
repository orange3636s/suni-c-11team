from __future__ import annotations

import pandas as pd
import pytest

from api.main import app
from src.ml.dataset import EXCLUDED_TARGET_COLUMNS, prepare_dataset


def _multipart_properties(path: str) -> set[str]:
    openapi = app.openapi()
    schema = openapi["paths"][path]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    component = schema["$ref"].rsplit("/", 1)[-1]
    return set(openapi["components"]["schemas"][component]["properties"])


def test_training_uses_only_server_side_y_contract() -> None:
    assert _multipart_properties("/api/train") == {"file"}
    assert _multipart_properties("/api/train/jobs") == {"file"}


def test_y1_through_y10_and_identifiers_never_become_features() -> None:
    rows = 12
    frame = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"LOT-{index // 3}_W{index}" for index in range(rows)],
            "Lot_ID": [f"LOT-{index // 3}" for index in range(rows)],
            "Wafer_ID": [f"W{index}" for index in range(rows)],
            "Wafer_Slot": list(range(rows)),
            "Step1_R1": [float(index) for index in range(rows)],
            "Step1_EQ": ["EQ-A" if index % 2 else "EQ-B" for index in range(rows)],
            "Y": [90.0 + index / 10 for index in range(rows)],
            **{column: [index % 2 for index in range(rows)] for column in EXCLUDED_TARGET_COLUMNS},
        }
    )

    dataset = prepare_dataset(frame, add_missing_indicators=False)

    forbidden = {"Y", *EXCLUDED_TARGET_COLUMNS, "Lot_Wafer_ID", "Lot_ID", "Wafer_ID", "Wafer_Slot"}
    assert forbidden.isdisjoint(dataset.feature_columns)
    assert dataset.feature_columns == ["Step1_R1", "Step1_EQ"]


def test_missing_y_has_actionable_korean_error() -> None:
    frame = pd.DataFrame(
        {
            "Lot_Wafer_ID": [f"LOT_W{index}" for index in range(12)],
            "Step1_R1": [float(index) for index in range(12)],
        }
    )

    with pytest.raises(ValueError) as error:
        prepare_dataset(frame)

    assert str(error.value) == (
        "학습 데이터에 최종 수율 컬럼 Y가 없습니다. "
        "Y 컬럼이 포함된 CSV 파일을 선택해주세요."
    )


def test_latest_model_metadata_route_is_registered() -> None:
    assert "get" in app.openapi()["paths"]["/api/model/latest"]
