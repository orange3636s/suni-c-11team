import asyncio
import inspect
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from api.routes.data import preprocess_csv, validate_csv


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


def _valid_csv() -> bytes:
    return (
        "Lot_Wafer_ID,Y,Step1_R1,Step1_D1,Step1_EQ\n"
        "LOT01_W01,98.1,1.0,0,EQ1\n"
        "LOT01_W02,97.5,,1,\n"
    ).encode("utf-8")


def test_non_csv_file_is_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(validate_csv(_upload("sample.txt", b"hello")))

    assert error.value.status_code == 400
    assert "CSV" in str(error.value.detail)


def test_valid_csv_validation_succeeds() -> None:
    response = asyncio.run(validate_csv(_upload("sample.csv", _valid_csv())))

    assert response.success is True
    assert response.row_count == 2
    assert response.column_count == 5
    assert response.validation.is_valid is True
    assert response.validation.detected_columns.r == ["Step1_R1"]


def test_valid_csv_preprocessing_succeeds() -> None:
    response = asyncio.run(preprocess_csv(_upload("sample.csv", _valid_csv())))

    assert response.success is True
    assert response.changes.filled_missing_values == 2
    assert len(response.preview) == 2
    assert response.preview[1]["Step1_R1"] is not None
    assert response.preview[1]["Step1_EQ"] == "UNKNOWN"


def test_file_field_is_required() -> None:
    file_parameter = inspect.signature(validate_csv).parameters["file"]

    assert file_parameter.default.is_required() is True


def test_unsupported_encoding_is_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            validate_csv(_upload("sample.csv", b"\xff\xfe\xff\xfe"))
        )

    assert error.value.status_code == 400
    assert "인코딩" in str(error.value.detail)


def test_preprocess_preview_is_json_safe() -> None:
    csv_content = (
        "Lot_Wafer_ID,Y,Step1_R1,Step1_EQ\n"
        "LOT01_W01,98.1,inf,EQ1\n"
    ).encode("utf-8")

    response = asyncio.run(
        preprocess_csv(_upload("sample.csv", csv_content))
    )
    serialized = response.model_dump(mode="json")

    assert serialized["preview"][0]["Step1_R1"] is None
