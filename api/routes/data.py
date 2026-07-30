from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas.data import (
    ColumnDetectionResult,
    DataSummary,
    PreprocessChanges,
    PreprocessResponse,
    ValidationResponse,
    ValidationResult,
)
from src.data_validation import load_data_schema, validate_dataframe
from src.preprocessing import preprocess_dataframe


router = APIRouter(prefix="/api", tags=["data"])

MAX_FILE_SIZE = 20 * 1024 * 1024
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")


async def _read_csv_upload(file: UploadFile) -> tuple[str, pd.DataFrame]:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV 파일을 선택해 주세요.",
        )
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV(.csv) 파일만 업로드할 수 있습니다.",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="업로드한 파일을 읽을 수 없습니다.",
        ) from exc
    finally:
        await file.close()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="파일 크기는 20MB 이하여야 합니다.",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="비어 있는 CSV 파일은 처리할 수 없습니다.",
        )

    for encoding in SUPPORTED_ENCODINGS:
        try:
            dataframe = pd.read_csv(BytesIO(content), encoding=encoding)
            return filename, dataframe
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV 파일에 읽을 수 있는 열이 없습니다.",
            ) from exc
        except pd.errors.ParserError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV 형식이 올바르지 않습니다. 행과 열 구분을 확인해 주세요.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV 파일을 읽는 중 오류가 발생했습니다.",
            ) from exc

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="CSV 인코딩을 확인해 주세요. utf-8-sig, utf-8, cp949를 지원합니다.",
    )


def _validation_payload(
    dataframe: pd.DataFrame,
    validation: dict[str, Any],
) -> ValidationResult:
    schema = load_data_schema()
    id_column = schema["id_column"]
    return ValidationResult(
        is_valid=bool(validation["is_valid"]),
        errors=list(validation["errors"]),
        warnings=list(validation["warnings"]),
        detected_columns=ColumnDetectionResult(
            id=[id_column] if id_column in dataframe.columns else [],
            r=list(validation["r_columns"]),
            d=list(validation["d_columns"]),
            eq=list(validation["eq_columns"]),
            targets=list(validation["target_columns"]),
        ),
        missing_required_columns=list(validation["missing_required_columns"]),
        duplicate_wafer_id_count=int(
            validation["duplicate_wafer_id_count"]
        ),
        total_missing_count=int(validation["total_missing_count"]),
        overall_missing_rate=float(validation["overall_missing_rate"]),
    )


def _json_safe_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _preview_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in dataframe.head(10).to_dict(orient="records"):
        records.append(
            {str(column): _json_safe_value(value) for column, value in row.items()}
        )
    return records


@router.post("/validate", response_model=ValidationResponse)
async def validate_csv(
    file: UploadFile = File(...),
) -> ValidationResponse:
    filename, dataframe = await _read_csv_upload(file)
    validation = validate_dataframe(dataframe)
    return ValidationResponse(
        filename=filename,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        validation=_validation_payload(dataframe, validation),
    )


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_csv(
    file: UploadFile = File(...),
) -> PreprocessResponse:
    filename, dataframe = await _read_csv_upload(file)
    validation = validate_dataframe(dataframe)
    if not validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "데이터 검증 오류로 전처리를 실행할 수 없습니다.",
                "errors": validation["errors"],
            },
        )

    processed, report = preprocess_dataframe(dataframe)
    filled_missing_values = sum(
        int(count) for count in report["imputed_counts"].values()
    )
    clipped_outliers = sum(
        int(count) for count in report["clipped_counts"].values()
    )

    return PreprocessResponse(
        filename=filename,
        before=DataSummary(
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            missing_count=int(dataframe.isna().sum().sum()),
        ),
        after=DataSummary(
            row_count=int(processed.shape[0]),
            column_count=int(processed.shape[1]),
            missing_count=int(processed.isna().sum().sum()),
        ),
        changes=PreprocessChanges(
            filled_missing_values=filled_missing_values,
            clipped_outliers=clipped_outliers,
            added_indicator_columns=list(report["added_indicator_columns"]),
        ),
        warnings=list(report["warnings"]),
        preview=_preview_records(processed),
    )
