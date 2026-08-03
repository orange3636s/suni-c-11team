from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import Response

from api.schemas.datasets import (
    DatasetDeleteResponse,
    DatasetListResponse,
    DatasetSchemaResponse,
    DatasetUploadResponse,
)
from api.settings import settings
from src.runtime.datasets import (
    BundledDatasetDeleteError,
    DatasetNotFoundError,
    DatasetRegistry,
)
from src.runtime.store import RuntimeStore
from src.upload_limits import max_upload_size_bytes, max_upload_size_mb

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def get_dataset_registry() -> DatasetRegistry:
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    return DatasetRegistry(store, settings.dataset_upload_dir, settings.bundled_dataset_dir)


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> dict[str, Any]:
    return {"items": get_dataset_registry().list_datasets()}


@router.post("", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV 파일만 업로드할 수 있습니다.")
    limit_bytes = max_upload_size_bytes()
    content = await file.read(limit_bytes + 1)
    if len(content) > limit_bytes:
        actual_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"파일이 너무 큽니다 (최대 {max_upload_size_mb()}MB). 현재 {actual_mb:.1f}MB",
        )
    return get_dataset_registry().upload(filename, content)


@router.delete("/{dataset_id}", response_model=DatasetDeleteResponse)
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        get_dataset_registry().delete(dataset_id)
    except BundledDatasetDeleteError as exc:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="내장 데이터셋은 삭제할 수 없습니다.",
        ) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc
    return {"success": True, "dataset_id": dataset_id}


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str) -> Response:
    """Raw CSV bytes so the frontend can feed the selected dataset straight
    into the existing /api/train(/jobs) file-upload contract without that
    endpoint needing to learn about dataset_id at all.
    """
    registry = get_dataset_registry()
    summary = registry.get_summary(dataset_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.")
    df = registry.get_dataframe(dataset_id)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    filename = summary["original_filename"]
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{dataset_id}/schema", response_model=DatasetSchemaResponse)
def get_dataset_schema(dataset_id: str) -> dict[str, Any]:
    registry = get_dataset_registry()
    try:
        df = registry.get_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc

    from src.analysis.screening.schema import parse_schema

    schema = parse_schema(df)
    missing_rates = {
        column: float(df[column].isna().mean())
        for column in [*schema.r_cols, *schema.d_cols, *schema.config_cols]
    }
    return {
        "dataset_id": dataset_id,
        "steps_present": schema.steps_present,
        "max_step": schema.max_step,
        "r_columns": schema.r_cols,
        "d_columns": schema.d_cols,
        "config_columns": schema.config_cols,
        "target_columns": schema.target_cols,
        "unmapped_columns": schema.unmapped,
        "missing_rates": missing_rates,
    }
