from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

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
from src.analysis.screening.schema import parse_schema
from src.analysis.target_hydration import inspect_target_status
from src.runtime.store import RuntimeStore
from src.upload_limits import max_upload_size_bytes, max_upload_size_mb

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def get_dataset_registry() -> DatasetRegistry:
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    return DatasetRegistry(store, settings.dataset_upload_dir, settings.bundled_dataset_dir)


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> dict[str, Any]:
    t0 = time.perf_counter()
    items = get_dataset_registry().list_datasets()
    logger.info("list_datasets %.1fms (n=%d)", (time.perf_counter() - t0) * 1000, len(items))
    return {"items": items}


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
    # CSV parsing + full-dataframe validation is CPU-bound; run off the
    # event loop so a large upload doesn't stall every other request on
    # this single-worker process.
    return await run_in_threadpool(get_dataset_registry().upload, filename, content)


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

    schema = parse_schema(df)
    missing_rates = {
        column: float(df[column].isna().mean())
        for column in [*schema.r_cols, *schema.d_cols, *schema.config_cols]
    }

    def _measurement_rate(columns: list[str]) -> float | None:
        # Mean of *per-column* rates (spec: "인자별 계측률의 평균을 쓴다.
        # 전체 셀 기준이 아니다") -- averaging over columns first, not
        # flattening every cell into one pool, so columns with very
        # different row-level measurement patterns don't let a handful of
        # densely-measured columns hide the rest.
        if not columns:
            return None
        return float(df[columns].notna().mean().mean() * 100.0)

    return {
        "dataset_id": dataset_id,
        "steps_present": schema.steps_present,
        "config_steps": [step for column in schema.config_cols if (step := schema.step_of(column)) is not None],
        "max_step": schema.max_step,
        "r_columns": schema.r_cols,
        "d_columns": schema.d_cols,
        "config_columns": schema.config_cols,
        "target_columns": schema.target_cols,
        "unmapped_columns": schema.unmapped,
        "missing_rates": missing_rates,
        "r_measurement_rate": _measurement_rate(schema.r_cols),
        "d_measurement_rate": _measurement_rate(schema.d_cols),
        "target_status": inspect_target_status(df).as_dict(),
    }
