from __future__ import annotations

import asyncio
import csv
import gc
import logging
import math
import os
from contextvars import ContextVar
from functools import partial
import threading
import time
from collections import Counter
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from starlette.concurrency import run_in_threadpool

from api.schemas.data import (
    ColumnDetectionResult,
    DataSummary,
    DatasetSplit,
    EvaluationSummary,
    ModelArtifacts,
    ModelComparisonItem,
    ModelDeleteResponse,
    ModelDetailMetrics,
    ModelDetailResponse,
    ModelListResponse,
    ModelMetrics,
    ModelSummary,
    PreprocessChanges,
    PreprocessResponse,
    TrainResponse,
    ValidationResponse,
    ValidationResult,
)
from api.schemas.jobs import TrainJobAccepted, TrainJobResult, TrainJobStatus
from api.settings import settings
from src.analysis.screening.schema import parse_schema
from src.data_validation import load_data_schema, validate_dataframe
from src.ml.dataset import (
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.ml.hybrid import save_hybrid_bundle
from src.ml.pipeline import (
    FINAL_YIELD_COLUMN,
    build_hybrid_training_result,
    target_metrics_summary,
    train_and_evaluate,
)
from src.ml.evaluation import evaluate_regression
from src.ml.inference import (
    InferenceInputError,
    InvalidModelIdError,
    ModelDeletionError,
    ModelNotFoundError,
    ModelLoadError,
    get_prediction_model_detail,
    delete_model_bundle,
    list_prediction_models,
    load_latest_model_bundle,
    get_latest_model_metadata,
)
from src.preprocessing import preprocess_dataframe
from src.runtime.store import RuntimeStore
from src.runtime.operation_coordinator import (
    HEAVY_JOB_MESSAGE,
    ActiveOperationError,
    operation_coordinator,
)
from src.runtime.training_jobs import (
    ProgressCallback,
    TrainingJobManager,
    new_training_job_id,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["data"])

MAX_FILE_SIZE = settings.max_upload_size_bytes
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")
MODEL_DIR = settings.model_dir
_TRAINING_LOCK = threading.Lock()
_TRAINING_PROGRESS: ContextVar[ProgressCallback | None] = ContextVar(
    "training_progress",
    default=None,
)
_TRAINING_JOB_MANAGER: TrainingJobManager | None = None
_TRAINING_JOB_MANAGER_LOCK = threading.Lock()


def _runtime_store() -> RuntimeStore:
    configured_model_dir = Path(settings.model_dir).resolve()
    resolved_model_dir = Path(MODEL_DIR).resolve()
    if resolved_model_dir != configured_model_dir:
        runtime_root = resolved_model_dir / ".runtime"
        return RuntimeStore(runtime_root / "dashboard.db", runtime_root)
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _latest_model() -> Any:
    try:
        return load_latest_model_bundle(_runtime_store(), MODEL_DIR)
    except InferenceInputError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


def get_training_job_manager() -> TrainingJobManager:
    global _TRAINING_JOB_MANAGER
    with _TRAINING_JOB_MANAGER_LOCK:
        if _TRAINING_JOB_MANAGER is None:
            store = RuntimeStore(
                settings.runtime_db_path,
                settings.runtime_artifact_dir,
            )
            _TRAINING_JOB_MANAGER = TrainingJobManager(
                store=store,
                input_root=settings.training_job_artifact_dir,
                coordinator=operation_coordinator,
            )
        return _TRAINING_JOB_MANAGER


def recover_interrupted_training_jobs() -> int:
    return get_training_job_manager().recover_interrupted()


def _report_training_progress(stage: str, progress: int) -> None:
    callback = _TRAINING_PROGRESS.get()
    if callback is None:
        return
    try:
        callback(stage, progress)
    except Exception:
        logger.warning("학습 Job 진행률 저장 실패", exc_info=True)


def _duplicate_names(values: list[str]) -> list[str]:
    counts = Counter(values)
    return list(
        dict.fromkeys(
            value for value in values if counts[value] > 1
        )
    )


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
        dataframe = await run_in_threadpool(
            _read_csv_stream,
            file.file,
        )
        return filename, dataframe
    finally:
        await file.close()


def _read_csv_stream(source: Any) -> pd.DataFrame:
    """Parse a seekable upload without retaining bytes and decoded text copies."""
    try:
        source.seek(0, os.SEEK_END)
        size = int(source.tell())
        source.seek(0)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="업로드한 파일을 읽을 수 없습니다.",
        ) from exc
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "파일 크기는 "
                f"{settings.max_upload_size_mb}MB 이하여야 합니다."
            ),
        )
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="비어 있는 CSV 파일은 처리할 수 없습니다.",
        )

    for encoding in SUPPORTED_ENCODINGS:
        wrapper: TextIOWrapper | None = None
        try:
            source.seek(0)
            wrapper = TextIOWrapper(source, encoding=encoding, newline="")
            header = next(csv.reader(wrapper), [])
            duplicate_columns = _duplicate_names(header)
            if duplicate_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "CSV에 중복된 컬럼명이 있습니다: "
                        + ", ".join(duplicate_columns)
                    ),
                )
            wrapper.detach()
            wrapper = None
            source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except HTTPException:
            raise
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
        finally:
            if wrapper is not None:
                try:
                    wrapper.detach()
                except (OSError, ValueError):
                    pass

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="CSV 인코딩을 확인해 주세요. utf-8-sig, utf-8, cp949를 지원합니다.",
    )


async def _persist_training_job_upload(
    file: UploadFile,
    destination: Path,
) -> str:
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

    temporary = destination.with_name(f"{destination.name}.tmp")
    total = 0
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "파일 크기는 "
                            f"{settings.max_upload_size_mb}MB 이하여야 합니다."
                        ),
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="비어 있는 CSV 파일은 처리할 수 없습니다.",
            )
        os.replace(temporary, destination)
        return filename
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 CSV를 작업 저장소에 보관하지 못했습니다.",
        ) from exc
    finally:
        await file.close()
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                logger.warning("학습 Job 임시 업로드 파일 정리 실패", exc_info=True)


def _training_job_result(response: TrainResponse) -> dict[str, Any]:
    test_metrics = response.metrics.get("test")
    model_id = response.model_id or Path(response.artifacts.metadata_file).stem
    return TrainJobResult(
        model_id=model_id,
        target=response.target,
        best_model=response.best_model,
        test_metrics=(
            test_metrics.model_dump(mode="json")
            if test_metrics is not None
            else None
        ),
        feature_count=response.feature_count,
        warning_count=len(response.warnings),
    ).model_dump(mode="json")


def _run_persisted_training_job(
    input_path: Path,
    filename: str,
    options: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    token = _TRAINING_PROGRESS.set(progress)
    try:
        with input_path.open("rb") as source:
            upload = UploadFile(file=source, filename=filename)
            response = asyncio.run(train_model(upload, **options))
        return _training_job_result(response)
    finally:
        _TRAINING_PROGRESS.reset(token)


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
            config=list(validation.get("config_columns", [])),
        ),
        missing_required_columns=list(validation["missing_required_columns"]),
        duplicate_wafer_id_count=int(
            validation["duplicate_wafer_id_count"]
        ),
        total_missing_count=int(validation["total_missing_count"]),
        overall_missing_rate=float(validation["overall_missing_rate"]),
        schema_version=validation.get("schema_version"),
        validation_mode=validation.get("validation_mode"),
        config_completeness_rate=float(validation.get("config_completeness_rate", 0.0)),
        r_measurement_coverage=float(validation.get("r_measurement_coverage", 0.0)),
        d_measurement_coverage=float(validation.get("d_measurement_coverage", 0.0)),
        required_field_error_count=int(validation.get("required_field_error_count", 0)),
        config_parse_error_count=int(validation.get("config_parse_error_count", 0)),
        target_consistency_rate=validation.get("target_consistency_rate"),
        lot_structure_consistency_rate=validation.get("lot_structure_consistency_rate"),
        duplicate_wafer_count=int(validation.get("duplicate_wafer_count", 0)),
        invalid_numeric_count=int(validation.get("invalid_numeric_count", 0)),
        structural_unmeasured_count=int(validation.get("structural_unmeasured_count", 0)),
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


@router.get("/model/latest")
def get_latest_model() -> dict[str, Any]:
    return {"latest_model": get_latest_model_metadata(_runtime_store())}


@router.post("/validate", response_model=ValidationResponse)
async def validate_csv(
    file: UploadFile = File(...),
    validation_mode: Annotated[str, Form()] = "training",
) -> ValidationResponse:
    filename, dataframe = await _read_csv_upload(file)
    validation = validate_dataframe(dataframe, validation_mode=validation_mode)
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

    preview_frame = processed.astype(object)
    maintained_columns = report.get("numeric_feature_columns", [])
    for column in maintained_columns:
        if column in preview_frame.columns:
            missing_mask = pd.isna(preview_frame[column])
            preview_frame.loc[missing_mask, column] = "NaN (유지)"

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
        preview=_preview_records(preview_frame),
        schema_version=validation.get("schema_version"),
        measurement_coverage={
            "r": float(validation.get("r_measurement_coverage", 0.0)),
            "d": float(validation.get("d_measurement_coverage", 0.0)),
        },
        preprocessing_policy=report.get("preprocessing_policy", {}),
        config_summary=report.get("config_parsing_result", {}),
        processing_summary=report.get("processing_summary", {}),
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    file: UploadFile = File(...),
) -> TrainResponse:
    training_started_at = time.perf_counter()
    filename, dataframe = await _read_csv_upload(file)
    if TARGET_COLUMN not in dataframe.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "학습 데이터에 최종 수율 컬럼 Y가 없습니다. "
                "Y 컬럼이 포함된 CSV 파일을 선택해주세요."
            ),
        )
    logger.info(
        "학습 CSV 읽기 완료: rows=%d, columns=%d",
        dataframe.shape[0],
        dataframe.shape[1],
    )
    _report_training_progress("학습 CSV 확인", 10)

    # Fixed server-side training contract.
    target = TARGET_COLUMN
    train_ratio, validation_ratio, test_ratio = 70, 15, 15
    missing_indicator = False
    compare_missingness = False
    ensemble_enabled = False

    split_ratios = (train_ratio, validation_ratio, test_ratio)
    if sum(split_ratios) != 100 or any(
        ratio < 5 or ratio > 90 for ratio in split_ratios
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Train/Validation/Test 비율은 각각 5~90%이며 "
                "합계가 100%여야 합니다."
            ),
        )

    try:
        validation = await run_in_threadpool(
            partial(validate_dataframe, dataframe, validation_mode="training")
        )
    except Exception as exc:
        logger.exception("학습 데이터 검증 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="데이터 검증 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    if not validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "데이터 검증 오류로 모델을 학습할 수 없습니다.",
                "errors": validation["errors"],
            },
        )
    logger.info("학습 데이터 검증 완료")
    _report_training_progress("학습 데이터 검증", 20)

    if target is None:
        # Hybrid training owns fold-scoped preprocessing.  Building another
        # full processed DataFrame here doubled peak memory without being used.
        preprocessing_report = {
            "warnings": [],
            "processing_summary": {
                "report_source": "hybrid_fold_scoped_preprocessing",
            },
        }
    else:
        try:
            processed, preprocessing_report = await run_in_threadpool(
                preprocess_dataframe, dataframe
            )
        except Exception as exc:
            logger.exception("학습 데이터 전처리 중 내부 오류")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="데이터 전처리 중 서버 내부 오류가 발생했습니다.",
            ) from exc
        logger.info(
            "학습 데이터 전처리 Summary 완료: rows=%d, columns=%d",
            processed.shape[0],
            processed.shape[1],
        )
        del processed
        gc.collect()
    _report_training_progress("학습 데이터 전처리", 30)

    schema = parse_schema(dataframe)
    if not schema.target_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Y1~Y5 타깃 열이 없어 학습할 수 없습니다.",
        )

    lot_column = "Lot_ID"
    has_groups = lot_column in dataframe.columns and dataframe[lot_column].nunique(dropna=True) >= 10
    if has_groups:
        groups = dataframe[lot_column].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
        train_indices, test_indices = next(splitter.split(dataframe, groups=groups))
        split_method = "group_shuffle_lot_85_15"
        group_split_used = True
    else:
        all_indices = np.arange(len(dataframe))
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.15, random_state=RANDOM_STATE
        )
        split_method = "random_shuffle_85_15"
        group_split_used = False

    internal_train = dataframe.iloc[train_indices].reset_index(drop=True)
    internal_test = dataframe.iloc[test_indices].reset_index(drop=True)
    train_lots = (
        sorted(internal_train[lot_column].astype(str).unique().tolist())
        if lot_column in dataframe.columns
        else []
    )
    test_lots = (
        sorted(internal_test[lot_column].astype(str).unique().tolist())
        if lot_column in dataframe.columns
        else []
    )

    _report_training_progress("인자 스크리닝 및 모델 학습", 40)
    try:
        evaluation = await run_in_threadpool(
            partial(train_and_evaluate, internal_train, internal_test, schema)
        )
    except Exception as exc:
        logger.exception("스크리닝 기반 파이프라인 학습 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 학습 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    _report_training_progress("모델 저장", 90)

    created_at = datetime.now().astimezone()
    model_id = f"SCREENING_GBDT_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
    hybrid_result = build_hybrid_training_result(
        evaluation,
        source_filename=filename,
        dataset_rows={"train": len(internal_train), "test": len(internal_test)},
        train_lots=train_lots,
        test_lots=test_lots,
        split_method=split_method,
    )

    try:
        bundle_path, metadata_path = await run_in_threadpool(
            partial(save_hybrid_bundle, hybrid_result, MODEL_DIR, model_id)
        )
    except Exception as exc:
        logger.exception("학습 모델 또는 메타데이터 저장 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 학습에는 성공했지만 파일 저장에 실패했습니다.",
        ) from exc
    logger.info("학습 모델 저장 완료")
    _report_training_progress("학습 결과 정리", 98)

    # Switch the active pointer only after the saved bundle is complete.  Any
    # failure above leaves the previous active model untouched.
    RuntimeStore(
        settings.runtime_db_path,
        settings.runtime_artifact_dir,
    ).promote_model(
        model_id=model_id,
        pipeline_version=str(hybrid_result.metadata["pipeline_version"]),
        dataset_version=0,
        metadata={
            "model_id": model_id,
            "target": TARGET_COLUMN,
            "model_name": hybrid_result.metadata["model_name"],
            "source_filename": filename,
            "metrics": hybrid_result.metadata["metrics"],
            "row_count": int(len(dataframe)),
            "feature_columns": hybrid_result.metadata["feature_columns"],
        },
    )

    final_metrics = evaluation.metrics[FINAL_YIELD_COLUMN]
    per_target_metrics = target_metrics_summary(evaluation)
    dummy_baseline = float(pd.to_numeric(internal_train[TARGET_COLUMN], errors="coerce").mean())
    dummy_actual = pd.to_numeric(internal_test[TARGET_COLUMN], errors="coerce").to_numpy()
    dummy_predicted = np.full(len(dummy_actual), dummy_baseline, dtype=np.float32)
    dummy_valid = np.isfinite(dummy_actual)
    dummy_test = evaluate_regression(
        pd.Series(dummy_actual[dummy_valid]),
        dummy_predicted[dummy_valid],
    )
    evaluation_summary = {
        "dummy_test_r2": dummy_test.r2,
        "dummy_test_rmse": dummy_test.rmse,
        "dummy_rmse_improvement": (
            dummy_test.rmse - final_metrics["rmse"]
            if dummy_test.rmse is not None
            else None
        ),
        "no_significant_factor_targets": [
            target for target, detail in per_target_metrics.items() if detail["no_significant_factor"]
        ],
    }

    warnings = list(
        dict.fromkeys(
            [
                *preprocessing_report["warnings"],
            ]
        )
    )
    try:
        response = TrainResponse(
            target=TARGET_COLUMN,
            best_model="HistGradientBoostingRegressor",
            split=DatasetSplit(
                train_rows=len(internal_train),
                validation_rows=0,
                test_rows=len(internal_test),
                group_split_used=group_split_used,
                split_method=split_method,
            ),
            metrics={
                "test": ModelMetrics(
                    r2=final_metrics["r2"],
                    rmse=final_metrics["rmse"],
                    mae=final_metrics["mae"],
                    mse=final_metrics["mse"],
                ),
            },
            model_comparison=[
                ModelComparisonItem(
                    model_name=f"{target}_HistGradientBoostingRegressor",
                    status="no_significant_factor" if detail["no_significant_factor"] else "trained",
                    validation=(
                        None
                        if detail["no_significant_factor"]
                        else ModelMetrics(r2=detail["r2"], rmse=detail["rmse"], mae=detail["mae"])
                    ),
                    selected=not detail["no_significant_factor"],
                    error_message=None,
                )
                for target, detail in per_target_metrics.items()
            ],
            feature_count=len(hybrid_result.metadata["feature_columns"]),
            warnings=warnings,
            artifacts=ModelArtifacts(
                model_file=f"{model_id}/bundle.joblib",
                metadata_file=f"{model_id}/metadata.json",
            ),
            evaluation_summary=EvaluationSummary(**evaluation_summary),
            model_id=model_id,
            model_type=str(hybrid_result.metadata["model_type"]),
            selected_final_output="derived",
            final_y_metrics={"test": final_metrics},
            target_metrics=per_target_metrics,
            preprocessing={
                "schema_version": "semicon_yield_v2",
                "strategy": "screening_selected_factors_raw_miss_dev",
                "missing_strategy": "native_nan_preserved",
                "outlier_strategy": "none",
                "measurement_coverage": {
                    "r": validation.get("r_measurement_coverage", 0.0),
                    "d": validation.get("d_measurement_coverage", 0.0),
                },
            },
        )
        response.model_dump_json()
    except Exception as exc:
        logger.exception("학습 API 응답 직렬화 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 결과 응답을 생성하는 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    logger.info("학습 API 응답 직렬화 완료")
    return response

    return response


@router.post("/train/jobs", response_model=TrainJobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_training_job(
    file: UploadFile = File(...),
) -> TrainJobAccepted:
    manager = get_training_job_manager()
    job_id = new_training_job_id()
    try:
        input_path = manager.allocate_input_path(job_id)
    except Exception as exc:
        logger.exception("학습 Job 임시 경로 생성 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 Job 저장소를 준비하지 못했습니다.",
        ) from exc

    try:
        filename = await _persist_training_job_upload(file, input_path)
    except Exception:
        manager.cleanup_input(job_id)
        raise

    options: dict[str, Any] = {}
    try:
        manager.submit(
            job_id=job_id,
            source_filename=filename,
            input_path=input_path,
            runner=partial(
                _run_persisted_training_job,
                input_path,
                filename,
                options,
            ),
        )
    except ActiveOperationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=HEAVY_JOB_MESSAGE,
        ) from exc
    except Exception as exc:
        logger.exception("학습 Job 등록 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 Job을 등록하지 못했습니다.",
        ) from exc
    return TrainJobAccepted(job_id=job_id)


@router.get("/train/jobs/{job_id}", response_model=TrainJobStatus)
def get_training_job(job_id: str) -> TrainJobStatus:
    row = get_training_job_manager().get(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="학습 Job을 찾을 수 없습니다.",
        )
    return TrainJobStatus(
        job_id=str(row["job_id"]),
        status=row["status"],
        stage=str(row["stage"]),
        progress=int(row["progress"]),
        elapsed_seconds=float(row["elapsed_seconds"]),
        result=row.get("result"),
        error=row.get("error_message"),
    )


@router.get("/models", response_model=ModelListResponse)
def get_models() -> ModelListResponse:
    try:
        models, warnings = list_prediction_models(MODEL_DIR)
        return ModelListResponse(
            models=[ModelSummary(**model) for model in models],
            total=len(models),
            warnings=warnings,
        )
    except ModelLoadError as exc:
        logger.exception("모델 목록 조회 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/models/{model_id}", response_model=ModelDetailResponse)
def get_model_detail(model_id: str) -> ModelDetailResponse:
    try:
        detail = get_prediction_model_detail(model_id, MODEL_DIR)
        detail_metrics = detail.pop("metrics")
        return ModelDetailResponse(
            **detail,
            metrics={
                name: ModelDetailMetrics(**values)
                for name, values in detail_metrics.items()
            },
        )
    except InferenceInputError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
                if "존재하지 않는 모델" in str(exc)
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(exc),
        ) from exc
    except ModelLoadError as exc:
        logger.exception("모델 상세 조회 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/models/{model_id}/references")
def get_model_references(model_id: str) -> dict[str, Any]:
    try:
        detail = get_prediction_model_detail(model_id, MODEL_DIR)
    except InferenceInputError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    counts = RuntimeStore(settings.runtime_db_path).model_reference_counts(model_id)
    return {"model_id": model_id, "model_name": detail.get("model_name"), "model_type": detail.get("model_type"), "created_at": detail.get("created_at"), **counts}


@router.delete(
    "/models/{model_id}",
    response_model=ModelDeleteResponse,
    responses={
        400: {"description": "삭제할 수 없는 모델"},
        404: {"description": "모델을 찾을 수 없음"},
        500: {"description": "모델 삭제 실패"},
    },
)
def delete_model(model_id: str) -> ModelDeleteResponse:
    try:
        store = RuntimeStore(
            settings.runtime_db_path,
            settings.runtime_artifact_dir,
        )
        # Runtime history tables and snapshot artifacts are intentionally not
        # mutated by model deletion.
        counts_before = store.model_reference_counts(model_id)
        result = delete_model_bundle(model_id, MODEL_DIR)
        return ModelDeleteResponse(
            model_id=model_id,
            deleted_files=result.deleted_files,
            missing_files=result.missing_files,
            metadata_deleted=result.metadata_deleted,
            bundle_deleted=result.bundle_deleted,
            removed_files=result.deleted_files,
            prediction_history_count=counts_before["prediction_history_count"],
            analysis_history_count=counts_before["analysis_history_count"],
        )
    except InvalidModelIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelDeletionError as exc:
        logger.exception(
            "모델 삭제 일부 또는 전체 실패: model_id=%s deleted_files=%s failed_files=%s",
            model_id,
            exc.deleted_files,
            exc.failed_files,
        )
        errors = []
        if exc.failed_files:
            errors.append("실패 파일: " + ", ".join(exc.failed_files))
        if exc.deleted_files:
            errors.append("이미 삭제된 파일: " + ", ".join(exc.deleted_files))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(exc), "errors": errors},
        ) from exc
    except InferenceInputError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ModelLoadError, OSError) as exc:
        logger.exception("모델 삭제 처리 실패: model_id=%s", model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 삭제 중 서버 내부 오류가 발생했습니다.",
        ) from exc


