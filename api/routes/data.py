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
from uuid import uuid4

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
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from api.schemas.data import (
    AnalyzeResponse,
    ColumnDetectionResult,
    DataSummary,
    DatasetSplit,
    ExplainAnalysisSummary,
    ExplainModelInfo,
    ExplainResponse,
    ModelArtifacts,
    ModelComparisonItem,
    ModelDeleteResponse,
    ModelDetailMetrics,
    ModelDetailResponse,
    ModelListResponse,
    ModelMetrics,
    ModelSummary,
    PredictionModelInfo,
    PredictionResponse,
    PredictionSummary,
    PreprocessChanges,
    PreprocessResponse,
    RelationshipAnalysisResponse,
    ReportResponse,
    TrainResponse,
    ValidationResponse,
    ValidationResult,
)
from api.schemas.jobs import TrainJobAccepted, TrainJobResult, TrainJobStatus
from api.settings import settings
from src.analytics.relationships import analyze_relationships
from src.analytics.lot_analysis import build_lot_cause_analysis
from src.analytics.analysis_result import (
    build_analysis_result,
    compose_multi_y_predictions,
    dataset_fingerprint,
)
from src.data_validation import load_data_schema, validate_dataframe
from src.automation.analyzer import build_automation_response
from src.ml.dataset import (
    ALLOWED_TARGETS,
    RANDOM_STATE,
    prepare_dataset,
    split_dataset,
)
from src.config_parser import CONFIG_PARSER_VERSION
from src.schema_compatibility import schema_fingerprint
from src.ml.model_io import save_model_bundle
from src.ml.hybrid import save_hybrid_bundle, train_hybrid_multi_y
from src.ml.ensemble import EnsembleOptions
from src.ml.inference import (
    DEFAULT_DANGER_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    InferenceInputError,
    InvalidModelIdError,
    ModelDeletionError,
    ModelNotFoundError,
    PredictionResult,
    ModelLoadError,
    get_prediction_model_detail,
    delete_model_bundle,
    list_prediction_models,
    load_prediction_model,
    load_prediction_model_target,
    predict_dataframe,
)
from src.ml.explainability import (
    DEFAULT_MAX_EXPLAIN_ROWS,
    DEFAULT_TOP_N,
    DEFAULT_WAFER_TOP_N,
    MAX_TOP_N,
    ExplainResult,
    explain_dataframe,
)
from src.ml.training import train_regression_models
from src.ml.evaluation import evaluate_regression
from src.preprocessing import preprocess_dataframe
from src.reporting.export import render_report_html
from src.reporting.report_builder import build_report
from src.runtime.store import RuntimeStore, safe_runtime_call
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
TRAINABLE_TARGETS = tuple(["Y", *[f"Y{index}" for index in range(1, 6)]])
PREDICTION_PREVIEW_ROWS = 10
_TRAINING_LOCK = threading.Lock()
_TRAINING_PROGRESS: ContextVar[ProgressCallback | None] = ContextVar(
    "training_progress",
    default=None,
)
_TRAINING_JOB_MANAGER: TrainingJobManager | None = None
_TRAINING_JOB_MANAGER_LOCK = threading.Lock()


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
    target: str | None = Form(None),
    train_ratio: Annotated[int, Form()] = 64,
    validation_ratio: Annotated[int, Form()] = 16,
    test_ratio: Annotated[int, Form()] = 20,
    missing_indicator: Annotated[bool, Form()] = True,
    compare_missingness: Annotated[bool, Form()] = False,
    ensemble_enabled: Annotated[bool, Form()] = True,
    ensemble_size: Annotated[str, Form()] = "auto",
    ensemble_method: Annotated[str, Form()] = "auto",
    ensemble_min_improvement: Annotated[float, Form()] = 0.01,
    diversity_check: Annotated[bool, Form()] = True,
    max_base_models: Annotated[int, Form()] = 3,
) -> TrainResponse:
    training_started_at = time.perf_counter()
    filename, dataframe = await _read_csv_upload(file)
    logger.info(
        "학습 CSV 읽기 완료: rows=%d, columns=%d",
        dataframe.shape[0],
        dataframe.shape[1],
    )
    _report_training_progress("학습 CSV 확인", 10)

    if target is not None and target not in TRAINABLE_TARGETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="지원하지 않는 목표 변수입니다. Y 또는 Y1~Y5만 학습할 수 있습니다.",
        )

    split_ratios = (train_ratio, validation_ratio, test_ratio)
    if target is not None and (sum(split_ratios) != 100 or any(
        ratio < 5 or ratio > 90 for ratio in split_ratios
    )):
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

    if target is None:
        if not _TRAINING_LOCK.acquire(blocking=False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="다른 모델 학습이 진행 중입니다. 완료 후 다시 시도해 주세요.",
            )
        try:
            try:
                _report_training_progress("Multi-Y 모델 학습", 40)
                hybrid = await run_in_threadpool(
                    partial(train_hybrid_multi_y, dataframe)
                )
            finally:
                _TRAINING_LOCK.release()
            _report_training_progress("Multi-Y 모델 저장", 90)
            created_at = datetime.now().astimezone()
            model_id = f"AUTO_MULTI_Y_HGBR_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
            raw_features = list(hybrid.metadata["raw_feature_columns"])
            hybrid_processing_summary = {
                **preprocessing_report.get("processing_summary", {}),
                **hybrid.metadata.get("preprocessing_summary", {}),
            }
            hybrid.metadata.update({
                "model_id": model_id,
                "created_at": created_at.isoformat(),
                "schema_version": "semicon_yield_v2",
                "schema_fingerprint": schema_fingerprint(raw_features),
                "raw_feature_columns": raw_features,
                "config_parser_version": CONFIG_PARSER_VERSION,
                "preprocessing_config": {
                    "missing_strategy": hybrid.metadata.get("missing_strategy"),
                    "outlier_strategy": hybrid.metadata.get("outlier_strategy"),
                    "missing_indicator": False,
                    "outlier_indicator": False,
                    "statistics_scope": "each_cv_training_fold_and_full_refit",
                    "fallback_used": hybrid.metadata.get("fallback_used", False),
                },
                "preprocessing_summary": hybrid_processing_summary,
                "measurement_coverage": {
                    "r": validation.get("r_measurement_coverage", 0.0),
                    "d": validation.get("d_measurement_coverage", 0.0),
                },
                "source_filename": filename,
                "training_time_seconds": time.perf_counter() - training_started_at,
            })
            bundle_path, metadata_path = save_hybrid_bundle(hybrid, MODEL_DIR, model_id)
            _report_training_progress("학습 결과 정리", 98)
            selected = hybrid.metadata["selected_final_output"]
            selected_metrics = hybrid.metadata["final_y_metrics"][selected]
            rows = hybrid.metadata["dataset_rows"]
            response = TrainResponse(
                target="Y",
                best_model="Auto Multi-Y HGBR",
                split=DatasetSplit(
                    train_rows=rows["train"],
                    validation_rows=rows["validation"],
                    test_rows=rows["test"],
                    group_split_used=True,
                    split_method=hybrid.metadata["split_method"],
                ),
                metrics={name: ModelMetrics(**values) for name, values in selected_metrics.items()},
                model_comparison=[],
                feature_count=len(hybrid.bundle.feature_columns),
                warnings=hybrid.warnings,
                artifacts=ModelArtifacts(
                    model_file=str(bundle_path.relative_to(MODEL_DIR)),
                    metadata_file=str(metadata_path.relative_to(MODEL_DIR)),
                ),
                missingness_sensitivity=None,
                evaluation_summary=hybrid.metadata["cv_protocol"],
                model_id=model_id,
                model_type="hybrid_multi_y",
                selected_final_output=selected,
                final_y_metrics=hybrid.metadata["final_y_metrics"],
                target_metrics=hybrid.metadata["target_metrics"],
                risk_metrics=hybrid.metadata["risk_metrics"],
                preprocessing={
                    "schema_version": "semicon_yield_v2",
                    "config_parser_version": CONFIG_PARSER_VERSION,
                    "measurement_coverage": hybrid.metadata["measurement_coverage"],
                    "policy": hybrid.metadata["preprocessing_config"],
                    **hybrid_processing_summary,
                },
                ensemble=None,
            )
            response.model_dump_json()
            return response
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Hybrid Multi-Y 학습 또는 Bundle 저장 실패")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Hybrid Multi-Y 모델 학습 또는 저장 중 서버 오류가 발생했습니다.",
            ) from exc

    try:
        dataset = prepare_dataset(
            dataframe,
            target=target,
            add_missing_indicators=missing_indicator,
        )
    except ValueError as exc:
        logger.warning("학습 데이터 준비 실패: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("학습 feature 탐지 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 feature를 준비하는 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    logger.info(
        "학습 feature 탐지 완료: target=%s, features=%d",
        target,
        len(dataset.feature_columns),
    )
    _report_training_progress(f"{target} Feature 준비", 38)

    try:
        split = split_dataset(
            dataset,
            random_state=RANDOM_STATE,
            train_ratio=train_ratio / 100,
            validation_ratio=validation_ratio / 100,
            test_ratio=test_ratio / 100,
        )
    except ValueError as exc:
        logger.warning("학습 데이터 분할 실패: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("학습 데이터 분할 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="학습 데이터를 분할하는 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    logger.info(
        "학습 데이터 분할 완료: method=%s, train=%d, validation=%d, test=%d",
        split.split_method,
        split.row_counts["train_rows"],
        split.row_counts["validation_rows"],
        split.row_counts["test_rows"],
    )
    _report_training_progress(f"{target} 학습 데이터 분할", 45)

    try:
        _report_training_progress(f"{target} 모델 학습", 50)
        training = train_regression_models(
            dataset,
            split,
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        logger.warning("모델 학습 실패: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("모델 학습 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 학습 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    logger.info("최적 모델 선정 완료: %s", training.best_model_name)
    _report_training_progress(f"{target} 모델 평가", 82)

    missingness_sensitivity: dict[str, Any] | None = None
    if compare_missingness:
        try:
            alternate_dataset = prepare_dataset(
                dataframe,
                target=target,
                add_missing_indicators=not missing_indicator,
            )
            alternate_split = split_dataset(
                alternate_dataset,
                random_state=RANDOM_STATE,
                train_ratio=train_ratio / 100,
                validation_ratio=validation_ratio / 100,
                test_ratio=test_ratio / 100,
            )
            alternate_training = train_regression_models(
                alternate_dataset,
                alternate_split,
                random_state=RANDOM_STATE,
            )
            primary_test = training.metrics["test"]
            alternate_test = alternate_training.metrics["test"]
            with_indicator = primary_test if missing_indicator else alternate_test
            without_indicator = alternate_test if missing_indicator else primary_test
            r2_difference = (
                (with_indicator.r2 - without_indicator.r2)
                if with_indicator.r2 is not None and without_indicator.r2 is not None
                else None
            )
            sensitivity_warnings: list[str] = []
            if r2_difference is not None and abs(r2_difference) >= 0.15:
                sensitivity_warnings.append(
                    "모델이 공정값뿐 아니라 측정 대상 선정 규칙을 학습했을 가능성이 있습니다."
                )
            missingness_sensitivity = {
                "with_indicator_test_r2": with_indicator.r2,
                "without_indicator_test_r2": without_indicator.r2,
                "r2_difference": r2_difference,
                "rmse_difference": (
                    with_indicator.rmse - without_indicator.rmse
                    if with_indicator.rmse is not None and without_indicator.rmse is not None
                    else None
                ),
                "mae_difference": (
                    with_indicator.mae - without_indicator.mae
                    if with_indicator.mae is not None and without_indicator.mae is not None
                    else None
                ),
                "indicator_in_feature_set": any(
                    column.endswith("_missing") for column in dataset.feature_columns
                ),
                "warnings": sensitivity_warnings,
            }
        except Exception as exc:
            logger.exception("측정 여부 Indicator 민감도 비교 실패")
            missingness_sensitivity = {
                "status": "failed",
                "warnings": ["측정 여부 Indicator 민감도 비교를 완료하지 못했습니다."],
            }

    metrics = {
        name: values.as_dict()
        for name, values in training.metrics.items()
    }
    for name, values in training.metrics.items():
        metrics[name]["mse"] = values.mse
    dummy_test = evaluate_regression(
        split.y_test,
        np.full(len(split.y_test), float(split.y_train.mean())),
    )
    train_r2 = training.metrics["train"].r2
    test_r2 = training.metrics["test"].r2
    evaluation_summary = {
        "generalization_gap": (
            train_r2 - test_r2 if train_r2 is not None and test_r2 is not None else None
        ),
        "dummy_test_r2": dummy_test.r2,
        "dummy_test_rmse": dummy_test.rmse,
        "dummy_rmse_improvement": (
            dummy_test.rmse - training.metrics["test"].rmse
            if dummy_test.rmse is not None and training.metrics["test"].rmse is not None
            else None
        ),
    }
    try:
        try:
            transformed_feature_columns = [
                str(name).split("__", 1)[-1]
                for name in training.best_model.named_steps["features"].get_feature_names_out()
            ]
        except Exception:
            transformed_feature_columns = list(dataset.feature_columns)
        raw_features = dataset.raw_feature_columns or dataset.feature_columns
        _report_training_progress(f"{target} 모델 저장", 92)
        model_path, metadata_path, saved_metadata = save_model_bundle(
            training.best_model,
            target=target,
            model_name=training.best_model_name,
            feature_columns=dataset.feature_columns,
            metrics=metrics,
            random_state=RANDOM_STATE,
            split_method=split.split_method,
            dataset_split={
                "train": train_ratio / 100,
                "validation": validation_ratio / 100,
                "test": test_ratio / 100,
            },
            dataset_rows={
                "train": split.row_counts["train_rows"],
                "validation": split.row_counts["validation_rows"],
                "test": split.row_counts["test_rows"],
            },
            training_time_seconds=(
                time.perf_counter() - training_started_at
            ),
            source_filename=filename,
            preprocessing_config={
                **preprocessing_report.get("preprocessing_policy", {}),
                "missing_strategy": training.missing_strategy,
                "outlier_strategy": training.outlier_strategy,
                "model_strategies": training.model_strategies,
                "model_outlier_strategies": training.model_outlier_strategies,
                "fallback_used": training.fallback_used,
            },
            metadata_extensions={
                "schema_version": "semicon_yield_v2",
                "schema_fingerprint": schema_fingerprint(raw_features),
                "raw_feature_columns": raw_features,
                "transformed_feature_columns": transformed_feature_columns,
                "config_parser_version": CONFIG_PARSER_VERSION,
                "preprocessing_strategy": f"{training.missing_strategy}_{training.outlier_strategy}_train_only",
                "missing_strategy": training.missing_strategy,
                "outlier_strategy": training.outlier_strategy,
                "missing_indicator_used": training.missing_indicator,
                "outlier_indicator_used": training.outlier_indicator,
                "outlier_policy": training.outlier_strategy,
                "fallback_used": training.fallback_used,
                "preprocessing_summary": {
                    **preprocessing_report.get("processing_summary", {}),
                    "missing_strategy": training.missing_strategy,
                    "outlier_strategy": training.outlier_strategy,
                    "missing_indicator": training.missing_indicator,
                    "outlier_indicator": training.outlier_indicator,
                    "model_strategies": training.model_strategies,
                    "model_outlier_strategies": training.model_outlier_strategies,
                    "fallback_used": training.fallback_used,
                },
                "split_strategy": split.split_method,
                "group_column": "Lot_ID" if "Lot_ID" in dataframe.columns else "Lot_Wafer_ID (legacy parsed)",
                "train_lot_count": int(split.train_groups.nunique(dropna=True)),
                "validation_lot_count": int(split.validation_groups.nunique(dropna=True)),
                "test_lot_count": int(split.test_groups.nunique(dropna=True)),
                "train_row_count": split.row_counts["train_rows"],
                "validation_row_count": split.row_counts["validation_rows"],
                "test_row_count": split.row_counts["test_rows"],
                "target_leakage_check": dataset.target_leakage_check,
                "data_summary": {
                    "row_count": int(dataframe.shape[0]),
                    "column_count": int(dataframe.shape[1]),
                    "lot_count": validation.get("lot_count", 0),
                },
                "measurement_coverage": {
                    "r": validation.get("r_measurement_coverage", 0.0),
                    "d": validation.get("d_measurement_coverage", 0.0),
                },
                "threshold_policy": {
                    "warning": 90.0,
                    "critical": 85.0,
                    "source": "data_distribution_reference_not_official_spec",
                },
                "missingness_sensitivity": missingness_sensitivity,
                "evaluation_summary": evaluation_summary,
            },
            model_dir=MODEL_DIR,
        )
    except Exception as exc:
        logger.exception("학습 모델 또는 메타데이터 저장 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 학습에는 성공했지만 파일 저장에 실패했습니다.",
        ) from exc
    logger.info("학습 모델 저장 완료")
    _report_training_progress("학습 결과 정리", 98)

    warnings = list(
        dict.fromkeys(
            [
                *preprocessing_report["warnings"],
                *dataset.warnings,
                *split.warnings,
                *training.warnings,
            ]
        )
    )
    try:
        response = TrainResponse(
            target=target,
            best_model=training.best_model_name,
            split=DatasetSplit(
                **split.row_counts,
                group_split_used=split.group_split_used,
                split_method=split.split_method,
            ),
            metrics={
                name: ModelMetrics(**values)
                for name, values in metrics.items()
            },
            model_comparison=[
                ModelComparisonItem(
                    model_name=item.model_name,
                    status=item.status,
                    validation=(
                        ModelMetrics(**item.validation.as_dict())
                        if item.validation is not None
                        else None
                    ),
                    selected=item.selected,
                    error_message=item.error_message,
                )
                for item in training.model_comparison
            ],
            feature_count=len(dataset.feature_columns),
            warnings=warnings,
            artifacts=ModelArtifacts(
                model_file=model_path.name,
                metadata_file=metadata_path.name,
            ),
            missingness_sensitivity=missingness_sensitivity,
            evaluation_summary=evaluation_summary,
            model_id=str(saved_metadata["model_id"]),
            model_type=str(saved_metadata.get("model_type") or training.best_model_name),
            preprocessing={
                "schema_version": "semicon_yield_v2",
                "config_parser_version": CONFIG_PARSER_VERSION,
                "measurement_coverage": {
                    "r": validation.get("r_measurement_coverage", 0.0),
                    "d": validation.get("d_measurement_coverage", 0.0),
                },
                **preprocessing_report.get("processing_summary", {}),
                "missing_strategy": training.missing_strategy,
                "outlier_strategy": training.outlier_strategy,
                "missing_indicator": training.missing_indicator,
                "outlier_indicator": training.outlier_indicator,
                "model_strategies": training.model_strategies,
                "model_outlier_strategies": training.model_outlier_strategies,
                "fallback_used": training.fallback_used,
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


@router.post(
    "/train/jobs",
    response_model=TrainJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={409: {"description": "Another heavy operation is running"}},
)
async def create_training_job(
    file: UploadFile = File(...),
    target: str | None = Form(None),
    train_ratio: Annotated[int, Form()] = 64,
    validation_ratio: Annotated[int, Form()] = 16,
    test_ratio: Annotated[int, Form()] = 20,
    missing_indicator: Annotated[bool, Form()] = True,
    compare_missingness: Annotated[bool, Form()] = False,
    ensemble_enabled: Annotated[bool, Form()] = True,
    ensemble_size: Annotated[str, Form()] = "auto",
    ensemble_method: Annotated[str, Form()] = "auto",
    ensemble_min_improvement: Annotated[float, Form()] = 0.01,
    diversity_check: Annotated[bool, Form()] = True,
    max_base_models: Annotated[int, Form()] = 3,
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

    options = {
        "target": target,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "missing_indicator": missing_indicator,
        "compare_missingness": compare_missingness,
        "ensemble_enabled": ensemble_enabled,
        "ensemble_size": ensemble_size,
        "ensemble_method": ensemble_method,
        "ensemble_min_improvement": ensemble_min_improvement,
        "diversity_check": diversity_check,
        "max_base_models": max_base_models,
    }
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


@router.get(
    "/train/jobs/{job_id}",
    response_model=TrainJobStatus,
    responses={404: {"description": "Training job not found"}},
)
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
        status.HTTP_400_BAD_REQUEST: {"description": "잘못된 model_id"},
        status.HTTP_404_NOT_FOUND: {"description": "존재하지 않거나 이미 삭제된 모델"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "모델 파일 삭제 실패"},
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


def _prediction_response(
    filename: str,
    result: PredictionResult,
    *,
    prediction_id: str | None = None,
    history_saved: bool = False,
    history_warning: str | None = None,
    artifact_available: bool | None = None,
) -> PredictionResponse:
    evaluation = (
        ModelMetrics(**result.evaluation.as_dict())
        if result.evaluation is not None
        else None
    )
    response = PredictionResponse(
        filename=filename,
        model=PredictionModelInfo(
            model_id=result.model_id,
            target=result.target,
            model_name=result.model_name,
        ),
        summary=PredictionSummary(
            total_rows=result.total_rows,
            average_prediction=result.average_prediction,
            normal_count=result.normal_count,
            warning_count=result.warning_count,
            danger_count=result.danger_count,
            evaluation=evaluation,
        ),
        identifier_column=result.identifier_column,
        predictions=result.predictions[:PREDICTION_PREVIEW_ROWS],
        warnings=result.warnings,
        truncated=(
            result.truncated
            or len(result.predictions) > PREDICTION_PREVIEW_ROWS
        ),
        preprocessing=result.preprocessing_summary,
        prediction_id=prediction_id,
        history_saved=history_saved,
        history_warning=history_warning,
        artifact_available=artifact_available,
        preview_row_count=min(
            len(result.predictions),
            PREDICTION_PREVIEW_ROWS,
        ),
    )
    response.model_dump_json()
    return response


def _prediction_history_summary(result: PredictionResult) -> dict[str, Any]:
    prediction_key = f"predicted_{result.target}"
    values = [
        float(row[prediction_key]) for row in result.predictions
        if row.get(prediction_key) is not None
    ]
    failure_rates: dict[str, list[float]] = {}
    fail_counts: dict[str, list[float]] = {}
    for row in result.predictions:
        for target, value in (row.get("failure_rates") or {}).items():
            failure_rates.setdefault(target, []).append(float(value))
        for target, value in (row.get("fail_bit_counts") or {}).items():
            fail_counts.setdefault(target, []).append(float(value))
    def mean_field(name: str) -> float | None:
        selected = [float(row[name]) for row in result.predictions if row.get(name) is not None]
        return float(np.mean(selected)) if selected else None
    combined_targets = {**failure_rates, **fail_counts}
    target_totals = {key: float(np.mean(entries)) for key, entries in combined_targets.items() if entries}
    evaluation = result.evaluation.as_dict() if result.evaluation is not None else {}
    lots = {str(row.get("Lot_ID")) for row in result.predictions if row.get("Lot_ID")}
    risk_lots = {
        str(row.get("Lot_ID")) for row in result.predictions
        if row.get("Lot_ID") and row.get("risk_level") in {"danger", "warning"}
    }
    return {
        "average_predicted_yield": float(np.mean(values)) if values else None,
        "minimum_predicted_yield": min(values) if values else None,
        "maximum_predicted_yield": max(values) if values else None,
        "median_predicted_yield": float(np.median(values)) if values else None,
        "direct_y_mean": mean_field("direct_y"),
        "derived_y_mean": mean_field("derived_y"),
        "hybrid_y_mean": mean_field("hybrid_y"),
        "critical_count": result.danger_count, "warning_count": result.warning_count,
        "normal_count": result.normal_count,
        "low_confidence_count": sum(row.get("confidence") == "low" for row in result.predictions),
        "top_failure_target": max(target_totals, key=target_totals.get) if target_totals else None,
        "failure_rates": {key: float(np.mean(entries)) for key, entries in failure_rates.items()},
        "fail_bit_counts": {key: float(np.mean(entries)) for key, entries in fail_counts.items()},
        "actual_y_available": result.evaluation is not None,
        "r2": evaluation.get("r2"), "rmse": evaluation.get("rmse"), "mae": evaluation.get("mae"),
        "lot_count": len(lots), "risk_lot_count": len(risk_lots),
    }


def _collect_multi_y_predictions(
    dataframe: pd.DataFrame,
    selected_model: Any,
    selected_prediction: PredictionResult,
) -> tuple[dict[str, Any], list[str]]:
    """Load only actually stored compatible target models for Hybrid Multi-Y."""
    if selected_model.metadata.get("model_type") == "hybrid_multi_y":
        try:
            bundle_model = load_prediction_model(selected_prediction.model_id, MODEL_DIR)
            bundle_prediction = predict_dataframe(dataframe, bundle_model, max_rows=None)
            if len(bundle_prediction.predictions) != len(selected_prediction.predictions):
                raise InferenceInputError(
                    "Hybrid Multi-Y Bundle과 선택 Target의 예측 행 수가 일치하지 않습니다."
                )
            auxiliary_fields = (
                "direct_y",
                "derived_y",
                "hybrid_y",
                "selected_final_output",
                "failure_rates",
                "fail_bit_counts",
                "critical_probability",
                "warning_probability",
                "confidence",
                "final_strategy",
                "ensemble_used",
                "base_model_count",
                "direct_y_ensemble",
                "derived_y_ensemble",
                "model_agreement",
            )
            for selected_row, bundle_row in zip(
                selected_prediction.predictions,
                bundle_prediction.predictions,
                strict=True,
            ):
                selected_identifier = selected_row.get(
                    selected_prediction.identifier_column
                )
                bundle_identifier = bundle_row.get(
                    bundle_prediction.identifier_column
                )
                if (
                    selected_identifier is not None
                    and bundle_identifier is not None
                    and str(selected_identifier) != str(bundle_identifier)
                ):
                    raise InferenceInputError(
                        "Hybrid Multi-Y Bundle과 선택 Target의 예측 행 순서가 일치하지 않습니다."
                    )
                for field in auxiliary_fields:
                    if field in bundle_row:
                        selected_row[field] = bundle_row[field]
            values_by_target: dict[str, list[float]] = {
                "Y": [float(row["direct_y"]) for row in bundle_prediction.predictions]
            }
            for target_name in [f"Y{index}" for index in range(1, 11)]:
                group = "failure_rates" if target_name in {"Y1", "Y2", "Y3", "Y4", "Y5"} else "fail_bit_counts"
                values_by_target[target_name] = [
                    float(row[group][target_name]) for row in bundle_prediction.predictions
                ]
            result = compose_multi_y_predictions(values_by_target, None)
            result["ensemble_y"] = [
                float(row["hybrid_y"]) for row in bundle_prediction.predictions
            ]
            result["ensemble_method"] = "oof_stacking_meta_model"
            result["selected_final_output"] = bundle_model.metadata.get("selected_final_output")
            return result, []
        except (InferenceInputError, ModelLoadError, KeyError, TypeError, ValueError) as exc:
            return compose_multi_y_predictions({}, None), [
                f"Hybrid Multi-Y Bundle 결과를 펼치지 못했습니다: {exc}"
            ]
    targets = ["Y", *[f"Y{index}" for index in range(1, 11)]]
    values_by_target: dict[str, list[float]] = {
        selected_prediction.target: [
            float(row[f"predicted_{selected_prediction.target}"])
            for row in selected_prediction.predictions
        ]
    }
    warnings: list[str] = []
    ensemble_weight = None
    if selected_prediction.target == "Y":
        ensemble_weight = selected_model.metadata.get("ensemble_weight")
    try:
        available, _ = list_prediction_models(MODEL_DIR)
    except ModelLoadError:
        available = []
    for target in targets:
        if target in values_by_target:
            continue
        candidate = next(
            (
                item for item in available
                if item.get("target") == target
                and item.get("compatibility") == "compatible"
            ),
            None,
        )
        if candidate is None:
            continue
        try:
            loaded = load_prediction_model(str(candidate["model_id"]), MODEL_DIR)
            predicted = predict_dataframe(dataframe, loaded, max_rows=None)
            values_by_target[target] = [
                float(row[f"predicted_{target}"])
                for row in predicted.predictions
            ]
            if target == "Y":
                ensemble_weight = loaded.metadata.get("ensemble_weight")
        except (InferenceInputError, ModelLoadError) as exc:
            warnings.append(f"{target} 모델을 Multi-Y 분석에서 제외했습니다: {exc}")
    return compose_multi_y_predictions(values_by_target, ensemble_weight), warnings


async def _run_prediction(
    file: UploadFile,
    model_id: str,
    warning_threshold: float,
    danger_threshold: float,
    *,
    max_rows: int | None,
    runtime_id: str | None = None,
) -> tuple[str, pd.DataFrame, Any, PredictionResult]:
    filename, dataframe = await _read_csv_upload(file)
    try:
        loaded = load_prediction_model(model_id, MODEL_DIR)
        result = predict_dataframe(
            dataframe,
            loaded,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            max_rows=max_rows,
        )
        analysis_id = runtime_id or f"prediction_{uuid4().hex}"
        safe_runtime_call(
            "record_prediction_alerts",
            analysis_id=analysis_id,
            model_id=model_id,
            model_version=loaded.metadata.get("model_version"),
            predictions=result.predictions,
            identifier_column=result.identifier_column,
        )
        return filename, dataframe, loaded, result
    except InferenceInputError as exc:
        logger.warning("예측 입력 오류: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelLoadError as exc:
        logger.exception("예측 모델 처리 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("예측 처리 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="예측 처리 중 서버 내부 오류가 발생했습니다.",
        ) from exc


@router.post("/predict", response_model=PredictionResponse)
async def predict_csv(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
) -> PredictionResponse:
    prediction_id = f"prediction_{uuid4().hex}"
    started_clock = time.perf_counter()
    history_started = safe_runtime_call(
        "start_prediction", prediction_id=prediction_id,
        source_filename=file.filename, model_id=model_id,
        warning_threshold=warning_threshold, critical_threshold=danger_threshold,
    ) is not None
    try:
        filename, dataframe, loaded, result = await _run_prediction(
            file, model_id, warning_threshold, danger_threshold,
            max_rows=None, runtime_id=prediction_id,
        )
        response = _prediction_response(
            filename,
            result,
            prediction_id=prediction_id,
            artifact_available=history_started,
        )
        history_warning = None
        history_saved = False
        if history_started:
            summary = _prediction_history_summary(result)
            metadata = loaded.metadata
            completed = safe_runtime_call(
                "complete_prediction", prediction_id=prediction_id,
                metadata={
                    "duration_ms": (time.perf_counter() - started_clock) * 1000,
                    "dataset_fingerprint": dataset_fingerprint(dataframe),
                    "model_name_snapshot": result.model_name,
                    "model_version_snapshot": metadata.get("model_version"),
                    "model_type_snapshot": metadata.get("model_type"),
                    "schema_version": metadata.get("schema_version"),
                    "row_count": result.total_rows, "lot_count": summary["lot_count"],
                    "final_strategy": metadata.get("selected_final_output"),
                },
                summary=summary, preprocessing=result.preprocessing_summary,
                artifact={
                    "metadata": {"prediction_id": prediction_id, "created_at": datetime.now().astimezone().isoformat()},
                    "summary": summary, "rows": result.predictions,
                    "warnings": result.warnings, "response": response.model_dump(mode="json"),
                },
                warnings=result.warnings,
            )
            history_saved = completed is True
            if not history_saved:
                response.artifact_available = False
                history_warning = "예측 결과는 생성했지만 이력 저장에 실패했습니다."
                safe_runtime_call("fail_prediction", prediction_id=prediction_id, message=history_warning)
        else:
            response.artifact_available = False
            history_warning = "예측 결과는 생성했지만 이력 저장소를 사용할 수 없습니다."
        response.history_saved = history_saved
        response.history_warning = history_warning
        if history_warning:
            response.warnings = list(dict.fromkeys([*response.warnings, history_warning]))
        return response
    except HTTPException as exc:
        if history_started:
            safe_runtime_call("fail_prediction", prediction_id=prediction_id, message=str(exc.detail))
        raise
    except Exception as exc:
        logger.exception("예측 응답 직렬화 실패")
        if history_started:
            safe_runtime_call("fail_prediction", prediction_id=prediction_id, message=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="예측 결과 응답을 생성하지 못했습니다.",
        ) from exc


@router.post("/predict/download")
async def download_predictions(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
) -> Response:
    _, _, _, result = await _run_prediction(
        file,
        model_id,
        warning_threshold,
        danger_threshold,
        max_rows=None,
    )
    output = pd.DataFrame(result.predictions)
    csv_content = output.to_csv(index=False).encode("utf-8-sig")
    filename = f"predictions_{result.model_id}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


def _explain_response(
    filename: str,
    result: ExplainResult,
) -> ExplainResponse:
    response = ExplainResponse(
        filename=filename,
        model=ExplainModelInfo(
            model_id=result.model_id,
            target=result.target,
            model_name=result.model_name,
        ),
        analysis_summary=ExplainAnalysisSummary(
            total_rows=result.total_rows,
            analyzed_rows=result.analyzed_rows,
            sampling_used=result.sampling_used,
            sampling_strategy=result.sampling_strategy,
            explanation_method=result.explanation_method,
            is_fallback=result.is_fallback,
        ),
        explanation_method=result.explanation_method,
        is_fallback=result.is_fallback,
        identifier_column=result.identifier_column,
        global_importance=result.global_importance,
        step_summary=result.step_summary,
        parameter_type_summary=result.parameter_type_summary,
        equipment_summary=result.equipment_summary,
        wafer_explanations=result.wafer_explanations,
        model_quality_warnings=result.model_quality_warnings,
        warnings=result.warnings,
    )
    response.model_dump_json()
    return response


def _compact_lot_analysis(
    value: dict[str, Any] | None,
    *,
    lot_limit: int = 5,
    wafer_limit: int = 50,
) -> dict[str, Any]:
    source = value or {}
    lots = source.get("lots")
    if not isinstance(lots, list):
        lots = []
    compact_lots: list[dict[str, Any]] = []
    for raw_lot in lots[:lot_limit]:
        if not isinstance(raw_lot, dict):
            continue
        lot = dict(raw_lot)
        wafers = raw_lot.get("wafer_list")
        wafer_rows = wafers if isinstance(wafers, list) else []
        lot["wafer_list"] = wafer_rows[:wafer_limit]
        lot["returned_wafer_count"] = len(lot["wafer_list"])
        lot["wafer_list_truncated"] = len(wafer_rows) > wafer_limit
        compact_lots.append(lot)
    return {
        **source,
        "lots": compact_lots,
        "returned_lot_count": len(compact_lots),
        "lot_list_truncated": len(lots) > lot_limit,
    }


def _compact_analysis_result(
    value: dict[str, Any],
    explanation: ExplainResult,
    compact_lot_analysis: dict[str, Any],
) -> dict[str, Any]:
    source_multi_y = value.get("multi_y")
    multi_y = source_multi_y if isinstance(source_multi_y, dict) else {}
    sampled_identifiers = {
        str(
            item.get("identifier")
            if isinstance(item, dict)
            else getattr(item, "identifier", None)
        )
        for item in explanation.wafer_explanations[:DEFAULT_MAX_EXPLAIN_ROWS]
    }
    wafer_results = multi_y.get("wafer_results")
    wafer_rows = wafer_results if isinstance(wafer_results, list) else []
    compact_wafers = [
        row
        for row in wafer_rows
        if isinstance(row, dict)
        and str(row.get("identifier")) in sampled_identifiers
    ][:DEFAULT_MAX_EXPLAIN_ROWS]
    compact_multi_y = {
        key: multi_y.get(key)
        for key in (
            "average_direct_y",
            "average_derived_y",
            "average_ensemble_y",
            "ensemble_weight",
            "ensemble_method",
            "selected_final_output",
            "failure_rate_averages",
            "fail_bit_count_averages",
        )
    }
    compact_multi_y["wafer_results"] = compact_wafers
    compact_multi_y["returned_wafer_count"] = len(compact_wafers)
    compact_multi_y["wafer_results_truncated"] = (
        len(compact_wafers) < len(wafer_rows)
    )
    return {
        **value,
        "multi_y": compact_multi_y,
        "lot_analysis": compact_lot_analysis,
        "lot_summary": (
            value.get("lot_summary", [])[:5]
            if isinstance(value.get("lot_summary"), list)
            else []
        ),
    }


async def _run_explanation(
    file: UploadFile,
    model_id: str,
    max_rows: int,
    top_n: int,
    per_wafer_top_n: int,
) -> tuple[str, ExplainResult]:
    filename, dataframe = await _read_csv_upload(file)
    try:
        loaded = load_prediction_model(model_id, MODEL_DIR)
        result = explain_dataframe(
            dataframe,
            loaded,
            max_rows=max_rows,
            top_n=top_n,
            per_wafer_top_n=per_wafer_top_n,
        )
        return filename, result
    except InferenceInputError as exc:
        logger.warning("설명 입력 오류: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelLoadError as exc:
        logger.exception("설명 모델 처리 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("원인 분석 처리 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="원인 분석 처리 중 서버 내부 오류가 발생했습니다.",
        ) from exc


@router.post("/explain", response_model=ExplainResponse)
async def explain_csv(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(DEFAULT_TOP_N),
    per_wafer_top_n: int = Form(DEFAULT_WAFER_TOP_N),
) -> ExplainResponse:
    filename, result = await _run_explanation(
        file,
        model_id,
        max_rows,
        top_n,
        per_wafer_top_n,
    )
    try:
        return _explain_response(filename, result)
    except Exception as exc:
        logger.exception("원인 분석 응답 직렬화 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="원인 분석 결과 응답을 생성하지 못했습니다.",
        ) from exc


@router.post(
    "/relationships",
    response_model=RelationshipAnalysisResponse,
)
async def analyze_feature_relationships(
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(10),
    per_wafer_top_n: int = Form(DEFAULT_WAFER_TOP_N),
    correlation_method: str = Form("pearson"),
    analysis_unit: str = Form("wafer_observed_only"),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
    analysis_target: str | None = Form(None),
    prediction_id: str | None = Form(None),
) -> RelationshipAnalysisResponse:
    analysis_id = f"analysis_{uuid4().hex}"
    resolved_prediction_id = prediction_id if isinstance(prediction_id, str) else None
    resolved_model_id = model_id.strip() if isinstance(model_id, str) else ""
    started_clock = time.perf_counter()
    history_started = False
    filename, dataframe = await _read_csv_upload(file)
    try:
        resolved_analysis_target = analysis_target if isinstance(analysis_target, str) else None
        if not resolved_model_id:
            analysis = analyze_relationships(
                dataframe,
                target=resolved_analysis_target or "Y",
                correlation_method=correlation_method,
                top_n=top_n,
                analysis_unit=analysis_unit,
            )
            response = RelationshipAnalysisResponse(
                filename=filename,
                explanation=None,
                analysis_result=None,
                report_snapshot=None,
                lot_analysis={},
                analysis_id=None,
                prediction_id=resolved_prediction_id,
                artifact_available=False,
                **analysis,
            )
            response.model_dump_json()
            return response

        history_started = safe_runtime_call(
            "start_analysis", analysis_id=analysis_id,
            prediction_id=resolved_prediction_id,
            source_filename=file.filename, model_id=resolved_model_id,
        ) is not None
        loaded = (
            load_prediction_model_target(resolved_model_id, resolved_analysis_target, MODEL_DIR)
            if resolved_analysis_target
            else load_prediction_model(resolved_model_id, MODEL_DIR)
        )
        prediction = predict_dataframe(
            dataframe,
            loaded,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            max_rows=None,
        )
        multi_y, multi_y_warnings = _collect_multi_y_predictions(
            dataframe,
            loaded,
            prediction,
        )
        explanation = explain_dataframe(
            dataframe,
            loaded,
            max_rows=max_rows,
            top_n=MAX_TOP_N,
            per_wafer_top_n=per_wafer_top_n,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            prediction_result=prediction,
        )
        analysis = analyze_relationships(
            dataframe,
            target=explanation.target,
            correlation_method=correlation_method,
            top_n=top_n,
            shap_importance=explanation.global_importance,
            analysis_unit=analysis_unit,
        )
        report = build_report(filename, loaded, prediction, explanation)
        lot_analysis = build_lot_cause_analysis(prediction, explanation)
        report["lot_analysis"] = lot_analysis
        report["target_analysis"] = {
            "target": analysis["target"],
            "rankings": analysis["rankings"],
            "pareto": analysis["pareto"],
            "statistics": analysis["statistics"],
        }
        report["relationship_analysis"] = {
            "relationship_paths": analysis["relationship_paths"],
            "statistics": analysis["statistics"],
        }
        common = build_analysis_result(
            filename=filename,
            dataframe=dataframe,
            loaded=loaded,
            prediction=prediction,
            explanation=explanation,
            relationships=analysis,
            report=report,
            multi_y=multi_y,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            analysis_unit=analysis_unit,
            analysis_id=analysis_id,
            lot_analysis=lot_analysis,
        )
        common["warnings"] = list(dict.fromkeys([*common["warnings"], *multi_y_warnings]))
        report["analysis_id"] = common["analysis_id"]
        report["snapshot_metadata"] = {
            "analysis_id": common["analysis_id"],
            "model_id": resolved_model_id,
            "dataset_fingerprint": common["dataset"]["fingerprint"],
            "target": explanation.target,
            "threshold": {
                "warning": warning_threshold,
                "critical": danger_threshold,
            },
            "analysis_unit": analysis_unit,
            "created_at": common["created_at"],
            "schema_version": common["model"]["schema_version"],
            "report_version": common["report"]["report_version"],
        }
        compact_lot_analysis = _compact_lot_analysis(lot_analysis)
        compact_common = _compact_analysis_result(
            common,
            explanation,
            compact_lot_analysis,
        )
        compact_report = {
            **report,
            "lot_analysis": compact_lot_analysis,
            "lot_summary": (
                report.get("lot_summary", [])[:5]
                if isinstance(report.get("lot_summary"), list)
                else []
            ),
        }
        response = RelationshipAnalysisResponse(
            filename=filename,
            explanation=_explain_response(filename, explanation),
            analysis_result=compact_common,
            report_snapshot=compact_report,
            lot_analysis=compact_lot_analysis,
            analysis_id=analysis_id,
            prediction_id=resolved_prediction_id,
            artifact_available=history_started,
            **analysis,
        )
        summary = _prediction_history_summary(prediction)
        failure_rate_averages = common["multi_y"].get("failure_rate_averages", {})
        if failure_rate_averages:
            summary["top_failure_target"] = max(
                failure_rate_averages,
                key=failure_rate_averages.get,
            )
        lots = {str(row.get("Lot_ID")) for row in prediction.predictions if row.get("Lot_ID")}
        history_warning = None
        history_saved = False
        if history_started:
            stored_response = response.model_copy(
                update={
                    "history_saved": True,
                    "history_warning": None,
                    "artifact_available": True,
                }
            )
            completed = safe_runtime_call(
                "complete_analysis", analysis_id=analysis_id,
                metadata={
                    "duration_ms": (time.perf_counter() - started_clock) * 1000,
                    "dataset_fingerprint": common["dataset"]["fingerprint"],
                    "model_name_snapshot": loaded.metadata.get("model_name"),
                    "model_version_snapshot": loaded.metadata.get("model_version"),
                    "model_type_snapshot": loaded.metadata.get("model_type"),
                    "schema_version": loaded.metadata.get("schema_version"),
                    "row_count": len(dataframe), "lot_count": len(lots),
                    "available_targets_json": RuntimeStore._json(["Y", *[f"Y{i}" for i in range(1, 11)]]),
                    "default_target": explanation.target,
                    "report_snapshot_available": 1,
                },
                summary=summary, methodology=common.get("methodology") or {},
                artifact={
                    "metadata": {"analysis_id": analysis_id, "prediction_id": resolved_prediction_id},
                    "analysis_result": common, "report_snapshot": report,
                    "response": stored_response.model_dump(mode="json"),
                },
                warnings=common["warnings"],
            )
            history_saved = completed is True
            if not history_saved:
                response.artifact_available = False
                history_warning = "원인 분석 결과는 생성했지만 이력 저장에 실패했습니다."
                safe_runtime_call("fail_analysis", analysis_id=analysis_id, message=history_warning)
        else:
            response.artifact_available = False
            history_warning = "원인 분석 결과는 생성했지만 이력 저장소를 사용할 수 없습니다."
        response.history_saved = history_saved
        response.history_warning = history_warning
        if history_warning:
            response.caveats = list(dict.fromkeys([*response.caveats, history_warning]))
        response.model_dump_json()
        return response
    except (InferenceInputError, ValueError) as exc:
        if history_started:
            safe_runtime_call("fail_analysis", analysis_id=analysis_id, message=str(exc))
        logger.warning("연관 분석 입력 오류: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelLoadError as exc:
        if history_started:
            safe_runtime_call("fail_analysis", analysis_id=analysis_id, message=str(exc))
        logger.exception("연관 분석 모델 처리 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        if history_started:
            safe_runtime_call("fail_analysis", analysis_id=analysis_id, message=str(exc))
        logger.exception("연관 분석 처리 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="연관 분석 결과를 생성하지 못했습니다.",
        ) from exc


@router.post("/explain/download")
async def download_explanation(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(DEFAULT_TOP_N),
    per_wafer_top_n: int = Form(DEFAULT_WAFER_TOP_N),
) -> Response:
    _, result = await _run_explanation(
        file,
        model_id,
        max_rows,
        top_n,
        per_wafer_top_n,
    )
    output = pd.DataFrame(result.global_importance)
    csv_content = output.to_csv(index=False).encode("utf-8-sig")
    filename = f"explanation_{result.model_id}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


async def _run_report(
    file: UploadFile,
    model_id: str,
    warning_threshold: float,
    danger_threshold: float,
    max_rows: int,
    top_n: int,
    per_wafer_top_n: int = DEFAULT_WAFER_TOP_N,
) -> dict[str, Any]:
    filename, dataframe = await _read_csv_upload(file)
    try:
        logger.info("보고서 모델 로드 시작: %s", model_id)
        loaded = load_prediction_model(model_id, MODEL_DIR)
        logger.info("보고서 예측 및 위험 분류 시작")
        prediction = predict_dataframe(
            dataframe,
            loaded,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            max_rows=None,
        )
        logger.info("보고서 SHAP 분석 시작")
        explanation = explain_dataframe(
            dataframe,
            loaded,
            max_rows=max_rows,
            top_n=top_n,
            per_wafer_top_n=per_wafer_top_n,
            warning_threshold=warning_threshold,
            danger_threshold=danger_threshold,
            prediction_result=prediction,
        )
        logger.info("규칙 기반 보고서 구성 시작")
        return build_report(
            filename,
            loaded,
            prediction,
            explanation,
        )
    except InferenceInputError as exc:
        logger.warning("보고서 입력 오류: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelLoadError as exc:
        logger.exception("보고서 모델 또는 SHAP 처리 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("자동 분석 보고서 생성 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="자동 분석 보고서를 생성하는 중 서버 오류가 발생했습니다.",
        ) from exc


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(DEFAULT_TOP_N),
) -> ReportResponse:
    report = await _run_report(
        file,
        model_id,
        warning_threshold,
        danger_threshold,
        max_rows,
        top_n,
    )
    try:
        response = ReportResponse(**report)
        response.model_dump_json()
        return response
    except Exception as exc:
        logger.exception("보고서 JSON 응답 직렬화 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="보고서 JSON 응답을 생성하지 못했습니다.",
        ) from exc


@router.post("/report/download")
async def download_report(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(DEFAULT_TOP_N),
) -> Response:
    report = await _run_report(
        file,
        model_id,
        warning_threshold,
        danger_threshold,
        max_rows,
        top_n,
    )
    html = render_report_html(report)
    timestamp = report["created_at"].replace("-", "").replace(":", "")
    timestamp = timestamp[:15].replace("T", "_")
    filename = f"manufacturing_ai_report_{timestamp}.html"
    return Response(
        content=html.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_csv(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    warning_threshold: float = Form(DEFAULT_WARNING_THRESHOLD),
    danger_threshold: float = Form(DEFAULT_DANGER_THRESHOLD),
    max_rows: int = Form(DEFAULT_MAX_EXPLAIN_ROWS),
    top_n: int = Form(DEFAULT_TOP_N),
    per_wafer_top_n: int = Form(DEFAULT_WAFER_TOP_N),
    include_report: bool = Form(True),
) -> AnalyzeResponse:
    report = await _run_report(
        file,
        model_id,
        warning_threshold,
        danger_threshold,
        max_rows,
        top_n,
        per_wafer_top_n,
    )
    try:
        response = AnalyzeResponse(
            **build_automation_response(
                report,
                include_report=include_report,
            )
        )
        response.model_dump_json()
        return response
    except Exception as exc:
        logger.exception("통합 분석 응답 생성 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="통합 분석 결과 응답을 생성하지 못했습니다.",
        ) from exc
