from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

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
    ModelListResponse,
    ModelMetrics,
    ModelSummary,
    PredictionModelInfo,
    PredictionResponse,
    PredictionSummary,
    PreprocessChanges,
    PreprocessResponse,
    ReportResponse,
    TrainResponse,
    ValidationResponse,
    ValidationResult,
)
from api.settings import settings
from src.data_validation import load_data_schema, validate_dataframe
from src.automation.analyzer import build_automation_response
from src.ml.dataset import (
    ALLOWED_TARGETS,
    RANDOM_STATE,
    prepare_dataset,
    split_dataset,
)
from src.ml.model_io import save_model_bundle
from src.ml.inference import (
    DEFAULT_DANGER_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    MAX_PREDICTION_ROWS,
    InferenceInputError,
    PredictionResult,
    ModelLoadError,
    list_prediction_models,
    load_prediction_model,
    predict_dataframe,
)
from src.ml.explainability import (
    DEFAULT_MAX_EXPLAIN_ROWS,
    DEFAULT_TOP_N,
    DEFAULT_WAFER_TOP_N,
    ExplainResult,
    explain_dataframe,
)
from src.ml.training import train_regression_models
from src.preprocessing import preprocess_dataframe
from src.reporting.export import render_report_html
from src.reporting.report_builder import build_report


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["data"])

MAX_FILE_SIZE = settings.max_upload_size_bytes
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")
MODEL_DIR = settings.model_dir


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
            detail=(
                "파일 크기는 "
                f"{settings.max_upload_size_mb}MB 이하여야 합니다."
            ),
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="비어 있는 CSV 파일은 처리할 수 없습니다.",
        )

    for encoding in SUPPORTED_ENCODINGS:
        try:
            decoded_content = content.decode(encoding)
            header = next(csv.reader(StringIO(decoded_content)), [])
            duplicate_columns = _duplicate_names(header)
            if duplicate_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "CSV에 중복된 컬럼명이 있습니다: "
                        + ", ".join(duplicate_columns)
                    ),
                )
            dataframe = pd.read_csv(StringIO(decoded_content))
            return filename, dataframe
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


@router.post("/train", response_model=TrainResponse)
async def train_model(
    file: UploadFile = File(...),
    target: str = Form("Y"),
) -> TrainResponse:
    _, dataframe = await _read_csv_upload(file)
    logger.info(
        "학습 CSV 읽기 완료: rows=%d, columns=%d",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    if target not in ALLOWED_TARGETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="지원하지 않는 목표 변수입니다. Y부터 Y10까지만 사용할 수 있습니다.",
        )

    try:
        validation = validate_dataframe(dataframe)
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

    try:
        processed, preprocessing_report = preprocess_dataframe(dataframe)
    except Exception as exc:
        logger.exception("학습 데이터 전처리 중 내부 오류")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="데이터 전처리 중 서버 내부 오류가 발생했습니다.",
        ) from exc
    logger.info(
        "학습 데이터 전처리 완료: rows=%d, columns=%d",
        processed.shape[0],
        processed.shape[1],
    )

    try:
        dataset = prepare_dataset(processed, target=target)
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

    try:
        split = split_dataset(dataset, random_state=RANDOM_STATE)
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

    try:
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

    metrics = {
        name: values.as_dict()
        for name, values in training.metrics.items()
    }
    try:
        model_path, metadata_path, _ = save_model_bundle(
            training.best_model,
            target=target,
            model_name=training.best_model_name,
            feature_columns=dataset.feature_columns,
            metrics=metrics,
            random_state=RANDOM_STATE,
            split_method=split.split_method,
            model_dir=MODEL_DIR,
        )
    except Exception as exc:
        logger.exception("학습 모델 또는 메타데이터 저장 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 학습에는 성공했지만 파일 저장에 실패했습니다.",
        ) from exc
    logger.info("학습 모델 저장 완료")

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


@router.get("/models", response_model=ModelListResponse)
def get_models() -> ModelListResponse:
    try:
        models, warnings = list_prediction_models(MODEL_DIR)
        return ModelListResponse(
            models=[ModelSummary(**model) for model in models],
            warnings=warnings,
        )
    except ModelLoadError as exc:
        logger.exception("모델 목록 조회 실패")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def _prediction_response(
    filename: str,
    result: PredictionResult,
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
        predictions=result.predictions,
        warnings=result.warnings,
        truncated=result.truncated,
    )
    response.model_dump_json()
    return response


async def _run_prediction(
    file: UploadFile,
    model_id: str,
    warning_threshold: float,
    danger_threshold: float,
    *,
    max_rows: int | None,
) -> tuple[str, PredictionResult]:
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
        return filename, result
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
    filename, result = await _run_prediction(
        file,
        model_id,
        warning_threshold,
        danger_threshold,
        max_rows=MAX_PREDICTION_ROWS,
    )
    try:
        return _prediction_response(filename, result)
    except Exception as exc:
        logger.exception("예측 응답 직렬화 실패")
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
    _, result = await _run_prediction(
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
