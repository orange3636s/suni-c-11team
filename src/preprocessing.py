"""YAML 설정에 따라 제조 공정 DataFrame을 전처리한다."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema
from src.config_parser import parse_config_columns


DEFAULT_PREPROCESSING_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "preprocessing.yaml"
)
LOT_PATTERN = re.compile(
    r"^(.+?)(?:[_-]?W(?:AFER|F)?\d+)$",
    re.IGNORECASE,
)
INDICATOR_OVERFIT_WARNING = (
    "결측 여부가 목표값을 과도하게 설명할 수 있으므로 모델 학습 시 "
    "과적합 검증이 필요합니다."
)
STRING_MISSING_VALUES = {"", "none", "null", "nan"}
SUPPORTED_MISSING_STRATEGIES = {"native", "median", "lot_mean"}
SUPPORTED_OUTLIER_STRATEGIES = {"flag_only", "iqr", "none"}
logger = logging.getLogger(__name__)


def _duplicate_column_names(columns: list[Any]) -> list[Any]:
    counts = Counter(columns)
    return list(
        dict.fromkeys(
            column for column in columns if counts[column] > 1
        )
    )


def load_preprocessing_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """YAML 파일에서 전처리 설정을 읽는다."""
    resolved_path = (
        Path(config_path) if config_path else DEFAULT_PREPROCESSING_PATH
    )
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("전처리 설정은 YAML 매핑이어야 합니다.")

    required_sections = ("missing", "outlier", "categorical")
    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise ValueError(
            "전처리 필수 설정이 누락되었습니다: "
            + ", ".join(missing_sections)
        )

    return config


def _extract_lot_values(id_values: pd.Series) -> pd.Series:
    """Wafer ID 끝의 Wafer 번호를 제거해 Lot 식별자를 추출한다."""
    string_values = id_values.astype("string")
    return string_values.str.extract(LOT_PATTERN, expand=False)


def _missing_summary(df: pd.DataFrame) -> tuple[int, float]:
    """DataFrame 전체 결측치 개수와 비율을 계산한다."""
    missing_count = int(df.isna().sum().sum())
    cell_count = int(df.shape[0] * df.shape[1])
    missing_rate = missing_count / cell_count if cell_count else 0.0
    return missing_count, missing_rate


def _standardize_missing_values(
    df: pd.DataFrame,
    columns: list[str],
) -> int:
    """feature 컬럼의 None 및 문자열 결측 표현을 np.nan으로 표준화한다."""
    standardized_string_count = 0
    for column in columns:
        series = df[column]
        string_missing_mask = series.map(
            lambda value: (
                isinstance(value, str)
                and value.strip().lower() in STRING_MISSING_VALUES
            )
        )
        standardized_string_count += int(string_missing_mask.sum())
        missing_mask = series.isna() | string_missing_mask
        if missing_mask.any():
            df.loc[missing_mask, column] = np.nan
    return standardized_string_count


def preprocess_dataframe(
    df: pd.DataFrame,
    schema_config: dict | None = None,
    preprocessing_config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """원본을 복사한 뒤 설정 기반 결측치 대체와 IQR clipping을 수행한다.

    Args:
        df: 전처리할 원본 pandas DataFrame.
        schema_config: 컬럼과 접미사 규칙. 지정하지 않으면 기본 데이터
            스키마 YAML을 읽는다.
        preprocessing_config: 전처리 규칙. 지정하지 않으면 기본 전처리
            YAML을 읽는다.

    Returns:
        전처리된 DataFrame과 처리 내역을 담은 리포트 딕셔너리.
    """
    schema = schema_config if schema_config is not None else load_data_schema()
    config = (
        preprocessing_config
        if preprocessing_config is not None
        else load_preprocessing_config()
    )
    processed_df = df.copy(deep=True)
    duplicate_columns = _duplicate_column_names(list(processed_df.columns))
    if duplicate_columns:
        raise ValueError(
            "전처리 데이터에 중복된 컬럼명이 있습니다: "
            + ", ".join(str(column) for column in duplicate_columns)
        )

    original_shape = tuple(df.shape)
    warnings: list[str] = []
    imputed_counts: dict[str, int] = {}
    clipped_counts: dict[str, int] = {}
    outlier_flagged_counts: dict[str, int] = {}
    added_indicator_columns: list[str] = []
    new_indicator_data: dict[str, pd.Series] = {}
    existing_indicator_updates: dict[str, pd.Series] = {}

    id_column = schema["id_column"]
    detected_columns = detect_feature_columns(
        list(processed_df.columns),
        schema,
    )
    r_columns = list(dict.fromkeys(detected_columns["r_columns"]))
    d_columns = list(dict.fromkeys(detected_columns["d_columns"]))
    categorical_feature_columns = list(
        dict.fromkeys(detected_columns["eq_columns"])
    )
    config_report = {
        "parser_version": schema.get("config_parser_version", "legacy"),
        "config_columns": [],
        "config_column_count": 0,
        "parsed_value_count": 0,
        "parse_error_count": 0,
        "parse_errors": [],
        "derived_columns": [],
    }
    if detected_columns.get("config_columns"):
        processed_df, config_report = parse_config_columns(processed_df, schema)
        categorical_feature_columns.extend(config_report["config_columns"])
    numeric_feature_columns = list(
        dict.fromkeys([*r_columns, *d_columns])
    )
    standardized_missing_count = _standardize_missing_values(
        processed_df,
        [*numeric_feature_columns, *categorical_feature_columns],
    )

    for column in numeric_feature_columns:
        processed_df[column] = pd.to_numeric(
            processed_df[column],
            errors="coerce",
        )

    missing_before, missing_rate_before = _missing_summary(processed_df)

    missing_config = config["missing"]
    add_indicator = bool(missing_config["add_indicator"])
    missing_strategy = missing_config["strategy"]
    fallback_strategy = missing_config["fallback"]
    fallback_used = False
    if missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError(f"지원하지 않는 결측치 전략입니다: {missing_strategy}")
    if fallback_strategy != "median":
        raise ValueError(f"지원하지 않는 결측치 fallback입니다: {fallback_strategy}")

    lot_values = pd.Series(pd.NA, index=processed_df.index, dtype="string")
    if id_column in processed_df.columns:
        lot_values = _extract_lot_values(processed_df[id_column])
    elif missing_strategy == "lot_mean":
        warnings.append(
            f"{id_column} 컬럼이 없어 Lot 평균을 사용할 수 없습니다. "
            "전체 중앙값을 사용합니다."
        )

    all_missing_columns = {
        column
        for column in numeric_feature_columns
        if processed_df[column].notna().sum() == 0
    }

    for column in numeric_feature_columns:
        missing_mask = processed_df[column].isna()
        missing_count = int(missing_mask.sum())

        if add_indicator:
            indicator_column = f"{column}_missing"
            if indicator_column in processed_df.columns:
                if missing_count:
                    logger.info(
                        f"{indicator_column} 컬럼이 이미 존재하여 값을 갱신했습니다."
                    )
                    existing_indicator_updates[indicator_column] = (
                        missing_mask.astype("int8")
                    )
            else:
                new_indicator_data[indicator_column] = missing_mask.astype(
                    "int8"
                )
                added_indicator_columns.append(indicator_column)

        if not missing_count:
            imputed_counts[column] = 0
            continue

        if missing_strategy == "native":
            imputed_counts[column] = 0
            continue
        if missing_strategy == "lot_mean" and id_column in processed_df.columns:
            extract_failed_mask = missing_mask & lot_values.isna()
            if extract_failed_mask.any():
                logger.info(
                    f"{column} 컬럼의 결측 행 중 Lot을 추출할 수 없는 "
                    f"{int(extract_failed_mask.sum())}개 행에 전체 중앙값 "
                    "fallback을 사용합니다."
                )

            valid_lot_mask = lot_values.notna()
            lot_means = (
                processed_df.loc[valid_lot_mask, column]
                .groupby(lot_values.loc[valid_lot_mask])
                .mean()
            )
            lot_fill_values = lot_values.map(lot_means)
            processed_df.loc[missing_mask, column] = (
                processed_df.loc[missing_mask, column].fillna(
                    lot_fill_values.loc[missing_mask]
                )
            )
        elif missing_strategy == "median":
            pass

        remaining_missing_mask = processed_df[column].isna()
        if remaining_missing_mask.any():
            if missing_strategy == "lot_mean":
                fallback_used = True
            median_value = processed_df[column].median(skipna=True)
            if pd.isna(median_value):
                median_value = 0
                warnings.append(
                    f"{column} 컬럼은 전체 중앙값을 계산할 수 없어 0으로 "
                    "결측치를 채웠습니다."
                )
            processed_df.loc[remaining_missing_mask, column] = median_value

        imputed_counts[column] = missing_count

    if existing_indicator_updates:
        update_df = pd.DataFrame(
            existing_indicator_updates,
            index=processed_df.index,
        )
        processed_df.loc[:, list(update_df.columns)] = update_df

    if new_indicator_data:
        indicator_df = pd.DataFrame(
            new_indicator_data,
            index=processed_df.index,
        )
        processed_df = pd.concat(
            [processed_df, indicator_df],
            axis=1,
        )
        duplicate_columns = _duplicate_column_names(
            list(processed_df.columns)
        )
        if duplicate_columns:
            raise ValueError(
                "전처리 indicator 결합 후 중복된 컬럼명이 있습니다: "
                + ", ".join(str(column) for column in duplicate_columns)
            )
        processed_df = processed_df.copy()

    categorical_fill_value = str(config["categorical"]["fill_value"])
    for column in categorical_feature_columns:
        missing_count = int(processed_df[column].isna().sum())
        processed_df[column] = (
            processed_df[column].astype("string").fillna(categorical_fill_value)
        )
        imputed_counts[column] = missing_count

    outlier_config = config["outlier"]
    outlier_method = outlier_config["method"]
    lower_multiplier = float(outlier_config["lower_multiplier"])
    upper_multiplier = float(outlier_config["upper_multiplier"])
    if outlier_method not in SUPPORTED_OUTLIER_STRATEGIES:
        raise ValueError(f"지원하지 않는 이상치 처리 전략입니다: {outlier_method}")

    for column in numeric_feature_columns:
        clipped_counts[column] = 0
        outlier_flagged_counts[column] = 0

        if outlier_method == "none":
            continue

        if column in all_missing_columns:
            logger.info(
                f"{column} 컬럼은 값이 전부 결측치였으므로 IQR clipping을 "
                "적용하지 않았습니다."
            )
            continue

        valid_values = processed_df[column].dropna()
        if len(valid_values) < 4:
            logger.info(
                f"{column} 컬럼은 유효값이 너무 적어 IQR clipping을 "
                "적용하지 않았습니다."
            )
            continue

        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            logger.info(
                f"{column} 컬럼은 IQR이 0이므로 clipping을 적용하지 않았습니다."
            )
            continue

        lower_bound = q1 - lower_multiplier * iqr
        upper_bound = q3 + upper_multiplier * iqr
        clip_mask = (
            (processed_df[column] < lower_bound)
            | (processed_df[column] > upper_bound)
        )
        outlier_flagged_counts[column] = int(clip_mask.sum())
        if outlier_method == "flag_only":
            # Flag counts are descriptive only here. Model-time boundaries must
            # be learned from the training split, never from the full dataset.
            pass
        else:
            clipped_counts[column] = int(clip_mask.sum())
            processed_df[column] = processed_df[column].clip(
                lower=lower_bound,
                upper=upper_bound,
            )

    missing_after, missing_rate_after = _missing_summary(processed_df)
    remaining_numeric_missing_count = int(
        processed_df[numeric_feature_columns].isna().sum().sum()
    ) if numeric_feature_columns else 0
    all_missing_indicators = [
        column for column in processed_df.columns if str(column).endswith("_missing")
    ]
    step_feature_columns = [
        column
        for column in numeric_feature_columns
        if str(column).lower().startswith("step")
        and not str(column).endswith("_missing")
    ]
    processing_summary = {
        "missing_strategy": missing_strategy,
        "outlier_strategy": outlier_method,
        "missing_indicator": add_indicator,
        "outlier_indicator": outlier_method == "flag_only",
        "missing_indicator_count": len(all_missing_indicators),
        "outlier_indicator_count": len(numeric_feature_columns) if outlier_method == "flag_only" else 0,
        "r_column_count": len(r_columns),
        "d_column_count": len(d_columns),
        "categorical_column_count": len(categorical_feature_columns),
        "config_parsed": config_report.get("parse_error_count", 0) == 0,
        "config_column_count": config_report.get("config_column_count", 0),
        "step_feature_count": len(step_feature_columns),
        "fallback_used": fallback_used,
    }
    report = {
        "original_shape": original_shape,
        "processed_shape": tuple(processed_df.shape),
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_rate_before": missing_rate_before,
        "missing_rate_after": missing_rate_after,
        "imputed_counts": imputed_counts,
        "clipped_counts": clipped_counts,
        "outlier_flagged_counts": outlier_flagged_counts,
        "outlier_policy": outlier_method,
        "nan_maintained_count": remaining_numeric_missing_count,
        "train_only_median_imputation_feature_count": (
            len(numeric_feature_columns) if missing_strategy == "native" else 0
        ),
        "observed_indicator_count": len(all_missing_indicators),
        "optional_clipping_count": sum(clipped_counts.values()) if outlier_method == "iqr" else 0,
        "config_parsing_result": config_report,
        "preprocessing_policy": {
            "missing_strategy": missing_strategy,
            "outlier_policy": outlier_method,
            "statistics_fit_scope": "model_train_split",
            "missing_indicator": add_indicator,
            "outlier_indicator": outlier_method == "flag_only",
            "fallback_used": fallback_used,
        },
        "processing_summary": processing_summary,
        "added_indicator_columns": added_indicator_columns,
        "standardized_missing_count": standardized_missing_count,
        "remaining_numeric_missing_count": remaining_numeric_missing_count,
        "detected_columns": detected_columns,
        "warnings": warnings,
    }
    return processed_df, report
