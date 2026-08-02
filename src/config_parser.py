"""Config normalization without token decomposition.

Config is one categorical process value.  Model, equipment, and chamber tokens
are deliberately not derived from it.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data_validation import load_data_schema


CONFIG_PARSER_VERSION = "3.0-frequency-no-decomposition"
MISSING_CONFIG_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def parse_config_value(
    column: str,
    value: Any,
    schema_config: dict[str, Any] | None = None,
) -> str:
    """Return the trimmed whole Config category; never parse embedded tokens."""
    schema = schema_config or load_data_schema()
    pattern = re.compile(schema["feature_patterns"]["config_column"], re.IGNORECASE)
    if pattern.fullmatch(column) is None:
        raise ValueError(f"Config 컬럼명이 올바르지 않습니다: {column}")
    if value is None or pd.isna(value):
        raise ValueError(f"{column} 값이 비어 있습니다.")
    normalized = str(value).strip()
    if normalized.lower() in MISSING_CONFIG_STRINGS:
        raise ValueError(f"{column} 값이 비어 있습니다.")
    return normalized


def parse_config_columns(
    dataframe: pd.DataFrame,
    schema_config: dict[str, Any] | None = None,
    *,
    keep_original: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize whole Config strings while preserving the original columns."""
    if not keep_original:
        raise ValueError("원본 Config 컬럼은 삭제할 수 없습니다.")
    schema = schema_config or load_data_schema()
    pattern = re.compile(schema["feature_patterns"]["config_column"], re.IGNORECASE)
    config_columns = [
        str(column)
        for column in dataframe.columns
        if isinstance(column, str) and pattern.fullmatch(column)
    ]
    result = dataframe.copy(deep=False)
    missing_count = 0
    category_counts: dict[str, int] = {}
    for column in config_columns:
        normalized = dataframe[column].map(
            lambda value: (
                pd.NA
                if value is None
                or pd.isna(value)
                or str(value).strip().lower() in MISSING_CONFIG_STRINGS
                else str(value).strip()
            )
        ).astype("string")
        result[column] = normalized
        missing_count += int(normalized.isna().sum())
        category_counts[column] = int(normalized.nunique(dropna=True))
    return result, {
        "parser_version": CONFIG_PARSER_VERSION,
        "config_parser_version": CONFIG_PARSER_VERSION,
        "normalization": "trim_whole_config_category",
        "config_columns": config_columns,
        "config_column_count": len(config_columns),
        "category_counts": category_counts,
        "total_category_count": int(sum(category_counts.values())),
        "missing_value_count": missing_count,
        "parsed_value_count": int(len(dataframe) * len(config_columns) - missing_count),
        "parse_error_count": 0,
        "parse_errors": [],
        "derived_columns": [],
        "config_strings_decomposed": False,
    }
