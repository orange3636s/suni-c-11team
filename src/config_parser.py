"""Config normalization and the canonical Model/EQ/Chamber parser.

The source Config column remains one categorical model feature. Hierarchy
tokens are derived only for analysis/display and never replace the raw value.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.schema_loader import load_data_schema


CONFIG_PARSER_VERSION = "4.0-canonical-hierarchy"
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


def parse_config_hierarchy_value(
    column: str,
    value: Any,
    schema_config: dict[str, Any] | None = None,
) -> dict[str, str | int | bool]:
    """Parse one Config value according to ``data_schema.yaml``.

    Invalid legacy categories are retained under an ``Unknown`` hierarchy so
    every observed wafer remains represented without invented tokens.
    """
    schema = schema_config or load_data_schema()
    normalized = parse_config_value(column, value, schema)
    column_match = re.compile(schema["feature_patterns"]["config_column"], re.IGNORECASE).fullmatch(column)
    value_match = re.compile(schema["feature_patterns"]["config_value"], re.IGNORECASE).fullmatch(normalized)
    step = int(column_match.group("step")) if column_match is not None else 0
    if value_match is None or int(value_match.group("step")) != step:
        return {
            "step": step,
            "model": "Unknown",
            "equipment": "Unknown",
            "chamber": normalized,
            "matched": False,
        }
    model = str(value_match.group("model"))
    return {
        "step": step,
        "model": model if model.lower().startswith("model") else f"Model{model}",
        "equipment": str(value_match.group("equipment")).upper(),
        "chamber": str(value_match.group("chamber")).upper(),
        "matched": True,
    }


def config_hierarchy_series(
    column: str,
    values: pd.Series,
    level: str,
    schema_config: dict[str, Any] | None = None,
) -> pd.Series:
    """Return one canonical hierarchy level while preserving missing values."""
    if level not in {"model", "equipment", "chamber"}:
        raise ValueError(f"지원하지 않는 Config 계층입니다: {level}")

    def _parse(value: Any) -> Any:
        try:
            return parse_config_hierarchy_value(column, value, schema_config)[level]
        except ValueError:
            return pd.NA

    return values.map(_parse).astype("string")


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
