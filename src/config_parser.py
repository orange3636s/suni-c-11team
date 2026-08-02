"""Deterministic parser for semicon yield schema-v2 Step_Config values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_validation import load_data_schema


CONFIG_PARSER_VERSION = "2.0"


@dataclass(frozen=True)
class ParsedConfig:
    step: int
    model: str
    equipment: str
    chamber: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "equipment": self.equipment,
            "chamber": self.chamber,
        }


def parse_config_value(
    column: str,
    value: Any,
    schema_config: dict[str, Any] | None = None,
) -> ParsedConfig:
    """Parse one Config value and verify its embedded step."""
    schema = schema_config or load_data_schema()
    column_match = re.fullmatch(
        schema["feature_patterns"]["config_column"], column
    )
    if column_match is None:
        raise ValueError(f"Config 컬럼명이 올바르지 않습니다: {column}")
    if pd.isna(value) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{column} 값이 비어 있습니다.")
    value_match = re.fullmatch(
        schema["feature_patterns"]["config_value"], value.strip()
    )
    if value_match is None:
        raise ValueError(f"{column} Config 형식이 올바르지 않습니다: {value}")
    column_step = int(column_match.group("step"))
    value_step = int(value_match.group("step"))
    if column_step != value_step:
        raise ValueError(
            f"{column}의 Step과 Config 값의 Step{value_step}이 일치하지 않습니다."
        )
    return ParsedConfig(
        step=value_step,
        model=f"Model{value_match.group('model')}",
        equipment=value_match.group("equipment"),
        chamber=value_match.group("chamber"),
    )


def parse_config_columns(
    dataframe: pd.DataFrame,
    schema_config: dict[str, Any] | None = None,
    *,
    keep_original: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand all detected Config columns without converting errors to categories."""
    schema = schema_config or load_data_schema()
    pattern = re.compile(schema["feature_patterns"]["config_column"])
    config_columns = [
        str(column) for column in dataframe.columns
        if isinstance(column, str) and pattern.fullmatch(column)
    ]
    result = dataframe.copy(deep=True)
    derived: dict[str, pd.Series] = {}
    errors: list[dict[str, Any]] = []
    parsed_count = 0
    for column in config_columns:
        step = int(pattern.fullmatch(column).group("step"))  # type: ignore[union-attr]
        names = {
            "model": f"Step{step}_Model",
            "equipment": f"Step{step}_Equipment",
            "chamber": f"Step{step}_Chamber",
        }
        collisions = [name for name in names.values() if name in result.columns]
        if collisions:
            raise ValueError(
                "Config 파생 컬럼명이 기존 컬럼과 충돌합니다: "
                + ", ".join(collisions)
            )
        values = {key: [] for key in names}
        for index, value in result[column].items():
            try:
                parsed = parse_config_value(column, value, schema)
                parsed_count += 1
                values["model"].append(parsed.model)
                values["equipment"].append(parsed.equipment)
                values["chamber"].append(parsed.chamber)
            except ValueError as exc:
                values["model"].append(pd.NA)
                values["equipment"].append(pd.NA)
                values["chamber"].append(pd.NA)
                errors.append({"row": index, "column": column, "message": str(exc)})
        for key, name in names.items():
            derived[name] = pd.Series(values[key], index=result.index, dtype="string")
    if derived:
        result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)
    if not keep_original and config_columns:
        result = result.drop(columns=config_columns)
    return result, {
        "parser_version": CONFIG_PARSER_VERSION,
        "config_columns": config_columns,
        "config_column_count": len(config_columns),
        "parsed_value_count": parsed_count,
        "parse_error_count": len(errors),
        "parse_errors": errors,
        "derived_columns": list(derived),
    }
