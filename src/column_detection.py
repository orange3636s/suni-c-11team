"""데이터 스키마의 정규표현식으로 공정 feature 컬럼을 탐지한다."""

import re

from src.schema_loader import load_data_schema


def detect_feature_columns(
    columns: list[str],
    schema_config: dict | None = None,
) -> dict[str, list[str]]:
    """R, D, EQ 및 목표 컬럼을 설정 기반으로 대소문자 구분 없이 탐지한다."""
    if schema_config is None:
        schema_config = load_data_schema()

    patterns = schema_config["feature_patterns"]
    response_pattern = re.compile(patterns["response"], re.IGNORECASE)
    defect_pattern = re.compile(patterns["defect"], re.IGNORECASE)
    equipment_pattern = re.compile(patterns.get("equipment", r"a^"), re.IGNORECASE)
    config_pattern = re.compile(patterns.get("config_column", r"a^"), re.IGNORECASE)
    target_names = {
        schema_config["yield_column"],
        *schema_config["fail_rate_columns"],
        *schema_config["fail_bit_columns"],
    }

    string_columns = [
        column
        for column in columns
        if isinstance(column, str)
        and not column.lower().endswith("_missing")
    ]
    return {
        "r_columns": [
            column
            for column in string_columns
            if response_pattern.fullmatch(column)
        ],
        "d_columns": [
            column
            for column in string_columns
            if defect_pattern.fullmatch(column)
        ],
        "eq_columns": [
            column
            for column in string_columns
            if equipment_pattern.fullmatch(column)
        ],
        "config_columns": [
            column for column in string_columns if config_pattern.fullmatch(column)
        ],
        "target_columns": [
            column for column in string_columns if column in target_names
        ],
    }
