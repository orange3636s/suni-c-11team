"""반도체 제조 공정 DataFrame의 기본 구조와 품질을 검증한다."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype


DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "data_schema.yaml"
)
REQUIRED_SCHEMA_KEYS = (
    "id_column",
    "yield_column",
    "fail_rate_columns",
    "fail_bit_columns",
    "response_suffix",
    "defect_suffix",
    "equipment_suffix",
)


def load_data_schema(
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """YAML 파일에서 데이터 컬럼 및 접미사 설정을 읽는다."""
    resolved_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with resolved_path.open(encoding="utf-8") as schema_file:
        schema = yaml.safe_load(schema_file)

    if not isinstance(schema, dict):
        raise ValueError("데이터 스키마 설정은 YAML 매핑이어야 합니다.")

    missing_keys = [key for key in REQUIRED_SCHEMA_KEYS if key not in schema]
    if missing_keys:
        raise ValueError(
            "데이터 스키마 필수 설정이 누락되었습니다: "
            + ", ".join(missing_keys)
        )

    return schema


def validate_dataframe(
    df: pd.DataFrame,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """원본을 변경하지 않고 업로드된 공정 DataFrame의 기본 품질을 검증한다.

    Args:
        df: 검증할 pandas DataFrame.
        schema_path: 데이터 컬럼 규칙을 정의한 YAML 파일 경로. 지정하지
            않으면 프로젝트 기본 스키마를 사용한다.

    Returns:
        행·열 수, 결측치, 중복 ID, 자동 탐지 컬럼, 경고와 오류를 포함한
        검증 결과 딕셔너리.
    """
    schema = load_data_schema(schema_path)
    id_column = schema["id_column"]
    yield_column = schema["yield_column"]
    fail_rate_columns = schema["fail_rate_columns"]
    fail_bit_columns = schema["fail_bit_columns"]
    response_suffix = schema["response_suffix"]
    defect_suffix = schema["defect_suffix"]
    equipment_suffix = schema["equipment_suffix"]

    row_count, column_count = df.shape
    required_columns = (id_column, yield_column)
    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    column_names = [column for column in df.columns if isinstance(column, str)]
    r_columns = [
        column for column in column_names if column.endswith(response_suffix)
    ]
    d_columns = [
        column for column in column_names if column.endswith(defect_suffix)
    ]
    eq_columns = [
        column for column in column_names if column.endswith(equipment_suffix)
    ]
    target_names = {
        yield_column,
        *fail_rate_columns,
        *fail_bit_columns,
    }
    target_columns = [column for column in column_names if column in target_names]

    total_missing_count = int(df.isna().sum().sum())
    total_cell_count = row_count * column_count
    overall_missing_rate = (
        total_missing_count / total_cell_count if total_cell_count else 0.0
    )

    duplicate_wafer_id_count = 0
    if id_column in df.columns:
        duplicate_wafer_id_count = int(df[id_column].duplicated().sum())

    warnings: list[str] = []
    errors: list[str] = []

    if df.empty:
        errors.append("DataFrame이 비어 있습니다.")

    if missing_required_columns:
        errors.append(
            "필수 컬럼이 누락되었습니다: "
            + ", ".join(missing_required_columns)
        )

    if not r_columns:
        warnings.append(
            f"{response_suffix}로 끝나는 Response 컬럼이 발견되지 않았습니다."
        )
    if not d_columns:
        warnings.append(
            f"{defect_suffix}로 끝나는 Defect 컬럼이 발견되지 않았습니다."
        )
    if not eq_columns:
        warnings.append(
            f"{equipment_suffix}로 끝나는 Equipment 컬럼이 발견되지 않았습니다."
        )

    if yield_column in df.columns and not is_numeric_dtype(df[yield_column]):
        warnings.append(f"{yield_column} 컬럼이 수치형이 아닙니다.")

    return {
        "is_valid": not errors,
        "row_count": row_count,
        "column_count": column_count,
        "missing_required_columns": missing_required_columns,
        "duplicate_wafer_id_count": duplicate_wafer_id_count,
        "total_missing_count": total_missing_count,
        "overall_missing_rate": overall_missing_rate,
        "r_columns": r_columns,
        "d_columns": d_columns,
        "eq_columns": eq_columns,
        "target_columns": target_columns,
        "warnings": warnings,
        "errors": errors,
    }
