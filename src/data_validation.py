"""반도체 제조 공정 DataFrame의 기본 구조와 품질을 검증한다."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

from src.column_detection import detect_feature_columns


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
    "feature_patterns",
)
VALIDATION_MODES = {"training", "inference", "analysis"}


def load_data_schema(
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """YAML 파일에서 데이터 컬럼 및 접미사 설정을 읽는다."""
    resolved_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with resolved_path.open(encoding="utf-8") as schema_file:
        schema = yaml.safe_load(schema_file)

    if not isinstance(schema, dict):
        raise ValueError("데이터 스키마 설정은 YAML 매핑이어야 합니다.")

    # V2 keeps aliases for old integrations, while custom v1 schemas remain valid.
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
    *,
    require_id: bool = True,
    require_yield: bool = True,
    validation_mode: str | None = None,
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
    if validation_mode is not None:
        if validation_mode not in VALIDATION_MODES:
            raise ValueError(
                "validation_mode는 training, inference, analysis 중 하나여야 합니다."
            )
        require_id = validation_mode == "training"
        require_yield = validation_mode == "training"
    id_column = schema["id_column"]
    yield_column = schema["yield_column"]

    row_count, column_count = df.shape
    required_columns = tuple(
        column
        for column, required in (
            (id_column, require_id),
            (yield_column, require_yield),
        )
        if required
    )
    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    column_names = [column for column in df.columns if isinstance(column, str)]
    detected_columns = detect_feature_columns(column_names, schema)
    r_columns = detected_columns["r_columns"]
    d_columns = detected_columns["d_columns"]
    eq_columns = detected_columns["eq_columns"]
    config_columns = detected_columns.get("config_columns", [])
    target_columns = detected_columns["target_columns"]

    total_missing_count = int(df.isna().sum().sum())
    total_cell_count = row_count * column_count
    overall_missing_rate = (
        total_missing_count / total_cell_count if total_cell_count else 0.0
    )

    duplicate_wafer_id_count = 0
    if id_column in df.columns:
        duplicate_wafer_id_count = int(df[id_column].duplicated().sum())

    lot_id_column = schema.get("lot_id_column", "Lot_ID")
    wafer_slot_column = schema.get("wafer_slot_column", "Wafer_Slot")
    invalid_numeric_count = 0
    for column in [*r_columns, *d_columns, *target_columns]:
        if column not in df.columns:
            continue
        source = df[column]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid_numeric_count += int((source.notna() & numeric.isna()).sum())

    def coverage(columns: list[str]) -> float:
        if not columns or row_count == 0:
            return 0.0
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        return float(numeric.notna().sum().sum() / (row_count * len(columns)))

    config_total = row_count * len(config_columns)
    config_nonempty = 0
    config_parse_error_count = 0
    if config_columns:
        config_nonempty = int(
            df[config_columns].apply(
                lambda series: series.notna() & series.astype("string").str.strip().ne("")
            ).sum().sum()
        )
        try:
            from src.config_parser import parse_config_columns

            _, config_report = parse_config_columns(df, schema)
            config_parse_error_count = int(config_report["parse_error_count"])
        except ValueError as exc:
            config_parse_error_count = config_total
            warnings = [str(exc)]
        else:
            warnings = []
    else:
        warnings = []
    config_completeness_rate = (
        config_nonempty / config_total if config_total else 0.0
    )

    fail_rates = [
        column for column in schema["fail_rate_columns"] if column in df.columns
    ]
    target_consistency_rate: float | None = None
    if yield_column in df.columns and len(fail_rates) == len(schema["fail_rate_columns"]):
        targets = df[[yield_column, *fail_rates]].apply(pd.to_numeric, errors="coerce")
        complete = targets.notna().all(axis=1)
        if complete.any():
            tolerance = float(schema.get("target_consistency_tolerance", 0.001))
            consistent = (targets.loc[complete].sum(axis=1) - 100.0).abs() <= tolerance
            target_consistency_rate = float(consistent.mean())

    lot_structure_consistency_rate: float | None = None
    wafers_per_lot: dict[str, int] = {}
    if lot_id_column in df.columns:
        counts = df.groupby(lot_id_column, dropna=False).size()
        wafers_per_lot = {str(key): int(value) for key, value in counts.items()}
        if len(counts):
            mode_count = int(counts.mode().iloc[0])
            lot_structure_consistency_rate = float((counts == mode_count).mean())

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
            "스키마 패턴과 일치하는 Response 컬럼이 발견되지 않았습니다."
        )
    if not d_columns:
        warnings.append(
            "스키마 패턴과 일치하는 Defect 컬럼이 발견되지 않았습니다."
        )
    if not eq_columns and not config_columns:
        warnings.append(
            "스키마 패턴과 일치하는 Config 또는 Equipment 컬럼이 발견되지 않았습니다."
        )

    if validation_mode == "training":
        if lot_id_column not in df.columns and config_columns:
            errors.append(f"필수 컬럼이 누락되었습니다: {lot_id_column}")
        if not r_columns and not config_columns:
            errors.append("학습에 사용할 R 또는 Config feature가 없습니다.")
        if config_parse_error_count:
            errors.append(
                f"Config parse error가 {config_parse_error_count}개 발견되었습니다."
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
        "config_columns": config_columns,
        "target_columns": target_columns,
        "schema_version": schema.get("schema_version", "legacy_v1"),
        "validation_mode": validation_mode,
        "config_completeness_rate": config_completeness_rate,
        "r_measurement_coverage": coverage(r_columns),
        "d_measurement_coverage": coverage(d_columns),
        "required_field_error_count": len(missing_required_columns),
        "config_parse_error_count": config_parse_error_count,
        "target_consistency_rate": target_consistency_rate,
        "lot_structure_consistency_rate": lot_structure_consistency_rate,
        "duplicate_wafer_count": duplicate_wafer_id_count,
        "invalid_numeric_count": invalid_numeric_count,
        "lot_count": int(df[lot_id_column].nunique(dropna=True)) if lot_id_column in df.columns else 0,
        "wafers_per_lot": wafers_per_lot,
        "structural_unmeasured_count": int(
            row_count * (len(r_columns) + len(d_columns))
            - df[[*r_columns, *d_columns]].notna().sum().sum()
        ) if r_columns or d_columns else 0,
        "warnings": warnings,
        "errors": errors,
    }
