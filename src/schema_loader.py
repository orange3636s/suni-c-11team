"""데이터 스키마(YAML) 로딩 -- 다른 어떤 src 모듈도 import하지 않는 leaf 모듈.

`column_detection.py`, `data_validation.py`, `config_parser.py` 모두 스키마
로딩이 필요하지만 서로를 import하면 순환 참조가 생긴다 (예: data_validation이
column_detection을 쓰고, column_detection의 기본 스키마 로딩이 다시
data_validation을 필요로 하는 식). `load_data_schema`를 이 leaf 모듈로 옮겨
셋 다 여기서만 가져오게 하면 순환이 원천적으로 생기지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
