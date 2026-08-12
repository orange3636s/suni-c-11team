from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema
from src.preprocessing import _extract_lot_values
from src.config_parser import parse_config_columns
from src.ml.memory_usage import log_memory_stage


logger = logging.getLogger(__name__)
ALLOWED_TARGETS = ("Y",)
EXCLUDED_TARGET_COLUMNS = tuple(f"Y{index}" for index in range(1, 11))
TARGET_COLUMN = "Y"
KNOWN_IDENTIFIER_COLUMNS = (
    "Lot_Wafer_ID", "LOT_WAFER_ID", "LOT_WF_ID", "Lot_ID", "LOT_ID",
    "Wafer_ID", "WAFER_ID", "Wafer_Slot", "WAFER_SLOT",
)
RANDOM_STATE = 42
MINIMUM_TRAINING_ROWS = 10


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return list(dict.fromkeys(value for value in values if counts[value] > 1))


def has_target_column(dataframe: pd.DataFrame, target: str = TARGET_COLUMN) -> bool:
    """단일 판정 기준 -- 학습
    (`prepare_dataset`)과 자동 수집(`src/automation/ingest.py`)이 이
    함수 하나로 "Y 있음"을 판정한다. 최종 수율 컬럼 `Y`의 유무만 본다 --
    Y1~Y5나 Y6~Y10 같은 부분 라벨은 이 기준에서 "Y 없음"이다(학습
    파이프라인이 애초에 `Y`만 목표 변수로 받아들이기 때문)."""
    return target in dataframe.columns


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    warnings: list[str] = field(default_factory=list)
    raw_feature_columns: list[str] = field(default_factory=list)
    config_report: dict = field(default_factory=dict)
    target_leakage_check: dict = field(default_factory=dict)


def prepare_dataset(
    dataframe: pd.DataFrame,
    target: str = "Y",
    *,
    add_missing_indicators: bool = True,
) -> PreparedDataset:
    if target != TARGET_COLUMN:
        raise ValueError(
            "지원하지 않는 목표 변수입니다. Y부터 Y10까지만 사용할 수 있습니다."
        )
    schema = load_data_schema()
    log_memory_stage(
        logger,
        "dataset_prepare_start",
        rows=len(dataframe),
        columns=len(dataframe.columns),
        target=target,
    )
    # The source frame is never mutated in place. A shallow frame avoids a full
    # CSV-sized duplicate; column assignment below creates only the new blocks.
    working = dataframe.copy(deep=False)
    config_report: dict = {}
    detected_raw = detect_feature_columns(list(dataframe.columns), schema)
    if detected_raw.get("config_columns"):
        working, config_report = parse_config_columns(dataframe, schema)
    column_names = list(working.columns)
    duplicate_columns = _duplicates(column_names)
    logger.info(
        "학습 DataFrame 컬럼 확인: total=%d, unique=%d, duplicates=%s",
        len(column_names),
        len(set(column_names)),
        duplicate_columns,
    )
    if column_names.count(target) > 1:
        raise ValueError(f"목표 변수 '{target}' 컬럼이 중복되어 있습니다.")
    if duplicate_columns:
        raise ValueError(
            "학습 데이터에 중복된 컬럼명이 있습니다: "
            + ", ".join(str(column) for column in duplicate_columns)
        )
    if not has_target_column(working, target):
        raise ValueError(
            "학습 데이터에 최종 수율 컬럼 Y가 없습니다. "
            "Y 컬럼이 포함된 CSV 파일을 선택해주세요."
        )

    id_column = schema["id_column"]
    if id_column not in working.columns:
        raise ValueError(f"식별자 컬럼 '{id_column}'이 없습니다.")

    detected = detect_feature_columns(column_names, schema)
    detected_feature_candidates = [
        *detected["r_columns"],
        *detected["d_columns"],
        *detected["eq_columns"],
    ]
    detected_duplicate_features = _duplicates(
        detected_feature_candidates
    )
    logger.info(
        "탐지 feature 중복 확인: total=%d, unique=%d, duplicates=%s",
        len(detected_feature_candidates),
        len(set(detected_feature_candidates)),
        detected_duplicate_features,
    )
    base_numeric_columns = list(
        dict.fromkeys(
            [
                *detected["r_columns"],
                *detected["d_columns"],
            ]
        )
    )
    indicator_columns = [
        f"{column}_missing"
        for column in base_numeric_columns
        if f"{column}_missing" in working.columns
    ]
    if add_missing_indicators:
        for column in base_numeric_columns:
            indicator = f"{column}_missing"
            if indicator not in working.columns:
                numeric = pd.to_numeric(working[column], errors="coerce")
                working[indicator] = numeric.isna().astype("int8")
                indicator_columns.append(indicator)
    lot_id_column = schema.get("lot_id_column", "Lot_ID")
    wafer_slot_column = schema.get("wafer_slot_column", "Wafer_Slot")
    excluded_columns = {
        "Y",
        *EXCLUDED_TARGET_COLUMNS,
        id_column,
        lot_id_column,
        wafer_slot_column,
        *KNOWN_IDENTIFIER_COLUMNS,
    }
    numeric_columns = list(
        dict.fromkeys(
            column
            for column in [*base_numeric_columns, *indicator_columns]
            if column not in excluded_columns and column in working.columns
        )
    )
    categorical_columns = list(
        dict.fromkeys(
            column for column in [
                *detected.get("config_columns", []),
                *detected["eq_columns"],
            ]
            if column not in excluded_columns and column in working.columns
        )
    )
    feature_columns = list(
        dict.fromkeys([*numeric_columns, *categorical_columns])
    )
    duplicate_features = _duplicates(feature_columns)
    logger.info(
        "학습 feature 확인: total=%d, unique=%d, duplicates=%s",
        len(feature_columns),
        len(set(feature_columns)),
        duplicate_features,
    )
    if not feature_columns:
        raise ValueError(
            "학습에 사용할 R, D, Config 계열 feature 컬럼이 없습니다."
        )

    numeric_target = pd.to_numeric(
        working[target],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    valid_target_mask = numeric_target.notna()
    if not valid_target_mask.any():
        raise ValueError(f"목표 변수 '{target}'의 값이 모두 결측입니다.")
    valid_target_count = int(valid_target_mask.sum())
    if valid_target_count < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "모델 학습에는 숫자로 변환 가능한 목표값이 "
            f"최소 {MINIMUM_TRAINING_ROWS}개 필요합니다."
        )

    warnings: list[str] = []
    dropped_target_rows = int((~valid_target_mask).sum())
    if dropped_target_rows:
        warnings.append(
            f"목표 변수 '{target}'가 결측인 {dropped_target_rows}개 행을 "
            "학습 데이터에서 제외했습니다."
        )

    features = working.loc[valid_target_mask, feature_columns].copy()
    for column in feature_columns:
        selected = features.loc[:, column]
        if not isinstance(selected, pd.Series):
            raise ValueError(
                f"feature '{column}' 컬럼이 중복되어 1차원으로 선택할 수 없습니다."
            )
        logger.debug(
            "학습 feature 선택 타입: column=%s, type=%s",
            column,
            type(selected).__name__,
        )

    numeric_features = features.loc[:, numeric_columns].apply(
        lambda series: pd.to_numeric(series, errors="coerce")
    )
    numeric_features = numeric_features.replace(
        [np.inf, -np.inf],
        np.nan,
    ).astype(np.float32)
    categorical_features = features.loc[:, categorical_columns].astype(
        "string"
    )
    features = pd.concat(
        [numeric_features, categorical_features],
        axis=1,
    )

    unusable_columns = [
        column
        for column in feature_columns
        if features[column].dropna().nunique() <= 1
    ]
    if unusable_columns:
        warnings.append(
            "전부 결측이거나 상수인 feature를 제외했습니다: "
            + ", ".join(unusable_columns)
        )
        features = features.drop(columns=unusable_columns)
        numeric_columns = [
            column for column in numeric_columns
            if column not in unusable_columns
        ]
        categorical_columns = [
            column for column in categorical_columns
            if column not in unusable_columns
        ]
        feature_columns = [
            column for column in feature_columns
            if column not in unusable_columns
        ]
    if not feature_columns:
        raise ValueError(
            "학습 가능한 feature가 없습니다. R, D, Config 컬럼이 "
            "전부 결측이거나 상수인지 확인해 주세요."
        )

    if lot_id_column in working.columns:
        groups = working.loc[valid_target_mask, lot_id_column].astype("string")
    else:
        groups = _extract_lot_values(
            working.loc[valid_target_mask, id_column]
        )
        logger.info(
            "%s 컬럼이 없어 Legacy 방식으로 %s에서 Lot을 추출했습니다.",
            lot_id_column,
            id_column,
        )
    leakage_columns = sorted(set(feature_columns) & {"Y", *EXCLUDED_TARGET_COLUMNS})
    leakage_check = {
        "passed": not leakage_columns,
        "target_column": "Y",
        "excluded_target_columns": list(EXCLUDED_TARGET_COLUMNS),
        "leakage_columns": leakage_columns,
        "excluded_identifiers": [id_column, lot_id_column, wafer_slot_column],
    }
    if leakage_columns:
        raise ValueError(
            "Target leakage가 탐지되어 학습을 중단했습니다: "
            + ", ".join(leakage_columns)
        )
    prepared = PreparedDataset(
        features=features.reset_index(drop=True),
        target=numeric_target.loc[valid_target_mask].reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        warnings=warnings,
        raw_feature_columns=list(dict.fromkeys([
            *detected_raw["r_columns"],
            *detected_raw["d_columns"],
            *detected_raw["eq_columns"],
            *detected_raw.get("config_columns", []),
        ])),
        config_report=config_report,
        target_leakage_check=leakage_check,
    )
    log_memory_stage(
        logger,
        "dataset_prepare_complete",
        rows=len(prepared.features),
        features=len(prepared.feature_columns),
        numeric_dtype="float32",
        target=target,
    )
    return prepared
