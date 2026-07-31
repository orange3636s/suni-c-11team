from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.column_detection import detect_feature_columns
from src.data_validation import load_data_schema
from src.preprocessing import _extract_lot_values


logger = logging.getLogger(__name__)
ALLOWED_TARGETS = tuple(["Y", *[f"Y{index}" for index in range(1, 11)]])
RANDOM_STATE = 42
MINIMUM_TRAINING_ROWS = 10


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return list(dict.fromkeys(value for value in values if counts[value] > 1))


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetSplit:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_groups: pd.Series
    validation_groups: pd.Series
    test_groups: pd.Series
    group_split_used: bool
    split_method: str
    warnings: list[str] = field(default_factory=list)

    @property
    def row_counts(self) -> dict[str, int]:
        return {
            "train_rows": len(self.x_train),
            "validation_rows": len(self.x_validation),
            "test_rows": len(self.x_test),
        }


def prepare_dataset(
    dataframe: pd.DataFrame,
    target: str = "Y",
) -> PreparedDataset:
    if target not in ALLOWED_TARGETS:
        raise ValueError(
            "지원하지 않는 목표 변수입니다. Y부터 Y10까지만 사용할 수 있습니다."
        )
    column_names = list(dataframe.columns)
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
    if target not in dataframe.columns:
        raise ValueError(f"목표 변수 '{target}' 컬럼이 없습니다.")

    schema = load_data_schema()
    id_column = schema["id_column"]
    if id_column not in dataframe.columns:
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
        if f"{column}_missing" in dataframe.columns
    ]
    excluded_columns = {*ALLOWED_TARGETS, id_column}
    numeric_columns = list(
        dict.fromkeys(
            column
            for column in [*base_numeric_columns, *indicator_columns]
            if column not in excluded_columns and column in dataframe.columns
        )
    )
    categorical_columns = list(
        dict.fromkeys(
            column
            for column in detected["eq_columns"]
            if column not in excluded_columns and column in dataframe.columns
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
            "학습에 사용할 R, D, EQ 계열 feature 컬럼이 없습니다."
        )

    numeric_target = pd.to_numeric(
        dataframe[target],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
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

    features = dataframe.loc[valid_target_mask, feature_columns].copy()
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
    )
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
            "학습 가능한 feature가 없습니다. R, D, EQ 컬럼이 "
            "전부 결측이거나 상수인지 확인해 주세요."
        )

    groups = _extract_lot_values(
        dataframe.loc[valid_target_mask, id_column]
    )
    return PreparedDataset(
        features=features.reset_index(drop=True),
        target=numeric_target.loc[valid_target_mask].reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        warnings=warnings,
    )


def _take_split(
    dataset: PreparedDataset,
    train_indices: list[int] | pd.Index,
    validation_indices: list[int] | pd.Index,
    test_indices: list[int] | pd.Index,
    *,
    group_split_used: bool,
    split_method: str,
    warnings: list[str],
) -> DatasetSplit:
    def take_frame(indices: list[int] | pd.Index) -> pd.DataFrame:
        return dataset.features.iloc[indices].reset_index(drop=True)

    def take_series(
        series: pd.Series,
        indices: list[int] | pd.Index,
    ) -> pd.Series:
        return series.iloc[indices].reset_index(drop=True)

    return DatasetSplit(
        x_train=take_frame(train_indices),
        x_validation=take_frame(validation_indices),
        x_test=take_frame(test_indices),
        y_train=take_series(dataset.target, train_indices),
        y_validation=take_series(dataset.target, validation_indices),
        y_test=take_series(dataset.target, test_indices),
        train_groups=take_series(dataset.groups, train_indices),
        validation_groups=take_series(dataset.groups, validation_indices),
        test_groups=take_series(dataset.groups, test_indices),
        group_split_used=group_split_used,
        split_method=split_method,
        warnings=warnings,
    )


def split_dataset(
    dataset: PreparedDataset,
    random_state: int = RANDOM_STATE,
    train_ratio: float = 0.64,
    validation_ratio: float = 0.16,
    test_ratio: float = 0.2,
) -> DatasetSplit:
    row_count = len(dataset.features)
    if row_count < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            "Train/Validation/Test 분리를 위해 유효한 데이터가 "
            f"최소 {MINIMUM_TRAINING_ROWS}행 필요합니다."
        )
    if (
        any(
            ratio <= 0
            for ratio in (train_ratio, validation_ratio, test_ratio)
        )
        or not np.isclose(
            train_ratio + validation_ratio + test_ratio,
            1.0,
        )
    ):
        raise ValueError(
            "Train/Validation/Test 비율은 양수이며 합계가 1이어야 합니다."
        )

    validation_relative_ratio = validation_ratio / (
        train_ratio + validation_ratio
    )

    split_warnings: list[str] = []
    groups_are_usable = (
        dataset.groups.notna().all() and dataset.groups.nunique() >= 5
    )
    if groups_are_usable:
        try:
            test_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_ratio,
                random_state=random_state,
            )
            remaining_indices, test_indices = next(
                test_splitter.split(
                    dataset.features,
                    dataset.target,
                    groups=dataset.groups,
                )
            )
            remaining_groups = dataset.groups.iloc[remaining_indices]
            if remaining_groups.nunique() < 2:
                raise ValueError("Validation 분리에 필요한 Lot 그룹이 부족합니다.")

            validation_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_relative_ratio,
                random_state=random_state,
            )
            train_relative, validation_relative = next(
                validation_splitter.split(
                    dataset.features.iloc[remaining_indices],
                    dataset.target.iloc[remaining_indices],
                    groups=remaining_groups,
                )
            )
            train_indices = remaining_indices[train_relative]
            validation_indices = remaining_indices[validation_relative]
            if min(
                len(train_indices),
                len(validation_indices),
                len(test_indices),
            ) < 2:
                raise ValueError(
                    "각 데이터셋의 평가에 필요한 행 수가 부족합니다."
                )
            return _take_split(
                dataset,
                train_indices,
                validation_indices,
                test_indices,
                group_split_used=True,
                split_method="group",
                warnings=split_warnings,
            )
        except ValueError:
            split_warnings.append(
                "Lot 그룹 수가 부족하여 random split으로 대체했습니다."
            )
    else:
        split_warnings.append(
            "일부 Lot 정보를 추출할 수 없거나 그룹 수가 부족하여 "
            "random split을 사용했습니다."
        )

    all_indices = list(range(row_count))
    remaining_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=random_state,
    )
    train_indices, validation_indices = train_test_split(
        remaining_indices,
        test_size=validation_relative_ratio,
        random_state=random_state,
    )
    return _take_split(
        dataset,
        train_indices,
        validation_indices,
        test_indices,
        group_split_used=False,
        split_method="random",
        warnings=split_warnings,
    )
