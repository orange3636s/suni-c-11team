import warnings

import pandas as pd
import pytest

from src.preprocessing import (
    _standardize_missing_values,
    preprocess_dataframe,
)
from src.data_validation import validate_dataframe


@pytest.fixture
def schema_config() -> dict:
    return {
        "id_column": "Lot_Wafer_ID",
        "yield_column": "Y",
        "fail_rate_columns": [f"Y{index}" for index in range(1, 6)],
        "fail_bit_columns": [f"Y{index}" for index in range(6, 11)],
        "response_suffix": "_R",
        "defect_suffix": "_D",
        "equipment_suffix": "_EQ",
        "feature_patterns": {
            "response": r"^Step\d+_R\d+$",
            "defect": r"^Step\d+_D.*$",
            "equipment": r"^Step\d+_EQ.*$",
        },
    }


@pytest.fixture
def preprocessing_config() -> dict:
    return {
        "missing": {
            "strategy": "lot_mean",
            "fallback": "median",
            "add_indicator": True,
        },
        "outlier": {
            "method": "iqr",
            "lower_multiplier": 1.5,
            "upper_multiplier": 1.5,
        },
        "categorical": {"fill_value": "UNKNOWN"},
    }


def test_original_dataframe_is_not_modified(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02"],
            "Y": [98.0, 97.0],
            "Step1_R1": [1.0, None],
            "Step1_EQ": ["EQ1", None],
        }
    )
    original = dataframe.copy(deep=True)

    preprocess_dataframe(dataframe, schema_config, preprocessing_config)

    pd.testing.assert_frame_equal(dataframe, original)


def test_numeric_missing_value_is_filled_with_lot_mean(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001-W02", "L002W01"],
            "Y": [98.0, 97.0, 96.0],
            "Step1_R1": [10.0, None, 30.0],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert processed.loc[1, "Step1_R1"] == pytest.approx(10.0)
    assert report["imputed_counts"]["Step1_R1"] == 1
    assert report["processing_summary"]["fallback_used"] is False


def test_numeric_missing_value_uses_global_median_fallback(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["INVALID", "L001_W01", "L002_W01"],
            "Y": [98.0, 97.0, 96.0],
            "Step1_D1": [None, 10.0, 20.0],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert processed.loc[0, "Step1_D1"] == pytest.approx(15.0)
    assert report["warnings"] == []
    assert report["processing_summary"]["fallback_used"] is True


def test_missing_indicator_column_is_added(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02"],
            "Y": [98.0, 97.0],
            "Step1_R1": [1.0, None],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert processed["Step1_R1_missing"].tolist() == [0, 1]
    assert report["added_indicator_columns"] == ["Step1_R1_missing"]
    assert report["warnings"] == []


def test_iqr_outlier_is_clipped(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": [
                "L001_W01",
                "L001_W02",
                "L001_W03",
                "L001_W04",
                "L001_W05",
            ],
            "Y": [98.0, 98.0, 98.0, 98.0, 98.0],
            "Step1_R1": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert processed.loc[4, "Step1_R1"] == pytest.approx(7.0)
    assert report["clipped_counts"]["Step1_R1"] == 1


def test_eq_missing_value_is_filled_with_unknown(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02"],
            "Y": [98.0, 97.0],
            "Step1_EQ": ["EQ1", None],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert processed.loc[1, "Step1_EQ"] == "UNKNOWN"
    assert report["imputed_counts"]["Step1_EQ"] == 1
    assert report["categorical_feature_columns"] == ["Step1_EQ"]


def test_target_columns_are_not_modified(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    target_data = {
        "Y": [98.0, None],
        **{
            f"Y{index}": [float(index), None]
            for index in range(1, 11)
        },
    }
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02"],
            **target_data,
            "Step1_R1": [1.0, None],
        }
    )
    original_targets = dataframe[
        ["Y", *[f"Y{index}" for index in range(1, 11)]]
    ].copy(deep=True)

    processed, _ = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    pd.testing.assert_frame_equal(
        processed[original_targets.columns],
        original_targets,
    )


def test_lot_wafer_id_is_not_modified(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", None, "CUSTOM_ID"],
            "Y": [98.0, 97.0, 96.0],
            "Step1_R1": [1.0, None, 3.0],
        }
    )
    original_ids = dataframe["Lot_Wafer_ID"].copy(deep=True)

    processed, _ = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    pd.testing.assert_series_equal(
        processed["Lot_Wafer_ID"],
        original_ids,
    )


def test_preprocessing_report_contains_required_keys(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01"],
            "Y": [98.0],
            "Step1_R1": [1.0],
            "Step1_EQ": ["EQ1"],
        }
    )
    required_keys = {
        "original_shape",
        "processed_shape",
        "numeric_feature_columns",
        "categorical_feature_columns",
        "missing_before",
        "missing_after",
        "missing_rate_before",
        "missing_rate_after",
        "imputed_counts",
        "clipped_counts",
        "added_indicator_columns",
        "warnings",
    }

    _, report = preprocess_dataframe(
        dataframe, schema_config, preprocessing_config
    )

    assert required_keys.issubset(report)


def test_string_none_is_standardized_to_nan() -> None:
    dataframe = pd.DataFrame({"Step1_R1": ["None", "1.0"]})

    standardized_count = _standardize_missing_values(
        dataframe,
        ["Step1_R1"],
    )

    assert pd.isna(dataframe.loc[0, "Step1_R1"])
    assert standardized_count == 1


def test_r_column_has_no_missing_value_after_preprocessing(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02", "L002_W01"],
            "Y": [98.0, 97.0, 96.0],
            "Step1_R1": ["1.0", "None", None],
        }
    )

    processed, report = preprocess_dataframe(
        dataframe,
        schema_config,
        preprocessing_config,
    )

    assert processed["Step1_R1"].isna().sum() == 0
    assert report["remaining_numeric_missing_count"] == 0
    assert report["standardized_missing_count"] == 1


def test_validation_and_preprocessing_use_same_detection(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02"],
            "Y": [98.0, 97.0],
            "Step1_R1": [1.0, None],
            "Step1_R2": [2.0, 3.0],
            "Step1_D1": [0.0, 1.0],
            "Step1_EQ": ["EQ1", None],
        }
    )

    validation_result = validate_dataframe(dataframe)
    _, preprocessing_report = preprocess_dataframe(
        dataframe,
        schema_config,
        preprocessing_config,
    )
    preprocessing_detection = preprocessing_report["detected_columns"]

    assert validation_result["r_columns"] == preprocessing_detection["r_columns"]
    assert validation_result["d_columns"] == preprocessing_detection["d_columns"]
    assert validation_result["eq_columns"] == preprocessing_detection["eq_columns"]


def test_preprocessing_twice_does_not_duplicate_indicators(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["L001_W01", "L001_W02", "L002_W01"],
            "Y": [98.0, 97.0, 96.0],
            "Step1_R1": [1.0, None, 3.0],
            "Step1_D1": [None, 1.0, 2.0],
        }
    )

    processed_once, _ = preprocess_dataframe(
        dataframe,
        schema_config,
        preprocessing_config,
    )
    processed_twice, _ = preprocess_dataframe(
        processed_once,
        schema_config,
        preprocessing_config,
    )
    first_indicators = [
        column
        for column in processed_once.columns
        if column.endswith("_missing")
    ]
    second_indicators = [
        column
        for column in processed_twice.columns
        if column.endswith("_missing")
    ]

    assert processed_once.columns.is_unique
    assert processed_twice.columns.is_unique
    assert second_indicators == first_indicators
    assert len(processed_twice.columns) == len(processed_once.columns)
    assert processed_twice.index.equals(processed_once.index)


def test_indicator_creation_does_not_emit_fragmentation_warning(
    schema_config: dict,
    preprocessing_config: dict,
) -> None:
    row_count = 4
    feature_data = {
        f"Step1_R{index}": [1.0, None, 2.0, 3.0]
        for index in range(1, 121)
    }
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": [
                f"L001_W{index:02d}" for index in range(1, row_count + 1)
            ],
            "Y": [98.0, 97.0, 96.0, 95.0],
            **feature_data,
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        processed, report = preprocess_dataframe(
            dataframe,
            schema_config,
            preprocessing_config,
        )

    performance_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
    assert performance_warnings == []
    assert processed.columns.is_unique
    assert len(report["added_indicator_columns"]) == 120


def test_default_native_flag_only_policy_preserves_values_without_strategy_warnings(
    schema_config: dict,
) -> None:
    dataframe = pd.DataFrame({
        "Lot_Wafer_ID": [f"L001_W{index:02d}" for index in range(1, 7)],
        "Y": [98.0] * 6,
        "Step1_R1": [1.0, 2.0, None, 3.0, 4.0, 100.0],
        "Step12_R2": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
    })

    processed, report = preprocess_dataframe(dataframe, schema_config)

    assert pd.isna(processed.loc[2, "Step1_R1"])
    assert processed.loc[5, "Step1_R1"] == 100.0
    assert report["clipped_counts"]["Step1_R1"] == 0
    assert report["outlier_flagged_counts"]["Step1_R1"] == 1
    assert report["processing_summary"]["missing_strategy"] == "native"
    assert report["processing_summary"]["outlier_strategy"] == "flag_only"
    assert report["processing_summary"]["outlier_indicator"] is True
    assert report["processing_summary"]["fallback_used"] is False
    assert report["processing_summary"]["step_feature_count"] == 2
    assert not any(
        token in warning
        for warning in report["warnings"]
        for token in ("native", "flag_only", "Step1_R1", "Step12_R2", "fallback", "지원하지 않는")
    )
