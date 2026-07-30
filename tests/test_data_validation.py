from pathlib import Path

import pandas as pd
import pytest

from src.data_validation import load_data_schema, validate_dataframe


def test_valid_dataframe_passes_validation() -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["W001", "W002"],
            "Y": [98.1, 97.5],
            "Step1_R1": [1.0, 1.1],
            "Step1_D1": [0, 1],
            "Step1_EQ": ["EQ1", "EQ2"],
        }
    )

    result = validate_dataframe(dataframe)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_empty_dataframe_fails_validation() -> None:
    result = validate_dataframe(pd.DataFrame())

    assert result["is_valid"] is False
    assert any("비어" in error for error in result["errors"])


def test_missing_lot_wafer_id_fails_validation() -> None:
    dataframe = pd.DataFrame({"Y": [98.1]})

    result = validate_dataframe(dataframe)

    assert result["is_valid"] is False
    assert "Lot_Wafer_ID" in result["missing_required_columns"]


def test_missing_y_fails_validation() -> None:
    dataframe = pd.DataFrame({"Lot_Wafer_ID": ["W001"]})

    result = validate_dataframe(dataframe)

    assert result["is_valid"] is False
    assert "Y" in result["missing_required_columns"]


def test_r_d_eq_columns_are_detected_by_suffix() -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["W001"],
            "Y": [98.1],
            "Step1_R1": [1.0],
            "Step2_R2": [2.0],
            "Step1_D1": [0],
            "Step1_EQ": ["EQ1"],
            "NOT_RESPONSE": [3.0],
        }
    )

    result = validate_dataframe(dataframe)

    assert result["r_columns"] == ["Step1_R1", "Step2_R2"]
    assert result["d_columns"] == ["Step1_D1"]
    assert result["eq_columns"] == ["Step1_EQ"]


def test_duplicate_lot_wafer_id_count() -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["W001", "W001", "W002", "W002", "W002"],
            "Y": [98.1, 98.0, 97.5, 97.4, 97.3],
        }
    )

    result = validate_dataframe(dataframe)

    assert result["duplicate_wafer_id_count"] == 3


def test_overall_missing_rate() -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["W001", "W002"],
            "Y": [98.1, None],
            "Step1_R1": [None, 1.1],
        }
    )

    result = validate_dataframe(dataframe)

    assert result["total_missing_count"] == 2
    assert result["overall_missing_rate"] == pytest.approx(2 / 6)


def test_target_columns_are_detected_from_yaml() -> None:
    dataframe = pd.DataFrame(
        {
            "Lot_Wafer_ID": ["W001"],
            "Y": [98.1],
            "Y1": [0.1],
            "Y6": [10],
            "Y11": [999],
        }
    )

    result = validate_dataframe(dataframe)

    assert result["target_columns"] == ["Y", "Y1", "Y6"]


def test_custom_yaml_schema_is_used() -> None:
    schema_path = (
        Path(__file__).parent / "fixtures" / "custom_data_schema.yaml"
    )
    dataframe = pd.DataFrame(
        {
            "Sample_ID": ["S001"],
            "Yield_Value": [98.1],
            "Rate_A": [0.1],
            "Bit_A": [10],
            "Unit1_RESP1": [1.0],
            "Unit1_DEF1": [0],
            "Unit1_MACHINE": ["M1"],
        }
    )

    result = validate_dataframe(dataframe, schema_path=schema_path)

    assert result["is_valid"] is True
    assert result["r_columns"] == ["Unit1_RESP1"]
    assert result["d_columns"] == ["Unit1_DEF1"]
    assert result["eq_columns"] == ["Unit1_MACHINE"]
    assert result["target_columns"] == ["Yield_Value", "Rate_A", "Bit_A"]


def test_default_schema_contains_expected_groups() -> None:
    schema = load_data_schema()

    assert len(schema["fail_rate_columns"]) == 5
    assert len(schema["fail_bit_columns"]) == 5
