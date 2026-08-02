from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config_parser import CONFIG_PARSER_VERSION, parse_config_columns, parse_config_value
from src.ml.hybrid import AutoFeaturePreprocessor, detect_auto_schema


def test_config_parser_preserves_whole_strings_without_derived_columns() -> None:
    original = "  Step1_Model3_EQC_CH2  "
    assert parse_config_value("Step1_Config", original) == "Step1_Model3_EQC_CH2"
    frame = pd.DataFrame({
        "Step1_Config": [original, "NULL", "", "custom-whole-value"],
        "Step1_R1": [1.0, 2.0, 3.0, 4.0],
    })
    parsed, summary = parse_config_columns(frame)
    assert list(parsed.columns) == list(frame.columns)
    assert parsed.loc[0, "Step1_Config"] == "Step1_Model3_EQC_CH2"
    assert pd.isna(parsed.loc[1, "Step1_Config"])
    assert pd.isna(parsed.loc[2, "Step1_Config"])
    assert parsed.loc[3, "Step1_Config"] == "custom-whole-value"
    assert summary["config_strings_decomposed"] is False
    assert summary["config_parser_version"] == CONFIG_PARSER_VERSION
    assert not any(token in parsed.columns for token in ("Model", "Equipment", "Chamber", "EQ"))


def test_train_only_frequency_numeric_imputation_and_asymmetric_clipping() -> None:
    train = pd.DataFrame({
        "Step1_R1": list(range(1, 101)),
        "Step1_D1": [0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 1000.0],
        "Step1_Config": ["A"] * 60 + ["B"] * 39 + [None],
    })
    transformer = AutoFeaturePreprocessor(
        response_columns=["Step1_R1"],
        defect_columns=["Step1_D1"],
        categorical_columns=["Step1_Config"],
    ).fit(train)
    inference = pd.DataFrame({
        "Step1_R1": [-100.0, np.nan, 1000.0],
        "Step1_D1": [0.0, np.nan, 9999.0],
        "Step1_Config": ["A", None, "UNSEEN"],
    })
    transformed = transformer.transform(inference)
    assert transformed.dtype == np.float32
    lower, upper = transformer.r_bounds_["Step1_R1"]
    assert transformed[0, 0] == pytest.approx(lower)
    assert transformed[2, 0] == pytest.approx(upper)
    assert transformed[0, 1] == 0.0
    assert transformed[2, 1] == pytest.approx(transformer.d_upper_bounds_["Step1_D1"])
    assert transformed[2, 2] == 0.0
    assert transformer.frequency_mappings_["Step1_Config"] == {"A": 0.61, "B": 0.39}
    assert transformer.config_modes_["Step1_Config"] == "A"


def test_auto_schema_uses_config_name_and_excludes_all_targets() -> None:
    frame = pd.DataFrame({
        "Lot_ID": ["L1", "L2", "L3"],
        "Step1_R1": [1.0, 2.0, 3.0],
        "Step1_D1": [0.0, 1.0, 0.0],
        "Step1_Config": ["FULL_A", "FULL_B", "FULL_C"],
        **{f"Y{i}": [float(i)] * 3 for i in range(1, 11)},
        "Y": [85.0, 85.0, 85.0],
    })
    schema = detect_auto_schema(frame)
    assert schema["config_columns"] == ["Step1_Config"]
    assert set(schema["feature_columns"]) == {"Step1_R1", "Step1_D1", "Step1_Config"}
    assert set(schema["feature_columns"]).isdisjoint({"Y", *[f"Y{i}" for i in range(1, 11)]})
