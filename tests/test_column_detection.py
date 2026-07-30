from src.column_detection import detect_feature_columns


SCHEMA_CONFIG = {
    "id_column": "Lot_Wafer_ID",
    "yield_column": "Y",
    "fail_rate_columns": [f"Y{index}" for index in range(1, 6)],
    "fail_bit_columns": [f"Y{index}" for index in range(6, 11)],
    "feature_patterns": {
        "response": r"^Step\d+_R\d+$",
        "defect": r"^Step\d+_D.*$",
        "equipment": r"^Step\d+_EQ.*$",
    },
}


def test_step1_r1_is_detected_as_response() -> None:
    result = detect_feature_columns(["Step1_R1"], SCHEMA_CONFIG)

    assert result["r_columns"] == ["Step1_R1"]


def test_step1_r2_is_detected_as_response() -> None:
    result = detect_feature_columns(["Step1_R2"], SCHEMA_CONFIG)

    assert result["r_columns"] == ["Step1_R2"]


def test_step10_r3_is_detected_as_response() -> None:
    result = detect_feature_columns(["Step10_R3"], SCHEMA_CONFIG)

    assert result["r_columns"] == ["Step10_R3"]


def test_step1_eq_is_detected_as_equipment() -> None:
    result = detect_feature_columns(["Step1_EQ"], SCHEMA_CONFIG)

    assert result["eq_columns"] == ["Step1_EQ"]


def test_step1_d1_is_detected_as_defect() -> None:
    result = detect_feature_columns(["Step1_D1"], SCHEMA_CONFIG)

    assert result["d_columns"] == ["Step1_D1"]
