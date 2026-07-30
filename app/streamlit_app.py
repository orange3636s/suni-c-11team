import hashlib
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import load_data_schema, validate_dataframe
from src.preprocessing import (
    load_preprocessing_config,
    preprocess_dataframe,
)


DATA_SCHEMA_PATH = PROJECT_ROOT / "config" / "data_schema.yaml"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing.yaml"

st.title("제조 공정 불량 예측 및 원인 분석 AI")
st.write("프로젝트 초기 환경 구축 완료")

uploaded_file = st.file_uploader("검증할 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 기본 데이터 검증을 시작합니다.")
else:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        dataframe = pd.read_csv(BytesIO(uploaded_bytes))
    except Exception as error:
        st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    else:
        upload_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()
        if st.session_state.get("upload_fingerprint") != upload_fingerprint:
            st.session_state["upload_fingerprint"] = upload_fingerprint
            st.session_state["original_dataframe"] = dataframe.copy(deep=True)
            st.session_state.pop("processed_dataframe", None)
            st.session_state.pop("preprocessing_report", None)

        validation_result = validate_dataframe(
            dataframe,
            schema_path=DATA_SCHEMA_PATH,
        )

        if validation_result["is_valid"]:
            st.success("데이터 기본 검증 완료")
        else:
            st.error("데이터 검증 실패")

        st.subheader("요약 지표")
        metric_columns = st.columns(4)
        metric_columns[0].metric("행 수", validation_result["row_count"])
        metric_columns[1].metric("열 수", validation_result["column_count"])
        metric_columns[2].metric(
            "전체 결측률",
            f"{validation_result['overall_missing_rate']:.2%}",
        )
        metric_columns[3].metric(
            "중복 Lot_Wafer_ID 수",
            validation_result["duplicate_wafer_id_count"],
        )

        st.subheader("컬럼 탐지 결과")
        detection_columns = st.columns(3)
        detection_columns[0].metric(
            "R 컬럼 개수", len(validation_result["r_columns"])
        )
        detection_columns[1].metric(
            "D 컬럼 개수", len(validation_result["d_columns"])
        )
        detection_columns[2].metric(
            "EQ 컬럼 개수", len(validation_result["eq_columns"])
        )
        detected_targets = validation_result["target_columns"]
        st.write(
            "탐지된 Target 컬럼:",
            ", ".join(detected_targets) if detected_targets else "없음",
        )
        st.write(
            "탐지된 R 컬럼명 일부:",
            validation_result["r_columns"][:10] or "없음",
        )
        st.write(
            "탐지된 D 컬럼명 일부:",
            validation_result["d_columns"][:10] or "없음",
        )

        st.subheader("오류 및 경고")
        errors = validation_result["errors"]
        warnings = validation_result["warnings"]
        missing_columns = validation_result["missing_required_columns"]

        st.write("오류")
        if errors:
            for message in errors:
                st.error(message)
        else:
            st.write("없음")

        st.write("경고")
        if warnings:
            for message in warnings:
                st.warning(message)
        else:
            st.write("없음")

        st.write(
            "필수 컬럼 누락:",
            ", ".join(missing_columns) if missing_columns else "없음",
        )

        st.subheader("데이터 미리보기")
        st.dataframe(dataframe.head(10))

        if validation_result["is_valid"]:
            if st.button("데이터 전처리 실행"):
                schema_config = load_data_schema(DATA_SCHEMA_PATH)
                preprocessing_config = load_preprocessing_config(
                    PREPROCESSING_CONFIG_PATH
                )
                processed_dataframe, preprocessing_report = (
                    preprocess_dataframe(
                        st.session_state["original_dataframe"],
                        schema_config=schema_config,
                        preprocessing_config=preprocessing_config,
                    )
                )
                st.session_state["processed_dataframe"] = processed_dataframe
                st.session_state["preprocessing_report"] = preprocessing_report

            if (
                "processed_dataframe" in st.session_state
                and "preprocessing_report" in st.session_state
            ):
                processed_dataframe = st.session_state["processed_dataframe"]
                preprocessing_report = st.session_state[
                    "preprocessing_report"
                ]

                st.subheader("전처리 결과")
                before_columns = st.columns(2)
                before_columns[0].metric(
                    "전처리 전 전체 결측치 수",
                    preprocessing_report["missing_before"],
                )
                before_columns[1].metric(
                    "전처리 전 전체 결측률",
                    f"{preprocessing_report['missing_rate_before']:.2%}",
                )

                after_columns = st.columns(4)
                after_columns[0].metric(
                    "전처리 후 전체 결측치 수",
                    preprocessing_report["missing_after"],
                )
                after_columns[1].metric(
                    "전처리 후 전체 결측률",
                    f"{preprocessing_report['missing_rate_after']:.2%}",
                )
                after_columns[2].metric(
                    "추가된 Missing Indicator 컬럼 수",
                    len(
                        preprocessing_report["added_indicator_columns"]
                    ),
                )
                after_columns[3].metric(
                    "총 클리핑된 값 수",
                    sum(preprocessing_report["clipped_counts"].values()),
                )
                detail_columns = st.columns(2)
                detail_columns[0].metric(
                    "전처리 후 R/D 컬럼 잔여 결측치 수",
                    preprocessing_report[
                        "remaining_numeric_missing_count"
                    ],
                )
                detail_columns[1].metric(
                    "문자열 결측값 표준화 개수",
                    preprocessing_report["standardized_missing_count"],
                )

                st.write(
                    "컬럼별 결측치 대체 개수",
                    preprocessing_report["imputed_counts"],
                )
                st.write(
                    "컬럼별 이상치 클리핑 개수",
                    preprocessing_report["clipped_counts"],
                )

                st.write("전처리 경고")
                if preprocessing_report["warnings"]:
                    for message in preprocessing_report["warnings"]:
                        st.warning(message)
                else:
                    st.write("없음")

                st.write("전처리된 데이터 미리보기")
                st.dataframe(processed_dataframe.head(10))

                st.download_button(
                    "전처리된 CSV 다운로드",
                    data=processed_dataframe.to_csv(index=False).encode(
                        "utf-8-sig"
                    ),
                    file_name="processed_data.csv",
                    mime="text/csv",
                )
