import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import validate_dataframe


DATA_SCHEMA_PATH = PROJECT_ROOT / "config" / "data_schema.yaml"

st.title("제조 공정 불량 예측 및 원인 분석 AI")
st.write("프로젝트 초기 환경 구축 완료")

uploaded_file = st.file_uploader("검증할 CSV 파일을 업로드하세요.", type=["csv"])

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 기본 데이터 검증을 시작합니다.")
else:
    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    else:
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
