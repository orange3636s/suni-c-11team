import asyncio
import json
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import UploadFile
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import api.routes.data as data_routes
from src.ml.dataset import prepare_dataset
from src.ml.explainability import explain_dataframe
from src.ml.inference import LoadedPredictionModel, predict_dataframe
from src.ml.model_io import save_model_bundle
from src.ml.training import _build_preprocessor
from src.preprocessing import preprocess_dataframe
from src.reporting.export import render_report_html
from src.reporting.report_builder import build_report


@pytest.fixture(scope="module")
def reporting_environment():
    path = Path(__file__).parent / "fixtures" / "training_sample.csv"
    dataframe = pd.read_csv(path)
    processed, _ = preprocess_dataframe(dataframe)
    dataset = prepare_dataset(processed)
    pipeline = Pipeline(
        [
            ("features", _build_preprocessor(dataset)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(dataset.features, dataset.target)
    metadata = {
        "target": "Y",
        "model_name": "Ridge",
        "feature_columns": dataset.feature_columns,
        "metrics": {
            "validation": {"r2": 0.7, "rmse": 0.4, "mae": 0.3},
            "test": {"r2": 0.6, "rmse": 0.5, "mae": 0.4},
        },
    }
    loaded = LoadedPredictionModel(
        model_id="report-ridge",
        model=pipeline,
        metadata=metadata,
    )
    prediction = predict_dataframe(
        dataframe,
        loaded,
        warning_threshold=96,
        danger_threshold=93,
        max_rows=None,
    )
    explanation = explain_dataframe(
        dataframe,
        loaded,
        max_rows=10,
        top_n=10,
        prediction_result=prediction,
    )
    report = build_report(
        "reporting_sample.csv",
        loaded,
        prediction,
        explanation,
    )
    return {
        "dataframe": dataframe,
        "dataset": dataset,
        "loaded": loaded,
        "prediction": prediction,
        "explanation": explanation,
        "report": report,
    }


def test_report_json_combines_prediction_and_shap(
    reporting_environment,
) -> None:
    report = reporting_environment["report"]

    assert json.dumps(report, ensure_ascii=False)
    assert report["executive_summary"]["total_wafers"] == 20
    assert report["top_features"]
    assert report["top_risk_wafers"]
    assert report["explanation_method"] == "shap_linear"


def test_executive_summary_is_calculated_from_prediction(
    reporting_environment,
) -> None:
    report = reporting_environment["report"]
    prediction = reporting_environment["prediction"]
    summary = report["executive_summary"]

    assert summary["average_predicted_yield"] == pytest.approx(
        prediction.average_prediction
    )
    assert summary["danger_count"] == prediction.danger_count
    assert summary["risk_ratio"] == pytest.approx(
        (prediction.danger_count + prediction.warning_count)
        / prediction.total_rows
    )
    assert summary["shap_sampling_used"] is True


def test_risk_wafers_are_sorted_by_risk_then_prediction(
    reporting_environment,
) -> None:
    rows = reporting_environment["report"]["top_risk_wafers"]
    order = {"danger": 0, "warning": 1, "normal": 2, None: 3}
    sort_values = [
        (order[row["risk_level"]], row["predicted_value"])
        for row in rows
    ]

    assert sort_values == sorted(sort_values)


def test_lot_summary_and_parameter_ratios(reporting_environment) -> None:
    report = reporting_environment["report"]

    assert report["lot_summary"]
    assert sum(item["wafer_count"] for item in report["lot_summary"]) == 20
    ratios = [
        item["ratio"]
        for item in report["parameter_type_summary"]
        if item["ratio"] is not None
    ]
    assert sum(ratios) == pytest.approx(1.0)
    assert report["top_steps"][0]["rank"] == 1


def test_recommendations_are_review_priorities_not_causal_claims(
    reporting_environment,
) -> None:
    report = reporting_environment["report"]
    text = " ".join(
        [
            *(
                item["description"]
                for item in report["key_findings"]
            ),
            *(
                item["description"]
                for item in report["recommendations"]
            ),
        ]
    )

    assert report["recommendations"]
    assert "원인으로 확정됨" not in text
    assert "자동 변경" not in text


def test_low_model_quality_creates_warning_and_retraining_advice(
    reporting_environment,
) -> None:
    loaded = reporting_environment["loaded"]
    original = loaded.metadata["metrics"]["test"]["r2"]
    loaded.metadata["metrics"]["test"]["r2"] = -0.1
    try:
        report = build_report(
            "low_quality.csv",
            loaded,
            reporting_environment["prediction"],
            reporting_environment["explanation"],
        )
    finally:
        loaded.metadata["metrics"]["test"]["r2"] = original

    assert any(
        item["title"] == "모델 개선 우선"
        for item in report["recommendations"]
    )


def test_html_report_is_standalone_utf8(reporting_environment) -> None:
    html = render_report_html(reporting_environment["report"])

    assert '<meta charset="utf-8">' in html
    assert "제조 공정 AI 자동 분석 보고서" in html
    assert "<style>" in html
    assert "cdn" not in html.lower()
    assert "실제 공정 인과관계를 확정하지 않습니다" in html


def test_report_api_and_html_download(
    reporting_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = reporting_environment["dataframe"]
    dataset = reporting_environment["dataset"]
    loaded = reporting_environment["loaded"]
    temporary_root = Path(__file__).parent / ".tmp_report_models"
    temporary_root.mkdir(exist_ok=True)
    model_dir = temporary_root / f"run_{uuid4().hex}"
    model_dir.mkdir()
    model_path, _, _ = save_model_bundle(
        loaded.model,
        target="Y",
        model_name="Ridge",
        feature_columns=dataset.feature_columns,
        metrics=loaded.metadata["metrics"],
        random_state=42,
        split_method="test",
        model_dir=model_dir,
    )
    monkeypatch.setattr(data_routes, "MODEL_DIR", model_dir)

    def upload() -> UploadFile:
        return UploadFile(
            file=BytesIO(dataframe.to_csv(index=False).encode("utf-8")),
            filename="report.csv",
        )

    try:
        response = asyncio.run(
            data_routes.generate_report(
                upload(),
                model_id=model_path.stem,
                warning_threshold=96,
                danger_threshold=93,
                max_rows=8,
                top_n=5,
            )
        )
        download = asyncio.run(
            data_routes.download_report(
                upload(),
                model_id=model_path.stem,
                warning_threshold=96,
                danger_threshold=93,
                max_rows=8,
                top_n=5,
            )
        )
        serialized = json.loads(response.model_dump_json())
        disposition = download.headers["content-disposition"]

        assert serialized["report_id"].startswith("report_")
        assert re.fullmatch(
            r'attachment; filename="manufacturing_ai_report_'
            r'\d{8}_\d{6}\.html"',
            disposition,
        )
        assert download.media_type == "text/html; charset=utf-8"
        assert "제조 공정 AI 자동 분석 보고서" in download.body.decode(
            "utf-8"
        )
    finally:
        for generated_file in model_dir.iterdir():
            generated_file.unlink()
        model_dir.rmdir()
        if not any(temporary_root.iterdir()):
            temporary_root.rmdir()
