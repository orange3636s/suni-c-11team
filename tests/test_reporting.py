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
from src.analytics.analysis_result import (
    build_analysis_result,
    compose_multi_y_predictions,
)
from src.analytics.relationships import analyze_relationships


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


def test_common_analysis_result_reuses_report_numbers(reporting_environment) -> None:
    relationships = analyze_relationships(
        reporting_environment["dataframe"],
        shap_importance=reporting_environment["explanation"].global_importance,
    )
    result = build_analysis_result(
        filename="reporting_sample.csv",
        dataframe=reporting_environment["dataframe"],
        loaded=reporting_environment["loaded"],
        prediction=reporting_environment["prediction"],
        explanation=reporting_environment["explanation"],
        relationships=relationships,
        report=reporting_environment["report"],
        warning_threshold=96,
        danger_threshold=93,
        analysis_unit="wafer_observed_only",
    )
    assert result["risk"]["critical_count"] == reporting_environment["report"]["executive_summary"]["danger_count"]
    assert result["risk_wafers"] == reporting_environment["report"]["top_risk_wafers"]
    assert result["lot_summary"] == reporting_environment["report"]["lot_summary"]
    assert result["dataset"]["fingerprint"]


def test_relationship_api_returns_one_shared_analysis_snapshot(
    reporting_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = reporting_environment["dataframe"]
    loaded = reporting_environment["loaded"]
    monkeypatch.setattr(
        data_routes,
        "load_prediction_model",
        lambda model_id, model_dir: loaded,
    )
    monkeypatch.setattr(
        data_routes,
        "list_prediction_models",
        lambda model_dir: ([], []),
    )
    stored_artifacts: list[dict[str, object]] = []

    def collect_before_lot(dataframe, selected_model, selected_prediction):
        for row in selected_prediction.predictions:
            row.update(
                {
                    "critical_probability": 0.2,
                    "warning_probability": 0.7,
                    "failure_rates": {"Y1": 1.0},
                    "fail_bit_counts": {"Y6": 10.0},
                }
            )
        return compose_multi_y_predictions({}, None), []

    def runtime_call(method: str, **values):
        if method == "start_analysis":
            return "started"
        if method == "complete_analysis":
            stored_artifacts.append(values["artifact"])
            return True
        return None

    monkeypatch.setattr(
        data_routes,
        "_collect_multi_y_predictions",
        collect_before_lot,
    )
    monkeypatch.setattr(data_routes, "safe_runtime_call", runtime_call)
    upload = UploadFile(
        file=BytesIO(dataframe.to_csv(index=False).encode("utf-8")),
        filename="shared-snapshot.csv",
    )

    response = asyncio.run(
        data_routes.analyze_feature_relationships(
            upload,
            model_id=loaded.model_id,
            max_rows=10,
            top_n=10,
            per_wafer_top_n=5,
            correlation_method="pearson",
            analysis_unit="wafer_observed_only",
            warning_threshold=96,
            danger_threshold=93,
        )
    )
    serialized = json.loads(response.model_dump_json())
    analysis = serialized["analysis_result"]
    report = serialized["report_snapshot"]

    assert analysis["analysis_id"] == report["analysis_id"]
    assert report["snapshot_metadata"]["analysis_id"] == analysis["analysis_id"]
    assert analysis["risk"]["critical_count"] == report["executive_summary"]["danger_count"]
    assert analysis["risk_wafers"] == report["top_risk_wafers"]
    assert analysis["lot_summary"] == report["lot_summary"]
    assert analysis["dataset"]["fingerprint"] == report["snapshot_metadata"]["dataset_fingerprint"]
    assert serialized["history_saved"] is True
    assert serialized["lot_analysis"]["lots"]
    assert all(
        lot["average_confidence"] == pytest.approx(0.5)
        and lot["top_failure_target"] == "Y1"
        for lot in serialized["lot_analysis"]["lots"]
    )
    assert stored_artifacts
    assert stored_artifacts[0]["response"]["history_saved"] is True


def test_relationship_api_without_model_returns_statistics_only(
    reporting_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("모델 없는 통계 분석에서 모델 또는 SHAP을 호출했습니다.")

    monkeypatch.setattr(
        data_routes,
        "load_prediction_model",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        data_routes,
        "load_prediction_model_target",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        data_routes,
        "predict_dataframe",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        data_routes,
        "explain_dataframe",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        data_routes,
        "safe_runtime_call",
        unexpected_model_call,
    )
    dataframe = reporting_environment["dataframe"]
    upload = UploadFile(
        file=BytesIO(dataframe.to_csv(index=False).encode("utf-8")),
        filename="statistics-only.csv",
    )

    response = asyncio.run(
        data_routes.analyze_feature_relationships(
            upload,
            model_id=None,
            max_rows=10,
            top_n=10,
            per_wafer_top_n=5,
            correlation_method="pearson",
            analysis_unit="wafer_observed_only",
            warning_threshold=96,
            danger_threshold=93,
            analysis_target=None,
            prediction_id=None,
        )
    )
    serialized = json.loads(response.model_dump_json())
    statistics = serialized["statistics"]

    assert serialized["success"] is True
    assert serialized["target"] == "Y"
    assert serialized["explanation"] is None
    assert serialized["analysis_result"] is None
    assert serialized["report_snapshot"] is None
    assert serialized["lot_analysis"] == {}
    assert {"pearson", "spearman", "anova", "kruskal"}.issubset(
        statistics["methods"]
    )
    assert statistics["numeric"]
    assert {"pearson", "spearman"}.issubset(statistics["numeric"][0])
    assert statistics["categorical"]
    assert {"anova", "kruskal"}.issubset(statistics["categorical"][0])


def test_multi_y_direct_derived_ensemble_and_count_separation() -> None:
    predictions = {
        "Y": [90.0, 80.0],
        "Y1": [1.0, 2.0],
        "Y2": [2.0, 3.0],
        "Y3": [3.0, 4.0],
        "Y4": [4.0, 5.0],
        "Y5": [5.0, 6.0],
        "Y6": [100.0, 200.0],
    }
    result = compose_multi_y_predictions(predictions, 0.25)
    assert result["derived_y"] == pytest.approx([85.0, 80.0])
    assert result["ensemble_y"] == pytest.approx([86.25, 80.0])
    assert result["failure_rates"] == {key: predictions[key] for key in ["Y1", "Y2", "Y3", "Y4", "Y5"]}
    assert result["fail_bit_counts"] == {"Y6": [100.0, 200.0]}


def test_multi_y_does_not_invent_alpha_or_missing_models() -> None:
    result = compose_multi_y_predictions({"Y": [90.0]}, None)
    assert result["derived_y"] is None
    assert result["ensemble_y"] is None
    assert result["ensemble_weight"] is None


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
