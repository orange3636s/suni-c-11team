import asyncio
import json
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import api.routes.data as data_routes
from src.automation.analyzer import build_automation_response
from src.ml.dataset import prepare_dataset
from src.ml.model_io import save_model_bundle
from src.ml.training import _build_preprocessor
from src.preprocessing import preprocess_dataframe


def _base_report() -> dict:
    return {
        "report_id": "report_20260731_000500_ab12cd34",
        "filename": "process.csv",
        "model": {
            "model_id": "model-1",
            "target": "Y",
            "model_name": "Ridge",
            "test_metrics": {"r2": 0.7, "rmse": 1.2, "mae": 0.9},
        },
        "executive_summary": {
            "total_wafers": 10,
            "average_predicted_yield": 94.5,
            "normal_count": 7,
            "warning_count": 2,
            "danger_count": 1,
            "risk_ratio": 0.3,
        },
        "key_findings": [],
        "top_risk_wafers": [
            {
                "identifier": "LOT01_WF01",
                "predicted_value": 88.2,
                "risk_level": "danger",
                "actual_value": None,
                "absolute_error": None,
                "top_harmful_features": ["Step1_D1"],
                "top_step": "Step1",
                "top_parameter_type": "D",
            }
        ],
        "top_features": [
            {
                "rank": 1,
                "feature": "Step1_D1",
                "step": "Step1",
                "parameter_type": "D",
                "parameter_name": "1",
                "mean_abs_shap": 1.0,
                "mean_harmful_contribution": 0.8,
                "direction": "yield_down",
            }
        ],
        "top_steps": [],
        "parameter_type_summary": [],
        "model_quality_warnings": [],
        "warnings": [],
    }


def test_alert_danger_branch_and_automation_message() -> None:
    response = build_automation_response(_base_report())

    assert response["alert"]["required"] is True
    assert response["alert"]["severity"] == "danger"
    assert "위험 1개, 주의 2개" in response["automation_message"]["summary"]
    assert "Step1_D1" in response["automation_message"]["top_cause"]
    assert re.fullmatch(
        r"analysis_\d{8}_\d{6}_[0-9a-f]{6}",
        response["analysis_id"],
    )


def test_alert_warning_branch() -> None:
    report = _base_report()
    report["executive_summary"].update(
        danger_count=0,
        warning_count=3,
        normal_count=7,
        risk_ratio=0.3,
    )

    response = build_automation_response(report)

    assert response["alert"]["required"] is True
    assert response["alert"]["severity"] == "warning"


def test_alert_normal_is_independent_of_model_quality() -> None:
    report = _base_report()
    report["executive_summary"].update(
        danger_count=0,
        warning_count=0,
        normal_count=10,
        risk_ratio=0.0,
    )
    report["model_quality_warnings"] = ["Test R²가 낮습니다."]

    response = build_automation_response(report, include_report=False)

    assert response["alert"]["required"] is False
    assert response["alert"]["severity"] == "normal"
    assert response["model_quality_warnings"]
    assert response["report"] == {
        "included": False,
        "report_id": None,
        "download_endpoint": None,
    }
    assert json.dumps(response, ensure_ascii=False)


@pytest.fixture(scope="module")
def analyze_api_environment():
    fixture = Path(__file__).parent / "fixtures" / "training_sample.csv"
    dataframe = pd.read_csv(fixture)
    processed, _ = preprocess_dataframe(dataframe)
    dataset = prepare_dataset(processed)
    pipeline = Pipeline(
        [
            ("features", _build_preprocessor(dataset)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(dataset.features, dataset.target)
    temporary_root = Path(__file__).parent / ".tmp_analyze_models"
    temporary_root.mkdir(exist_ok=True)
    model_dir = temporary_root / f"run_{uuid4().hex}"
    model_dir.mkdir()
    model_path, _, _ = save_model_bundle(
        pipeline,
        target="Y",
        model_name="Ridge",
        feature_columns=dataset.feature_columns,
        metrics={
            "validation": {"r2": 0.7, "rmse": 0.4, "mae": 0.3},
            "test": {"r2": 0.6, "rmse": 0.5, "mae": 0.4},
        },
        random_state=42,
        split_method="test",
        model_dir=model_dir,
    )
    yield {
        "dataframe": dataframe,
        "model_dir": model_dir,
        "model_id": model_path.stem,
    }
    for generated_file in model_dir.iterdir():
        generated_file.unlink()
    model_dir.rmdir()
    if not any(temporary_root.iterdir()):
        temporary_root.rmdir()


def _upload(dataframe: pd.DataFrame) -> UploadFile:
    return UploadFile(
        file=BytesIO(dataframe.to_csv(index=False).encode("utf-8")),
        filename="analysis.csv",
    )


def test_analyze_api_success(
    analyze_api_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        analyze_api_environment["model_dir"],
    )

    response = asyncio.run(
        data_routes.analyze_csv(
            _upload(analyze_api_environment["dataframe"]),
            model_id=analyze_api_environment["model_id"],
            warning_threshold=96,
            danger_threshold=93,
            max_rows=8,
            top_n=5,
            per_wafer_top_n=3,
            include_report=True,
        )
    )
    serialized = json.loads(response.model_dump_json())

    assert serialized["success"] is True
    assert serialized["summary"]["total_wafers"] == 20
    assert serialized["top_features"]
    assert serialized["report"]["download_endpoint"] == (
        "/api/report/download"
    )


@pytest.mark.parametrize(
    ("model_id", "warning", "danger", "expected_detail"),
    [
        ("missing-model", 95, 90, "존재하지"),
        ("valid", 90, 95, "주의 기준값"),
    ],
)
def test_analyze_api_rejects_invalid_request(
    analyze_api_environment,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    warning: float,
    danger: float,
    expected_detail: str,
) -> None:
    monkeypatch.setattr(
        data_routes,
        "MODEL_DIR",
        analyze_api_environment["model_dir"],
    )
    selected_model = (
        analyze_api_environment["model_id"]
        if model_id == "valid"
        else model_id
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            data_routes.analyze_csv(
                _upload(analyze_api_environment["dataframe"]),
                model_id=selected_model,
                warning_threshold=warning,
                danger_threshold=danger,
                max_rows=8,
                top_n=5,
                per_wafer_top_n=3,
                include_report=True,
            )
        )

    assert error.value.status_code == 400
    assert expected_detail in str(error.value.detail)


def test_n8n_workflow_is_valid_and_has_required_nodes() -> None:
    path = (
        Path(__file__).parents[1]
        / "workflows"
        / "n8n_manufacturing_ai_workflow.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["name"]: node for node in workflow["nodes"]}
    required = {
        "Webhook",
        "Validate Input",
        "Prepare Request",
        "HTTP Request - FastAPI Analyze",
        "Analysis Success?",
        "Alert Required?",
        "Danger or Warning?",
        "Slack Alert - Danger",
        "Slack Alert - Warning",
        "No Alert",
        "Respond to Webhook",
        "Error Response",
    }

    assert workflow["name"] == "제조 공정 예측 및 불량분석 AI 자동화"
    assert required.issubset(nodes)
    assert nodes["Webhook"]["parameters"]["path"] == (
        "manufacturing-ai-analysis"
    )
    assert "$env.FASTAPI_BASE_URL" in nodes[
        "HTTP Request - FastAPI Analyze"
    ]["parameters"]["url"]
    assert nodes["Slack Alert - Danger"]["continueOnFail"] is True
    assert "credentials" not in nodes["Slack Alert - Danger"]
