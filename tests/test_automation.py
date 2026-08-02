import asyncio
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import api.routes.data as data_routes
from src.ml.dataset import prepare_dataset
from src.ml.model_io import save_model_bundle
from src.ml.training import _build_preprocessor
from src.preprocessing import preprocess_dataframe


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
        )
    )
    serialized = json.loads(response.model_dump_json())

    assert serialized["success"] is True
    assert serialized["summary"]["total_wafers"] == 20
    assert serialized["top_features"]
    assert "report" not in serialized


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
