from __future__ import annotations

from api.routes.runtime import _normalize_analysis_history_detail


def test_history_detail_builds_nullable_safe_response_from_outer_snapshots() -> None:
    analysis_result = {
        "analysis_id": "analysis_legacy",
        "relationships": [{"step": 1, "valid_count": 20}],
        "statistics": None,
        "lot_analysis": {"target": "Y", "lots": [{"lot_id": "L001"}]},
        "data_quality": {"selection_bias_warnings": None},
    }
    report_snapshot = {
        "report_id": "report_legacy",
        "analysis_id": "analysis_legacy",
    }
    detail = {
        "metadata": {"analysis_id": "analysis_legacy", "status": "completed"},
        "artifact": {
            "analysis_result": analysis_result,
            "report_snapshot": report_snapshot,
        },
        "source_prediction_deleted": False,
    }

    normalized = _normalize_analysis_history_detail(detail)
    response = normalized["artifact"]["response"]

    assert response["analysis_result"] == analysis_result
    assert "report_snapshot" not in response
    assert response["lot_analysis"] == analysis_result["lot_analysis"]
    assert response["relationship_paths"] == analysis_result["relationships"]
    assert response["selection_bias_warnings"] == []
    assert response["statistics"] == {
        "methods": [],
        "numeric": [],
        "categorical": [],
        "scatter_data": [],
        "boxplot_data": [],
        "categorical_relationships": [],
    }
    assert set(response["rankings"]) == {"shap", "correlation"}
    assert all(
        rows == []
        for groups in response["rankings"].values()
        for rows in groups.values()
    )


def test_history_detail_filters_null_and_invalid_optional_collections() -> None:
    detail = {
        "metadata": {"analysis_id": "analysis_partial", "status": "partial"},
        "artifact": {
            "response": {
                "selection_bias_warnings": ["keep", None, 3, ""],
                "statistics": {
                    "methods": ["pearson", None, 4],
                    "numeric": [{"feature": "Step1_R1"}, None, "bad"],
                    "categorical": None,
                    "scatter_data": [{"x": 1.0, "y": 2.0}, None, "bad"],
                    "boxplot_data": [{"category": "A", "count": 4}, None],
                    "categorical_relationships": [
                        {"feature": "Step1_Config"}, None, "bad",
                    ],
                },
                "rankings": {
                    "shap": {
                        "all": [{"feature": "Step1_R1"}, None, "bad"],
                        "r": None,
                    },
                    "correlation": None,
                },
                "available_steps": [1, True, "2", None],
                "caveats": None,
            },
            "analysis_result": None,
            "report_snapshot": None,
        },
        "source_prediction_deleted": True,
    }

    normalized = _normalize_analysis_history_detail(detail)
    response = normalized["artifact"]["response"]

    assert response["analysis_result"] is None
    assert "report_snapshot" not in response
    assert response["lot_analysis"] == {}
    assert response["selection_bias_warnings"] == ["keep"]
    assert response["statistics"] == {
        "methods": ["pearson"],
        "numeric": [{"feature": "Step1_R1"}],
        "categorical": [],
        "scatter_data": [{"x": 1.0, "y": 2.0}],
        "boxplot_data": [{"category": "A", "count": 4}],
        "categorical_relationships": [{"feature": "Step1_Config"}],
    }
    assert response["rankings"]["shap"]["all"] == [
        {"feature": "Step1_R1"},
    ]
    assert response["rankings"]["shap"]["r"] == []
    assert all(
        rows == [] for rows in response["rankings"]["correlation"].values()
    )
    assert response["available_steps"] == [1]
    assert response["caveats"] == []


def test_history_detail_restores_outer_aliases_from_response_only_snapshot() -> None:
    analysis_result = {
        "analysis_id": "analysis_response_only",
        "relationships": [{"step": 3, "valid_count": 12}],
        "statistics": {
            "methods": ["pearson", "anova"],
            "numeric": [
                {
                    "feature": "Step1_R1",
                    "scatter_data": [{"x": 1.0, "y": 90.0}],
                },
                {
                    "feature": "Step2_D1",
                    "scatter_data": [{"x": 2.0, "y": 88.0}],
                },
            ],
            "categorical": [
                {
                    "feature": "Step1_Config",
                    "category_summary": [{"category": "A", "count": 7}],
                },
                {
                    "feature": "Step2_Config",
                    "boxplot_data": [{"category": "B", "count": 5}],
                },
            ],
        },
        "lot_analysis": {
            "target": "Y1",
            "lots": [{"lot_id": "L003", "risk_level": "warning"}],
        },
    }
    report_snapshot = {
        "report_id": "report_response_only",
        "analysis_id": "analysis_response_only",
    }
    detail = {
        "metadata": {
            "analysis_id": "analysis_response_only",
            "status": "partial",
        },
        "artifact": {
            "response": {
                "analysis_result": analysis_result,
                "report_snapshot": report_snapshot,
            },
        },
        "source_prediction_deleted": False,
    }

    normalized = _normalize_analysis_history_detail(detail)
    artifact = normalized["artifact"]

    assert "explanation" not in artifact["response"]
    assert artifact["response"]["analysis_result"] == analysis_result
    assert "report_snapshot" not in artifact["response"]
    assert artifact["response"]["lot_analysis"] == analysis_result["lot_analysis"]
    assert artifact["analysis_result"] == analysis_result
    assert "report_snapshot" not in artifact
    assert artifact["lot_analysis"] == analysis_result["lot_analysis"]
    assert artifact["response"]["statistics"]["scatter_data"] == [
        {"x": 1.0, "y": 90.0},
        {"x": 2.0, "y": 88.0},
    ]
    assert artifact["response"]["statistics"]["boxplot_data"] == [
        {"category": "A", "count": 7},
        {"category": "B", "count": 5},
    ]
    assert artifact["response"]["statistics"][
        "categorical_relationships"
    ] == analysis_result["statistics"]["categorical"]
