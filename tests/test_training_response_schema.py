from api.schemas.data import TrainResponse


def _base_response() -> dict:
    metrics = {"r2": 0.8, "rmse": 1.2, "mae": 0.9, "mse": 1.44}
    return {
        "target": "Y",
        "best_model": "Ridge",
        "split": {
            "train_rows": 60,
            "validation_rows": 20,
            "test_rows": 20,
            "group_split_used": True,
            "split_method": "group_holdout",
        },
        "metrics": {name: metrics for name in ("train", "validation", "test")},
        "model_comparison": [],
        "feature_count": 10,
        "artifacts": {"model_file": "model.joblib", "metadata_file": "model.json"},
    }


def test_legacy_holdout_evaluation_summary_is_preserved() -> None:
    payload = _base_response()
    payload["evaluation_summary"] = {"generalization_gap": 0.1, "dummy_test_r2": 0.0}
    response = TrainResponse(**payload).model_dump(mode="json")

    assert response["evaluation_summary"]["generalization_gap"] == 0.1
    assert response["evaluation_summary"]["metric_summary"] is None
    assert response["cv"] is None


def test_nested_cv_metric_summary_is_aliased_without_recalculation() -> None:
    payload = _base_response()
    summary = {
        "r2": {"mean": 0.81, "std": 0.03},
        "rmse": {"mean": 1.1, "std": 0.2},
        "mae": {"mean": 0.8, "std": 0.1},
    }
    payload["evaluation_summary"] = {
        "name": "nested_group_kfold",
        "outer_folds": 5,
        "inner_folds": 3,
        "metric_summary": summary,
    }
    response = TrainResponse(**payload).model_dump(mode="json")

    assert response["evaluation_summary"]["metric_summary"] == summary | {"mse": None}
    assert response["cv"]["aggregate_metrics"] == summary | {"mse": None}
    assert response["cv_summary"]["metric_summary"] == summary | {"mse": None}
    assert response["cv"]["name"] == "nested_group_kfold"
    assert response["cv"]["outer_folds"] == 5


def test_new_cv_aggregate_metrics_populates_legacy_alias() -> None:
    payload = _base_response()
    summary = {"r2": {"mean": 0.76, "std": 0.04}}
    payload["evaluation_summary"] = None
    payload["cv"] = {"name": "nested_group_kfold", "aggregate_metrics": summary}
    payload["ensemble"] = {"enabled": True, "selected": True}
    response = TrainResponse(**payload).model_dump(mode="json")

    expected = summary | {"rmse": None, "mae": None, "mse": None}
    assert response["evaluation_summary"]["metric_summary"] == expected
    assert response["evaluation_summary"]["name"] == "nested_group_kfold"
    assert response["cv"]["aggregate_metrics"] == expected
    assert response["ensemble"]["selected"] is True
