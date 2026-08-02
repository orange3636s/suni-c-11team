from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src.ml.ensemble import (
    EnsembleOptions,
    EnsembleRegressor,
    optimize_non_negative_weights,
    select_target_ensemble,
)


class ColumnRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, column: str):
        self.column = column

    def fit(self, features: pd.DataFrame, target: np.ndarray):
        del features, target
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features[self.column].to_numpy(dtype=float)


def _group_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    random = np.random.default_rng(7)
    rows = 60
    features = pd.DataFrame({
        "a": random.normal(size=rows),
        "b": random.normal(size=rows),
        "c": random.normal(size=rows),
    })
    groups = np.repeat(np.arange(12), 5)
    return features, (features["a"] + features["b"]).to_numpy() / 2, groups


def test_non_negative_weights_are_constrained_to_simplex() -> None:
    predictions = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    weights = optimize_non_negative_weights(predictions, np.array([1.0, 2.0, 3.0]))

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)


def test_two_model_ensemble_is_selected_from_group_oof_only() -> None:
    features, target, groups = _group_data()
    selection = select_target_ensemble(
        features,
        target,
        groups,
        lambda: {
            "ColumnA": ColumnRegressor("a"),
            "ColumnB": ColumnRegressor("b"),
            "ColumnC": ColumnRegressor("c"),
        },
        options=EnsembleOptions(
            size="2", method="simple_average", min_improvement=0.0
        ),
        folds=3,
    )

    assert selection.selected_type == "2-model-ensemble"
    assert set(selection.base_models) == {"ColumnA", "ColumnB"}
    assert sum(selection.weights.values()) == pytest.approx(1.0)
    assert selection.metrics["rmse"] == pytest.approx(0.0)


def test_three_model_bundle_predicts_and_reports_spread() -> None:
    features, _, _ = _group_data()
    model = EnsembleRegressor(
        models={
            "a": ColumnRegressor("a"),
            "b": ColumnRegressor("b"),
            "c": ColumnRegressor("c"),
        },
        weights={"a": 0.2, "b": 0.3, "c": 0.5},
    )

    prediction = model.predict(features)
    expected = features.to_numpy() @ np.array([0.2, 0.3, 0.5])
    assert prediction == pytest.approx(expected)
    assert np.all(model.prediction_spread(features) >= 0)
