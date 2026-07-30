from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float | None
    rmse: float | None
    mae: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {"r2": self.r2, "rmse": self.rmse, "mae": self.mae}


def _finite_float(value: float) -> float | None:
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def evaluate_regression(
    expected: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> RegressionMetrics:
    expected_values = np.asarray(expected, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    r2 = (
        r2_score(expected_values, predicted_values)
        if len(expected_values) >= 2
        else float("nan")
    )
    rmse = math.sqrt(
        mean_squared_error(expected_values, predicted_values)
    )
    mae = mean_absolute_error(expected_values, predicted_values)
    return RegressionMetrics(
        r2=_finite_float(r2),
        rmse=_finite_float(rmse),
        mae=_finite_float(mae),
    )
