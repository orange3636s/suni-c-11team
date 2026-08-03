"""GroupKFold(5) preprocessing comparison on train.CSV: this file's output
IS a deliverable, not a scratch script -- it documents why the production
pipeline (src/ml/pipeline.py) preserves NaN and engineers screening-derived
features instead of imputing/clipping/frequency-encoding everything.

  A: current-style preprocessing -- train-median impute R/D, R clipped at
     [1,99] percentile, D clipped at upper 99.9 percentile, Config
     frequency-encoded (reuses src.ml.hybrid.AutoFeaturePreprocessor)
  B: same full R+D+Config feature set, but NaN preserved (R unclipped, D
     still upper-clipped), Config as a native categorical column
  C: only the factor(s) src.analysis.screening selects per target (BH-FDR
     -> 80% cumulative cut), with a missingness flag and, for u_shape
     factors, a |value - optimal_center| deviation column

Factor selection for C is re-run inside every fold on that fold's training
split only (never on the held-out fold), matching src/ml/pipeline.py's
leakage rules. Run: python scripts/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import select_pareto_factors
from src.ml.hybrid import AutoFeaturePreprocessor
from src.ml.pipeline import FAIL_RATE_TARGETS, HGBR_PARAMS, build_features

TRAIN_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.CSV"
N_FOLDS = 5
FINAL_YIELD_COLUMN = "Y"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _hgbr() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**HGBR_PARAMS)


def method_a_features(schema, train_fold: pd.DataFrame, test_fold: pd.DataFrame):
    preprocessor = AutoFeaturePreprocessor(
        response_columns=list(schema.r_cols),
        defect_columns=list(schema.d_cols),
        categorical_columns=list(schema.config_cols),
    )
    preprocessor.fit(train_fold)
    return preprocessor.transform(train_fold), preprocessor.transform(test_fold)


def _d_upper_bound(observed: pd.Series, quantile: float = 0.999) -> float | None:
    if len(observed) < 10 or observed.nunique() <= 2:
        return None
    upper = float(observed.quantile(quantile))
    maximum = float(observed.max())
    if not np.isfinite(upper) or upper >= maximum or np.isclose(upper, maximum):
        return None
    return upper


def method_b_features(schema, train_fold: pd.DataFrame, test_fold: pd.DataFrame):
    """Full R+D feature set, NaN preserved. Config is excluded rather than
    encoded -- it carries no signal (screening never selects one; see
    method C), and a 36-category-per-column frequency/category encoding
    only adds noise splits for a model this size. This is the "제외" option
    the preprocessing spec explicitly allows for Config in this variant.
    """
    train_columns: dict[str, pd.Series] = {}
    test_columns: dict[str, pd.Series] = {}

    for column in schema.r_cols:
        train_columns[column] = _numeric(train_fold[column]).astype("float32")
        test_columns[column] = _numeric(test_fold[column]).astype("float32")

    for column in schema.d_cols:
        train_values = _numeric(train_fold[column])
        test_values = _numeric(test_fold[column])
        upper = _d_upper_bound(train_values.dropna())
        if upper is not None:
            train_values = train_values.clip(upper=upper)
            test_values = test_values.clip(upper=upper)
        train_columns[column] = train_values.astype("float32")
        test_columns[column] = test_values.astype("float32")

    x_train = pd.DataFrame(train_columns, index=train_fold.index)
    x_test = pd.DataFrame(test_columns, index=test_fold.index)
    return x_train, x_test


def method_c_predict(schema, train_fold: pd.DataFrame, test_fold: pd.DataFrame, target: str):
    selection = select_pareto_factors(train_fold, schema, target)
    if selection.no_significant_factor or not selection.factors:
        baseline = float(_numeric(train_fold[target]).mean())
        return np.full(len(test_fold), baseline, dtype=np.float32), selection
    x_train = build_features(train_fold, selection.factors)
    x_test = build_features(test_fold, selection.factors).reindex(columns=x_train.columns)
    model = _hgbr()
    model.fit(x_train, _numeric(train_fold[target]))
    return np.asarray(model.predict(x_test), dtype=np.float32), selection


def run_benchmark(df: pd.DataFrame) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, list[list[str]]]]:
    schema = parse_schema(df)
    groups = df["Lot_ID"]
    splitter = GroupKFold(n_splits=N_FOLDS)

    oof = {
        "A": {target: np.full(len(df), np.nan, dtype=np.float64) for target in FAIL_RATE_TARGETS},
        "B": {target: np.full(len(df), np.nan, dtype=np.float64) for target in FAIL_RATE_TARGETS},
        "C": {target: np.full(len(df), np.nan, dtype=np.float64) for target in FAIL_RATE_TARGETS},
    }
    fold_factor_choice: dict[str, list[list[str]]] = {target: [] for target in FAIL_RATE_TARGETS}

    for train_idx, test_idx in splitter.split(df, groups=groups):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        test_fold = df.iloc[test_idx].reset_index(drop=True)
        test_original_idx = df.index[test_idx]

        x_train_a, x_test_a = method_a_features(schema, train_fold, test_fold)
        x_train_b, x_test_b = method_b_features(schema, train_fold, test_fold)

        for target in FAIL_RATE_TARGETS:
            y_train = _numeric(train_fold[target])

            model_a = _hgbr()
            model_a.fit(x_train_a, y_train)
            oof["A"][target][test_original_idx] = model_a.predict(x_test_a)

            model_b = _hgbr()
            model_b.fit(x_train_b, y_train)
            oof["B"][target][test_original_idx] = model_b.predict(x_test_b)

            prediction_c, selection = method_c_predict(schema, train_fold, test_fold, target)
            oof["C"][target][test_original_idx] = prediction_c
            chosen = (
                [factor.feature for factor in selection.factors]
                if not selection.no_significant_factor
                else []
            )
            fold_factor_choice[target].append(chosen)

    def compute_metrics(method_oof: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        for target in FAIL_RATE_TARGETS:
            actual = _numeric(df[target]).to_numpy()
            predicted = method_oof[target]
            metrics[target] = {
                "r2": float(r2_score(actual, predicted)),
                "mae": float(mean_absolute_error(actual, predicted)),
            }
        clipped_sum = sum(np.clip(method_oof[target], 0.0, None) for target in FAIL_RATE_TARGETS)
        derived = np.clip(100.0 - clipped_sum, 0.0, 100.0)
        actual_y = _numeric(df[FINAL_YIELD_COLUMN]).to_numpy()
        metrics[FINAL_YIELD_COLUMN] = {
            "r2": float(r2_score(actual_y, derived)),
            "mae": float(mean_absolute_error(actual_y, derived)),
        }
        return metrics

    results = {
        "A_median_impute_clip_freq": compute_metrics(oof["A"]),
        "B_full_factors_nan_preserved": compute_metrics(oof["B"]),
        "C_pareto_dev_mask": compute_metrics(oof["C"]),
    }
    return results, fold_factor_choice


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    results, fold_factor_choice = run_benchmark(df)

    columns = [*FAIL_RATE_TARGETS, FINAL_YIELD_COLUMN]
    header = f"{'method':32s}" + "".join(f"{c + ' R2':>10s}" for c in columns)
    print(header)
    for name, metrics in results.items():
        row = "".join(f"{metrics[c]['r2']:10.3f}" for c in columns)
        print(f"{name:32s}{row}")
    print()
    print(f"{'method':32s}{'Y R2':>10s}{'Y MAE':>10s}")
    for name, metrics in results.items():
        print(f"{name:32s}{metrics[FINAL_YIELD_COLUMN]['r2']:10.3f}{metrics[FINAL_YIELD_COLUMN]['mae']:10.3f}")

    print()
    print("Per-fold selected factor (method C) -- should be identical across all 5 folds per target:")
    for target, choices in fold_factor_choice.items():
        print(f"  {target}: {choices}")


if __name__ == "__main__":
    main()
