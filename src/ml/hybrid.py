from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from src.ml.dataset import RANDOM_STATE
from src.ml.model_io import to_json_safe


PIPELINE_VERSION = "auto_multi_y_hgbr_v1"
TARGETS = ["Y", *[f"Y{index}" for index in range(1, 11)]]
FAIL_RATE_TARGETS = [f"Y{index}" for index in range(1, 6)]
COUNT_TARGETS = [f"Y{index}" for index in range(6, 11)]
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "-"}


def normalized_failure_rates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 100.0)
    totals = clipped.sum(axis=1)
    normalized = clipped.copy()
    overflow = totals > 100.0
    if overflow.any():
        normalized[overflow] *= 100.0 / totals[overflow, None]
    derived = np.clip(100.0 - normalized.sum(axis=1), 0.0, 100.0)
    return normalized, derived, int(overflow.sum())


def _canonical(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_auto_schema(dataframe: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in dataframe.columns]
    canonical = {_canonical(column): column for column in columns}
    target_columns: dict[str, str] = {}
    target_aliases = {"Y": ("y", "finalyield")}
    target_aliases.update({f"Y{i}": (f"y{i}",) for i in range(1, 11)})
    for target, aliases in target_aliases.items():
        for alias in aliases:
            if alias in canonical:
                target_columns[target] = canonical[alias]
                break

    identifier_columns = [
        column
        for column in columns
        if _canonical(column) in {
            "lotid", "waferid", "waferslot", "lotwaferid", "rowid", "sampleid"
        }
        or _canonical(column).endswith("identifier")
    ]
    config_pattern = re.compile(
        r"(?i)^step\d+_(?:config|eq(?:uipment)?.*)$"
    )
    response_pattern = re.compile(r"(?i)^step\d+_r(?:esponse)?\d*$")
    defect_pattern = re.compile(r"(?i)^step\d+_d(?:efect)?\d*$")
    config_columns = [
        column
        for column in columns
        if config_pattern.fullmatch(column)
        or ("equipment" in column.lower() and re.search(r"(?i)step\d+", column))
    ]
    response_columns = [column for column in columns if response_pattern.fullmatch(column)]
    defect_columns = [column for column in columns if defect_pattern.fullmatch(column)]
    feature_columns = list(dict.fromkeys([
        *config_columns,
        *response_columns,
        *defect_columns,
    ]))
    target_source_columns = set(target_columns.values())
    feature_columns = [
        column for column in feature_columns
        if column not in target_source_columns and column not in identifier_columns
    ]
    return {
        "identifier_columns": identifier_columns,
        "config_columns": config_columns,
        "response_columns": response_columns,
        "defect_columns": defect_columns,
        "target_columns": target_columns,
        "feature_columns": feature_columns,
    }


def validate_y_formula(targets: pd.DataFrame, tolerance: float = 1e-6) -> dict[str, Any]:
    derived = 100.0 - targets[FAIL_RATE_TARGETS].sum(axis=1)
    difference = targets["Y"] - derived
    valid = difference.replace([np.inf, -np.inf], np.nan).dropna()
    absolute = valid.abs()
    return {
        "formula": "100 - (Y1 + Y2 + Y3 + Y4 + Y5)",
        "tolerance": tolerance,
        "valid_row_count": int(len(valid)),
        "mean_absolute_error": float(absolute.mean()) if len(absolute) else None,
        "maximum_absolute_error": float(absolute.max()) if len(absolute) else None,
        "exact_match_ratio": float((absolute == 0).mean()) if len(absolute) else None,
        "within_tolerance_ratio": float((absolute <= tolerance).mean()) if len(absolute) else None,
        "formula_consistent": bool(len(absolute) and (absolute <= tolerance).mean() >= 0.999),
    }


def _normalize_category(value: object) -> object:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text.lower() in MISSING_STRINGS else text


class AutoFeaturePreprocessor(TransformerMixin, BaseEstimator):
    """Fold-local numeric cleanup, winsorization, imputation and ordinal encoding."""

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    @staticmethod
    def _frame(x: Any) -> pd.DataFrame:
        return x.copy() if isinstance(x, pd.DataFrame) else pd.DataFrame(x)

    def fit(self, x: Any, y: Any = None) -> "AutoFeaturePreprocessor":
        frame = self._frame(x)
        numeric = pd.DataFrame(index=frame.index)
        for column in self.numeric_columns:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            numeric[column] = pd.to_numeric(source, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

        self.numeric_columns_ = []
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_bounds_: dict[str, tuple[float, float] | None] = {}
        self.all_missing_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        self.near_constant_columns_: list[str] = []
        for column in self.numeric_columns:
            observed = numeric[column].dropna()
            if observed.empty:
                self.all_missing_columns_.append(column)
                continue
            counts = observed.value_counts(dropna=False)
            if len(counts) <= 1:
                self.constant_columns_.append(column)
                continue
            if len(observed) >= 100 and float(counts.iloc[0] / len(observed)) >= 0.999:
                self.near_constant_columns_.append(column)
                continue
            self.numeric_columns_.append(column)
            self.numeric_medians_[column] = float(observed.median())
            if len(observed) >= 20 and observed.nunique() > 2:
                lower, upper = observed.quantile(
                    [self.lower_quantile, self.upper_quantile]
                ).to_numpy(dtype=float)
                self.numeric_bounds_[column] = (
                    (float(lower), float(upper))
                    if np.isfinite(lower) and np.isfinite(upper) and lower < upper
                    else None
                )
            else:
                self.numeric_bounds_[column] = None

        categorical = pd.DataFrame(index=frame.index)
        self.categorical_columns_ = []
        for column in self.categorical_columns:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category)
            if normalized.dropna().empty:
                self.all_missing_columns_.append(column)
                continue
            if normalized.dropna().nunique() <= 1:
                self.constant_columns_.append(column)
                continue
            categorical[column] = normalized.fillna("__MISSING__")
            self.categorical_columns_.append(column)
        self.encoder_ = None
        if self.categorical_columns_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
                dtype=np.float32,
            )
            self.encoder_.fit(categorical[self.categorical_columns_])

        self.feature_names_out_ = [*self.numeric_columns_, *self.categorical_columns_]
        if not self.feature_names_out_:
            raise ValueError("Fold 학습 데이터에 사용할 수 있는 공정 Feature가 없습니다.")
        self.summary_ = {
            "numeric_feature_count": len(self.numeric_columns_),
            "categorical_config_count": len(self.categorical_columns_),
            "removed_all_missing_columns": list(self.all_missing_columns_),
            "removed_constant_columns": list(self.constant_columns_),
            "removed_near_constant_columns": list(self.near_constant_columns_),
            "missing_imputed_columns": int(sum(numeric[c].isna().any() for c in self.numeric_columns_)),
            "winsorized_columns": int(sum(v is not None for v in self.numeric_bounds_.values())),
            "winsorization_quantiles": [self.lower_quantile, self.upper_quantile],
        }
        return self

    def transform(self, x: Any) -> np.ndarray:
        frame = self._frame(x)
        parts: list[np.ndarray] = []
        if self.numeric_columns_:
            output = np.empty((len(frame), len(self.numeric_columns_)), dtype=np.float32)
            for index, column in enumerate(self.numeric_columns_):
                source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
                values = pd.to_numeric(source, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                bounds = self.numeric_bounds_[column]
                if bounds is not None:
                    values = values.clip(lower=bounds[0], upper=bounds[1])
                output[:, index] = values.fillna(self.numeric_medians_[column]).to_numpy(
                    dtype=np.float32
                )
            parts.append(output)
        if self.categorical_columns_ and self.encoder_ is not None:
            categorical = pd.DataFrame(index=frame.index)
            for column in self.categorical_columns_:
                source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
                categorical[column] = source.map(_normalize_category).fillna("__MISSING__")
            parts.append(self.encoder_.transform(categorical[self.categorical_columns_]))
        return np.hstack(parts).astype(np.float32, copy=False)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


def _estimator(row_count: int) -> HistGradientBoostingRegressor:
    if row_count < 200:
        max_iter, min_samples_leaf = 80, 5
    elif row_count < 1000:
        max_iter, min_samples_leaf = 120, 10
    else:
        max_iter, min_samples_leaf = 200, 20
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def _pipeline(schema: dict[str, Any], row_count: int) -> Pipeline:
    return Pipeline([
        ("features", AutoFeaturePreprocessor(
            numeric_columns=[*schema["response_columns"], *schema["defect_columns"]],
            categorical_columns=schema["config_columns"],
        )),
        ("model", _estimator(row_count)),
    ])


def _metrics(actual: Any, predicted: Any) -> dict[str, float | None]:
    truth = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(estimate)
    truth, estimate = truth[valid], estimate[valid]
    if not len(truth):
        return {name: None for name in ("r2", "rmse", "mae", "mse", "pearson", "spearman")}
    mse = float(mean_squared_error(truth, estimate))
    pearson = None
    spearman = None
    if len(truth) > 1 and np.std(truth) > 0 and np.std(estimate) > 0:
        pearson = float(np.corrcoef(truth, estimate)[0, 1])
        spearman_value = spearmanr(truth, estimate).statistic
        spearman = float(spearman_value) if np.isfinite(spearman_value) else None
    return {
        "r2": float(r2_score(truth, estimate)) if len(truth) > 1 else None,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(truth, estimate)),
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
    }


def _splitter(features: pd.DataFrame, groups: pd.Series) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    unique_groups = groups.dropna().astype(str).nunique()
    if unique_groups >= 3:
        splits = list(GroupKFold(n_splits=3).split(features, groups=groups.astype(str)))
        return splits, "group_3_fold"
    if len(features) >= 12:
        return list(KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE).split(features)), "kfold_3"
    folds = 2 if len(features) >= 4 else 0
    if not folds:
        raise ValueError("자동 교차검증에는 유효한 행이 최소 4개 필요합니다.")
    return list(KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE).split(features)), "kfold_2"


def _hybrid_weights(direct: dict[str, Any], derived: dict[str, Any]) -> tuple[float, float, str]:
    direct_r2 = direct.get("r2")
    derived_r2 = derived.get("r2")
    if direct_r2 is not None and direct_r2 < 0 <= (derived_r2 if derived_r2 is not None else -1):
        return 0.0, 1.0, "derived_only_direct_invalid"
    if derived_r2 is not None and derived_r2 < 0 <= (direct_r2 if direct_r2 is not None else -1):
        return 1.0, 0.0, "direct_only_derived_invalid"
    direct_rmse = float(direct.get("rmse") or np.inf)
    derived_rmse = float(derived.get("rmse") or np.inf)
    if (direct_r2 is not None and direct_r2 < 0) and (derived_r2 is not None and derived_r2 < 0):
        return (1.0, 0.0, "direct_only_both_low_confidence") if direct_rmse <= derived_rmse else (0.0, 1.0, "derived_only_both_low_confidence")
    inverse_direct = 1.0 / max(direct_rmse, 1e-12)
    inverse_derived = 1.0 / max(derived_rmse, 1e-12)
    direct_weight = inverse_direct / (inverse_direct + inverse_derived)
    direct_weight = float(np.clip(direct_weight, 0.1, 0.9))
    return direct_weight, 1.0 - direct_weight, "inverse_oof_rmse"


@dataclass
class HybridMultiYBundle:
    feature_columns: list[str]
    direct_model: Any
    target_models: dict[str, Any]
    risk_classifiers: dict[str, Any] | None = None
    meta_model: Any | None = None
    selected_final_output: str = "hybrid"
    warning_threshold: float = 90.0
    critical_threshold: float = 85.0
    direct_weight: float = 0.5
    derived_weight: float = 0.5
    pipeline_version: str = PIPELINE_VERSION

    def predict_components(self, features: pd.DataFrame) -> dict[str, Any]:
        direct = np.clip(np.asarray(self.direct_model.predict(features), dtype=float), 0.0, 100.0)
        target_predictions: dict[str, np.ndarray] = {}
        for target, model in self.target_models.items():
            raw = np.asarray(model.predict(features), dtype=float)
            upper = 100.0 if target in FAIL_RATE_TARGETS else None
            target_predictions[target] = np.clip(raw, 0.0, upper)
        rate_matrix = np.column_stack([target_predictions[target] for target in FAIL_RATE_TARGETS])
        normalized_rates, derived, normalization_count = normalized_failure_rates(rate_matrix)
        for index, target in enumerate(FAIL_RATE_TARGETS):
            target_predictions[target] = normalized_rates[:, index]

        if getattr(self, "pipeline_version", None) == PIPELINE_VERSION:
            hybrid = np.clip(
                getattr(self, "direct_weight", 0.5) * direct
                + getattr(self, "derived_weight", 0.5) * derived,
                0.0,
                100.0,
            )
            selected = hybrid
            critical_probability = 1.0 / (1.0 + np.exp((hybrid - self.critical_threshold) / 3.0))
            warning_probability = 1.0 / (1.0 + np.exp((hybrid - self.warning_threshold) / 3.0))
        else:
            meta_features = np.column_stack([
                direct, derived,
                *[target_predictions[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS],
            ])
            hybrid = np.clip(np.asarray(self.meta_model.predict(meta_features), dtype=float), 0.0, 100.0)
            selected = {"direct": direct, "derived": derived, "hybrid": hybrid}[self.selected_final_output]
            probabilities: dict[str, np.ndarray] = {}
            for name, classifier in (self.risk_classifiers or {}).items():
                values = np.asarray(classifier.predict_proba(meta_features), dtype=float)
                classes = list(classifier.classes_)
                probabilities[name] = values[:, classes.index(1)] if 1 in classes else np.zeros(len(features))
            critical_probability = probabilities.get("critical", np.zeros(len(features)))
            warning_probability = probabilities.get("warning", np.zeros(len(features)))
        return {
            "selected": selected,
            "direct": direct,
            "derived": derived,
            "hybrid": hybrid,
            "targets": target_predictions,
            "critical_probability": critical_probability,
            "warning_probability": warning_probability,
            "normalization_count": normalization_count,
            "model_agreement": {"available": False, "mean_spread": None, "target_spread": {}},
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_components(features)["selected"]


@dataclass
class HybridTrainingResult:
    bundle: HybridMultiYBundle
    metadata: dict[str, Any]
    warnings: list[str]
    oof_predictions: dict[str, list[float]] | None = None


def train_hybrid_multi_y(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float = 0.64,
    validation_ratio: float = 0.16,
    test_ratio: float = 0.20,
    missing_indicator: bool = False,
    oof_folds: int = 3,
    outer_folds: int = 3,
    ensemble_options: Any | None = None,
) -> HybridTrainingResult:
    del train_ratio, validation_ratio, test_ratio, missing_indicator, oof_folds, outer_folds, ensemble_options
    schema = detect_auto_schema(dataframe)
    missing_targets = [target for target in TARGETS if target not in schema["target_columns"]]
    if missing_targets:
        raise ValueError("자동 Multi-Y 학습에 필요한 Target이 없습니다: " + ", ".join(missing_targets))
    if not schema["feature_columns"]:
        raise ValueError("Config/EQ, R, D 공정 Feature를 탐지하지 못했습니다.")

    targets = pd.DataFrame({
        target: pd.to_numeric(dataframe[source], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for target, source in schema["target_columns"].items()
    })
    valid = targets[TARGETS].notna().all(axis=1)
    if int(valid.sum()) < 15:
        raise ValueError("Y와 Y1~Y10이 모두 유효한 행이 최소 15개 필요합니다.")
    working = dataframe.loc[valid].reset_index(drop=True)
    targets = targets.loc[valid, TARGETS].reset_index(drop=True)
    features = working[schema["feature_columns"]].copy()
    lot_column = next((c for c in schema["identifier_columns"] if _canonical(c) == "lotid"), None)
    if lot_column:
        groups = working[lot_column].astype("string").fillna("__MISSING_LOT__")
    else:
        identifier = next((c for c in schema["identifier_columns"] if _canonical(c) == "lotwaferid"), None)
        groups = (
            working[identifier].astype("string").str.extract(r"^([^_]+)", expand=False)
            if identifier else pd.Series(np.arange(len(working)).astype(str))
        ).fillna("__MISSING_LOT__")
    splits, split_method = _splitter(features, groups)

    oof: dict[str, np.ndarray] = {}
    production_models: dict[str, Any] = {}
    target_metrics: dict[str, Any] = {}
    fold_metrics: list[dict[str, Any]] = []
    fold_assignments = [{
        "fold": index + 1,
        "train_groups": sorted(set(groups.iloc[train].astype(str))),
        "holdout_groups": sorted(set(groups.iloc[holdout].astype(str))),
    } for index, (train, holdout) in enumerate(splits)]

    for target in TARGETS:
        prediction = np.zeros(len(features), dtype=float)
        per_fold: list[dict[str, Any]] = []
        for fold_number, (train_index, holdout_index) in enumerate(splits, start=1):
            model = _pipeline(schema, len(train_index))
            with threadpool_limits(limits=1):
                model.fit(features.iloc[train_index], targets[target].iloc[train_index])
                fold_prediction = np.asarray(model.predict(features.iloc[holdout_index]), dtype=float)
            fold_prediction = np.clip(
                fold_prediction,
                0.0,
                100.0 if target in ["Y", *FAIL_RATE_TARGETS] else None,
            )
            prediction[holdout_index] = fold_prediction
            metrics = _metrics(targets[target].iloc[holdout_index], fold_prediction)
            per_fold.append({"fold": fold_number, **metrics})
        oof[target] = prediction
        overall = _metrics(targets[target], prediction)
        target_metrics[target] = {
            "oof": overall,
            "validation": overall,
            "test": overall,
            "folds": per_fold,
        }
        production = _pipeline(schema, len(features))
        with threadpool_limits(limits=1):
            production.fit(features, targets[target])
        production_models[target] = production

    oof_rates, derived_oof, normalization_count = normalized_failure_rates(
        np.column_stack([oof[target] for target in FAIL_RATE_TARGETS])
    )
    for index, target in enumerate(FAIL_RATE_TARGETS):
        oof[target] = oof_rates[:, index]
    direct_oof = np.clip(oof["Y"], 0.0, 100.0)
    direct_metrics = _metrics(targets["Y"], direct_oof)
    derived_metrics = _metrics(targets["Y"], derived_oof)
    direct_weight, derived_weight, weight_method = _hybrid_weights(
        direct_metrics, derived_metrics
    )
    hybrid_oof = np.clip(
        direct_weight * direct_oof + derived_weight * derived_oof, 0.0, 100.0
    )
    hybrid_metrics = _metrics(targets["Y"], hybrid_oof)
    final_y_metrics = {
        name: {"train": metrics, "validation": metrics, "test": metrics, "oof": metrics}
        for name, metrics in {
            "direct": direct_metrics,
            "derived": derived_metrics,
            "hybrid": hybrid_metrics,
        }.items()
    }
    for fold_number, (_, holdout_index) in enumerate(splits, start=1):
        fold_metrics.append({
            "fold": fold_number,
            **_metrics(targets["Y"].iloc[holdout_index], hybrid_oof[holdout_index]),
            "strategy_metrics": {
                "direct": _metrics(targets["Y"].iloc[holdout_index], direct_oof[holdout_index]),
                "derived": _metrics(targets["Y"].iloc[holdout_index], derived_oof[holdout_index]),
                "hybrid": _metrics(targets["Y"].iloc[holdout_index], hybrid_oof[holdout_index]),
            },
        })
    metric_summary = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_metrics if fold[metric] is not None])),
            "std": float(np.std([fold[metric] for fold in fold_metrics if fold[metric] is not None])),
        }
        for metric in ("r2", "rmse", "mae", "mse")
    }

    direct_model = production_models.pop("Y")
    preprocessing_summary = direct_model.named_steps["features"].summary_
    formula_validation = validate_y_formula(targets)
    first_train, first_holdout = splits[0]
    target_configs = {
        target: {
            "selected_type": "single",
            "base_models": ["HistGradientBoostingRegressor"],
            "weights": {"HistGradientBoostingRegressor": 1.0},
            "best_single_metrics": target_metrics[target]["oof"],
            "ensemble_metrics": target_metrics[target]["oof"],
            "improvement_over_single": {"rmse_relative": 0.0},
            "agreement": {"mean_pairwise_correlation": None},
        }
        for target in TARGETS
    }
    bundle = HybridMultiYBundle(
        feature_columns=schema["feature_columns"],
        direct_model=direct_model,
        target_models=production_models,
        direct_weight=direct_weight,
        derived_weight=derived_weight,
    )
    metadata = to_json_safe({
        "schema_version": "semicon_yield_v2",
        "pipeline_version": PIPELINE_VERSION,
        "model_version": PIPELINE_VERSION,
        "model_type": "hybrid_multi_y",
        "bundle_type": "hybrid_multi_y",
        "target": "Y",
        "model_name": "Auto Multi-Y HGBR",
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_columns": schema["feature_columns"],
        "raw_feature_columns": schema["feature_columns"],
        "feature_count": len(schema["feature_columns"]),
        "feature_schema": schema,
        "feature_groups": {
            "config": schema["config_columns"],
            "response": schema["response_columns"],
            "defect": schema["defect_columns"],
        },
        "identifier_columns": schema["identifier_columns"],
        "target_columns": schema["target_columns"],
        "selected_final_output": "hybrid",
        "direct_weight": direct_weight,
        "derived_weight": derived_weight,
        "weight_method": weight_method,
        "final_y_metrics": final_y_metrics,
        "target_metrics": target_metrics,
        "metrics": final_y_metrics["hybrid"],
        "formula_validation": formula_validation,
        "risk_metrics": {},
        "ensemble_enabled": False,
        "ensemble_mode": "disabled",
        "ensemble_method": "automatic_hybrid_weight",
        "target_ensemble_configs": target_configs,
        "direct_y_ensemble": target_configs["Y"],
        "base_model_names": ["HistGradientBoostingRegressor"],
        "missing_strategy": "fold_train_median",
        "outlier_strategy": "fold_train_winsor_0.5_99.5",
        "missing_indicator_used": False,
        "outlier_indicator_used": False,
        "fallback_used": False,
        "preprocessing_strategy": "fold_local_auto_numeric_and_ordinal_config",
        "preprocessing_summary": {
            **preprocessing_summary,
            "missing_strategy": "fold_train_median",
            "outlier_strategy": "fold_train_winsor_0.5_99.5",
            "missing_indicator": False,
            "outlier_indicator": False,
            "missing_indicator_count": 0,
            "outlier_indicator_count": 0,
            "fallback_used": False,
            "categorical_column_count": len(schema["config_columns"]),
            "r_column_count": len(schema["response_columns"]),
            "d_column_count": len(schema["defect_columns"]),
            "config_column_count": len(schema["config_columns"]),
            "training_row_count": len(features),
            "lot_count": int(groups.nunique()),
            "split_method": split_method,
            "pipeline_version": PIPELINE_VERSION,
        },
        "cv_protocol": {
            "name": split_method,
            "group_column": lot_column,
            "outer_folds": len(splits),
            "inner_folds": None,
            "seed": RANDOM_STATE,
            "selection_target": "Hybrid Y OOF",
            "fold_metrics": fold_metrics,
            "metric_summary": metric_summary,
            "outer_group_assignments": fold_assignments,
        },
        "oof_folds": len(splits),
        "oof_group_assignments": fold_assignments,
        "normalization_count": normalization_count,
        "dataset_rows": {
            "train": len(first_train),
            "validation": len(first_holdout),
            "test": 0,
        },
        "training_row_count": len(features),
        "lot_count": int(groups.nunique()),
        "split_method": split_method,
        "group_column": lot_column,
        "target_leakage_check": {
            "passed": not bool(set(schema["feature_columns"]) & set(schema["target_columns"].values())),
            "excluded_targets": list(schema["target_columns"].values()),
            "excluded_identifiers": schema["identifier_columns"],
            "leakage_columns": [],
        },
        "training_config": {
            "model": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": _estimator(len(features)).max_iter,
            "max_leaf_nodes": 31,
            "min_samples_leaf": _estimator(len(features)).min_samples_leaf,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "n_iter_no_change": 20,
            "random_state": RANDOM_STATE,
        },
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "sklearn_version": sklearn.__version__,
        "scikit_learn_version": sklearn.__version__,
        "available_targets": TARGETS,
    })
    return HybridTrainingResult(
        bundle=bundle,
        metadata=metadata,
        warnings=[],
        oof_predictions={
            **{target: values.tolist() for target, values in oof.items()},
            "derived_Y": derived_oof.tolist(),
            "hybrid_Y": hybrid_oof.tolist(),
        },
    )


def save_hybrid_bundle(
    result: HybridTrainingResult,
    model_dir: str | Path,
    model_id: str,
) -> tuple[Path, Path]:
    bundle_dir = Path(model_dir) / model_id
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = bundle_dir / "bundle.joblib"
    metadata_path = bundle_dir / "metadata.json"
    oof_path = bundle_dir / "oof_predictions.json.gz"
    folds_path = bundle_dir / "fold_assignments.json.gz"
    joblib.dump(result.bundle, bundle_path)
    metadata_path.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with gzip.open(oof_path, "wt", encoding="utf-8") as handle:
        json.dump(result.oof_predictions or {}, handle, ensure_ascii=False)
    with gzip.open(folds_path, "wt", encoding="utf-8") as handle:
        json.dump(result.metadata.get("oof_group_assignments", []), handle, ensure_ascii=False)
    return bundle_path, metadata_path
