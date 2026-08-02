from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.ml.dataset import RANDOM_STATE, prepare_dataset
from src.ml.evaluation import evaluate_regression
from src.ml.ensemble import (
    EnsembleOptions,
    EnsembleRegressor,
    selection_metadata,
    select_target_ensemble,
)
from src.ml.model_io import to_json_safe
from src.ml.training import (
    _build_preprocessor,
    missing_strategy_for_model,
    outlier_strategy_for_model,
)


TARGETS = ["Y", *[f"Y{index}" for index in range(1, 11)]]
FAIL_RATE_TARGETS = [f"Y{index}" for index in range(1, 6)]
COUNT_TARGETS = [f"Y{index}" for index in range(6, 11)]


def _seeded_group_splits(
    features: pd.DataFrame,
    groups: pd.Series,
    *,
    folds: int,
    seed: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build reproducible, group-disjoint folds without depending on row order."""
    unique_groups = np.asarray(sorted(groups.astype(str).unique()))
    if len(unique_groups) < folds:
        raise ValueError(f"GroupKFold {folds}개를 만들려면 최소 {folds}개 Lot 그룹이 필요합니다.")
    shuffled = unique_groups.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    group_rank = {name: index for index, name in enumerate(shuffled)}
    seeded_groups = groups.astype(str).map(group_rank)
    return list(GroupKFold(n_splits=folds).split(features, groups=seeded_groups))


def normalized_failure_rates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    totals = clipped.sum(axis=1)
    normalized = clipped.copy()
    overflow = totals > 100.0
    if overflow.any():
        normalized[overflow] = normalized[overflow] * (100.0 / totals[overflow, None])
    derived = np.clip(100.0 - normalized.sum(axis=1), 0.0, 100.0)
    return normalized, derived, int(overflow.sum())


def _metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    result = evaluate_regression(actual, predicted)
    return {**result.as_dict(), "mse": result.mse}


def _candidate_pipelines(dataset: Any, *, count: bool) -> dict[str, Pipeline]:
    estimators: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=80, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    if count:
        estimators["HistGradientBoostingPoisson"] = HistGradientBoostingRegressor(
            loss="poisson", random_state=RANDOM_STATE
        )
    try:
        from xgboost import XGBRegressor

        estimators["XGBoostRegressor"] = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            objective="count:poisson" if count else "reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        estimators["CatBoostRegressor"] = CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.05,
            loss_function="Poisson" if count else "RMSE",
            random_seed=RANDOM_STATE,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:
        pass
    return {
        name: Pipeline([
            (
                "features",
                _build_preprocessor(
                    dataset,
                    missing_strategy=missing_strategy_for_model(name),
                    outlier_strategy=outlier_strategy_for_model(name),
                ),
            ),
            ("model", estimator),
        ])
        for name, estimator in estimators.items()
    }


def _best_model(
    dataset: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    count: bool,
) -> tuple[str, Pipeline, dict[str, float | None], list[str]]:
    candidates: list[tuple[float, float, str, Pipeline, dict[str, float | None]]] = []
    warnings: list[str] = []
    for name, pipeline in _candidate_pipelines(dataset, count=count).items():
        try:
            pipeline.fit(x_train, y_train)
            prediction = np.asarray(pipeline.predict(x_validation), dtype=float)
            if count:
                prediction = np.clip(prediction, 0.0, None)
            metrics = _metrics(y_validation, prediction)
            candidates.append((metrics["rmse"] or float("inf"), -(metrics["r2"] or float("-inf")), name, pipeline, metrics))
        except Exception as exc:
            warnings.append(f"{name} 후보 제외: {type(exc).__name__}")
    if not candidates:
        raise ValueError("학습 가능한 Hybrid Multi-Y 후보 모델이 없습니다.")
    _, _, name, pipeline, metrics = min(candidates, key=lambda item: (item[0], item[1]))
    return name, pipeline, metrics, warnings


@dataclass
class HybridMultiYBundle:
    feature_columns: list[str]
    direct_model: Any
    target_models: dict[str, Any]
    risk_classifiers: dict[str, Any]
    meta_model: Any
    selected_final_output: str
    warning_threshold: float = 90.0
    critical_threshold: float = 85.0

    def predict_components(self, features: pd.DataFrame) -> dict[str, Any]:
        direct = np.clip(np.asarray(self.direct_model.predict(features), dtype=float), 0.0, 100.0)
        target_predictions: dict[str, np.ndarray] = {}
        for target, model in self.target_models.items():
            raw = np.asarray(model.predict(features), dtype=float)
            target_predictions[target] = np.clip(raw, 0.0, None)
        rate_matrix = np.column_stack([target_predictions[target] for target in FAIL_RATE_TARGETS])
        normalized_rates, derived, normalization_count = normalized_failure_rates(rate_matrix)
        for index, target in enumerate(FAIL_RATE_TARGETS):
            target_predictions[target] = normalized_rates[:, index]
        meta_features = np.column_stack(
            [direct, derived, *[target_predictions[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS]]
        )
        hybrid = np.clip(np.asarray(self.meta_model.predict(meta_features), dtype=float), 0.0, 100.0)
        outputs = {"direct": direct, "derived": derived, "hybrid": hybrid}
        selected = outputs[self.selected_final_output]
        risk_probabilities: dict[str, np.ndarray] = {}
        for name, classifier in self.risk_classifiers.items():
            probabilities = np.asarray(classifier.predict_proba(meta_features), dtype=float)
            classes = list(classifier.classes_)
            risk_probabilities[name] = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(features))
        return {
            "selected": selected,
            "direct": direct,
            "derived": derived,
            "hybrid": hybrid,
            "targets": target_predictions,
            "critical_probability": risk_probabilities["critical"],
            "warning_probability": risk_probabilities["warning"],
            "normalization_count": normalization_count,
            "model_agreement": self._agreement(features),
        }

    def _agreement(self, features: pd.DataFrame) -> dict[str, Any]:
        spreads: dict[str, list[float]] = {}
        models = {"Y": self.direct_model, **self.target_models}
        for target, model in models.items():
            if isinstance(model, EnsembleRegressor):
                spreads[target] = model.prediction_spread(features).tolist()
        if not spreads:
            return {"available": False, "mean_spread": None, "target_spread": {}}
        values = np.concatenate([np.asarray(item) for item in spreads.values()])
        return {
            "available": True,
            "mean_spread": float(np.mean(values)),
            "target_spread": spreads,
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
    # Deprecated compatibility inputs. Hybrid training uses the fixed CV
    # protocol below instead of percentage-based splitting.
    train_ratio: float = 0.64,
    validation_ratio: float = 0.16,
    test_ratio: float = 0.20,
    missing_indicator: bool = True,
    oof_folds: int = 3,
    outer_folds: int = 5,
    ensemble_options: EnsembleOptions | None = None,
) -> HybridTrainingResult:
    ensemble_options = ensemble_options or EnsembleOptions()
    ensemble_options.validate()
    missing = [target for target in TARGETS if target not in dataframe.columns]
    if missing:
        raise ValueError("Hybrid Multi-Y 학습에 필요한 Target이 없습니다: " + ", ".join(missing))
    numeric_targets = dataframe[TARGETS].apply(pd.to_numeric, errors="coerce")
    valid = numeric_targets.notna().all(axis=1)
    if int(valid.sum()) < 15:
        raise ValueError("Hybrid Multi-Y 학습에는 Y와 Y1~Y10이 모두 유효한 행이 최소 15개 필요합니다.")
    working = dataframe.loc[valid].reset_index(drop=True)
    numeric_targets = numeric_targets.loc[valid].reset_index(drop=True)
    dataset = prepare_dataset(working, target="Y", add_missing_indicators=missing_indicator)
    groups = dataset.groups.reset_index(drop=True)
    if groups.nunique() < outer_folds:
        raise ValueError(f"Nested Group K-Fold를 위해 최소 {outer_folds}개 Lot 그룹이 필요합니다.")

    outer_splits = _seeded_group_splits(
        dataset.features, groups, folds=outer_folds, seed=RANDOM_STATE
    )
    development_index, test_index = outer_splits[0]
    split_features = dataset.features.iloc[development_index].reset_index(drop=True)
    split_groups = groups.iloc[development_index].reset_index(drop=True)
    reporting_folds = min(oof_folds, int(split_groups.nunique()))
    train_relative, validation_relative_index = _seeded_group_splits(
        split_features,
        split_groups,
        folds=reporting_folds,
        seed=RANDOM_STATE,
    )[0]
    train_index = development_index[train_relative]
    validation_index = development_index[validation_relative_index]
    development_features = dataset.features.iloc[development_index].reset_index(drop=True)
    development_targets = numeric_targets.iloc[development_index].reset_index(drop=True)
    development_groups = groups.iloc[development_index].reset_index(drop=True)
    x_test = dataset.features.iloc[test_index]
    warnings = list(dataset.warnings)

    selections: dict[str, Any] = {}
    target_metrics: dict[str, Any] = {}
    selected_estimators: dict[str, str] = {}
    oof: dict[str, np.ndarray] = {}
    target_ensemble_configs: dict[str, Any] = {}
    folds = min(oof_folds, int(development_groups.nunique()))
    if folds < 2:
        raise ValueError("OOF prediction을 생성할 Lot 그룹이 부족합니다.")
    for target in TARGETS:
        transform = (
            (lambda values: np.clip(values, 0.0, None))
            if target != "Y"
            else (lambda values: np.clip(values, 0.0, 100.0))
        )
        selection = select_target_ensemble(
            development_features,
            development_targets[target],
            development_groups,
            lambda count=target in COUNT_TARGETS: _candidate_pipelines(
                dataset, count=count
            ),
            options=ensemble_options,
            folds=folds,
            prediction_transform=transform,
        )
        selections[target] = selection
        oof[target] = transform(selection.oof_prediction)
        target_ensemble_configs[target] = selection_metadata(selection)
        target_ensemble_configs[target]["preprocessing"] = {
            "missing_strategies": {
                name: missing_strategy_for_model(name)
                for name in selection.base_models
            },
            "outlier_strategies": {
                name: outlier_strategy_for_model(name)
                for name in selection.base_models
            },
            "missing_indicator": missing_indicator,
            "outlier_indicator": any(
                outlier_strategy_for_model(name) == "flag_only"
                for name in selection.base_models
            ),
            "fallback_used": False,
        }
        selected_estimators[target] = (
            selection.best_single_name
            if selection.selected_type == "single"
            else " + ".join(selection.base_models)
        )
        warnings.extend(f"{target}: {warning}" for warning in selection.warnings)

    # Outer folds are never used for model selection. Each fold performs its
    # own Inner Group CV selection and evaluates only its held-out Lots.
    outer_fold_metrics: list[dict[str, Any]] = []
    outer_group_assignments: list[dict[str, Any]] = []
    for fold_number, (outer_train, outer_holdout) in enumerate(outer_splits, start=1):
        fold_groups = groups.iloc[outer_train].reset_index(drop=True)
        fold_features = dataset.features.iloc[outer_train].reset_index(drop=True)
        fold_targets = numeric_targets.iloc[outer_train].reset_index(drop=True)
        fold_selections: dict[str, Any] = {}
        fold_oof: dict[str, np.ndarray] = {}
        fold_holdout_predictions: dict[str, np.ndarray] = {}
        for target in TARGETS:
            transform = (
                (lambda values: np.clip(values, 0.0, None))
                if target != "Y"
                else (lambda values: np.clip(values, 0.0, 100.0))
            )
            selection = selections[target] if fold_number == 1 else select_target_ensemble(
                fold_features,
                fold_targets[target],
                fold_groups,
                lambda count=target in COUNT_TARGETS: _candidate_pipelines(dataset, count=count),
                options=ensemble_options,
                folds=min(oof_folds, int(fold_groups.nunique())),
                prediction_transform=transform,
            )
            fold_selections[target] = selection
            fold_oof[target] = transform(selection.oof_prediction)
            fold_holdout_predictions[target] = transform(
                np.asarray(selection.model.predict(dataset.features.iloc[outer_holdout]), dtype=float)
            )

        fold_oof_rates, fold_oof_derived, _ = normalized_failure_rates(
            np.column_stack([fold_oof[target] for target in FAIL_RATE_TARGETS])
        )
        holdout_rates, holdout_derived, _ = normalized_failure_rates(
            np.column_stack([fold_holdout_predictions[target] for target in FAIL_RATE_TARGETS])
        )
        for index, target in enumerate(FAIL_RATE_TARGETS):
            fold_oof[target] = fold_oof_rates[:, index]
            fold_holdout_predictions[target] = holdout_rates[:, index]
        fold_direct = np.clip(fold_oof["Y"], 0.0, 100.0)
        holdout_direct = np.clip(fold_holdout_predictions["Y"], 0.0, 100.0)
        fold_meta_features = np.column_stack([
            fold_direct, fold_oof_derived,
            *[fold_oof[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS],
        ])
        holdout_meta_features = np.column_stack([
            holdout_direct, holdout_derived,
            *[fold_holdout_predictions[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS],
        ])
        fold_meta_model = Ridge(alpha=1.0).fit(fold_meta_features, fold_targets["Y"])
        fold_hybrid = np.clip(fold_meta_model.predict(fold_meta_features), 0.0, 100.0)
        holdout_hybrid = np.clip(fold_meta_model.predict(holdout_meta_features), 0.0, 100.0)
        train_candidates = {
            "direct": fold_direct,
            "derived": fold_oof_derived,
            "hybrid": fold_hybrid,
        }
        selected_strategy = min(
            train_candidates,
            key=lambda name: _metrics(fold_targets["Y"], train_candidates[name])["rmse"] or float("inf"),
        )
        holdout_candidates = {
            "direct": holdout_direct,
            "derived": holdout_derived,
            "hybrid": holdout_hybrid,
        }
        outer_fold_metrics.append({
            "fold": fold_number,
            "selected_strategy": selected_strategy,
            "strategy_metrics": {
                name: _metrics(numeric_targets["Y"].iloc[outer_holdout], prediction)
                for name, prediction in holdout_candidates.items()
            },
            **_metrics(
                numeric_targets["Y"].iloc[outer_holdout],
                holdout_candidates[selected_strategy],
            ),
        })
        outer_group_assignments.append({
            "fold": fold_number,
            "train_groups": sorted(set(groups.iloc[outer_train].astype(str))),
            "holdout_groups": sorted(set(groups.iloc[outer_holdout].astype(str))),
        })
    nested_metric_summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in outer_fold_metrics if row[metric] is not None])),
            "std": float(np.std([row[metric] for row in outer_fold_metrics if row[metric] is not None])),
        }
        for metric in ("r2", "rmse", "mae", "mse")
    }

    validation_positions = np.asarray(validation_relative_index)
    fold_assignments: list[dict[str, list[str]]] = []
    inner_splits = list(
        GroupKFold(n_splits=folds).split(
            development_features, groups=development_groups
        )
    )
    for fold_train, fold_holdout in inner_splits:
        fold_assignments.append({
            "train_groups": sorted(set(development_groups.iloc[fold_train].astype(str))),
            "holdout_groups": sorted(set(development_groups.iloc[fold_holdout].astype(str))),
        })

    oof_rates, oof_derived, normalization_count = normalized_failure_rates(
        np.column_stack([oof[target] for target in FAIL_RATE_TARGETS])
    )
    for index, target in enumerate(FAIL_RATE_TARGETS):
        oof[target] = oof_rates[:, index]
    oof_direct = np.clip(oof["Y"], 0.0, 100.0)
    oof_meta = np.column_stack([oof_direct, oof_derived, *[oof[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS]])
    meta_oof = np.zeros(len(development_features), dtype=float)
    for fold_train, fold_holdout in inner_splits:
        fold_meta = Ridge(alpha=1.0).fit(
            oof_meta[fold_train], development_targets["Y"].iloc[fold_train]
        )
        meta_oof[fold_holdout] = fold_meta.predict(oof_meta[fold_holdout])
    meta_oof = np.clip(meta_oof, 0.0, 100.0)
    meta_model = Ridge(alpha=1.0).fit(oof_meta, development_targets["Y"])
    risk_labels = {
        "critical": (development_targets["Y"] < 85.0).astype(int),
        "warning": (development_targets["Y"] < 90.0).astype(int),
    }
    risk_classifiers: dict[str, Any] = {}
    for name, labels in risk_labels.items():
        classifier: Any
        if labels.nunique() > 1:
            classifier = VotingClassifier(
                estimators=[
                    ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                    ("random_forest", RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
                ],
                voting="soft",
            )
        else:
            classifier = DummyClassifier(strategy="most_frequent")
        classifier.fit(oof_meta, labels)
        risk_classifiers[name] = classifier

    for target, selection in selections.items():
        predicted = np.asarray(selection.model.predict(x_test), dtype=float)
        if target in FAIL_RATE_TARGETS + COUNT_TARGETS:
            predicted = np.clip(predicted, 0.0, None)
        target_metrics[target] = {
            "inner_oof": selection.metrics,
            "validation": _metrics(
                development_targets[target].iloc[validation_positions],
                oof[target][validation_positions],
            ),
            "test": _metrics(numeric_targets[target].iloc[test_index], predicted),
        }

    final_metrics = {
        "direct": {
            "train": _metrics(development_targets["Y"], oof_direct),
            "validation": _metrics(development_targets["Y"].iloc[validation_positions], oof_direct[validation_positions]),
        },
        "derived": {
            "train": _metrics(development_targets["Y"], oof_derived),
            "validation": _metrics(development_targets["Y"].iloc[validation_positions], oof_derived[validation_positions]),
        },
        "hybrid": {
            "train": _metrics(development_targets["Y"], meta_oof),
            "validation": _metrics(development_targets["Y"].iloc[validation_positions], meta_oof[validation_positions]),
        },
    }
    selected_final_output = min(
        final_metrics,
        key=lambda name: (
            final_metrics[name]["train"]["rmse"]
            if final_metrics[name]["train"]["rmse"] is not None
            else float("inf"),
            -(
                final_metrics[name]["train"]["r2"]
                if final_metrics[name]["train"]["r2"] is not None
                else float("-inf")
            ),
        ),
    )
    direct_test = np.clip(selections["Y"].model.predict(x_test), 0.0, 100.0)
    test_targets = {target: np.clip(selections[target].model.predict(x_test), 0.0, None) for target in FAIL_RATE_TARGETS + COUNT_TARGETS}
    test_rates, derived_test, test_normalized = normalized_failure_rates(
        np.column_stack([test_targets[target] for target in FAIL_RATE_TARGETS])
    )
    for index, target in enumerate(FAIL_RATE_TARGETS):
        test_targets[target] = test_rates[:, index]
    test_meta = np.column_stack([direct_test, derived_test, *[test_targets[target] for target in FAIL_RATE_TARGETS + COUNT_TARGETS]])
    hybrid_test = np.clip(meta_model.predict(test_meta), 0.0, 100.0)
    for name, predicted in {
        "direct": direct_test,
        "derived": derived_test,
        "hybrid": hybrid_test,
    }.items():
        final_metrics[name]["test"] = _metrics(numeric_targets["Y"].iloc[test_index], predicted)
    risk_metrics: dict[str, Any] = {}
    for name, labels in risk_labels.items():
        classifier = risk_classifiers[name]
        prediction = classifier.predict(oof_meta)
        probability = classifier.predict_proba(oof_meta)
        classes = list(classifier.classes_)
        positive = probability[:, classes.index(1)] if 1 in classes else np.zeros(len(prediction))
        risk_metrics[name] = {
            "recall": float(recall_score(labels, prediction, zero_division=0)),
            "f1": float(f1_score(labels, prediction, zero_division=0)),
            "f2": float(fbeta_score(labels, prediction, beta=2, zero_division=0)),
            "pr_auc": float(average_precision_score(labels, positive)) if labels.nunique() > 1 else None,
        }
    # Production base estimators are retrained on all available rows after the
    # untouched outer-test evaluation; development OOF weights remain fixed.
    production_models: dict[str, Any] = {}
    for target, selection in selections.items():
        if isinstance(selection.model, EnsembleRegressor):
            fitted_members: dict[str, Any] = {}
            for model_name, model in selection.model.models.items():
                fitted = clone(model)
                fitted.fit(dataset.features, numeric_targets[target])
                fitted_members[model_name] = fitted
            production_models[target] = EnsembleRegressor(
                models=fitted_members,
                weights=selection.model.weights,
                method=selection.model.method,
                meta_model=selection.model.meta_model,
            )
        else:
            fitted = clone(selection.model)
            fitted.fit(dataset.features, numeric_targets[target])
            production_models[target] = fitted

    bundle = HybridMultiYBundle(
        feature_columns=dataset.feature_columns,
        direct_model=production_models.pop("Y"),
        target_models=production_models,
        risk_classifiers=risk_classifiers,
        meta_model=meta_model,
        selected_final_output=selected_final_output,
    )
    selected_model_names = {
        name for selection in selections.values() for name in selection.base_models
    }
    selected_outlier_strategies = {
        outlier_strategy_for_model(name) for name in selected_model_names
    }
    applied_outlier_strategy = (
        next(iter(selected_outlier_strategies))
        if len(selected_outlier_strategies) == 1
        else "model_specific"
    )
    outlier_indicator_used = "flag_only" in selected_outlier_strategies
    metadata = to_json_safe({
        "model_type": "hybrid_multi_y",
        "bundle_type": "hybrid_multi_y",
        "target": "Y",
        "model_name": "Hybrid Multi-Y Ensemble" if ensemble_options.enabled else "Hybrid Multi-Y",
        "feature_columns": dataset.feature_columns,
        "feature_count": len(dataset.feature_columns),
        "selected_final_output": selected_final_output,
        "final_y_metrics": final_metrics,
        "target_metrics": target_metrics,
        "risk_metrics": risk_metrics,
        "selected_estimators": selected_estimators,
        "ensemble_enabled": ensemble_options.enabled,
        "ensemble_mode": ensemble_options.size,
        "ensemble_method": ensemble_options.method,
        "ensemble_selection_method": "inner_group_oof_rmse_stability",
        "target_ensemble_configs": target_ensemble_configs,
        "direct_y_ensemble": target_ensemble_configs["Y"],
        "fail_rate_ensembles": {target: target_ensemble_configs[target] for target in FAIL_RATE_TARGETS},
        "fail_bit_ensembles": {target: target_ensemble_configs[target] for target in COUNT_TARGETS},
        "risk_ensemble": {"method": "soft_voting", "base_models": ["LogisticRegression", "RandomForestClassifier"]},
        "base_model_names": sorted({name for selection in selections.values() for name in selection.base_models}),
        "ensemble_weights": {target: selection.weights for target, selection in selections.items()},
        "improvement_over_best_single": {target: selection.improvement_over_single for target, selection in selections.items()},
        "prediction_correlations": {target: selection.prediction_correlations for target, selection in selections.items()},
        "residual_correlations": {target: selection.residual_correlations for target, selection in selections.items()},
        "model_agreement_summary": {target: selection.agreement for target, selection in selections.items()},
        "production_ensemble_retrained": True,
        "missing_strategy": "model_specific",
        "outlier_strategy": applied_outlier_strategy,
        "missing_indicator_used": missing_indicator,
        "outlier_indicator_used": outlier_indicator_used,
        "fallback_used": False,
        "preprocessing_strategy": "model_specific_native_or_median_and_flag_only_or_iqr_train_only",
        "preprocessing_summary": {
            "missing_strategy": "model_specific",
            "outlier_strategy": applied_outlier_strategy,
            "missing_indicator": missing_indicator,
            "outlier_indicator": outlier_indicator_used,
            "fallback_used": False,
            "r_column_count": len([name for name in dataset.numeric_columns if "_R" in name and not name.endswith("_missing")]),
            "d_column_count": len([name for name in dataset.numeric_columns if "_D" in name and not name.endswith("_missing")]),
            "step_feature_count": len([name for name in dataset.numeric_columns if name.lower().startswith("step") and not name.endswith("_missing")]),
            "categorical_column_count": len(dataset.categorical_columns),
            "config_parsed": dataset.config_report.get("parse_error_count", 0) == 0,
            "config_column_count": dataset.config_report.get("config_column_count", 0),
            "model_strategies": {
                target: config["preprocessing"]["missing_strategies"]
                for target, config in target_ensemble_configs.items()
            },
            "model_outlier_strategies": {
                target: config["preprocessing"]["outlier_strategies"]
                for target, config in target_ensemble_configs.items()
            },
        },
        "cv_protocol": {
            "name": "nested_group_kfold",
            "group_column": "Lot_ID",
            "outer_folds": outer_folds,
            "inner_folds": oof_folds,
            "seed": RANDOM_STATE,
            "selection_target": "Hybrid Multi-Y (Y, Y1-Y10)",
            "fold_metrics": outer_fold_metrics,
            "metric_summary": nested_metric_summary,
            "outer_group_assignments": outer_group_assignments,
        },
        "oof_folds": folds,
        "oof_group_assignments": fold_assignments,
        "meta_model": "Ridge",
        "normalization_count": normalization_count + test_normalized,
        "dataset_rows": {"train": len(train_index), "validation": len(validation_index), "test": len(test_index)},
        "split_method": "nested_group_kfold",
        "group_column": "Lot_ID",
        "metrics": final_metrics[selected_final_output],
        "sklearn_version": sklearn.__version__,
        "scikit_learn_version": sklearn.__version__,
    })
    return HybridTrainingResult(
        bundle=bundle,
        metadata=metadata,
        warnings=list(dict.fromkeys(warnings)),
        oof_predictions={target: values.tolist() for target, values in oof.items()},
    )


def save_hybrid_bundle(result: HybridTrainingResult, model_dir: str | Path, model_id: str) -> tuple[Path, Path]:
    bundle_dir = Path(model_dir) / model_id
    bundle_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = bundle_dir / "bundle.joblib"
    metadata_path = bundle_dir / "metadata.json"
    oof_path = bundle_dir / "oof_predictions.json.gz"
    folds_path = bundle_dir / "fold_assignments.json.gz"
    joblib.dump(result.bundle, bundle_path)
    metadata_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(oof_path, "wt", encoding="utf-8") as handle:
        json.dump(result.oof_predictions or {}, handle, ensure_ascii=False)
    with gzip.open(folds_path, "wt", encoding="utf-8") as handle:
        json.dump(result.metadata.get("oof_group_assignments", []), handle, ensure_ascii=False)
    return bundle_path, metadata_path
