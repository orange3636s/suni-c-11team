from __future__ import annotations

import gc
import gzip
from dataclasses import dataclass, field, replace
from datetime import datetime
from io import BytesIO
import json
import logging
from pathlib import Path
import re
import shutil
from threading import Lock
import time
from typing import Any, Callable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.ml.dataset import RANDOM_STATE
from src.ml.memory_usage import log_memory_stage
from src.ml.model_io import to_json_safe


PIPELINE_VERSION = "auto_y1_y5_frequency_v2"
FAIL_RATE_TARGETS = [f"Y{index}" for index in range(1, 6)]
COUNT_TARGETS = [f"Y{index}" for index in range(6, 11)]
# Only Y1~Y5 have fitted estimators. Y and Y6~Y10 remain analysis fields.
TARGETS = list(FAIL_RATE_TARGETS)
TRAINING_TARGET_ORDER = list(FAIL_RATE_TARGETS)
TARGET_MODEL_ARTIFACTS = {
    target: f"target_{target}.joblib" for target in TRAINING_TARGET_ORDER
}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "-"}
logger = logging.getLogger(__name__)
_STAGING_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ACTIVE_STAGING_PATHS: set[Path] = set()
_STAGING_LOCK = Lock()
ProgressCallback = Callable[[str, int], None]


def normalized_failure_rates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the production final-y rule without changing valid fail rates."""
    rates = np.maximum(np.asarray(values, dtype=np.float32), np.float32(0.0))
    totals = rates.sum(axis=1)
    overflow = totals > 100.0
    derived = np.clip(
        np.float32(100.0) - totals,
        np.float32(0.0),
        np.float32(100.0),
    ).astype(np.float32, copy=False)
    return rates, derived, int(overflow.sum())


def _canonical(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_auto_schema(dataframe: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in dataframe.columns]
    canonical = {_canonical(column): column for column in columns}
    target_columns: dict[str, str] = {}
    aliases = {"Y": ("y", "finalyield")}
    aliases.update({f"Y{i}": (f"y{i}",) for i in range(1, 11)})
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in canonical:
                target_columns[target] = canonical[candidate]
                break

    identifier_columns = [
        column
        for column in columns
        if _canonical(column)
        in {"lotid", "waferid", "waferslot", "lotwaferid", "rowid", "sampleid"}
        or _canonical(column).endswith("identifier")
    ]
    config_pattern = re.compile(r"(?i)^step\d+_config$")
    # Legacy Step_EQ columns are accepted as whole categories, never tokenized,
    # and are exposed through the Config group in new metadata.
    legacy_config_pattern = re.compile(r"(?i)^step\d+_eq.*$")
    response_pattern = re.compile(r"(?i)^step\d+_r\d*$")
    defect_pattern = re.compile(r"(?i)^step\d+_d\d*$")
    config_columns = [
        column
        for column in columns
        if config_pattern.fullmatch(column) or legacy_config_pattern.fullmatch(column)
    ]
    response_columns = [column for column in columns if response_pattern.fullmatch(column)]
    defect_columns = [column for column in columns if defect_pattern.fullmatch(column)]
    feature_columns = list(
        dict.fromkeys([*config_columns, *response_columns, *defect_columns])
    )
    excluded = {*target_columns.values(), *identifier_columns}
    feature_columns = [column for column in feature_columns if column not in excluded]
    return {
        "identifier_columns": identifier_columns,
        "config_columns": config_columns,
        "response_columns": response_columns,
        "defect_columns": defect_columns,
        "target_columns": target_columns,
        "feature_columns": feature_columns,
    }


def validate_y_formula(targets: pd.DataFrame, tolerance: float = 1e-6) -> dict[str, Any]:
    if "Y" not in targets or any(target not in targets for target in FAIL_RATE_TARGETS):
        return {
            "formula": "100 - (Y1 + Y2 + Y3 + Y4 + Y5)",
            "tolerance": tolerance,
            "valid_row_count": 0,
            "mean_absolute_error": None,
            "maximum_absolute_error": None,
            "exact_match_ratio": None,
            "within_tolerance_ratio": None,
            "formula_consistent": False,
        }
    derived = 100.0 - targets[FAIL_RATE_TARGETS].sum(axis=1)
    difference = pd.to_numeric(targets["Y"], errors="coerce") - derived
    absolute = difference.replace([np.inf, -np.inf], np.nan).dropna().abs()
    return {
        "formula": "100 - (Y1 + Y2 + Y3 + Y4 + Y5)",
        "tolerance": tolerance,
        "valid_row_count": int(len(absolute)),
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
    """Train-only imputation, R/D clipping, and Config frequency encoding."""

    def __init__(
        self,
        response_columns: list[str],
        defect_columns: list[str],
        categorical_columns: list[str],
        r_lower_quantile: float = 0.01,
        r_upper_quantile: float = 0.99,
        d_upper_quantile: float = 0.999,
        unknown_frequency: float = 0.0,
    ) -> None:
        self.response_columns = response_columns
        self.defect_columns = defect_columns
        self.categorical_columns = categorical_columns
        self.r_lower_quantile = r_lower_quantile
        self.r_upper_quantile = r_upper_quantile
        self.d_upper_quantile = d_upper_quantile
        self.unknown_frequency = unknown_frequency

    @staticmethod
    def _frame(x: Any) -> pd.DataFrame:
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def fit(self, x: Any, y: Any = None) -> "AutoFeaturePreprocessor":
        del y
        frame = self._frame(x)
        self.response_columns_ = []
        self.defect_columns_ = []
        self.numeric_medians_: dict[str, float] = {}
        self.r_bounds_: dict[str, tuple[float, float] | None] = {}
        self.d_upper_bounds_: dict[str, float | None] = {}
        self.d_zero_ratios_: dict[str, float] = {}
        self.all_missing_columns_: list[str] = []
        self.constant_columns_: list[str] = []
        train_missing_count = 0

        for group, configured in (
            ("R", self.response_columns),
            ("D", self.defect_columns),
        ):
            for column in configured:
                values = self._numeric(frame, column)
                observed = values.dropna()
                if observed.empty:
                    self.all_missing_columns_.append(column)
                    continue
                if observed.nunique() <= 1:
                    self.constant_columns_.append(column)
                    continue
                train_missing_count += int(values.isna().sum())
                self.numeric_medians_[column] = float(observed.median())
                if group == "R":
                    self.response_columns_.append(column)
                    bounds: tuple[float, float] | None = None
                    if len(observed) >= 10 and observed.nunique() > 2:
                        lower, upper = observed.quantile(
                            [self.r_lower_quantile, self.r_upper_quantile]
                        ).to_numpy(dtype=float)
                        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                            bounds = (float(lower), float(upper))
                    self.r_bounds_[column] = bounds
                else:
                    self.defect_columns_.append(column)
                    self.d_zero_ratios_[column] = float((observed == 0).mean())
                    upper_bound: float | None = None
                    if len(observed) >= 10 and observed.nunique() > 2:
                        upper = float(observed.quantile(self.d_upper_quantile))
                        maximum = float(observed.max())
                        if np.isfinite(upper) and upper < maximum and not np.isclose(upper, maximum):
                            upper_bound = upper
                    self.d_upper_bounds_[column] = upper_bound

        self.categorical_columns_ = []
        self.config_modes_: dict[str, str] = {}
        self.frequency_mappings_: dict[str, dict[str, float]] = {}
        config_missing_count = 0
        for column in self.categorical_columns:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category)
            observed = normalized.dropna().astype(str)
            if observed.empty:
                self.all_missing_columns_.append(column)
                continue
            mode = str(observed.mode(dropna=True).iloc[0])
            filled = normalized.fillna(mode).astype(str)
            config_missing_count += int(normalized.isna().sum())
            frequency = filled.value_counts(dropna=False) / max(len(filled), 1)
            self.categorical_columns_.append(column)
            self.config_modes_[column] = mode
            self.frequency_mappings_[column] = {
                str(category): float(value) for category, value in frequency.items()
            }

        self.feature_names_out_ = [
            *self.response_columns_,
            *self.defect_columns_,
            *self.categorical_columns_,
        ]
        if not self.feature_names_out_:
            raise ValueError("Train 데이터에 학습 가능한 R, D, Config Feature가 없습니다.")
        self.summary_ = {
            "r_column_count": len(self.response_columns_),
            "d_column_count": len(self.defect_columns_),
            "config_column_count": len(self.categorical_columns_),
            "numeric_feature_count": len(self.response_columns_) + len(self.defect_columns_),
            "categorical_column_count": len(self.categorical_columns_),
            "total_category_count": int(sum(len(v) for v in self.frequency_mappings_.values())),
            "numeric_missing_fill_count_train": train_missing_count,
            "config_missing_fill_count_train": config_missing_count,
            "removed_all_missing_columns": list(self.all_missing_columns_),
            "removed_constant_columns": list(self.constant_columns_),
            "r_clipped_column_count": int(sum(v is not None for v in self.r_bounds_.values())),
            "d_upper_clipped_column_count": int(sum(v is not None for v in self.d_upper_bounds_.values())),
            "r_clipping_quantiles": [self.r_lower_quantile, self.r_upper_quantile],
            "d_upper_clipping_quantile": self.d_upper_quantile,
            "r_bounds": self.r_bounds_,
            "d_upper_bounds": self.d_upper_bounds_,
            "d_zero_ratios": self.d_zero_ratios_,
            "config_frequency_mappings": self.frequency_mappings_,
            "unknown_frequency": self.unknown_frequency,
            "config_encoding": "frequency",
            "config_strings_decomposed": False,
        }
        return self

    def transform(self, x: Any) -> np.ndarray:
        frame = self._frame(x)
        output = np.empty((len(frame), len(self.feature_names_out_)), dtype=np.float32)
        position = 0
        for column in self.response_columns_:
            values = self._numeric(frame, column).fillna(self.numeric_medians_[column])
            bounds = self.r_bounds_[column]
            if bounds is not None:
                values = values.clip(lower=bounds[0], upper=bounds[1])
            output[:, position] = values.to_numpy(dtype=np.float32)
            position += 1
        for column in self.defect_columns_:
            values = self._numeric(frame, column).fillna(self.numeric_medians_[column])
            upper = self.d_upper_bounds_[column]
            if upper is not None:
                values = values.clip(upper=upper)
            output[:, position] = values.to_numpy(dtype=np.float32)
            position += 1
        for column in self.categorical_columns_:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category).fillna(self.config_modes_[column]).astype(str)
            mapping = self.frequency_mappings_[column]
            output[:, position] = normalized.map(mapping).fillna(self.unknown_frequency).to_numpy(dtype=np.float32)
            position += 1
        return output

    def audit(self, x: Any) -> dict[str, Any]:
        frame = self._frame(x)
        r_adjusted = 0
        d_adjusted = 0
        unknown_by_column: dict[str, int] = {}
        for column in self.response_columns_:
            values = self._numeric(frame, column)
            bounds = self.r_bounds_[column]
            if bounds is not None:
                r_adjusted += int(((values < bounds[0]) | (values > bounds[1])).sum())
        for column in self.defect_columns_:
            values = self._numeric(frame, column)
            upper = self.d_upper_bounds_[column]
            if upper is not None:
                d_adjusted += int((values > upper).sum())
        for column in self.categorical_columns_:
            source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            normalized = source.map(_normalize_category).fillna(self.config_modes_[column]).astype(str)
            unknown_by_column[column] = int((~normalized.isin(self.frequency_mappings_[column])).sum())
        return {
            "r_adjusted_value_count": r_adjusted,
            "d_adjusted_value_count": d_adjusted,
            "unknown_config_count": int(sum(unknown_by_column.values())),
            "unknown_config_by_column": unknown_by_column,
        }

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.feature_names_out_, dtype=object)


def _hgbr(row_count: int) -> HistGradientBoostingRegressor:
    max_iter = 80 if row_count < 200 else 120 if row_count < 1000 else 150
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )


def _random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        max_samples=0.8,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )


def _pipeline(schema: dict[str, Any], estimator: Any) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                AutoFeaturePreprocessor(
                    response_columns=list(schema["response_columns"]),
                    defect_columns=list(schema["defect_columns"]),
                    categorical_columns=list(schema["config_columns"]),
                ),
            ),
            ("model", estimator),
        ]
    )


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
        pearson_value = float(np.corrcoef(truth, estimate)[0, 1])
        pearson = pearson_value if np.isfinite(pearson_value) else None
        ranked = spearmanr(truth, estimate).statistic
        spearman = float(ranked) if np.isfinite(ranked) else None
    return {
        "r2": float(r2_score(truth, estimate)) if len(truth) > 1 else None,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(truth, estimate)),
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
    }


def _lot_groups(dataframe: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.Series, str | None]:
    lot_column = next(
        (column for column in schema["identifier_columns"] if _canonical(column) == "lotid"),
        None,
    )
    if lot_column is not None:
        values = dataframe[lot_column].astype("string").str.strip()
    else:
        combined = next(
            (column for column in schema["identifier_columns"] if _canonical(column) == "lotwaferid"),
            None,
        )
        values = (
            dataframe[combined].astype("string").str.extract(
                r"^(.+?)(?:[_-]?(?:WAFER|WF|W)[_-]?\d+)$",
                expand=False,
            )
            if combined is not None
            else pd.Series(pd.NA, index=dataframe.index, dtype="string")
        )
    missing = values.isna() | values.eq("")
    if missing.any():
        values = values.copy()
        values.loc[missing] = [f"__ROW_{index}" for index in dataframe.index[missing]]
    return values.astype(str).reset_index(drop=True), lot_column


def _yield_proxy(dataframe: pd.DataFrame, schema: dict[str, Any]) -> pd.Series:
    actual_column = schema["target_columns"].get("Y")
    actual = (
        pd.to_numeric(dataframe[actual_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if actual_column is not None
        else pd.Series(np.nan, index=dataframe.index)
    )
    derived = pd.Series(np.nan, index=dataframe.index, dtype=float)
    if all(target in schema["target_columns"] for target in FAIL_RATE_TARGETS):
        rates = pd.DataFrame(
            {
                target: pd.to_numeric(dataframe[schema["target_columns"][target]], errors="coerce")
                for target in FAIL_RATE_TARGETS
            }
        )
        complete = rates.notna().all(axis=1)
        derived.loc[complete] = np.clip(100.0 - rates.loc[complete].sum(axis=1), 0.0, 100.0)
    return actual.fillna(derived).reset_index(drop=True)


def _partition_lots(
    lot_table: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[list[str], list[str], list[str], str, int | None]:
    lot_ids = lot_table["lot_id"].astype(str).to_numpy()
    for bins in (5, 4, 3, 2):
        if len(lot_table) < bins * 4 or lot_table["yield"].notna().sum() < bins * 4:
            continue
        try:
            labels = pd.qcut(
                lot_table["yield"].rank(method="first"),
                q=bins,
                labels=False,
            ).to_numpy()
            first = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
            train_index, temporary_index = next(first.split(lot_ids, labels))
            temporary_labels = labels[temporary_index]
            second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
            validation_relative, test_relative = next(
                second.split(lot_ids[temporary_index], temporary_labels)
            )
            return (
                lot_ids[train_index].tolist(),
                lot_ids[temporary_index[validation_relative]].tolist(),
                lot_ids[temporary_index[test_relative]].tolist(),
                "stratified_group_holdout",
                bins,
            )
        except ValueError:
            continue

    random = np.random.default_rng(random_state)
    shuffled = lot_ids.copy()
    random.shuffle(shuffled)
    count = len(shuffled)
    if count < 3:
        raise ValueError("Train/Validation/Test Lot 분할에는 유효한 Lot이 최소 3개 필요합니다.")
    validation_count = max(1, int(round(count * 0.15)))
    test_count = max(1, int(round(count * 0.15)))
    if validation_count + test_count >= count:
        validation_count = 1
        test_count = 1
    train_count = count - validation_count - test_count
    return (
        shuffled[:train_count].tolist(),
        shuffled[train_count : train_count + validation_count].tolist(),
        shuffled[train_count + validation_count :].tolist(),
        "group_shuffle_fallback",
        None,
    )


def split_lot_dataset(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, np.ndarray], pd.Series, dict[str, Any]]:
    groups, lot_column = _lot_groups(dataframe, schema)
    proxy = _yield_proxy(dataframe, schema)
    lot_table = pd.DataFrame({"lot_id": groups, "yield": proxy}).groupby(
        "lot_id", as_index=False
    )["yield"].mean()
    if lot_table["yield"].notna().any():
        lot_table["yield"] = lot_table["yield"].fillna(float(lot_table["yield"].median()))
    train_lots, validation_lots, test_lots, method, bins = _partition_lots(
        lot_table, random_state=random_state
    )
    lot_sets = {
        "train": set(train_lots),
        "validation": set(validation_lots),
        "test": set(test_lots),
    }
    indices = {
        name: np.flatnonzero(groups.isin(lots).to_numpy()) for name, lots in lot_sets.items()
    }
    if any(len(values) == 0 for values in indices.values()):
        raise ValueError("Lot 기준 Train/Validation/Test 분할 결과가 비어 있습니다.")
    overlap = {
        "train_validation": sorted(lot_sets["train"] & lot_sets["validation"]),
        "train_test": sorted(lot_sets["train"] & lot_sets["test"]),
        "validation_test": sorted(lot_sets["validation"] & lot_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError("같은 Lot이 여러 Split에 포함되었습니다.")

    try:
        risk_bins = pd.qcut(lot_table["yield"].rank(method="first"), 3, labels=["low", "mid", "high"])
        lot_table = lot_table.assign(risk_bin=risk_bins.astype(str))
    except ValueError:
        lot_table = lot_table.assign(risk_bin="unknown")
    split_statistics: dict[str, Any] = {}
    for name, selected_lots in lot_sets.items():
        rows = proxy.iloc[indices[name]].dropna()
        selected_table = lot_table[lot_table["lot_id"].isin(selected_lots)]
        shares = selected_table["risk_bin"].value_counts(normalize=True).to_dict()
        split_statistics[name] = {
            "row_count": int(len(indices[name])),
            "lot_count": int(len(selected_lots)),
            "yield_mean": float(rows.mean()) if len(rows) else None,
            "yield_std": float(rows.std(ddof=0)) if len(rows) else None,
            "yield_bin_proportions": {str(key): float(value) for key, value in shares.items()},
        }
    metadata = {
        "split_method": method,
        "stratification_bins": bins,
        "random_state": random_state,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "group_column": lot_column or "Lot_Wafer_ID-derived",
        "lot_overlap": overlap,
        "lot_overlap_count": int(sum(len(values) for values in overlap.values())),
        "lot_overlap_check_passed": not any(overlap.values()),
        "statistics": split_statistics,
        "lot_assignments": {name: sorted(values) for name, values in lot_sets.items()},
    }
    return indices, groups, metadata


def _splitter(features: pd.DataFrame, groups: pd.Series) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Compatibility helper that now returns holdout splits, never K-Fold."""
    proxy = pd.Series(np.arange(len(features), dtype=float))
    table = pd.DataFrame({"lot_id": groups.astype(str), "yield": proxy}).groupby(
        "lot_id", as_index=False
    )["yield"].mean()
    train, validation, test, method, _ = _partition_lots(table, random_state=RANDOM_STATE)
    train_index = np.flatnonzero(groups.astype(str).isin(set(train)).to_numpy())
    validation_index = np.flatnonzero(groups.astype(str).isin(set(validation)).to_numpy())
    test_index = np.flatnonzero(groups.astype(str).isin(set(test)).to_numpy())
    return [(train_index, validation_index), (np.concatenate([train_index, validation_index]), test_index)], method


def _serialized_size(model: Any) -> int:
    buffer = BytesIO()
    joblib.dump(model, buffer, compress=3)
    return len(buffer.getvalue())


def _fit_candidate(
    name: str,
    schema: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[Pipeline, dict[str, Any], np.ndarray]:
    estimator = _hgbr(len(x_train)) if name == "HistGradientBoostingRegressor" else _random_forest()
    model = _pipeline(schema, estimator)
    with threadpool_limits(limits=1):
        model.fit(x_train, y_train)
        start = time.perf_counter()
        prediction = np.asarray(model.predict(x_validation), dtype=np.float32)
        inference_seconds = time.perf_counter() - start
    prediction = np.maximum(prediction, 0.0).astype(np.float32, copy=False)
    return model, {
        **_metrics(y_validation, prediction),
        "model_file_size": _serialized_size(model),
        "validation_inference_seconds": inference_seconds,
    }, prediction


@dataclass
class ModelArtifactRef:
    artifact_path: str
    artifact_root: str | None = field(default=None, repr=False)

    def attach(self, root: str | Path) -> None:
        self.artifact_root = str(Path(root).resolve())

    def resolved_path(self) -> Path:
        configured = Path(self.artifact_path)
        if configured.is_absolute():
            candidate = configured.resolve()
        else:
            if self.artifact_root is None:
                raise FileNotFoundError("모델 Artifact 기준 경로가 설정되지 않았습니다.")
            root = Path(self.artifact_root).resolve()
            candidate = (root / configured).resolve()
            if candidate.parent != root:
                raise ValueError("모델 Artifact 경로가 Bundle 외부를 가리킵니다.")
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError("모델 Target Artifact를 찾을 수 없습니다.")
        return candidate

    def load(self) -> Any:
        return joblib.load(self.resolved_path())

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = self.load()
        try:
            return np.asarray(model.predict(features), dtype=np.float32)
        finally:
            del model
            gc.collect()


def _is_junction(path: Path) -> bool:
    check = getattr(path, "is_junction", None)
    return bool(check()) if callable(check) else False


def _cleanup_stale_staging(root: Path) -> None:
    if not root.is_dir() or root.is_symlink() or _is_junction(root):
        return
    allowed = set(TARGET_MODEL_ARTIFACTS.values())
    for candidate in root.iterdir():
        if (
            candidate in _ACTIVE_STAGING_PATHS
            or not _STAGING_NAME_PATTERN.fullmatch(candidate.name)
            or candidate.is_symlink()
            or _is_junction(candidate)
            or not candidate.is_dir()
        ):
            continue
        entries = list(candidate.iterdir())
        if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):
            logger.warning("안전하지 않은 ML staging 경로를 보존합니다: %s", candidate.name)
            continue
        shutil.rmtree(candidate)


class ModelStagingDirectory:
    def __init__(self) -> None:
        workspace = Path.cwd().resolve()
        self.root = workspace / ".ml-training-staging"
        if self.root.exists() and (self.root.is_symlink() or _is_junction(self.root) or not self.root.is_dir()):
            raise ValueError("ML staging root가 안전하지 않습니다.")
        self.root.mkdir(exist_ok=True)
        with _STAGING_LOCK:
            _cleanup_stale_staging(self.root)
            self.path = self.root / uuid4().hex
            self.path.mkdir(exist_ok=False)
            _ACTIVE_STAGING_PATHS.add(self.path)
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        resolved = self.path.resolve()
        if resolved.parent != self.root or resolved.is_symlink():
            raise ValueError("모델 임시 저장 경로가 안전하지 않습니다.")
        with _STAGING_LOCK:
            if resolved.is_dir():
                entries = list(resolved.iterdir())
                allowed = set(TARGET_MODEL_ARTIFACTS.values())
                if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):
                    raise ValueError("모델 임시 저장 경로에 허용되지 않은 파일이 있습니다.")
                shutil.rmtree(resolved)
            _ACTIVE_STAGING_PATHS.discard(self.path)
            try:
                self.root.rmdir()
            except OSError:
                pass
            self._cleaned = True

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


@dataclass
class HybridMultiYBundle:
    feature_columns: list[str]
    target_models: dict[str, Any]
    selected_final_output: str = "derived"
    warning_threshold: float = 90.0
    critical_threshold: float = 85.0
    pipeline_version: str = PIPELINE_VERSION

    def attach_artifact_root(self, root: str | Path) -> None:
        for stored in self.target_models.values():
            if isinstance(stored, ModelArtifactRef):
                stored.attach(root)

    def model_for_target(self, target: str) -> Any:
        stored = self.target_models.get(target)
        return stored.load() if isinstance(stored, ModelArtifactRef) else stored

    def predict_components(self, features: pd.DataFrame) -> dict[str, Any]:
        predictions: dict[str, np.ndarray] = {}
        for target in FAIL_RATE_TARGETS:
            stored = self.target_models.get(target)
            if stored is None:
                raise ValueError(f"{target} 모델 Artifact가 없습니다.")
            model = stored.load() if isinstance(stored, ModelArtifactRef) else stored
            try:
                values = np.asarray(model.predict(features), dtype=np.float32)
            finally:
                if isinstance(stored, ModelArtifactRef):
                    del model
                    gc.collect()
            predictions[target] = np.maximum(values, 0.0).astype(np.float32, copy=False)
        rates, derived, overflow_count = normalized_failure_rates(
            np.column_stack([predictions[target] for target in FAIL_RATE_TARGETS])
        )
        for position, target in enumerate(FAIL_RATE_TARGETS):
            predictions[target] = rates[:, position]
        critical_probability = 1.0 / (1.0 + np.exp((derived - self.critical_threshold) / 3.0))
        warning_probability = 1.0 / (1.0 + np.exp((derived - self.warning_threshold) / 3.0))
        return {
            "selected": derived,
            "targets": predictions,
            "critical_probability": critical_probability,
            "warning_probability": warning_probability,
            "normalization_count": 0,
            "sum_over_100_count": overflow_count,
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_components(features)["selected"]


@dataclass
class HybridTrainingResult:
    bundle: HybridMultiYBundle
    metadata: dict[str, Any]
    warnings: list[str]
    oof_predictions: dict[str, np.ndarray] | None = None
    _staging_directory: ModelStagingDirectory | None = field(default=None, repr=False)


def _selection_reason(
    hgbr_metrics: dict[str, Any],
    rf_metrics: dict[str, Any],
) -> tuple[str, str]:
    hgbr_rmse = float(hgbr_metrics["rmse"])
    rf_rmse = float(rf_metrics["rmse"])
    relative = abs(hgbr_rmse - rf_rmse) / max(min(hgbr_rmse, rf_rmse), 1e-12)
    if relative <= 0.01:
        if int(hgbr_metrics["model_file_size"]) != int(rf_metrics["model_file_size"]):
            selected = (
                "HistGradientBoostingRegressor"
                if int(hgbr_metrics["model_file_size"]) < int(rf_metrics["model_file_size"])
                else "RandomForestRegressor"
            )
            return selected, "validation_rmse_within_1_percent_smaller_model"
        if float(hgbr_metrics["validation_inference_seconds"]) != float(rf_metrics["validation_inference_seconds"]):
            selected = (
                "HistGradientBoostingRegressor"
                if float(hgbr_metrics["validation_inference_seconds"]) < float(rf_metrics["validation_inference_seconds"])
                else "RandomForestRegressor"
            )
            return selected, "validation_rmse_within_1_percent_faster_inference"
        return "HistGradientBoostingRegressor", "validation_rmse_within_1_percent_hgbr_tiebreak"
    return (
        ("HistGradientBoostingRegressor", "lower_validation_rmse")
        if hgbr_rmse < rf_rmse
        else ("RandomForestRegressor", "lower_validation_rmse")
    )


def _should_run_random_forest(
    hgbr_model: Any | None,
    hgbr_metrics: dict[str, Any],
    hgbr_prediction: np.ndarray | None,
    baseline_metrics: dict[str, Any],
) -> tuple[bool, str | None]:
    if hgbr_model is None:
        return True, "hgbr_training_failed"
    hgbr_rmse = hgbr_metrics.get("rmse")
    if hgbr_rmse is None or not np.isfinite(float(hgbr_rmse)):
        return True, "hgbr_metric_invalid"
    if hgbr_prediction is None or float(np.var(hgbr_prediction)) <= 1e-12:
        return True, "hgbr_prediction_variance_near_zero"
    baseline_rmse = baseline_metrics.get("rmse")
    if baseline_rmse is None or not np.isfinite(float(baseline_rmse)):
        return True, "baseline_metric_invalid"
    if float(hgbr_rmse) > float(baseline_rmse) * 0.95:
        return True, "hgbr_baseline_improvement_below_5_percent"
    return False, None


def train_hybrid_multi_y(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    missing_indicator: bool = False,
    oof_folds: int = 0,
    outer_folds: int = 0,
    ensemble_options: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HybridTrainingResult:
    del missing_indicator, oof_folds, outer_folds, ensemble_options
    if not np.allclose([train_ratio, validation_ratio, test_ratio], [0.70, 0.15, 0.15]):
        raise ValueError("자동 학습 분할 비율은 Train 70% / Validation 15% / Test 15%로 고정됩니다.")
    log_memory_stage(logger, "auto_training_csv_loaded", rows=len(dataframe), columns=len(dataframe.columns))
    schema = detect_auto_schema(dataframe)
    missing_targets = [target for target in FAIL_RATE_TARGETS if target not in schema["target_columns"]]
    if missing_targets:
        raise ValueError("자동 학습에 필요한 Target이 없습니다: " + ", ".join(missing_targets))
    if not schema["feature_columns"]:
        raise ValueError("Config, R, D 공정 Feature를 탐지하지 못했습니다.")
    features = dataframe.reindex(columns=schema["feature_columns"]).reset_index(drop=True)
    indices, groups, split_metadata = split_lot_dataset(dataframe.reset_index(drop=True), schema)
    staging = ModelStagingDirectory()
    production_models: dict[str, ModelArtifactRef] = {}
    target_metrics: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    preprocessing_by_target: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        for target_position, target in enumerate(TRAINING_TARGET_ORDER):
            source = schema["target_columns"][target]
            values = pd.to_numeric(dataframe[source], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
            valid = values.notna().to_numpy()
            selected_indices = {
                name: split_indices[valid[split_indices]] for name, split_indices in indices.items()
            }
            if len(selected_indices["train"]) < 10 or len(selected_indices["validation"]) < 2 or len(selected_indices["test"]) < 2:
                raise ValueError(f"{target}의 유효 Train/Validation/Test 행이 부족합니다.")
            x_train = features.iloc[selected_indices["train"]]
            y_train = values.iloc[selected_indices["train"]]
            x_validation = features.iloc[selected_indices["validation"]]
            y_validation = values.iloc[selected_indices["validation"]]
            baseline_value = float(y_train.mean())
            baseline_prediction = np.full(len(y_validation), baseline_value, dtype=np.float32)
            baseline_metrics = _metrics(y_validation, baseline_prediction)

            progress = 35 + target_position * 11
            if progress_callback:
                progress_callback(f"{target} HistGradientBoosting 검증", progress)
            hgbr_model: Pipeline | None = None
            hgbr_metrics: dict[str, Any]
            hgbr_prediction: np.ndarray | None = None
            try:
                hgbr_model, hgbr_metrics, hgbr_prediction = _fit_candidate(
                    "HistGradientBoostingRegressor", schema, x_train, y_train, x_validation, y_validation
                )
            except Exception as exc:
                logger.exception("%s HistGradientBoosting 학습 실패", target)
                hgbr_metrics = {"r2": None, "rmse": None, "mae": None, "mse": None, "error": str(exc)}

            run_rf, rf_reason = _should_run_random_forest(
                hgbr_model,
                hgbr_metrics,
                hgbr_prediction,
                baseline_metrics,
            )
            rf_model: Pipeline | None = None
            rf_metrics: dict[str, Any] | None = None
            rf_prediction: np.ndarray | None = None
            if run_rf:
                if progress_callback:
                    progress_callback(f"{target} RandomForest 비교", progress + 5)
                rf_model, rf_metrics, rf_prediction = _fit_candidate(
                    "RandomForestRegressor", schema, x_train, y_train, x_validation, y_validation
                )

            if rf_model is None or rf_metrics is None or rf_prediction is None:
                if hgbr_model is None or hgbr_prediction is None:
                    raise ValueError(f"{target}에서 학습에 성공한 모델이 없습니다.")
                selected_name = "HistGradientBoostingRegressor"
                selection_reason = "hgbr_improves_baseline_at_least_5_percent"
                selected_validation_prediction = hgbr_prediction
                selected_validation_metrics = hgbr_metrics
            else:
                if hgbr_model is None or hgbr_prediction is None or hgbr_metrics.get("rmse") is None:
                    selected_name = "RandomForestRegressor"
                    selection_reason = "hgbr_unavailable"
                    selected_validation_prediction = rf_prediction
                    selected_validation_metrics = rf_metrics
                else:
                    selected_name, selection_reason = _selection_reason(hgbr_metrics, rf_metrics)
                    if selected_name == "HistGradientBoostingRegressor":
                        selected_validation_prediction = hgbr_prediction
                        selected_validation_metrics = hgbr_metrics
                    else:
                        selected_validation_prediction = rf_prediction
                        selected_validation_metrics = rf_metrics

            validation_predictions[target] = np.maximum(
                (hgbr_model if selected_name == "HistGradientBoostingRegressor" else rf_model).predict(
                    features.iloc[indices["validation"]]
                ),
                0.0,
            ).astype(np.float32)
            del selected_validation_prediction

            combined_indices = np.concatenate([selected_indices["train"], selected_indices["validation"]])
            final_model = _pipeline(
                schema,
                _hgbr(len(combined_indices)) if selected_name == "HistGradientBoostingRegressor" else _random_forest(),
            )
            with threadpool_limits(limits=1):
                final_model.fit(features.iloc[combined_indices], values.iloc[combined_indices])
                test_target_prediction = np.maximum(
                    final_model.predict(features.iloc[selected_indices["test"]]), 0.0
                ).astype(np.float32)
                test_predictions[target] = np.maximum(
                    final_model.predict(features.iloc[indices["test"]]), 0.0
                ).astype(np.float32)
            preprocessor = final_model.named_steps["features"]
            preprocessing_by_target[target] = {
                **preprocessor.summary_,
                "validation_audit": preprocessor.audit(features.iloc[indices["validation"]]),
                "test_audit": preprocessor.audit(features.iloc[indices["test"]]),
            }
            target_path = staging.path / TARGET_MODEL_ARTIFACTS[target]
            joblib.dump(final_model, target_path, compress=3)
            production_models[target] = ModelArtifactRef(str(target_path))
            final_size = int(target_path.stat().st_size)
            target_metrics[target] = {
                "baseline_validation": baseline_metrics,
                "hist_gradient_boosting_validation": hgbr_metrics,
                "random_forest_executed": run_rf,
                "random_forest_reason": rf_reason,
                "random_forest_validation": rf_metrics,
                "selected_model": selected_name,
                "selection_reason": selection_reason,
                "validation": selected_validation_metrics,
                "test": _metrics(values.iloc[selected_indices["test"]], test_target_prediction),
                "valid_rows": {
                    "train": int(len(selected_indices["train"])),
                    "validation": int(len(selected_indices["validation"])),
                    "test": int(len(selected_indices["test"])),
                    "excluded": int((~valid).sum()),
                },
                "model_file_size": final_size,
            }
            del final_model, hgbr_model, rf_model, test_target_prediction
            gc.collect()
            log_memory_stage(logger, "auto_target_released", target=target, selected=selected_name)

        validation_rates, validation_y, validation_overflow = normalized_failure_rates(
            np.column_stack([validation_predictions[target] for target in FAIL_RATE_TARGETS])
        )
        test_rates, test_y, test_overflow = normalized_failure_rates(
            np.column_stack([test_predictions[target] for target in FAIL_RATE_TARGETS])
        )
        del validation_rates, test_rates
        actual_y_column = schema["target_columns"].get("Y")
        if actual_y_column is not None:
            actual_y = pd.to_numeric(dataframe[actual_y_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            validation_actual = actual_y.iloc[indices["validation"]]
            test_actual = actual_y.iloc[indices["test"]]
            validation_y_metrics = _metrics(validation_actual, validation_y)
            test_y_metrics = _metrics(test_actual, test_y)
        else:
            validation_y_metrics = _metrics([], [])
            test_y_metrics = _metrics([], [])

        bundle = HybridMultiYBundle(
            feature_columns=list(schema["feature_columns"]),
            target_models=production_models,
        )
        dataset_rows = {name: int(len(value)) for name, value in indices.items()}
        dataset_lots = {
            name: int(split_metadata["statistics"][name]["lot_count"])
            for name in ("train", "validation", "test")
        }
        selected_models = {
            target: target_metrics[target]["selected_model"] for target in FAIL_RATE_TARGETS
        }
        formula_targets = pd.DataFrame(
            {
                target: pd.to_numeric(dataframe[schema["target_columns"][target]], errors="coerce")
                for target in FAIL_RATE_TARGETS
            }
        )
        if "Y" in schema["target_columns"]:
            formula_targets["Y"] = pd.to_numeric(dataframe[schema["target_columns"]["Y"]], errors="coerce")
        metadata = to_json_safe(
            {
                "schema_version": "semicon_yield_v2",
                "pipeline_version": PIPELINE_VERSION,
                "model_version": PIPELINE_VERSION,
                "model_type": "hybrid_multi_y",
                "bundle_type": "y1_y5_derived_y",
                "target": "Y",
                "model_name": "Y1~Y5 자동 수율 모델",
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
                "available_targets": FAIL_RATE_TARGETS,
                "analysis_only_targets": ["Y", *COUNT_TARGETS],
                "selected_final_output": "derived",
                "final_y_formula": "clip(100 - sum(max(predicted_Y1..predicted_Y5, 0)), 0, 100)",
                "final_y_metrics": {
                    "derived": {
                        "validation": validation_y_metrics,
                        "test": test_y_metrics,
                    }
                },
                "metrics": {
                    "train": _metrics([], []),
                    "validation": validation_y_metrics,
                    "test": test_y_metrics,
                },
                "target_metrics": target_metrics,
                "risk_metrics": {},
                "selected_models": selected_models,
                "formula_validation": validate_y_formula(formula_targets),
                "dataset_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "dataset_rows": dataset_rows,
                "dataset_lots": dataset_lots,
                "split_method": split_metadata["split_method"],
                "split_metadata": split_metadata,
                "group_column": split_metadata["group_column"],
                "random_state": RANDOM_STATE,
                "preprocessing_strategy": "train_only_frequency_r01_99_d_upper_999",
                "preprocessing_summary": {
                    "by_target": preprocessing_by_target,
                    "config_encoding": "frequency",
                    "config_strings_decomposed": False,
                    "r_clipping": "train_percentile_1_99",
                    "d_clipping": "train_upper_percentile_99.9",
                },
                "missing_strategy": "train_median_and_train_config_mode",
                "outlier_strategy": "r_1_99_d_upper_99_9",
                "target_leakage_check": {
                    "passed": not bool(set(schema["feature_columns"]) & set(schema["target_columns"].values())),
                    "excluded_targets": list(schema["target_columns"].values()),
                    "excluded_identifiers": schema["identifier_columns"],
                    "leakage_columns": [],
                },
                "training_config": {
                    "primary_model": "HistGradientBoostingRegressor",
                    "comparison_model": "RandomForestRegressor",
                    "rf_condition": "hgbr_validation_rmse_not_5_percent_better_than_baseline_or_invalid",
                    "tie_rule": "validation_rmse_within_1_percent_choose_smaller_model",
                    "target_training": "sequential",
                    "random_state": RANDOM_STATE,
                },
                "cv_protocol": {
                    "name": "fixed_lot_holdout_70_15_15",
                    "outer_folds": 0,
                    "inner_folds": 0,
                    "seed": RANDOM_STATE,
                    "selection_data": "validation_only",
                    "test_used_for_selection": False,
                },
                "target_model_artifacts": TARGET_MODEL_ARTIFACTS,
                "sum_over_100_count": {
                    "validation": validation_overflow,
                    "test": test_overflow,
                },
                "memory_policy": {
                    "numeric_dtype": "float32",
                    "categorical_encoder": "FrequencyEncoding",
                    "target_training": "sequential",
                    "candidate_training": "sequential",
                    "random_forest_n_jobs": 1,
                    "production_models": "disk_backed_lazy_target_shards",
                },
                "scikit_learn_version": sklearn.__version__,
                "sklearn_version": sklearn.__version__,
                "library_versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
            }
        )
        oof_predictions = {
            **validation_predictions,
            "validation_predicted_Y": validation_y,
            **{f"test_{target}": values for target, values in test_predictions.items()},
            "test_predicted_Y": test_y,
        }
        return HybridTrainingResult(
            bundle=bundle,
            metadata=metadata,
            warnings=warnings,
            oof_predictions=oof_predictions,
            _staging_directory=staging,
        )
    except Exception:
        staging.cleanup()
        raise


def _write_oof_predictions(path: Path, predictions: dict[str, np.ndarray] | None) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                str(target): np.asarray(values, dtype=np.float32).tolist()
                for target, values in (predictions or {}).items()
            },
            handle,
            ensure_ascii=False,
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
    predictions_path = bundle_dir / "oof_predictions.json.gz"
    assignments_path = bundle_dir / "fold_assignments.json.gz"
    target_paths = {
        target: bundle_dir / filename for target, filename in TARGET_MODEL_ARTIFACTS.items()
    }
    try:
        persistent: dict[str, ModelArtifactRef] = {}
        for target in TRAINING_TARGET_ORDER:
            stored = result.bundle.target_models[target]
            destination = target_paths[target]
            if isinstance(stored, ModelArtifactRef):
                shutil.copy2(stored.resolved_path(), destination)
            else:
                joblib.dump(stored, destination, compress=3)
            persistent[target] = ModelArtifactRef(TARGET_MODEL_ARTIFACTS[target])
        bundle = replace(result.bundle, target_models=persistent)
        joblib.dump(bundle, bundle_path, compress=3)
        bundle.attach_artifact_root(bundle_dir)
        metadata_path.write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_oof_predictions(predictions_path, result.oof_predictions)
        with gzip.open(assignments_path, "wt", encoding="utf-8") as handle:
            json.dump(
                result.metadata.get("split_metadata", {}).get("lot_assignments", {}),
                handle,
                ensure_ascii=False,
            )
    except Exception:
        for generated in [bundle_path, metadata_path, predictions_path, assignments_path, *target_paths.values()]:
            if generated.is_file():
                generated.unlink()
        try:
            bundle_dir.rmdir()
        except OSError:
            pass
        raise
    result.bundle = bundle
    result.oof_predictions = None
    if result._staging_directory is not None:
        result._staging_directory.cleanup()
        result._staging_directory = None
    gc.collect()
    return bundle_path, metadata_path
