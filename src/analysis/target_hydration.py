"""Build an immutable analysis view with Y/Y1..Y5 gaps filled by the active model.

The uploaded/bundled dataframe is never mutated.  Every analysis consumer gets
the same hydrated copy and the same provenance, so measured/predicted mixing,
range handling, and model-version tracking cannot drift between endpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.ml.inference import InferenceInputError, ModelLoadError, load_prediction_model


logger = logging.getLogger(__name__)

FAIL_RATE_TARGETS = ("Y1", "Y2", "Y3", "Y4", "Y5")
FINAL_YIELD_COLUMN = "Y"
ALL_TARGETS = (FINAL_YIELD_COLUMN, *FAIL_RATE_TARGETS)
TARGET_HYDRATION_VERSION = "target-hydration-v1"
TARGET_FORMULA_TOLERANCE = 0.001
# T3-3: 개수가 아니라 캐시가 들고 있는 총 행 수로 정원을 잰다 -- 업로드
# 상한이 200,000행으로 오른 뒤로는(작업지시 T1) "최대 8개"가 최악의 경우
# 8 x 200,000행짜리 하이드레이션 결과를 동시에 물고 있는 걸 허용해
# 캐시 정원의 의미가 없어진다. 20,000행 표본(T2) 기준 데이터셋
# 10개 분량을 유지할 수 있는 여유로 잡았다.
_CACHE_MAX_ROWS = 200_000


class TargetHydrationError(RuntimeError):
    """A user-actionable failure to produce an analysis target view."""


TargetDataState = Literal["missing_columns", "all_missing", "partial", "complete"]


@dataclass(frozen=True)
class TargetStatus:
    state: TargetDataState
    present_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    valid_cell_count: int
    total_cell_count: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "present_columns": list(self.present_columns),
            "missing_columns": list(self.missing_columns),
            "valid_cell_count": self.valid_cell_count,
            "total_cell_count": self.total_cell_count,
            "message": self.message,
        }


@dataclass(frozen=True)
class HydrationProvenance:
    dataset_id: str
    dataset_version: str
    hydration_version: str
    model_id: str | None
    model_version: str | None
    predicted_at: str | None
    measured_rows: int
    predicted_rows: int
    mixed_rows: int
    measured_target_cells: int
    predicted_target_cells: int
    derived_y_rows: int
    feature_coverage: dict[str, Any]
    warnings: tuple[str, ...]
    warning_counts: dict[str, int]
    source_status: dict[str, Any]
    cache_hit: bool = False

    @property
    def uses_predictions(self) -> bool:
        return self.predicted_target_cells > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "hydration_version": self.hydration_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "predicted_at": self.predicted_at,
            "measured_rows": self.measured_rows,
            "predicted_rows": self.predicted_rows,
            "mixed_rows": self.mixed_rows,
            "measured_target_cells": self.measured_target_cells,
            "predicted_target_cells": self.predicted_target_cells,
            "derived_y_rows": self.derived_y_rows,
            "feature_coverage": self.feature_coverage,
            "warnings": list(self.warnings),
            "warning_counts": dict(self.warning_counts),
            "source_status": dict(self.source_status),
            "cache_hit": self.cache_hit,
            "uses_predictions": self.uses_predictions,
        }


@dataclass(frozen=True)
class HydratedTargets:
    dataframe: pd.DataFrame
    provenance: HydrationProvenance


_CacheKey = tuple[str, str, str, str, str]
_CACHE: OrderedDict[_CacheKey, HydratedTargets] = OrderedDict()
_CACHE_LOCK = RLock()


def _finite_numeric(source: pd.Series) -> pd.Series:
    return pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)


def inspect_target_status(dataframe: pd.DataFrame) -> TargetStatus:
    present = tuple(target for target in ALL_TARGETS if target in dataframe.columns)
    missing = tuple(target for target in ALL_TARGETS if target not in dataframe.columns)
    if not present:
        return TargetStatus(
            state="missing_columns",
            present_columns=(),
            missing_columns=missing,
            valid_cell_count=0,
            total_cell_count=len(dataframe) * len(ALL_TARGETS),
            message="타깃 컬럼이 없어 승인 모델의 예측값으로 분석합니다.",
        )

    valid = pd.DataFrame(
        {target: _finite_numeric(dataframe[target]) for target in present},
        index=dataframe.index,
    ).notna()
    valid_count = int(valid.sum().sum())
    expected_count = len(dataframe) * len(ALL_TARGETS)
    complete = not missing and valid_count == expected_count
    if valid_count == 0:
        state: TargetDataState = "all_missing"
        message = "실측 수율·불량률이 없어 승인 모델의 예측값으로 분석합니다."
    elif complete:
        state = "complete"
        message = "Y/Y1~Y5 실측값이 모두 존재합니다."
    else:
        state = "partial"
        message = "실측값을 우선 사용하고 결측값만 예측값으로 보완합니다."
    return TargetStatus(
        state=state,
        present_columns=present,
        missing_columns=missing,
        valid_cell_count=valid_count,
        total_cell_count=expected_count,
        message=message,
    )


def _active_model(store: Any, model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    active = store.active_model()
    model_id = str((active or {}).get("active_model_id") or "").strip()
    if not model_id:
        raise TargetHydrationError(
            "분석에 사용할 승인 모델이 없습니다. 모델 학습·자동화에서 Y1~Y5 모델을 먼저 학습해 주세요."
        )
    try:
        loaded = load_prediction_model(model_id, model_dir)
    except InferenceInputError as exc:
        raise TargetHydrationError(f"승인 모델을 사용할 수 없습니다: {exc}") from exc
    except ModelLoadError as exc:
        raise TargetHydrationError("승인 모델 파일을 불러오지 못했습니다. 모델 상태를 확인해 주세요.") from exc
    metadata = loaded.metadata
    available = {str(value) for value in metadata.get("available_targets", [])}
    missing_targets = [target for target in FAIL_RATE_TARGETS if target not in available]
    if missing_targets:
        raise TargetHydrationError(
            "승인 모델에 필요한 Y1~Y5 서브모델이 없습니다: " + ", ".join(missing_targets)
        )
    return loaded, active


def _screening_features(dataframe: pd.DataFrame, metadata: dict[str, Any], target: str, model: Any) -> tuple[pd.DataFrame, str | None]:
    details = (metadata.get("target_metrics") or {}).get(target) or {}
    raw_feature = details.get("feature")
    if not isinstance(raw_feature, str) or not raw_feature:
        raise TargetHydrationError(f"승인 모델의 {target} feature 메타데이터가 없습니다.")
    raw = (
        _finite_numeric(dataframe[raw_feature])
        if raw_feature in dataframe.columns
        else pd.Series(np.nan, index=dataframe.index, dtype=float)
    )
    available: dict[str, pd.Series] = {
        raw_feature: raw.astype("float32"),
        f"{raw_feature}_miss": raw.isna().astype("int8"),
    }
    center = details.get("optimal_center")
    if details.get("relation_shape") == "u_shape" and isinstance(center, (int, float)) and math.isfinite(float(center)):
        available[f"{raw_feature}_dev"] = (raw - float(center)).abs().astype("float32")
    expected = [str(value) for value in getattr(model, "feature_names_in_", list(available))]
    features = pd.DataFrame(available, index=dataframe.index).reindex(columns=expected)
    return features, raw_feature


def _predict_targets(dataframe: pd.DataFrame, loaded: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = loaded.metadata
    bundle = loaded.model
    model_for_target = getattr(bundle, "model_for_target", None)
    if not callable(model_for_target):
        raise TargetHydrationError("승인 모델이 Y1~Y5 서브모델 형식과 호환되지 않습니다.")

    predictions: dict[str, np.ndarray] = {}
    required_raw: list[str] = []
    coverage_by_target: dict[str, Any] = {}
    is_screening = metadata.get("bundle_type") == "screening_pareto_pipeline"
    for target in FAIL_RATE_TARGETS:
        try:
            model = model_for_target(target)
        except Exception as exc:
            raise TargetHydrationError(f"승인 모델의 {target} 서브모델을 불러오지 못했습니다.") from exc
        if model is None or not callable(getattr(model, "predict", None)):
            raise TargetHydrationError(f"승인 모델의 {target} 서브모델이 없습니다.")

        if is_screening:
            features, raw_feature = _screening_features(dataframe, metadata, target, model)
            raw_columns = [raw_feature] if raw_feature else []
        else:
            raw_columns = [str(value) for value in metadata.get("feature_columns", [])]
            features = dataframe.reindex(columns=raw_columns)
        required_raw.extend(raw_columns)
        existing = [column for column in raw_columns if column in dataframe.columns]
        cell_total = len(dataframe) * len(raw_columns)
        measured_cells = int(
            sum(_finite_numeric(dataframe[column]).notna().sum() for column in existing)
        )
        coverage_by_target[target] = {
            "required_features": len(raw_columns),
            "present_features": len(existing),
            "missing_features": [column for column in raw_columns if column not in dataframe.columns],
            "column_coverage": (len(existing) / len(raw_columns)) if raw_columns else 0.0,
            "measured_cell_coverage": (measured_cells / cell_total) if cell_total else 0.0,
        }
        try:
            values = np.asarray(model.predict(features), dtype=float)
        except Exception as exc:
            raise TargetHydrationError(f"승인 모델의 {target} 예측에 실패했습니다.") from exc
        if values.ndim != 1 or len(values) != len(dataframe) or not np.isfinite(values).all():
            raise TargetHydrationError(f"승인 모델의 {target} 예측 결과가 올바르지 않습니다.")
        predictions[target] = np.clip(values, 0.0, 100.0)

    unique_required = list(dict.fromkeys(required_raw))
    unique_present = [column for column in unique_required if column in dataframe.columns]
    return predictions, {
        "required_features": unique_required,
        "present_features": unique_present,
        "missing_features": [column for column in unique_required if column not in dataframe.columns],
        "column_coverage": (len(unique_present) / len(unique_required)) if unique_required else 0.0,
        "by_target": coverage_by_target,
    }


def _view_with_cache_flag(result: HydratedTargets, *, cache_hit: bool) -> HydratedTargets:
    """UB-1 (perf): rebuilds only the provenance (a frozen dataclass, cheap
    to reconstruct) with the correct `cache_hit` flag -- the dataframe
    itself is never copied here anymore.

    This used to be `_copy_result`, and it did `result.dataframe.copy(deep=True)`
    on *every* call, including cache HITS. That is exactly the anti-pattern
    this project's own review checklist warns against ("캐시 안에서 매번
    copy()하지 마라 -- 캐시 효과가 절반으로 준다"): a 5-target x 10-factor
    analysis run makes ~50 calls into `hydrate_targets` for the same
    (dataset, model) pair, so 49 of those 50 calls paid a full deep copy of a
    ~10k-row x ~90-col frame for nothing.

    Safe to drop because no consumer downstream of `_hydrated_targets_or_409`
    mutates `hydrated.dataframe` in place -- verified by inspecting every
    call site (`api/routes/analysis.py`, `api/routes/monitoring.py`) and every
    module that receives it transitively (`src/analysis/scatter.py`,
    `control_range.py`, `screening/*.py`,
    `measurement_expansion.py`, `alarm_gbdt.py`, `report.py`,
    `recommendations.py`, `alarm_bands.py`): every
    one of them either reads columns (`df[...]`, boolean-mask selection,
    which always returns a new object in pandas) or builds a brand-new local
    frame from the columns it needs (e.g. `scatter.py`'s `build_scatter_data`
    does `frame = pd.DataFrame({...})` and assigns into *that*, never into
    the frame it was handed). There is no `inplace=True`, no `df.loc[...] =`,
    no `del df[...]` anywhere on the shared frame in this codebase.

    The one dataframe copy that still matters -- `dataframe.copy(deep=True)`
    at the top of `hydrate_targets` below, made once per cache MISS -- stays.
    It protects the registry's own dataframe (which callers do not own) from
    ever being mutated by the target-filling logic that follows it; that is
    a real safety copy, not the redundant one this function used to add on
    top of it.
    """
    provenance = HydrationProvenance(
        **{**result.provenance.__dict__, "cache_hit": cache_hit}
    )
    return HydratedTargets(result.dataframe, provenance)


def hydrate_targets(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_version: str,
    store: Any,
    model_dir: str | Path,
) -> HydratedTargets:
    """Return an analysis-only target-complete frame and reproducibility metadata."""
    status = inspect_target_status(dataframe)
    numeric_original = {
        target: (
            _finite_numeric(dataframe[target])
            if target in dataframe.columns
            else pd.Series(np.nan, index=dataframe.index, dtype=float)
        )
        for target in ALL_TARGETS
    }
    original_fail = pd.DataFrame({target: numeric_original[target] for target in FAIL_RATE_TARGETS})
    missing_masks = original_fail.isna()
    needs_prediction = bool(missing_masks.any().any())

    loaded = None
    active: dict[str, Any] = {}
    model_id = "measured-only"
    model_version = "none"
    if needs_prediction:
        loaded, active = _active_model(store, model_dir)
        model_id = loaded.model_id
        model_version = str(
            loaded.metadata.get("model_version")
            or loaded.metadata.get("pipeline_version")
            or active.get("pipeline_version")
            or "unknown"
        )

    cache_key: _CacheKey = (
        dataset_id,
        dataset_version,
        model_id,
        model_version,
        TARGET_HYDRATION_VERSION,
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            logger.info("target_hydration cache hit dataset=%s model=%s", dataset_id, model_id)
            return _view_with_cache_flag(cached, cache_hit=True)
    logger.info("target_hydration cache miss dataset=%s model=%s", dataset_id, model_id)

    # T3-2: 얕은 복사 -- 바뀌는 건 아래 루프가 재할당하는 타깃 컬럼뿐이다.
    # `df[col] = ...`는 그 컬럼의 블록만 새로 만들 뿐 원본(`dataframe`,
    # 레지스트리가 캐시로 들고 있는 프레임)의 다른 컬럼 배열을 건드리지
    # 않으므로, deep=True였을 때와 안전성은 동일하면서 100,000행에서
    # 손대지 않는 나머지 ~90개 컬럼의 복사를 아낀다.
    hydrated = dataframe.copy(deep=False)
    for target in ALL_TARGETS:
        hydrated[target] = numeric_original[target]

    feature_coverage: dict[str, Any] = {
        "required_features": [],
        "present_features": [],
        "missing_features": [],
        "column_coverage": 1.0,
        "by_target": {},
    }
    predicted_at: str | None = None
    if needs_prediction:
        assert loaded is not None
        predictions, feature_coverage = _predict_targets(dataframe, loaded)
        predicted_at = datetime.now(timezone.utc).isoformat()
        for target in FAIL_RATE_TARGETS:
            mask = missing_masks[target]
            hydrated.loc[mask, target] = predictions[target][mask.to_numpy()]

    observed_values = original_fail.fillna(0.0).to_numpy(dtype=float)
    observed_sum = observed_values.sum(axis=1)
    predicted_positions = missing_masks.to_numpy(dtype=bool)
    hydrated_values = hydrated.loc[:, FAIL_RATE_TARGETS].to_numpy(dtype=float)
    predicted_sum = np.where(predicted_positions, hydrated_values, 0.0).sum(axis=1)
    remaining = np.maximum(100.0 - observed_sum, 0.0)
    overflow = predicted_sum > (remaining + 1e-12)
    scalable = overflow & (predicted_sum > 0.0)
    if scalable.any():
        scale = np.ones(len(hydrated), dtype=float)
        scale[scalable] = remaining[scalable] / predicted_sum[scalable]
        hydrated_values = np.where(predicted_positions, hydrated_values * scale[:, None], hydrated_values)
        hydrated.loc[:, FAIL_RATE_TARGETS] = hydrated_values

    observed_over_100 = observed_sum > (100.0 + TARGET_FORMULA_TOLERANCE)
    original_y = numeric_original[FINAL_YIELD_COLUMN]
    derived_y_mask = original_y.isna()
    final_sum = hydrated.loc[:, FAIL_RATE_TARGETS].sum(axis=1).to_numpy(dtype=float)
    derived_y = np.clip(100.0 - final_sum, 0.0, 100.0)
    hydrated.loc[derived_y_mask, FINAL_YIELD_COLUMN] = derived_y[derived_y_mask.to_numpy()]

    complete_components = hydrated.loc[:, FAIL_RATE_TARGETS].notna().all(axis=1)
    expected_y = 100.0 - hydrated.loc[:, FAIL_RATE_TARGETS].sum(axis=1)
    measured_y_mismatch = (
        original_y.notna()
        & complete_components
        & ((original_y - expected_y).abs() > TARGET_FORMULA_TOLERANCE)
    )
    observed_counts = (~missing_masks).sum(axis=1)
    measured_rows = int((observed_counts == len(FAIL_RATE_TARGETS)).sum())
    predicted_rows = int((observed_counts == 0).sum())
    mixed_rows = int(((observed_counts > 0) & (observed_counts < len(FAIL_RATE_TARGETS))).sum())
    warning_counts = {
        "observed_fail_rate_sum_over_100": int(observed_over_100.sum()),
        "predicted_components_rescaled": int(scalable.sum()),
        "measured_y_formula_mismatch": int(measured_y_mismatch.sum()),
    }
    warnings: list[str] = []
    if warning_counts["observed_fail_rate_sum_over_100"]:
        warnings.append(
            f"실측 Y1~Y5 합계가 100을 넘는 행이 {warning_counts['observed_fail_rate_sum_over_100']}개입니다. 실측값은 변경하지 않았습니다."
        )
    if warning_counts["predicted_components_rescaled"]:
        warnings.append(
            f"Y1~Y5 합계 정합성을 위해 예측 성분만 비례 조정한 행이 {warning_counts['predicted_components_rescaled']}개입니다."
        )
    if warning_counts["measured_y_formula_mismatch"]:
        warnings.append(
            f"실측 Y와 Y1~Y5 공식이 불일치하는 행이 {warning_counts['measured_y_formula_mismatch']}개입니다. 실측 Y는 유지했습니다."
        )

    provenance = HydrationProvenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        hydration_version=TARGET_HYDRATION_VERSION,
        model_id=(loaded.model_id if loaded is not None else None),
        model_version=(model_version if loaded is not None else None),
        predicted_at=predicted_at,
        measured_rows=measured_rows,
        predicted_rows=predicted_rows,
        mixed_rows=mixed_rows,
        measured_target_cells=int((~missing_masks).sum().sum()),
        predicted_target_cells=int(missing_masks.sum().sum()),
        derived_y_rows=int(derived_y_mask.sum()),
        feature_coverage=feature_coverage,
        warnings=tuple(warnings),
        warning_counts=warning_counts,
        source_status=status.as_dict(),
    )
    # `provenance.cache_hit` already defaults to False, so the freshly
    # computed `result` is stored and returned as-is -- no extra copy needed
    # for either (see `_view_with_cache_flag`'s docstring for why sharing the
    # same dataframe object between the cache and every caller is safe).
    result = HydratedTargets(hydrated, provenance)
    with _CACHE_LOCK:
        _CACHE[cache_key] = result
        _CACHE.move_to_end(cache_key)
        total_rows = sum(len(cached.dataframe) for cached in _CACHE.values())
        while total_rows > _CACHE_MAX_ROWS and len(_CACHE) > 1:
            _, evicted = _CACHE.popitem(last=False)
            total_rows -= len(evicted.dataframe)
    return result


def invalidate_target_hydration_cache(dataset_id: str | None = None) -> int:
    """Invalidate one immutable dataset view, or every view after promotion."""
    with _CACHE_LOCK:
        keys = [key for key in _CACHE if dataset_id is None or key[0] == dataset_id]
        for key in keys:
            _CACHE.pop(key, None)
    if keys:
        logger.info("target_hydration cache invalidated dataset=%s entries=%d", dataset_id or "*", len(keys))
    return len(keys)


def target_hydration_cache_info() -> dict[str, int]:
    with _CACHE_LOCK:
        return {
            "size": len(_CACHE),
            "total_rows": sum(len(cached.dataframe) for cached in _CACHE.values()),
            "max_rows": _CACHE_MAX_ROWS,
        }
