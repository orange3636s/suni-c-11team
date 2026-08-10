from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import datetime
from functools import lru_cache, wraps
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response, status

from api.routes.datasets import get_dataset_registry
from api.schemas.analysis import (
    AlarmListResponse,
    AlertsDataResponse,
    AnalysisContextResponse,
    AnalysisReportResponse,
    CategoricalScatterResponse,
    ControlRangeListResponse,
    HeatmapResponse,
    MeasurementExpansionResponse,
    ModelPerformanceResponse,
    ParetoRankingResponse,
    PreprocessingComparisonResponse,
    ReliabilityResponse,
    ScreeningScatterResponse,
    YieldPredictionResponse,
)
from api.settings import APP_VERSION, settings
from src.analysis import alarm_gbdt, distribution_shift, preprocessing_compare, reliability, target_fallback, warning_line
from src.analysis.alarm_bands import classify_measured_bands
from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.measurement_expansion import MIN_ACTION_BLOCKED_SHARE, compute_measurement_expansion
from src.analysis.recommendations import FactorRecommendation, compute_factor_recommendation
from src.analysis.report import build_analysis_report, build_chat_context
from src.analysis.rounding import round_floats
from src.analysis.scatter import build_categorical_data, build_scatter_data
from src.analysis.screening.fmea import build_fmea_table
from src.analysis.screening.heatmap import HeatmapData, build_categorical_heatmap, build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.target_hydration import (
    TARGET_HYDRATION_VERSION,
    HydratedTargets,
    TargetHydrationError,
    hydrate_targets,
)
from src.analysis.screening.selector import (
    DEFAULT_MIN_N_CATEGORICAL,
    DEFAULT_MIN_N_D,
    DEFAULT_MIN_N_R,
    PARETO_TOP_N,
    ParetoFactor,
    effective_confidence_tier,
    score_all_factors,
)
from src.analysis.screening.selector import _ranked_rows_with_contribution as _ranked_rows
from src.analysis.screening.selector import _row_to_factor
from src.ml.inference import get_latest_model_metadata
from src.runtime.datasets import DatasetNotFoundError
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])

# The report always evaluates alarms/eval_result against the bundled "test"
# set -- the root-cause tab (the report button's only caller) has a single
# dataset selector (train only), the same convention /api/alarms already
# defaults to.
REPORT_EVAL_DATASET_ID = "test"


def _dataframe_or_404(dataset_id: str):
    registry = get_dataset_registry()
    try:
        return registry.get_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc


def _hydrated_targets_or_409(dataset_id: str) -> HydratedTargets:
    registry = get_dataset_registry()
    try:
        dataframe = registry.get_dataframe(dataset_id)
        dataset_version = registry.content_version(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="데이터셋을 찾을 수 없습니다.") from exc
    try:
        return hydrate_targets(
            dataframe,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            store=RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir),
            model_dir=settings.model_dir,
        )
    except TargetHydrationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


SCHEMA_CACHE_DATASETS = 2


@lru_cache(maxsize=SCHEMA_CACHE_DATASETS)
def _cached_schema(dataset_id: str, dataset_version: str) -> Any:
    """UB-2 (perf): `parse_schema` re-walks every column of the dataframe
    (regex match x ~90 columns) on every call -- individually cheap, but
    `get_screening_scatter` alone calls it fresh 50 times in one 5-target x
    10-factor analysis run (target_hydration.py's own cache fixed the much
    bigger cost -- model load/predict -- but left this uncached).

    `dataset_version` MUST stay in the cache key even though the function
    body never reads it -- this project has hit the "cache key missing the
    version" bug three times before (uploading a replacement dataset under
    the same id must produce a fresh schema, not a stale one). Keying by
    version alone is enough: content changes bump `dataset_version`, which
    is itself a fresh key, so old entries just age out via `maxsize`
    LRU eviction -- no explicit invalidation call is needed for this cache
    (see the UB-3 invalidation-audit note in the perf commit message).
    """
    del dataset_version  # part of the cache key only, see docstring above
    df = get_dataset_registry().get_dataframe(dataset_id)
    return parse_schema(df)


@lru_cache(maxsize=SCHEMA_CACHE_DATASETS)
def _cached_gbdt_features(dataset_id: str, dataset_version: str) -> tuple[str, ...]:
    """Depends only on `_cached_schema` (a list comprehension over its
    r_cols/d_cols) -- cached and keyed the same way, for the same reason."""
    schema = _cached_schema(dataset_id, dataset_version)
    return tuple(alarm_gbdt.feature_columns(schema))


def _single_flight(fn):
    """B-4: `lru_cache`만으로는 같은 키가 아직 캐시되지 않은 상태에서
    동시에 두 스레드가 들어오면(예: 일일 발송 스케줄러 잡과 사용자의
    `/alarms` 요청이 겹치는 경우) 둘 다 캐시 미스를 보고 같은 무거운
    GBDT를 이중으로 적합시킨다. 키별 락으로 감싸 두 번째 호출은 첫 번째가
    끝날 때까지 기다렸다가 이미 채워진 캐시를 그대로 받아가게 한다 --
    서로 다른 키는 잠그지 않으므로 별개 데이터셋 요청끼리는 막지 않는다.
    데코레이터를 적용받는 함수는 반드시 위치 인자만 받아야 한다(캐시
    키가 그 튜플 그대로다).
    """
    locks: dict[tuple, threading.Lock] = {}
    locks_guard = threading.Lock()

    @wraps(fn)
    def wrapper(*args):
        with locks_guard:
            lock = locks.setdefault(args, threading.Lock())
        with lock:
            return fn(*args)

    return wrapper


def _find_cached_factor(
    dataset: str,
    df: pd.DataFrame,
    target: str,
    feature: str,
    provenance: Any | None = None,
) -> ParetoFactor | None:
    """B-2: `find_factor`가 직접 부르는 `_ranked_rows_with_contribution`
    (88인자 ANOVA+FDR 전수 스코어링)을 다시 돌리지 않고, Pareto/heatmap이
    이미 채워 뒀을 `_cached_ranked_rows(dataset, target)`를 재사용한다.
    같은 (dataset, target) 기본 파라미터(fdr_alpha=0.05, min_n=100/20)로
    스코어링하므로 결과는 find_factor와 동일하다 -- 다만 분석 실행이
    5타깃×10인자 산점도를 동시에 요청하면 이 캐시 덕분에 실제 스코어링은
    타깃당 한 번만 일어난다."""
    rows = _ranked_rows_for_provenance(dataset, target, provenance) if provenance is not None else _cached_ranked_rows(dataset, target)
    row = next((r for r in rows if r["feature"] == feature), None)
    return _row_to_factor(df, target, row) if row is not None else None


@router.get("/screening/scatter", response_model=ScreeningScatterResponse)
def get_screening_scatter(dataset: str, target: str, feature: str) -> dict[str, Any]:
    # UA-1 (perf measurement): per-phase timing for a scatter request -- a
    # single "분석 실행" fires 5 targets x 10 factors of these, so a slow
    # phase here is a slow phase x50. See UB's commit message for the
    # before/after numbers this instrumentation produced.
    t_start = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset)
    df = hydrated.dataframe
    dataset_version = hydrated.provenance.dataset_version
    t_hydrate = time.perf_counter()

    schema = _cached_schema(dataset, dataset_version)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃 결과가 없습니다.")
    # Resolves any of the 88 factors regardless of Pareto rank -- a heatmap
    # cell click can open a scatter for a factor outside the top 5.
    factor = _find_cached_factor(dataset, df, target, feature, hydrated.provenance)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")
    if factor.kind == "Config":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{feature}'은(는) Config(범주형) 인자입니다. /api/screening/scatter/categorical을 사용하세요.",
        )
    t_factor = time.perf_counter()

    reference_model = _cached_reference_model(dataset)
    gbdt_features = list(_cached_gbdt_features(dataset, dataset_version))
    t_refmodel = time.perf_counter()

    data = build_scatter_data(df, df, factor, dataset_id=dataset, reference_model=reference_model, gbdt_features=gbdt_features)
    t_build = time.perf_counter()
    logger.debug(
        "scatter %s/%s hydrate=%.3f factor=%.3f refmodel=%.3f build=%.3f total=%.3f",
        target, feature,
        t_hydrate - t_start, t_factor - t_hydrate, t_refmodel - t_factor, t_build - t_refmodel,
        t_build - t_start,
    )
    # Only the bulky per-point/per-bin arrays are rounded -- they're what
    # actually drives payload size (108KB for 1,470 points); scalar stats
    # (p_value/q_value/eps2) keep full precision since a very small
    # p-value (e.g. 7.7e-66) rounded to 4 decimals would collapse to a
    # meaningless 0.0 in the "p<0.001" exponential display.
    return {
        "points": round_floats(data.points),
        "reference_lines": round_floats(data.reference_lines),
        "normal_range": round_floats(data.normal_range),
        "bins": round_floats(data.bins),
        "optimal_center": data.optimal_center,
        "optimal_center_dropped_reason": data.optimal_center_dropped_reason,
        "eps2": data.eps2,
        "spearman_r": data.spearman_r,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "under_sampled": data.under_sampled,
        "relation_shape": data.relation_shape,
        "n": data.n,
        "axis": data.axis,
        "methods": round_floats(data.methods),
        "target_provenance": hydrated.provenance.as_dict(),
    }


@router.get("/screening/scatter/categorical", response_model=CategoricalScatterResponse)
def get_screening_scatter_categorical(dataset: str, target: str, feature: str) -> dict[str, Any]:
    """Per-category box-plot data for a Config factor. Config never gets a
    numeric normal-range (a category has no "range"), so this is a
    separate response shape from the numeric scatter endpoint above
    rather than an overloaded variant of it.
    """
    hydrated = _hydrated_targets_or_409(dataset)
    df = hydrated.dataframe
    schema = _cached_schema(dataset, hydrated.provenance.dataset_version)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    if feature not in schema.config_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}'은(는) Config 인자가 아닙니다.")

    factor = _find_cached_factor(dataset, df, target, feature, hydrated.provenance)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")

    data = build_categorical_data(df, factor)
    return {
        "groups": round_floats([vars(group) for group in data.groups]),
        "eps2": data.eps2,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "n": data.n,
        "axis": data.axis,
        "target_provenance": hydrated.provenance.as_dict(),
    }


HEATMAP_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones
# TC-4/TC-3: numeric은 더 이상 metric 토글이 없다(ε²+rho를 한 응답에 함께
# 낸다) -- numeric 1 + categorical 1 = 2.
HEATMAP_VARIANTS = 2

@lru_cache(maxsize=HEATMAP_CACHE_DATASETS * HEATMAP_VARIANTS)
def _cached_heatmap(
    dataset_id: str,
    kind: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    hydration_version: str,
) -> HeatmapData:
    # 캐시 키에 kind까지 넣는다 (spec E) -- 빠뜨리면 같은 dataset_id로
    # 수치형을 먼저 조회한 뒤 범주형을 조회했을 때 lru_cache가 수치형
    # 결과를 그대로 돌려준다. 데이터셋 내용은 dataset_id가 존재하는 한
    # 불변이므로(업로드는 매번 새 uuid, 번들 파일은 정적) 이 캐시는 최근
    # 2개 데이터셋만 LRU로 유지해 무한정 커지지 않는다.
    del model_id, model_version, hydration_version
    df = _hydrated_targets_or_409(dataset_id).dataframe
    schema = _cached_schema(dataset_id, dataset_version)
    if kind == "categorical":
        return build_categorical_heatmap(df, schema)
    return build_heatmap(df, schema)


@router.get("/screening/heatmap", response_model=HeatmapResponse)
def get_screening_heatmap(
    dataset: str = "train",
    kind: Literal["numeric", "categorical"] = "numeric",
) -> dict[str, Any]:
    """The correlation heatmap used identically by both the training tab
    and the root-cause tab. Two independent views (spec E), never merged
    into one grid or one FDR family: numeric (R+D x Y1~Y5, always both ε²
    and rho -- TC-4) and categorical (Config x Y1~Y5, ε² only -- rho isn't
    defined for an unordered category).
    """
    t0 = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset)
    provenance = hydrated.provenance
    hits_before = _cached_heatmap.cache_info().hits
    heatmap = _cached_heatmap(
        dataset,
        kind,
        provenance.dataset_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )
    cached = _cached_heatmap.cache_info().hits > hits_before
    logger.info(
        "screening_heatmap %.1fms (cached=%s, dataset=%s, kind=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset, kind,
    )
    return {
        "dataset_id": dataset,
        "metric": "eps2",
        "kind": kind,
        "features": heatmap.features,
        "targets": heatmap.targets,
        "values": heatmap.values,
        "rho": heatmap.rho,
        "n": heatmap.n,
        "q": heatmap.q,
        "significant": heatmap.significant,
        "tier": heatmap.tier,
        "gate_excluded": heatmap.gate_excluded,
        "scale": {"min": heatmap.scale["min"], "max": heatmap.scale["max"]},
        "excluded_configs": heatmap.excluded_configs,
        "target_provenance": provenance.as_dict(),
    }


PARETO_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones
PARETO_TARGETS = 5  # Y1..Y5

@lru_cache(maxsize=PARETO_CACHE_DATASETS * PARETO_TARGETS)
def _cached_ranked_rows_versioned(
    dataset_id: str,
    target: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    hydration_version: str,
) -> tuple[dict, ...]:
    # Cached per (dataset_id, target), capped at the 2 most-recent
    # datasets (LRU-evicted): the training tab and the root-cause tab both
    # request the same (dataset, target) pair and must see byte-identical
    # results -- this cache is exactly what guarantees that, not just a
    # performance nicety. Dataset content is immutable once a dataset_id
    # exists (see the heatmap cache's docstring for why that's safe).
    del dataset_version, model_id, model_version, hydration_version
    df = _hydrated_targets_or_409(dataset_id).dataframe
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    return tuple(_ranked_rows(df, schema, target, 0.05, DEFAULT_MIN_N_R, DEFAULT_MIN_N_D, DEFAULT_MIN_N_CATEGORICAL))


def _ranked_rows_for_provenance(dataset_id: str, target: str, provenance: Any) -> tuple[dict, ...]:
    return _cached_ranked_rows_versioned(
        dataset_id,
        target,
        provenance.dataset_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )


def _cached_ranked_rows(dataset_id: str, target: str) -> tuple[dict, ...]:
    hydrated = _hydrated_targets_or_409(dataset_id)
    return _ranked_rows_for_provenance(dataset_id, target, hydrated.provenance)


REFERENCE_MODEL_CACHE_DATASETS = 2


@lru_cache(maxsize=REFERENCE_MODEL_CACHE_DATASETS)
def _cached_reference_model(dataset_id: str):
    """경고선(§C) 계산 전용 단일 GBDT -- 데이터셋당 한 번만 학습해(약 5~10초)
    이후 모든 인자의 산점도 요청이 재사용한다. 결측(Y가 전부 비어 있거나
    R+D 인자가 하나도 없는) 데이터셋에서는 None을 반환해 호출자가 경고선
    없이 진행하게 한다.
    """
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    features = alarm_gbdt.feature_columns(schema)
    if not features or alarm_gbdt.FINAL_YIELD_COLUMN not in df.columns:
        return None
    valid_y = pd.to_numeric(df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce").notna()
    if int(valid_y.sum()) < 10:
        return None
    t0 = time.perf_counter()
    model = warning_line.fit_reference_model(df, features)
    logger.info("warning_line reference model fit %.1fms (dataset=%s)", (time.perf_counter() - t0) * 1000, dataset_id)
    return model


@lru_cache(maxsize=REFERENCE_MODEL_CACHE_DATASETS)
def _cached_all_warning_lines(dataset_id: str):
    """§A-3 알람 사유용 -- 전체 R+D 인자의 경고선을 데이터셋당 한 번만
    계산해 캐시한다 (인자 58개 기준 반복 호출 시 수십 초가 걸린다).
    """
    model = _cached_reference_model(dataset_id)
    if model is None:
        return {}
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    features = alarm_gbdt.feature_columns(schema)
    t0 = time.perf_counter()
    lines = warning_line.compute_all_warning_lines(model, df, features)
    logger.info(
        "warning_line all-feature computation %.1fms (dataset=%s, n_lines=%d)",
        (time.perf_counter() - t0) * 1000, dataset_id, len(lines),
    )
    return lines


BOOTSTRAP_MODELS_CACHE_TRAINS = 2
BOOTSTRAP_PREDICTION_CACHE_ENTRIES = 8  # (train, eval, max_step) 조합 -- 스텝 슬라이더가 여러 값을 오갈 수 있어 기존 pair 캐시보다 넉넉히 둔다


@_single_flight
@lru_cache(maxsize=BOOTSTRAP_MODELS_CACHE_TRAINS)
def _cached_bootstrap_models(train_dataset_id: str):
    """§A-1 부트스트랩 앙상블 -- train_dataset_id 하나로 한 번만 적합해
    캐시한다 (spec §A-1: "분석 실행 시 한 번만 수행하고 캐시한다").

    지시서 작업 2(특정 스텝까지의 정보만으로 예측): 적합(fit)만 여기서
    캐시하고, eval/max_step에 따라 달라지는 예측은
    `_cached_bootstrap_prediction`이 이 캐시를 재사용해 가볍게 계산한다
    -- max_step을 바꿔도 GBDT 30회를 다시 학습하지 않는다.
    """
    train_df = _dataframe_or_404(train_dataset_id)
    schema = parse_schema(train_df)
    features = alarm_gbdt.feature_columns(schema)
    if not features or alarm_gbdt.FINAL_YIELD_COLUMN not in train_df.columns:
        return None
    t0 = time.perf_counter()
    result = alarm_gbdt.fit_bootstrap_models(train_df, features, compute_step_profile=True)
    logger.info(
        "alarm_gbdt bootstrap ensemble fit %.1fms (train=%s)",
        (time.perf_counter() - t0) * 1000, train_dataset_id,
    )
    return result


@_single_flight
@lru_cache(maxsize=BOOTSTRAP_PREDICTION_CACHE_ENTRIES)
def _cached_bootstrap_prediction(train_dataset_id: str, eval_dataset_id: str, max_step: int | None = None):
    """지시서 작업 2 -- 이미 적합된 모델(`_cached_bootstrap_models`)로
    eval을 예측한다. 재학습이 없어 가볍지만(모델 30개의 predict뿐), 같은
    (train, eval, max_step) 조합을 반복 조회할 때는 여전히 캐시를 그대로
    쓴다."""
    bundle = _cached_bootstrap_models(train_dataset_id)
    if bundle is None:
        return None
    eval_df = _dataframe_or_404(eval_dataset_id)
    t0 = time.perf_counter()
    result = alarm_gbdt.predict_with_bootstrap_models(bundle, eval_df, max_step=max_step)
    logger.info(
        "alarm_gbdt bootstrap predict %.1fms (train=%s, eval=%s, max_step=%s)",
        (time.perf_counter() - t0) * 1000, train_dataset_id, eval_dataset_id, max_step,
    )
    return result


AUC_GATE_CACHE_PAIRS = 4


@_single_flight
@lru_cache(maxsize=AUC_GATE_CACHE_PAIRS)
def _cached_transfer_auc_folds(train_dataset_id: str, eval_dataset_id: str) -> tuple[float, ...] | None:
    """알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-1/§A-2) -- (train, eval)
    쌍마다 한 번만 계산해 캐시한다 (5-fold GBDT 적합이 주 비용이다).

    train==eval이면 같은 wafer로 학습하고 평가하는 누출을 피하려 기존
    out-of-fold self-CV(`cross_validate_auc`)를 쓴다. train과 eval이
    다르면 "이 train으로 학습한 모델이 이 eval 분포에서도 통하는가"를
    직접 재는 전이 AUC(`cross_validate_transfer_auc`)를 쓴다 -- self-CV는
    eval이 무엇이든 같은 값이 나와 분포 이동을 감지하지 못한다.
    """
    train_df = _dataframe_or_404(train_dataset_id)
    schema = parse_schema(train_df)
    features = alarm_gbdt.feature_columns(schema)
    if not features or alarm_gbdt.FINAL_YIELD_COLUMN not in train_df.columns:
        return None
    if train_dataset_id == eval_dataset_id:
        fold_aucs = alarm_gbdt.cross_validate_auc(train_df, features)
    else:
        eval_df = _dataframe_or_404(eval_dataset_id)
        fold_aucs = alarm_gbdt.cross_validate_transfer_auc(train_df, eval_df, features)
    return tuple(fold_aucs) if fold_aucs else None


def _auc_gate(train_dataset_id: str, eval_dataset_id: str) -> tuple[float | None, bool]:
    """Returns (auc_lower_bound, gate_passed). 표본 부족 등으로 AUC 자체를
    산출할 수 없으면 (None, False) -- 신뢰도를 확인할 수 없으니 통과시키지
    않는다 (spec §A-2: 기본값은 "알람을 내지 않는다")."""
    fold_aucs = _cached_transfer_auc_folds(train_dataset_id, eval_dataset_id)
    if not fold_aucs:
        return None, False
    auc_lo = float(np.percentile(fold_aucs, 5))
    return auc_lo, auc_lo >= alarm_gbdt.AUC_GATE


def _pareto_payload(dataset_id: str, target: str, top_n: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    hydrated = _hydrated_targets_or_409(dataset_id)
    hits_before = _cached_ranked_rows_versioned.cache_info().hits
    ranked = list(_ranked_rows_for_provenance(dataset_id, target, hydrated.provenance))
    cached = _cached_ranked_rows_versioned.cache_info().hits > hits_before
    logger.info(
        "screening_pareto %.1fms (cached=%s, dataset=%s, target=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset_id, target,
    )
    top = ranked[:top_n]
    items = [
        {
            "feature": row["feature"],
            "kind": row["kind"],
            "step": row["step"],
            "eps2": row["eps2"],
            "p_value": row["p_value"],
            "q_value": row["q_value"],
            "significant": row["significant"],
            "confidence_tier": effective_confidence_tier(row["eps2"], row["p_value"], under_sampled=row.get("under_sampled", False)),
            "n_observed": row["n_observed"],
            "contribution_pct": row["contribution_pct"],
            "cumulative_pct": row["cumulative_pct"],
            # QA-2: 배제 대신 "표본 부족" 배지로 표시 -- 하한(30) 이상이지만
            # 종류별 정상 판정 임계 미만인 경우 True.
            "under_sampled": row.get("under_sampled", False),
        }
        for row in top
    ]
    n80 = next((index + 1 for index, row in enumerate(ranked) if row["cumulative_pct"] >= 80.0), None)
    # 차트 표시 규칙(spec §B)의 0개-타깃 안내 문구가 쓰는 전체 풀 집계치 --
    # 화면에 노출되는 top-5만으로는 "58건 중 FDR 통과 0건"을 계산할 수 없어
    # 여기서 전체 ranked 풀을 기준으로 함께 내려보낸다.
    fdr_pass_count = sum(1 for row in ranked if row["significant"])
    effect_size_pass_count = sum(
        1 for row in ranked if effective_confidence_tier(row["eps2"], row["p_value"], under_sampled=row.get("under_sampled", False)) != "reference"
    )
    max_eps2 = max((row["eps2"] for row in ranked), default=None)
    return {
        "dataset_id": dataset_id,
        "target": target,
        "total_factor_count": len(ranked),
        "n80": n80,
        "fdr_pass_count": fdr_pass_count,
        "effect_size_pass_count": effect_size_pass_count,
        "max_eps2": max_eps2,
        "items": items,
        "analyzable_target_samples": int(pd.to_numeric(hydrated.dataframe[target], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()),
        "model_available": bool(hydrated.provenance.model_id) or not hydrated.provenance.uses_predictions,
        "factor_measurement_insufficient": len(ranked) == 0,
        "target_provenance": hydrated.provenance.as_dict(),
    }


def _fmea_payload(dataset_id: str, targets: tuple[str, ...]) -> dict[str, Any]:
    """FMEA 분석표 (모니터링 홈, 작업 지시서 WE) -- 타깃별 파레토 기여율
    10% 이상인 인자를 전부 스냅샷에 담는다(YG). 온디맨드 REST 엔드포인트는
    두지 않는다(지시서 IA-5: "자동 갱신 파이프라인에서 함께 계산되게
    하라, 별도 조회 금지") -- `src/automation/refresh.py`가 이 함수를
    직접 호출해 다른 분석 결과와 같은 스냅샷 저장 시점에 함께 계산한다.
    같은 시점에 WL(데이터 한계 진단: MNAR 계측 편향 + 분산 분해)도 함께
    계산해 담는다 -- 같은 eval 프레임을 재사용하므로 별도 조회가 없다.
    """
    hydrated = _hydrated_targets_or_409(dataset_id)
    df = hydrated.dataframe
    schema = parse_schema(df)
    usable_targets = [t for t in targets if t in schema.target_cols]
    rows_by_target = {
        t: list(_ranked_rows_for_provenance(dataset_id, t, hydrated.provenance)) for t in usable_targets
    }
    table = build_fmea_table(df, rows_by_target, usable_targets, dataset_id=dataset_id)

    # WL: 데이터 한계 진단(계측 편향 + 분산 분해) -- FMEA와 같은 eval
    # 프레임·같은 (타깃, 인자) 쌍을 재사용해 별도 조회 없이 함께 낸다.
    from src.analysis.data_limitations import build_mnar_rate_report, compute_variance_decomposition

    mnar_report = build_mnar_rate_report(df, [(f.target, f.feature) for f in table.items])
    variance_decomposition = compute_variance_decomposition(df)

    return round_floats(
        {
            "dataset_id": dataset_id,
            "total_wafers": table.total_wafers,
            "measurement_shortage_wafers": table.measurement_shortage_wafers,
            "correlation_shortage_wafers": table.correlation_shortage_wafers,
            "no_qualifying_factor": [
                {"target": n.target, "max_contribution_pct": n.max_contribution_pct} for n in table.no_qualifying_factor
            ],
            "mnar_rate_report": [
                {
                    "target": r.target,
                    "feature": r.feature,
                    "overall_rate_pct": r.overall_rate_pct,
                    "worst_decile_rate_pct": r.worst_decile_rate_pct,
                    "ratio": r.ratio,
                }
                for r in mnar_report
            ],
            "variance_decomposition": (
                {
                    "lot_count": variance_decomposition.lot_count,
                    "wafers_per_lot": variance_decomposition.wafers_per_lot,
                    "between_lot_pct": variance_decomposition.between_lot_pct,
                    "within_lot_pct": variance_decomposition.within_lot_pct,
                    "no_effect_expected_pct": variance_decomposition.no_effect_expected_pct,
                    "icc": variance_decomposition.icc,
                }
                if variance_decomposition is not None
                else None
            ),
            "items": [
                {
                    "target": f.target,
                    "feature": f.feature,
                    "kind": f.kind,
                    "step": f.step,
                    "eps2": f.eps2,
                    "relation_shape": f.relation_shape,
                    "factor_value": f.factor_value,
                    "range_lo": f.range_lo,
                    "range_hi": f.range_hi,
                    "measurement_rate": f.measurement_rate,
                    "deviation_rate_pct": f.deviation_rate_pct,
                    "detection_method": f.detection_method,
                    "detection_kind": f.detection_kind,
                    # 지시서 KA-1: 아래 둘은 타깃 컬럼(Y1~Y5) 기준 불량률이지
                    # 수율이 아니다 -- 이름을 그렇게 정정했다.
                    "expected_defect_rate_pct": f.expected_defect_rate_pct,
                    "defect_rate_deviation_pct": f.defect_rate_deviation_pct,
                    # 진짜 수율(최종 Y 컬럼) 기준 -- 위 불량률과 다른 값이다.
                    "expected_yield_pct": f.expected_yield_pct,
                    # WE-2/WE-3: 행 선정 근거(파레토 기여율)와, 최악 10%
                    # wafer에서의 계측률(WL의 MNAR 계측 편향과 같은 지표).
                    "contribution_pct": f.contribution_pct,
                    "worst_decile_measurement_rate_pct": f.worst_decile_measurement_rate_pct,
                    "mnar_gap_pp": f.mnar_gap_pp,
                }
                for f in table.items
            ],
            "target_provenance": hydrated.provenance.as_dict(),
        }
    )


@router.get("/screening/pareto", response_model=ParetoRankingResponse)
def get_screening_pareto(dataset: str = "train", target: str = "Y1", top_n: int = PARETO_TOP_N) -> dict[str, Any]:
    """The top-eps2 Pareto ranking for one target across the full
    R+D+Config pool -- the shared source for both the training tab's
    screening table and the root-cause tab's Pareto chart, which show
    different counts (5 vs 10) and so must each pass their own `top_n`
    explicitly rather than rely on this default drifting under them.
    Not gated by FDR significance: every returned row is included
    regardless of p-value, tiered by confidence_tier instead of
    filtered out. `n80` reports the rank (across the FULL pool, not
    just `top_n`) at which cumulative contribution first reaches 80%,
    so the caller can render "80%에 도달하지 못했습니다 -- N개 더 필요"
    without a second request.
    """
    return _pareto_payload(dataset, target, top_n)


def _alarm_factors(
    train_df,
    schema,
    train_dataset_id: str,
    provenance: Any | None = None,
) -> tuple[list[ParetoFactor], list[str]]:
    """Per-target alarm-eligible factors: every BH-FDR-significant factor
    (see select_fdr_significant_factors's docstring -- deliberately kept
    unchanged so the golden 19-alarm-wafer count doesn't move). Screen
    display no longer gates on significance, but alarm generation still
    does.

    B-3: reuses `_cached_ranked_rows(train_dataset_id, target)` instead of
    `select_fdr_significant_factors` (which reruns the full 88-factor
    ANOVA+FDR scoring every call with identical default parameters) --
    same rows, same significance flags, just not recomputed from scratch.
    """
    factors: list[ParetoFactor] = []
    no_alarm_factor: list[str] = []
    for target in schema.target_cols:
        rows = (
            _ranked_rows_for_provenance(train_dataset_id, target, provenance)
            if provenance is not None
            else _cached_ranked_rows(train_dataset_id, target)
        )
        target_factors = [_row_to_factor(train_df, target, row) for row in rows if row["significant"]]
        if not target_factors:
            no_alarm_factor.append(target)
            continue
        factors.extend(target_factors)
    return factors, no_alarm_factor


def _control_range_dict(control_range) -> dict[str, Any]:
    data = vars(control_range).copy()
    data["reference_lines"] = [vars(line) for line in control_range.reference_lines]
    return round_floats(data)


@router.get("/control-ranges", response_model=ControlRangeListResponse)
def get_control_ranges(dataset: str = "train") -> dict[str, Any]:
    hydrated = _hydrated_targets_or_409(dataset)
    train_df = hydrated.dataframe
    schema = parse_schema(train_df)
    factors, no_significant = _alarm_factors(train_df, schema, dataset, hydrated.provenance)
    items = [_control_range_dict(compute_control_range(train_df, factor)) for factor in factors]
    return {
        "train_dataset_id": dataset,
        "items": items,
        "no_significant_factor_targets": no_significant,
    }


MEASURED_IDS_CACHE_PAIRS = 8


@lru_cache(maxsize=MEASURED_IDS_CACHE_PAIRS)
def _measured_ids_for_alarm_factors_versioned(
    train: str,
    eval: str,
    train_version: str,
    eval_version: str,
    model_id: str,
    model_version: str,
    hydration_version: str,
) -> frozenset[str]:
    """알람 판정에 쓸 "선정 인자 계측 여부" -- 기존 unmeasured_id_set과
    동일한 기준(FDR-유의 인자 중 하나라도 계측)이다 (spec 사전 알람 로그
    전면 개편 §B-2 "미계측").

    B-3: `/alarms`와 `/alarms/predictions` 둘 다 이 경로를 지나는데, 매
    호출마다 5타깃 유의 인자 선정(`_alarm_factors`) + 관리한계 계산 +
    eval 전수 판정을 다시 돌렸다. (train, eval) 쌍마다 한 번만 계산해
    캐시하고, 내부는 `_cached_ranked_rows`를 재사용하는 `_alarm_factors`를
    그대로 쓴다.
    """
    hydrated = _hydrated_targets_or_409(train)
    train_df = hydrated.dataframe
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)
    factors, _ = _alarm_factors(train_df, schema, train, hydrated.provenance)
    control_ranges = [compute_control_range(train_df, factor) for factor in factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)
    return frozenset(v.lot_wafer_id for v in verdicts if v.status != "unmeasured")


def _measured_ids_for_alarm_factors(train: str, eval: str) -> frozenset[str]:
    train_view = _hydrated_targets_or_409(train)
    provenance = train_view.provenance
    eval_version = get_dataset_registry().content_version(eval)
    return _measured_ids_for_alarm_factors_versioned(
        train,
        eval,
        provenance.dataset_version,
        eval_version,
        provenance.model_id or "measured-only",
        provenance.model_version or "none",
        TARGET_HYDRATION_VERSION,
    )


def _scored_wafers(
    train: str,
    eval: str,
    eval_df: pd.DataFrame,
    *,
    target: float = alarm_gbdt.DEFAULT_TARGET_YIELD,
    sensitivity: float = alarm_gbdt.DEFAULT_SENSITIVITY,
    max_step: int | None = None,
) -> tuple[list, float | None, bool, dict[str, Any]]:
    """공유 파이프라인 -- `get_alarms`와 `compute_alarm_notification_items`가
    똑같이 게이트를 적용하도록 한 곳에 모았다 (spec 알람 신뢰도 게이트
    §A-2: "신뢰할 수 없으면 0건이 된다"가 어디서 알람을 불러오든 항상
    성립해야 한다). 사전 알람 로그 전면 개편 이후로는 목표 수율/민감도
    기준 classify_wafer를 쓴다 -- 이 두 파라미터를 직접 조절하는 화면이
    없는 호출자(원인 분석 탭의 알람 삼각형 마커, 알림 발송)는 기본값
    (목표 88.0·민감도 0.2)을 그대로 쓴다.

    게이트 미달이어도 정상/판별불가는 여전히 계산해야 하므로(spec §B-4)
    부트스트랩 예측 자체는 건너뛰지 않는다 -- 이전(품질 게이트 시
    예측조차 안 함)과 달라진 부분이다.

    지시서 작업 2(특정 스텝까지의 정보만으로 예측): `max_step`이 주어지면
    alarm_gbdt의 (마스킹된) 앙상블 예측을 target_hydration의 실측/모델
    보강 뷰로 덮어쓰지 않는다 -- 그 뷰는 "Step N까지만 진행됐다"는 상태를
    전혀 모르므로, 그대로 두면 max_step 마스킹 자체가 화면에 아무 영향을
    주지 못한다.
    """
    auc_lo, gate_passed = _auc_gate(train, eval)

    prediction = _cached_bootstrap_prediction(train, eval, max_step)
    if prediction is None:
        prediction = _uncalibrated_hydrated_prediction(eval, eval_df)
        prediction, target_sources, provenance = _prediction_from_hydrated_targets(eval, eval_df, prediction)
    elif max_step is not None:
        target_sources = ["predicted"] * len(prediction.lot_wafer_id)
        provenance = None
    else:
        prediction, target_sources, provenance = _prediction_from_hydrated_targets(eval, eval_df, prediction)

    # 지시서 작업 3(스텝별 신뢰도 게이트) -- max_step 모드에서 그 스텝의
    # OOF AUC가 게이트 미만이면(또는 표본 부족으로 산출 불가면) 심각/
    # 위험은 주의로 낮춘다. 알람 자체를 막지는 않는다(gate_passed=False와
    # 다르다) -- 조기 예측은 본질적으로 정확도가 낮을 뿐이다.
    cap_at_caution = False
    if max_step is not None:
        bundle = _cached_bootstrap_models(train)
        step_gate_passed = alarm_gbdt.gate_for_step(bundle.step_auc_profile, max_step)[0] if bundle is not None else None
        cap_at_caution = step_gate_passed is not True

    measured_ids = _measured_ids_for_alarm_factors(train, eval)
    scored = alarm_gbdt.score_wafers(
        eval_df, prediction,
        target=target, sensitivity=sensitivity,
        gate_passed=True, measured_ids=measured_ids, target_sources=target_sources,
        cap_at_caution=cap_at_caution,
    )
    return scored, auc_lo, gate_passed, provenance


def _uncalibrated_hydrated_prediction(
    eval_dataset_id: str,
    raw_eval_df: pd.DataFrame,
) -> alarm_gbdt.BootstrapPrediction:
    """Build a display-only prediction when no alarm training Y exists.

    The interval is deliberately [0, 100] and the external AUC gate remains
    closed. This permits risk ranking/history without presenting an
    uncalibrated interval as trusted evidence.
    """
    hydrated = _hydrated_targets_or_409(eval_dataset_id)
    point = pd.to_numeric(hydrated.dataframe["Y"], errors="coerce").to_numpy(dtype=float)
    identifier = "Lot_Wafer_ID"
    lot_wafer_id = (
        raw_eval_df[identifier].astype(str).tolist()
        if identifier in raw_eval_df.columns
        else [str(index) for index in raw_eval_df.index]
    )
    aggregate = float(np.mean(point)) if len(point) else 0.0
    return alarm_gbdt.BootstrapPrediction(
        lot_wafer_id=lot_wafer_id,
        pred_mean=point,
        pred_lo=np.zeros(len(point), dtype=float),
        pred_hi=np.full(len(point), 100.0, dtype=float),
        conformal_q=None,
        coverage_target=alarm_gbdt.CONFORMAL_TARGET_COVERAGE,
        coverage_actual=None,
        holdout_oof_actual=None,
        holdout_oof_pred=None,
        conformal_q_agg=None,
        pred_agg_mean=aggregate,
        pred_agg_lo=None,
        pred_agg_hi=None,
    )


def _prediction_from_hydrated_targets(
    eval_dataset_id: str,
    raw_eval_df: pd.DataFrame,
    prediction: alarm_gbdt.BootstrapPrediction,
) -> tuple[alarm_gbdt.BootstrapPrediction, list[str], dict[str, Any]]:
    """Use the common measured/predicted Y view for alert risk scoring.

    The independent alarm model remains the source of calibrated interval
    width and AUC evidence. Its point estimate is replaced by the common Y
    view so missing-target datasets cannot diverge across analyses.
    """
    hydrated = _hydrated_targets_or_409(eval_dataset_id)
    if len(hydrated.dataframe) != len(prediction.pred_mean):
        raise TargetHydrationError("알람 예측 행 수와 타깃 보강 행 수가 일치하지 않습니다.")

    point = pd.to_numeric(hydrated.dataframe["Y"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(point).all():
        raise TargetHydrationError("보강 후에도 최종 수율(Y)이 비어 있어 위험도를 계산할 수 없습니다.")
    point = np.clip(point, 0.0, 100.0)
    if prediction.conformal_q is not None:
        width = np.full(len(point), float(prediction.conformal_q), dtype=float)
    else:
        width = np.maximum(
            np.asarray(prediction.pred_hi, dtype=float) - np.asarray(prediction.pred_mean, dtype=float),
            np.asarray(prediction.pred_mean, dtype=float) - np.asarray(prediction.pred_lo, dtype=float),
        )
    pred_lo = np.clip(point - width, 0.0, 100.0)
    pred_hi = np.clip(point + width, 0.0, 100.0)

    raw_y = (
        pd.to_numeric(raw_eval_df["Y"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "Y" in raw_eval_df.columns
        else pd.Series(np.nan, index=raw_eval_df.index)
    )
    raw_components = pd.DataFrame(
        {
            target: (
                pd.to_numeric(raw_eval_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if target in raw_eval_df.columns
                else pd.Series(np.nan, index=raw_eval_df.index)
            )
            for target in ("Y1", "Y2", "Y3", "Y4", "Y5")
        },
        index=raw_eval_df.index,
    )
    target_sources = [
        "measured" if pd.notna(y_value) else ("derived_measured" if components_complete else "predicted")
        for y_value, components_complete in zip(raw_y.tolist(), raw_components.notna().all(axis=1).tolist(), strict=True)
    ]
    return (
        replace(prediction, pred_mean=point, pred_lo=pred_lo, pred_hi=pred_hi),
        target_sources,
        hydrated.provenance.as_dict(),
    )


def _reason_for(score, eval_by_id, warning_lines) -> str:
    # spec §BC-2: 계측 없이 등급이 매겨진 wafer는 선정 인자 근거를 댈 수
    # 없다 -- 정상 사유 대신 이 문구를 쓰고(배지 구분 표기는 프런트가
    # 담당), compute_alarm_notification_items가 이 wafer를 자동 발송
    # 대상에서 뺀다.
    if not score.measured:
        return alarm_gbdt.NO_REASON_UNMEASURED
    row = None
    if eval_by_id is not None and score.lot_wafer_id in eval_by_id.index:
        match = eval_by_id.loc[score.lot_wafer_id]
        row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
    return (
        warning_line.build_alarm_reason(row, warning_lines)
        if row is not None and warning_lines
        else warning_line.NO_EXCEEDANCE_REASON
    )


@router.get("/alarms/history")
def get_alarm_snapshot_history(limit: int = 20) -> dict[str, Any]:
    """Immutable alert snapshots; later promotions never rewrite old rows."""
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    return {"items": store.list_alert_snapshots(limit)}


@router.get("/alarms", response_model=AlarmListResponse)
def get_alarms(
    train: str = "train",
    eval: str = "test",
    grade: str | None = None,
    target: float | None = None,
    sensitivity: float | None = None,
    max_step: int | None = Query(None, ge=1, le=30, description="이 스텝까지의 정보만으로 판정합니다."),
) -> dict[str, Any]:
    """알람 판정 GBDT 전환 (spec §A) + 민감도 슬라이더를 실제 트레이드오프로
    (spec §CA-1) -- 부트스트랩 앙상블로 예측한 최종 수율(Y)의 점추정이
    목표 수율 - 민감도 margin(%p) 아래인 wafer만 알람으로 낸다.

    `target`/`sensitivity`는 선택 파라미터다 -- 지시서: 원인 분석 탭의
    알람 삼각형 마커가 수율 예측 탭에서 저장한 값을 그대로 넘겨 두
    화면의 판정 기준을 일치시킨다. 생략하면(둘 다 또는 하나만) 그
    파라미터는 기본값(88.0/0.2)을 쓴다 -- 파라미터 없이 부르던 기존
    호출부(알림 발송 등)는 동작이 그대로다. 사전 알람 로그 화면 자체는
    `/alarms/predictions`에서 원시 예측치를 받아 클라이언트가 실시간으로
    재분류하므로 이 라우트를 거치지 않는다.

    알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-2) -- train→eval 전이
    AUC 하한이 0.65 미만이면 알람을 아예 내지 않는다.

    지시서 작업 2(특정 스텝까지의 정보만으로 예측): `max_step`을 생략하면
    (기본값 None) 기존 동작과 완전히 같다.
    """
    # train 존재 여부는 _scored_wafers -> _auc_gate/_cached_bootstrap_prediction이
    # 검증한다(둘 다 첫 줄에서 _dataframe_or_404(train)을 부른다) -- 여기서는
    # eval_df 자체가 필요한 곳(warning_lines 조회, id_column 인덱싱)에만 fetch한다.
    eval_df = _dataframe_or_404(eval)

    resolved_target = target if target is not None else alarm_gbdt.DEFAULT_TARGET_YIELD
    resolved_sensitivity = sensitivity if sensitivity is not None else alarm_gbdt.DEFAULT_SENSITIVITY
    # _scored_wafers/score_wafers는 캐시되지 않는다 -- 매 호출마다 새로
    # 계산하므로 target/sensitivity를 캐시 키에 빠뜨릴 위험 자체가 없다
    # (부트스트랩 예측 자체를 캐시하는 _cached_bootstrap_prediction은
    # target/sensitivity와 무관한 원시 예측치라 그대로 재사용해도 안전
    # 하다 -- 분류 임계값은 이후 score_wafers에서 매번 새로 적용된다).
    scored, auc_lo, gate_passed, provenance = _scored_wafers(
        train, eval, eval_df, target=resolved_target, sensitivity=resolved_sensitivity, max_step=max_step
    )
    alarm_scored = [s for s in scored if s.grade in ("심각", "위험", "주의")]

    warning_lines = _cached_all_warning_lines(train)
    id_column = "Lot_Wafer_ID"
    eval_by_id = eval_df.set_index(id_column, drop=False) if id_column in eval_df.columns else None

    items: list[dict[str, Any]] = []
    for score in alarm_scored:
        if grade and score.grade != grade:
            continue
        items.append(
            {
                "lot_wafer_id": score.lot_wafer_id,
                "lot_id": score.lot_id,
                "grade": score.grade,
                "risk_percentile": score.risk_percentile,
                "reason": _reason_for(score, eval_by_id, warning_lines),
                "target_source": score.target_source,
            }
        )
    items.sort(key=lambda item: item["risk_percentile"])

    alarm_total = len(alarm_scored)
    evaluated_total = len(eval_df)
    alarm_share_warning = (
        evaluated_total > 0 and (alarm_total / evaluated_total) > alarm_gbdt.ALARM_SHARE_WARNING_THRESHOLD
    )

    return round_floats(
        {
            "train_dataset_id": train,
            "eval_dataset_id": eval,
            "items": items,
            "total": len(items),
            "alarm_total": alarm_total,
            "evaluated_total": evaluated_total,
            "alarm_share_warning": alarm_share_warning,
            "auc_lower_bound": auc_lo,
            "auc_gate_passed": gate_passed,
            "auc_gate_threshold": alarm_gbdt.AUC_GATE,
            "target_provenance": provenance,
            "external_delivery_suppressed_reason": (
                None if gate_passed else (
                    f"AUC 하한 {auc_lo:.3f}가 발송 기준 {alarm_gbdt.AUC_GATE:.2f} 미만입니다."
                    if auc_lo is not None else "AUC 하한을 산출할 수 없어 외부 알림을 차단했습니다."
                )
            ),
        }
    )


def compute_alarm_notification_items(
    train: str,
    eval: str,
    *,
    target: float = alarm_gbdt.DEFAULT_TARGET_YIELD,
    sensitivity: float = alarm_gbdt.DEFAULT_SENSITIVITY,
) -> list[dict[str, Any]] | None:
    """알림 발송(src.notifications.dispatch)이 쓰는 알람 목록 -- `get_alarms`와
    같은 파이프라인(게이트)을 그대로 재사용한다. `target`/`sensitivity`는
    호출부(notify.py)가 수율 예측 탭에 저장된 alarms_state.payload에서
    읽어 넘긴다 -- 생략하면(저장된 조회가 없을 때) 기본값(88.0/0.2)을
    쓴다. 데이터셋을 찾을 수 없으면 예외 대신 None을 반환한다 -- 알림
    발송은 best-effort라 404로 스케줄러 잡 전체를 죽이면 안 된다.

    spec §BC-2: 계측 없이(measured=False) 등급이 매겨진 wafer는 자동
    발송 대상에서 제외한다 -- 사유를 댈 수 없는 알람을 폰으로 보내면
    받는 사람이 조치할 수 없다. `get_alarms`(화면 표시)는 이 wafer를
    "사유 제시 불가" 표시와 함께 그대로 보여준다 -- 여기서만 거른다.
    """
    try:
        # train 존재 여부는 _dataframe_or_404을 직접 결과 없이 불러서만
        # 검증한다 -- 실제 사용은 eval_df뿐이다(_scored_wafers 내부에서
        # train을 다시 조회한다).
        _dataframe_or_404(train)
        eval_df = _dataframe_or_404(eval)
    except HTTPException:
        return None

    scored, _auc_lo, gate_passed, _provenance = _scored_wafers(
        train, eval, eval_df, target=target, sensitivity=sensitivity
    )
    if not gate_passed:
        return []
    alarm_scored = [s for s in scored if s.grade in ("심각", "위험", "주의") and s.measured]

    warning_lines = _cached_all_warning_lines(train)
    id_column = "Lot_Wafer_ID"
    eval_by_id = eval_df.set_index(id_column, drop=False) if id_column in eval_df.columns else None

    items: list[dict[str, Any]] = []
    for score in alarm_scored:
        items.append(
            {
                "lot_wafer_id": score.lot_wafer_id,
                "lot_id": score.lot_id,
                "grade": score.grade,
                "risk_percentile": score.risk_percentile,
                "reason": _reason_for(score, eval_by_id, warning_lines),
                "target_source": score.target_source,
            }
        )
    return items


@router.get("/alerts/ranking", response_model=YieldPredictionResponse)
def get_alerts_ranking(train: str = "train", eval: str = "test") -> dict[str, Any]:
    """VA~VD: 수율 예측 순위 목록 -- y(=100 − Σ Y1~Y5, RC-3 실측 우선
    규칙으로 채운 뒤 재계산) 오름차순 전체(신뢰도==0 웨이퍼는 제외)를
    내려보낸다. 상위 10/전체 보기·검색·정렬 9종은 프런트가 이 목록
    위에서 수행한다(VB그룹) -- `top_n`으로 서버가 미리 자르면 검색이
    상위 10 밖의 웨이퍼를 찾지 못한다(VB-4: "검색 중에는 상위 10 제한을
    해제한다").
    """
    from src.analysis.yield_prediction import build_yield_prediction_table

    train_df = _dataframe_or_404(train)
    eval_view = _hydrated_targets_or_409(eval)
    eval_df = _dataframe_or_404(eval)

    table = build_yield_prediction_table(
        train_df,
        eval_df,
        eval_view.dataframe,
        dataset_id=eval,
        train_dataset_id=train,
        train_dataset_version=get_dataset_registry().content_version(train),
    )
    return {
        "train_dataset_id": train,
        "eval_dataset_id": eval,
        "total_wafers": table.total_wafers,
        "candidates": [
            {
                "lot_wafer_id": c.lot_wafer_id,
                "lot_id": c.lot_id,
                "y": c.y,
                "y_components": c.y_components,
                "cells": c.cells,
                "core_factors": {
                    target: {
                        "feature": cell.feature,
                        "contribution_pct": cell.contribution_pct,
                        "rank_used": cell.rank_used,
                        "factor_value": cell.factor_value,
                    }
                    for target, cell in c.core_factors.items()
                },
                "reliability": {
                    "count": c.reliability.count,
                    "measured": [{"target": t, "feature": f} for t, f in c.reliability.measured],
                    "unmeasured": [{"target": t, "feature": f} for t, f in c.reliability.unmeasured],
                },
                "recommendation": {
                    "text": c.recommendation.text,
                    "adjustable_targets": list(c.recommendation.adjustable_targets),
                    "measurement_gap_targets": list(c.recommendation.measurement_gap_targets),
                },
            }
            for c in table.candidates
        ],
        "unmeasured_wafer_ids": table.unmeasured_wafer_ids,
        "unmeasured_count": len(table.unmeasured_wafer_ids),
        "fallback_summary": {
            "rank_counts": {str(rank): count for rank, count in table.fallback_summary.rank_counts.items()},
            "none_count": table.fallback_summary.none_count,
            "total_combinations": table.fallback_summary.total_combinations,
        },
        # WB/WC/WD: 모니터링 홈 요약 카드·모드별 손실 막대·수율 분포
        # 히스토그램이 그대로 쓰는 서버 계산 결과.
        "yield_summary": {
            "predicted_mean": table.summary.predicted_mean,
            "predicted_min": table.summary.predicted_min,
            "predicted_max": table.summary.predicted_max,
            "bottom_n": table.summary.bottom_n,
            "bottom_mean": table.summary.bottom_mean,
            "judgeable_count": table.summary.judgeable_count,
            "total_wafers": table.summary.total_wafers,
            "histogram": [
                {
                    "label": b.label,
                    "lo": None if b.lo == float("-inf") else b.lo,
                    "hi": None if b.hi == float("inf") else b.hi,
                    "judgeable_count": b.judgeable_count,
                    "not_judgeable_count": b.not_judgeable_count,
                }
                for b in table.summary.histogram
            ],
            "mode_loss": [
                {
                    "target": m.target,
                    "feature": m.feature,
                    "avg_loss_pct": m.avg_loss_pct,
                    "train_avg_loss_pct": m.train_avg_loss_pct,
                    "contribution_pct": m.contribution_pct,
                }
                for m in table.summary.mode_loss
            ],
        },
        "target_provenance": eval_view.provenance.as_dict(),
    }


@router.get("/alarms/predictions", response_model=AlertsDataResponse)
def get_alarms_predictions(
    train: str = "train",
    eval: str = "test",
    max_step: int | None = Query(None, ge=1, le=30, description="이 스텝까지의 정보만으로 예측합니다."),
) -> dict[str, Any]:
    """사전 알람 로그 전면 개편 (spec §A-3) -- 등급을 서버가 매겨 내려주지
    않고, wafer별 원시 예측치(pred_mean/pred_lo/pred_hi)와 목표 수율 조정에
    필요한 학습 Y 분위수를 내려준다. 목표 수율·민감도를 조절할 때마다 이
    응답을 다시 받아올 필요가 없다 -- frontend의 classifyWafer가
    `src.analysis.alarm_gbdt.classify_wafer`와 동일한 공식으로 클라이언트
    에서 즉시 재분류한다(§A-3: "API를 재호출하지 마라").

    민감도 슬라이더를 실제 트레이드오프로 (spec §CA-4) -- 홀드아웃 OOF
    (실제 Y, 예측값) 층화 샘플도 함께 내려 클라이언트가 슬라이더를 움직일
    때마다 정밀도·재현율을 즉시 추정하게 한다(`_cached_bootstrap_prediction`이
    이미 계산해 캐시한 것을 그대로 읽을 뿐이라 이 요청에서 추가 계산이
    없다).

    지시서 작업 2(특정 스텝까지의 정보만으로 예측) -- `max_step`을
    생략하면(기본값 None) 기존 동작과 완전히 같다. 주어지면 alarm_gbdt의
    (마스킹된) 앙상블 예측을 그대로 쓰고 target_hydration의 실측/모델
    보강 뷰로 덮어쓰지 않는다 -- 그 뷰는 스텝 진행 상태를 모르므로 덮어쓰면
    max_step이 화면에 아무 영향을 주지 못한다(`_scored_wafers`와 같은
    이유).
    """
    eval_df = _dataframe_or_404(eval)
    train_view = _hydrated_targets_or_409(train)
    train_df = train_view.dataframe

    auc_lo, gate_passed = _auc_gate(train, eval)
    prediction = _cached_bootstrap_prediction(train, eval, max_step)
    target_sources: list[str] = []
    eval_provenance: dict[str, Any] | None = None
    if prediction is None:
        prediction = _uncalibrated_hydrated_prediction(eval, eval_df)
        prediction, target_sources, eval_provenance = _prediction_from_hydrated_targets(eval, eval_df, prediction)
    elif max_step is not None:
        target_sources = ["predicted"] * len(prediction.lot_wafer_id)
        eval_provenance = None
    else:
        prediction, target_sources, eval_provenance = _prediction_from_hydrated_targets(eval, eval_df, prediction)

    # 지시서 작업 3(스텝별 신뢰도 게이트) -- max_step이 주어졌을 때만
    # 계산한다(그 외에는 전체 스텝 기준 AUC 게이트만 쓴다). bundle은
    # _cached_bootstrap_prediction이 이미 _cached_bootstrap_models로
    # 한 번 캐시해 둔 것을 그대로 읽으므로 추가 계산이 없다.
    step_auc: float | None = None
    step_auc_gate_passed: bool | None = None
    if max_step is not None:
        bundle = _cached_bootstrap_models(train)
        if bundle is not None:
            step_auc_gate_passed, step_auc = alarm_gbdt.gate_for_step(bundle.step_auc_profile, max_step)

    # "미계측" 정의는 그대로 둔다 (spec §B-2) -- 선정 인자가 하나도
    # 계측되지 않은 wafer는 예측이 있어도 신뢰할 수 없어 등급을 매기지
    # 않는다.
    measured_id_set = _measured_ids_for_alarm_factors(train, eval)

    warning_lines = _cached_all_warning_lines(train)
    id_column = "Lot_Wafer_ID"
    eval_by_id = eval_df.set_index(id_column, drop=False) if id_column in eval_df.columns else None

    predictions: list[dict[str, Any]] = []
    if prediction is not None:
        lot_column = "Lot_ID"
        lot_ids = (
            eval_df[lot_column].astype(str).where(eval_df[lot_column].notna(), None).tolist()
            if lot_column in eval_df.columns
            else [None] * len(prediction.lot_wafer_id)
        )
        for i, wafer_id in enumerate(prediction.lot_wafer_id):
            measured = wafer_id in measured_id_set
            reason = None
            if measured and eval_by_id is not None and wafer_id in eval_by_id.index:
                match = eval_by_id.loc[wafer_id]
                row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
                if warning_lines:
                    reason = warning_line.build_alarm_reason(row, warning_lines)
            predictions.append(
                {
                    "lot_wafer_id": wafer_id,
                    "lot_id": lot_ids[i],
                    "measured": measured,
                    "pred_mean": float(prediction.pred_mean[i]),
                    "pred_lo": float(prediction.pred_lo[i]),
                    "pred_hi": float(prediction.pred_hi[i]),
                    "reason": reason,
                    "target_source": target_sources[i],
                }
            )

    y_train = pd.to_numeric(train_df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce").dropna()
    has_y = len(y_train) > 0

    # B-1(과거)/§CA-4(현재): factor_bands/measurement_bias(개별 인자
    # 88개 전수 밴드)는 되살리지 않는다 -- 렌더하는 화면이 없다. 반면
    # 홀드아웃 OOF (실제, 예측) 쌍은 §CA-4가 요구하는 정밀도·재현율
    # 실시간 추정의 재료라 되살렸다 -- 단, `_cached_bootstrap_prediction`
    # 안에서 conformal q와 함께 "한 번만" 계산해 캐시된 것을 읽을 뿐,
    # 이 요청에서 GroupKFold를 다시 돌리지 않는다(이전에 문제였던 매
    # 요청 무캐시 재계산과 다르다).
    holdout_oof_actual = (
        prediction.holdout_oof_actual.tolist() if prediction is not None and prediction.holdout_oof_actual is not None else []
    )
    holdout_oof_predicted = (
        prediction.holdout_oof_pred.tolist() if prediction is not None and prediction.holdout_oof_pred is not None else []
    )

    return round_floats(
        {
            "train_dataset_id": train,
            "eval_dataset_id": eval,
            "total_wafers": len(eval_df),
            "train_y_min": float(y_train.min()) if has_y else 0.0,
            "train_y_max": float(y_train.max()) if has_y else 0.0,
            "train_y_median": float(y_train.median()) if has_y else 0.0,
            "train_y_p1": float(y_train.quantile(0.01)) if has_y else 0.0,
            "train_y_p99": float(y_train.quantile(0.99)) if has_y else 0.0,
            "predictions": predictions,
            "holdout_oof_actual": holdout_oof_actual,
            "holdout_oof_predicted": holdout_oof_predicted,
            "auc_lower_bound": auc_lo,
            "auc_gate_passed": gate_passed,
            "display_prediction_allowed": prediction is not None,
            "auc_gate_threshold": alarm_gbdt.AUC_GATE,
            "interval_coverage_target": prediction.coverage_target if prediction is not None else alarm_gbdt.CONFORMAL_TARGET_COVERAGE,
            "interval_coverage_actual": prediction.coverage_actual if prediction is not None else None,
            "interval_conformal_q": prediction.conformal_q if prediction is not None else None,
            "interval_conformal_q_agg": prediction.conformal_q_agg if prediction is not None else None,
            "effective_max_step": max_step,
            "max_step_auc": step_auc,
            "max_step_auc_gate_passed": step_auc_gate_passed,
            "target_provenance": eval_provenance,
            "external_delivery_suppressed_reason": (
                None if gate_passed else (
                    f"AUC 하한 {auc_lo:.3f}가 발송 기준 {alarm_gbdt.AUC_GATE:.2f} 미만입니다."
                    if auc_lo is not None else "AUC 하한을 산출할 수 없어 외부 알림을 차단했습니다."
                )
            ),
        }
    )


def _build_report_payload(dataset: str) -> dict[str, Any]:
    """The one function backing both /api/analysis/report (JSON download)
    and /api/analysis/context (SUNI chatbot context) -- same dict, so the
    chatbot never narrates a number the download button wouldn't also show.
    """
    registry = get_dataset_registry()
    train_view = _hydrated_targets_or_409(dataset)
    eval_view = _hydrated_targets_or_409(REPORT_EVAL_DATASET_ID)
    train_meta = registry.get_summary(dataset) or {}
    eval_meta = registry.get_summary(REPORT_EVAL_DATASET_ID) or {}
    report = build_analysis_report(
        train_view.dataframe,
        eval_view.dataframe,
        train_dataset_id=dataset,
        eval_dataset_id=REPORT_EVAL_DATASET_ID,
        train_meta=train_meta,
        eval_meta=eval_meta,
        app_version=APP_VERSION,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    report.setdefault("meta", {})["target_provenance"] = {
        "train": train_view.provenance.as_dict(),
        "eval": eval_view.provenance.as_dict(),
    }
    return report


@router.get("/analysis/report", response_model=AnalysisReportResponse)
def get_analysis_report(dataset: str = "train", *, response: Response) -> dict[str, Any]:
    """Full JSON analysis report -- always denominated by the full
    R+D+Config pool, matching the screen now that the R/D/Config split
    view has been removed entirely (see build_analysis_report's
    docstring for why the factor list and the alarm list use different,
    deliberately non-interchangeable factor sets).
    """
    response.headers["Content-Disposition"] = f'attachment; filename="analysis_report_{dataset}.json"'
    return _build_report_payload(dataset)


@router.get("/analysis/measurement-expansion", response_model=MeasurementExpansionResponse)
def get_measurement_expansion(dataset: str = "train") -> dict[str, Any]:
    """'계측 확대 권고' 카드 (spec 문구 전수 검토 PART B) -- 원인 분석 탭이
    "원인 분석 실행" 직후 한 번만 호출해 결과를 상태에 저장한다 (spec §B-7:
    "카드를 열 때마다 재계산하지 마라").

    JSON 보고서와 달리 eval을 고정된 REPORT_EVAL_DATASET_ID("test")로 두지
    않고 선택된 데이터셋 자기 자신을 판정 대상으로 삼는다. 이 카드가 답하는
    질문은 "이 데이터셋 자체의 계측을 늘리면 어떻게 되는가"이므로, 인자
    구성이 아예 다른 다른 데이터셋(예: 업로드 데이터셋의 특정 인자가
    test.csv에는 없는 컬럼인 경우)을 판정 기준으로 쓰면 그 데이터셋의
    실제 계측률과 무관하게 전량 "판정불가"로 나와 §B-6 축소 조건이 항상
    빗나간다 -- 원인 분석 탭 자체가 데이터셋 선택기 하나뿐이라는 점과도
    맞는다.
    """
    hydrated = _hydrated_targets_or_409(dataset)
    train_df = hydrated.dataframe
    eval_df = train_df
    schema = parse_schema(train_df)

    # 타깃당 전체 R+D+Config 풀 스코어링(88개 인자 x ANOVA)을 한 번씩만
    # 돌려 재사용한다 -- select_primary_factor/select_fdr_significant_factors/
    # score_all_factors를 각각 부르면 같은 스코어링을 3번 반복하게 되어
    # (원인 분석 실행이 이미 호출한 /api/screening/pareto와도 별개로) 이
    # 카드 하나 때문에 원인 분석 실행이 수십 초 느려진다. `_cached_ranked_rows`는
    # (dataset, target) 기준 프로세스 전역 캐시라 Pareto 화면이 이미 조회한
    # 타깃이면 사실상 즉시 반환된다.
    rows_by_target: dict[str, list[dict]] = {
        target: list(_ranked_rows_for_provenance(dataset, target, hydrated.provenance))
        for target in schema.target_cols
    }

    # "판정 가능 여부"(조치 불가/추가 판정)는 알람 목록과 동일한 FDR-유의
    # 인자 전체 집합으로 판단한다 -- get_alarms_predictions가 쓰는 것과
    # 같은 개념(_alarm_factors). 타깃마다 1위 인자 하나만 쓰면(select_primary_factor)
    # 여러 타깃이 같은 인자를 1위로 뽑는 데이터셋(치우친 스코어링 결과)에서
    # 사실상 서로 다른 컬럼 1개만 보는 셈이 되어, 그 인자 하나의 계측률만으로
    # "조치 불가"가 결정돼 데이터셋 전체 계측률과 동떨어진 값이 나온다.
    judgment_factors: list[ParetoFactor] = [
        _row_to_factor(train_df, target, row)
        for target, rows in rows_by_target.items()
        for row in rows
        if row["significant"]
    ]
    control_ranges = [compute_control_range(train_df, factor) for factor in judgment_factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)
    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]
    total_wafers = len(verdicts)

    # §B-6 축소 조건은 여기서 이미 판정 가능하다 (unmeasured_ids까지만
    # 있으면 됨) -- 카드가 어차피 한 줄로 축소될 데이터셋(계측률이 충분한
    # 경우)에서 아래 권장구간 계산(SPC/ML 부트스트랩, 인자 수가 많으면
    # 수십 초)까지 돌리는 건 낭비다.
    show_full_card = total_wafers > 0 and (len(unmeasured_ids) / total_wafers) >= MIN_ACTION_BLOCKED_SHARE

    if not show_full_card:
        summary = compute_measurement_expansion(
            train_df, eval_df, {}, {}, {}, classify_measured_bands(
                train_df, eval_df, alarm_ids, normal_ids, unmeasured_ids, [], [], dataset_id=dataset
            ), [], total_wafers=total_wafers,
        )
    else:
        bands = classify_measured_bands(
            train_df, eval_df, alarm_ids, normal_ids, unmeasured_ids, judgment_factors, control_ranges,
            dataset_id=dataset,
        )

        # B-3 인자별 우선순위 표는 스펙 예시("Step1_D1 -> Y3")대로 타깃당
        # 1위 인자 하나씩, 5행으로 보여준다 -- 위 판정 가능 여부 집합과는
        # 별개다.
        primary_factors: dict[str, ParetoFactor] = {
            target: _row_to_factor(train_df, target, rows[0]) for target, rows in rows_by_target.items() if rows
        }

        # 인자별 권장구간(SPC/ML 채택 방식)만 필요하다 -- 개선 권장 목록
        # (per-wafer 행)은 삭제됐으므로 compute_factor_recommendation을
        # 직접 부른다 (spec 알람 신뢰도 게이트 §B-2).
        factor_summaries: dict[str, FactorRecommendation] = {}
        for target, factor in primary_factors.items():
            control_range = compute_control_range(train_df, factor)
            factor_summary = compute_factor_recommendation(train_df, factor, control_range, dataset_id=dataset)
            if factor_summary is not None:
                factor_summaries[target] = factor_summary

        judgment_features = [factor.feature for factor in judgment_factors]
        summary = compute_measurement_expansion(
            train_df,
            eval_df,
            rows_by_target,
            primary_factors,
            factor_summaries,
            bands,
            judgment_features,
            total_wafers=total_wafers,
        )

    return round_floats(
        {
            "train_dataset_id": dataset,
            "eval_dataset_id": dataset,
            "action_blocked_wafers": summary.action_blocked_wafers,
            "total_wafers": summary.total_wafers,
            "additional_judged": summary.additional_judged,
            "action_target": summary.action_target,
            "expected_yield_gain_pp": summary.expected_yield_gain_pp,
            "show_full_card": summary.show_full_card,
            "priorities": [
                {
                    "feature": p.feature,
                    "target": p.target,
                    "measurement_rate": p.measurement_rate,
                    "recommendation": p.recommendation,
                    "reason": p.reason,
                    "additional_judged": p.additional_judged,
                    "yield_contribution_pp": p.yield_contribution_pp,
                }
                for p in summary.priorities
            ],
            "new_factor_discoveries": [
                {"feature": d.feature, "target": d.target, "kind": d.kind} for d in summary.new_factor_discoveries
            ],
            "target_provenance": hydrated.provenance.as_dict(),
        }
    )


RELIABILITY_CACHE_PAIRS = 8


@_single_flight
@lru_cache(maxsize=RELIABILITY_CACHE_PAIRS)
def _cached_reliability(dataset_id: str, eval_dataset_id: str) -> dict[str, Any]:
    """spec §E: 5개 지표 100점 -- (train, eval) 쌍마다 한 번만 계산해
    캐시한다 (GroupKFold 5-fold 검증이 주 비용이다).

    AUC 지표는 알람 신뢰도 게이트 §A-1/§A-2가 도입된 뒤로 train 자기
    자신만의 self-CV가 아니라 **선택된 eval에 대한 전이 AUC**다
    (`_cached_transfer_auc_folds`, `/api/alarms`와 동일한 값을 공유) --
    같은 train이라도 eval 분포가 다르면 다른 값이 나와야 게이트가 실제로
    잡아내려는 상황(분포 이동)을 신뢰도 점수에도 반영할 수 있다.
    """
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)

    fallback = target_fallback.select_analysis_targets(df)

    n_sig_features: set[str] = set()
    max_eps2: float | None = None
    for target in fallback.targets:
        rows = score_all_factors(df, schema, target)
        for row in rows:
            if row["significant"]:
                n_sig_features.add(row["feature"])
            max_eps2 = row["eps2"] if max_eps2 is None else max(max_eps2, row["eps2"])

    fold_aucs = _cached_transfer_auc_folds(dataset_id, eval_dataset_id)

    bad_threshold = (
        pd.to_numeric(df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce").quantile(alarm_gbdt.BAD_LABEL_QUANTILE)
        if alarm_gbdt.FINAL_YIELD_COLUMN in df.columns
        else None
    )
    bad_sample_size = (
        int((pd.to_numeric(df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce") <= bad_threshold).sum())
        if bad_threshold is not None
        else 0
    )

    # "판정 커버리지": 전체 R+D 인자 중 하나라도 계측된 행의 비율 --
    # GBDT는 값이 하나도 없어도 예측 자체는 내놓지만, 그런 행은 사실상
    # 모델의 편향(bias)만으로 예측한 것이라 신뢰성 점수에 반영한다.
    features = alarm_gbdt.feature_columns(schema)
    coverage_pct: float | None = None
    if features:
        present = [f for f in features if f in df.columns]
        if present:
            coverage_pct = float(df[present].notna().any(axis=1).mean() * 100.0)

    breakdown = reliability.compute_reliability(
        fold_aucs=list(fold_aucs) if fold_aucs else None,
        n_significant_factors=len(n_sig_features),
        max_eps2=max_eps2,
        n_train=len(df),
        coverage_pct=coverage_pct,
        bad_sample_size=bad_sample_size,
    )

    # 지시서 작업 4(분포 이동 감지) -- eval에는 실측 Y가 없는 것이 정상이라
    # AUC 게이트만으로는 "이 train으로 이 eval을 판정해도 되는가"를 항상
    # 잡아내지 못한다(실측 검증 결과: 같은 게이트 판정이 실제로는 반대로
    # 나온 조합이 있었다). train==eval이면 자기 자신과 비교라 의미가 없어
    # 건너뛴다.
    shift_report = None
    if features and dataset_id != eval_dataset_id:
        eval_df = _dataframe_or_404(eval_dataset_id)
        shift_report = distribution_shift.compute_distribution_shift(df, eval_df, features)

    return {
        "dataset_id": dataset_id,
        "eval_dataset_id": eval_dataset_id,
        "grade": breakdown.grade,
        "total_score": breakdown.total_score,
        "auc_lower_bound": breakdown.auc_lower_bound,
        "auc_score": breakdown.auc_score,
        "auc_gate_passed": breakdown.auc_gate_passed,
        "auc_gate_message": breakdown.auc_gate_message,
        "n_significant_factors": breakdown.n_significant_factors,
        "n_significant_score": breakdown.n_significant_score,
        "max_eps2": breakdown.max_eps2,
        "max_eps2_score": breakdown.max_eps2_score,
        "n_train": breakdown.n_train,
        "n_train_score": breakdown.n_train_score,
        "coverage_pct": breakdown.coverage_pct,
        "coverage_score": breakdown.coverage_score,
        "deduction_reasons": reliability.deduction_reasons(breakdown),
        "low_holdout_sample": breakdown.low_holdout_sample,
        "target_fallback_tier": fallback.tier,
        "target_fallback_message": fallback.message,
        "distribution_shift": (
            {
                "median": shift_report.median,
                "max": shift_report.max,
                "worst_feature": shift_report.worst_feature,
                "level": shift_report.level,
                "missing_rate_gap": shift_report.missing_rate_gap,
                "missing_rate_worst_feature": shift_report.missing_rate_worst_feature,
            }
            if shift_report is not None
            else None
        ),
    }


PREPROCESSING_COMPARISON_CACHE_DATASETS = 8


@lru_cache(maxsize=PREPROCESSING_COMPARISON_CACHE_DATASETS)
def _cached_preprocessing_comparison(dataset_id: str) -> dict[str, Any] | None:
    """설정 패널 신설 §E: 데이터셋마다 실측한 전처리 A/B/C 비교. 데이터셋당
    한 번만 계산해 캐시한다(§E-6: "탭을 열 때마다 재계산하지 마라") -- LOT
    70/30 홀드아웃 1회 × 3방식이라 4초 안팎이지만, 그 값을 매 탭 전환마다
    다시 치르지 않는다.
    """
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    features = alarm_gbdt.feature_columns(schema)
    registry = get_dataset_registry()
    summary = registry.get_summary(dataset_id)
    dataset_label = summary["original_filename"] if summary else dataset_id

    comparison = preprocessing_compare.compute_preprocessing_comparison(
        df, features, dataset_id=dataset_id, dataset_label=dataset_label
    )
    if comparison is None:
        return None
    return {
        "dataset_id": comparison.dataset_id,
        "dataset_label": comparison.dataset_label,
        "results": [
            {"mode": r.mode, "label": r.label, "r2": r.r2, "adopted": r.adopted} for r in comparison.results
        ],
        "winner": comparison.winner,
        "b_equals_c": comparison.b_equals_c,
        "holdout_note": comparison.holdout_note,
        "winner_note": comparison.winner_note,
    }


@router.get("/training/preprocessing-comparison", response_model=PreprocessingComparisonResponse)
def get_preprocessing_comparison(dataset: str) -> dict[str, Any]:
    result = _cached_preprocessing_comparison(dataset)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="전처리 비교를 계산하기에 데이터가 부족합니다.",
        )
    return round_floats(result)


@router.get("/analysis/reliability", response_model=ReliabilityResponse)
def get_reliability(dataset: str = "train", eval: str = "test") -> dict[str, Any]:
    return round_floats(_cached_reliability(dataset, eval))


@router.get("/analysis/context", response_model=AnalysisContextResponse)
def get_analysis_context(dataset: str = "train", *, response: Response) -> dict[str, Any]:
    """The SUNI chatbot's grounding context -- same underlying report as
    /api/analysis/report, but `alarms` is reshaped into `{summary,
    records}` (see build_chat_context's docstring) so the chatbot can
    answer a question about one specific wafer's alarm, not just the
    aggregate counts. No Content-Disposition, and explicitly non-cached
    since the frontend calls this right before /api/chat.
    """
    response.headers["Cache-Control"] = "no-store"
    return build_chat_context(_build_report_payload(dataset))


@router.get("/models/performance", response_model=ModelPerformanceResponse)
def get_model_performance(dataset: str | None = None) -> dict[str, Any]:
    del dataset  # Performance reflects whatever was last trained, not a live recompute.
    store = RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)
    metadata = get_latest_model_metadata(store)
    if metadata is None:
        return {"model_id": None, "trained_at": None, "source_filename": None, "targets": [], "final_yield": None}

    target_metrics = metadata.get("target_metrics") or {}
    targets = [
        {
            "target": target,
            "no_factor_available": bool(detail.get("no_factor_available")),
            "feature": detail.get("feature"),
            "kind": detail.get("kind"),
            "eps2": detail.get("eps2"),
            "contribution_pct": detail.get("contribution_pct"),
            "relation_shape": detail.get("relation_shape"),
            "optimal_center": detail.get("optimal_center"),
            "p_value": detail.get("p_value"),
            "confidence_tier": detail.get("confidence_tier"),
            "r2": detail.get("r2"),
            "rmse": detail.get("rmse"),
            "mae": detail.get("mae"),
            "n": detail.get("n"),
        }
        for target, detail in target_metrics.items()
    ]
    final_test_metrics = (metadata.get("final_y_metrics") or {}).get("test") or {}
    final_yield = (
        {
            "target": "Y",
            "no_factor_available": False,
            "feature": None,
            "kind": None,
            "eps2": None,
            "contribution_pct": None,
            "relation_shape": None,
            "optimal_center": None,
            "p_value": None,
            "confidence_tier": None,
            "r2": final_test_metrics.get("r2"),
            "rmse": final_test_metrics.get("rmse"),
            "mae": final_test_metrics.get("mae"),
            "n": final_test_metrics.get("n"),
        }
        if final_test_metrics
        else None
    )
    feature_columns = metadata.get("feature_columns") or []
    return {
        "model_id": metadata.get("model_id"),
        "trained_at": metadata.get("created_at"),
        "source_filename": metadata.get("source_filename"),
        "targets": targets,
        "final_yield": final_yield,
        "row_count": metadata.get("row_count"),
        "feature_count": len(feature_columns) or None,
    }
