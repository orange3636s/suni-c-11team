from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status

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
)
from api.settings import APP_VERSION, settings
from src.analysis import alarm_gbdt, preprocessing_compare, reliability, target_fallback, warning_line
from src.analysis.alarm_bands import classify_measured_bands, compute_factor_band
from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.llm_stats import per_factor_measurement_bias, summarize_measurement_bias
from src.analysis.measurement_expansion import MIN_ACTION_BLOCKED_SHARE, compute_measurement_expansion
from src.analysis.recommendations import FactorRecommendation, compute_factor_recommendation
from src.analysis.report import build_analysis_report, build_chat_context
from src.analysis.rounding import round_floats
from src.analysis.scatter import build_categorical_data, build_scatter_data
from src.analysis.screening.heatmap import HeatmapData, build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import (
    DEFAULT_TOP_N,
    ParetoFactor,
    confidence_tier,
    find_factor,
    score_all_factors,
    select_fdr_significant_factors,
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


@router.get("/screening/scatter", response_model=ScreeningScatterResponse)
def get_screening_scatter(dataset: str, target: str, feature: str) -> dict[str, Any]:
    df = _dataframe_or_404(dataset)
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃 결과가 없습니다.")
    # Resolves any of the 88 factors regardless of Pareto rank -- a heatmap
    # cell click can open a scatter for a factor outside the top 5.
    factor = find_factor(df, schema, target, feature)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")
    if factor.kind == "Config":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{feature}'은(는) Config(범주형) 인자입니다. /api/screening/scatter/categorical을 사용하세요.",
        )

    reference_model = _cached_reference_model(dataset)
    gbdt_features = alarm_gbdt.feature_columns(schema)
    data = build_scatter_data(df, df, factor, reference_model=reference_model, gbdt_features=gbdt_features)
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
        "relation_shape": data.relation_shape,
        "n": data.n,
        "axis": data.axis,
        "methods": round_floats(data.methods),
    }


@router.get("/screening/scatter/categorical", response_model=CategoricalScatterResponse)
def get_screening_scatter_categorical(dataset: str, target: str, feature: str) -> dict[str, Any]:
    """Per-category box-plot data for a Config factor. Config never gets a
    numeric normal-range (a category has no "range"), so this is a
    separate response shape from the numeric scatter endpoint above
    rather than an overloaded variant of it.
    """
    df = _dataframe_or_404(dataset)
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    if feature not in schema.config_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}'은(는) Config 인자가 아닙니다.")

    factor = find_factor(df, schema, target, feature)
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
    }


HEATMAP_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones
HEATMAP_METRICS = 2  # spearman + eps2

@lru_cache(maxsize=HEATMAP_CACHE_DATASETS * HEATMAP_METRICS)
def _cached_heatmap(dataset_id: str, metric: str) -> HeatmapData:
    # Cached per (dataset_id, metric), capped at the 2 most-recent
    # datasets (LRU-evicted) so this never grows unbounded in server
    # memory across a long-running process. Dataset content is immutable
    # once a dataset_id exists (uploads mint a fresh uuid; bundled files
    # are static). One heatmap per dataset now -- the R/D/Config split
    # view was removed, so there is no more per-kind cache dimension.
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    return build_heatmap(df, schema, metric=metric)  # type: ignore[arg-type]


@router.get("/screening/heatmap", response_model=HeatmapResponse)
def get_screening_heatmap(
    dataset: str = "train",
    metric: Literal["spearman", "eps2"] = "spearman",
) -> dict[str, Any]:
    """The one shared correlation heatmap (R+D x Y1~Y5, Config excluded --
    rho isn't defined for a category) used identically by both the
    training tab and the root-cause tab.
    """
    t0 = time.perf_counter()
    hits_before = _cached_heatmap.cache_info().hits
    heatmap = _cached_heatmap(dataset, metric)
    cached = _cached_heatmap.cache_info().hits > hits_before
    logger.info(
        "screening_heatmap %.1fms (cached=%s, dataset=%s, metric=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset, metric,
    )
    return {
        "dataset_id": dataset,
        "metric": metric,
        "features": heatmap.features,
        "targets": heatmap.targets,
        "values": heatmap.values,
        "n": heatmap.n,
        "q": heatmap.q,
        "significant": heatmap.significant,
        "tier": heatmap.tier,
        "scale": {"min": heatmap.scale["min"], "max": heatmap.scale["max"]},
        "excluded_configs": heatmap.excluded_configs,
    }


PARETO_CACHE_DATASETS = 2  # keep the most recent 2 datasets' worth, evict older ones
PARETO_TARGETS = 5  # Y1..Y5

@lru_cache(maxsize=PARETO_CACHE_DATASETS * PARETO_TARGETS)
def _cached_ranked_rows(dataset_id: str, target: str) -> tuple[dict, ...]:
    # Cached per (dataset_id, target), capped at the 2 most-recent
    # datasets (LRU-evicted): the training tab and the root-cause tab both
    # request the same (dataset, target) pair and must see byte-identical
    # results -- this cache is exactly what guarantees that, not just a
    # performance nicety. Dataset content is immutable once a dataset_id
    # exists (see the heatmap cache's docstring for why that's safe).
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    return tuple(_ranked_rows(df, schema, target, 0.05, 100, 20))


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


BOOTSTRAP_CACHE_PAIRS = 2


@lru_cache(maxsize=BOOTSTRAP_CACHE_PAIRS)
def _cached_bootstrap_prediction(train_dataset_id: str, eval_dataset_id: str):
    """§A-1 부트스트랩 앙상블 -- (train, eval) 쌍마다 한 번만 계산해
    캐시한다 (spec §A-1: "분석 실행 시 한 번만 수행하고 캐시한다")."""
    train_df = _dataframe_or_404(train_dataset_id)
    eval_df = _dataframe_or_404(eval_dataset_id)
    schema = parse_schema(train_df)
    features = alarm_gbdt.feature_columns(schema)
    if not features or alarm_gbdt.FINAL_YIELD_COLUMN not in train_df.columns:
        return None
    t0 = time.perf_counter()
    result = alarm_gbdt.fit_bootstrap_ensemble(train_df, eval_df, features)
    logger.info(
        "alarm_gbdt bootstrap ensemble fit %.1fms (train=%s, eval=%s)",
        (time.perf_counter() - t0) * 1000, train_dataset_id, eval_dataset_id,
    )
    return result


AUC_GATE_CACHE_PAIRS = 4


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


def _pareto_payload(dataset_id: str, target: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    hits_before = _cached_ranked_rows.cache_info().hits
    ranked = list(_cached_ranked_rows(dataset_id, target))
    cached = _cached_ranked_rows.cache_info().hits > hits_before
    logger.info(
        "screening_pareto %.1fms (cached=%s, dataset=%s, target=%s)",
        (time.perf_counter() - t0) * 1000, cached, dataset_id, target,
    )
    top = ranked[: DEFAULT_TOP_N]
    items = [
        {
            "feature": row["feature"],
            "kind": row["kind"],
            "step": row["step"],
            "eps2": row["eps2"],
            "p_value": row["p_value"],
            "q_value": row["q_value"],
            "significant": row["significant"],
            "confidence_tier": confidence_tier(row["eps2"], row["p_value"]),
            "n_observed": row["n_observed"],
            "contribution_pct": row["contribution_pct"],
            "cumulative_pct": row["cumulative_pct"],
        }
        for row in top
    ]
    n80 = next((index + 1 for index, row in enumerate(ranked) if row["cumulative_pct"] >= 80.0), None)
    # 차트 표시 규칙(spec §B)의 0개-타깃 안내 문구가 쓰는 전체 풀 집계치 --
    # 화면에 노출되는 top-5만으로는 "58건 중 FDR 통과 0건"을 계산할 수 없어
    # 여기서 전체 ranked 풀을 기준으로 함께 내려보낸다.
    fdr_pass_count = sum(1 for row in ranked if row["significant"])
    effect_size_pass_count = sum(1 for row in ranked if confidence_tier(row["eps2"], row["p_value"]) != "reference")
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
    }


@router.get("/screening/pareto", response_model=ParetoRankingResponse)
def get_screening_pareto(dataset: str = "train", target: str = "Y1") -> dict[str, Any]:
    """The fixed top-5-by-eps2 Pareto ranking for one target across the
    full R+D+Config pool -- the single shared source for both the
    training tab's and the root-cause tab's Pareto chart. Not gated by
    FDR significance: every one of the 5 is included regardless of
    p-value, tiered by confidence_tier instead of filtered out. `n80`
    reports the rank (across the FULL pool, not just these 5) at which
    cumulative contribution first reaches 80%, so the caller can render
    "80%에 도달하지 못했습니다 -- N개 더 필요" without a second request.
    """
    return _pareto_payload(dataset, target)


def _alarm_factors(train_df, schema) -> tuple[list[ParetoFactor], list[str]]:
    """Per-target alarm-eligible factors: every BH-FDR-significant factor
    (see select_fdr_significant_factors's docstring -- deliberately kept
    unchanged so the golden 19-alarm-wafer count doesn't move). Screen
    display no longer gates on significance, but alarm generation still
    does.
    """
    factors: list[ParetoFactor] = []
    no_alarm_factor: list[str] = []
    for target in schema.target_cols:
        target_factors = select_fdr_significant_factors(train_df, schema, target)
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
    train_df = _dataframe_or_404(dataset)
    schema = parse_schema(train_df)
    factors, no_significant = _alarm_factors(train_df, schema)
    items = [_control_range_dict(compute_control_range(train_df, factor)) for factor in factors]
    return {
        "train_dataset_id": dataset,
        "items": items,
        "no_significant_factor_targets": no_significant,
    }


def _measured_ids_for_alarm_factors(train_df: pd.DataFrame, eval_df: pd.DataFrame, schema) -> set[str]:
    """알람 판정에 쓸 "선정 인자 계측 여부" -- 기존 unmeasured_id_set과
    동일한 기준(FDR-유의 인자 중 하나라도 계측)이다 (spec 사전 알람 로그
    전면 개편 §B-2 "미계측").
    """
    factors, _ = _alarm_factors(train_df, schema)
    control_ranges = [compute_control_range(train_df, factor) for factor in factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)
    return {v.lot_wafer_id for v in verdicts if v.status != "unmeasured"}


def _scored_wafers(
    train: str,
    eval: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    target: float = alarm_gbdt.DEFAULT_TARGET_YIELD,
    sensitivity: float = alarm_gbdt.DEFAULT_SENSITIVITY,
) -> tuple[list, float | None, bool]:
    """공유 파이프라인 -- `get_alarms`와 `compute_alarm_notification_items`가
    똑같이 게이트를 적용하도록 한 곳에 모았다 (spec 알람 신뢰도 게이트
    §A-2: "신뢰할 수 없으면 0건이 된다"가 어디서 알람을 불러오든 항상
    성립해야 한다). 사전 알람 로그 전면 개편 이후로는 목표 수율/민감도
    기준 classify_wafer를 쓴다 -- 이 두 파라미터를 직접 조절하는 화면이
    없는 호출자(원인 분석 탭의 알람 삼각형 마커, 알림 발송)는 기본값
    (목표 85.0·민감도 0.5)을 그대로 쓴다.

    게이트 미달이어도 정상/판별불가는 여전히 계산해야 하므로(spec §B-4)
    부트스트랩 예측 자체는 건너뛰지 않는다 -- 이전(품질 게이트 시
    예측조차 안 함)과 달라진 부분이다.
    """
    auc_lo, gate_passed = _auc_gate(train, eval)

    prediction = _cached_bootstrap_prediction(train, eval)
    if prediction is None:
        return [], auc_lo, gate_passed

    schema = parse_schema(train_df)
    measured_ids = _measured_ids_for_alarm_factors(train_df, eval_df, schema)
    sigma = float(pd.to_numeric(train_df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce").std())
    scored = alarm_gbdt.score_wafers(
        eval_df, prediction,
        target=target, sensitivity=sensitivity, sigma=sigma,
        gate_passed=gate_passed, measured_ids=measured_ids,
    )
    return scored, auc_lo, gate_passed


def _reason_for(score, eval_by_id, warning_lines) -> str:
    row = None
    if eval_by_id is not None and score.lot_wafer_id in eval_by_id.index:
        match = eval_by_id.loc[score.lot_wafer_id]
        row = match.iloc[0] if isinstance(match, pd.DataFrame) else match
    return (
        warning_line.build_alarm_reason(row, warning_lines)
        if row is not None and warning_lines
        else warning_line.NO_EXCEEDANCE_REASON
    )


@router.get("/alarms", response_model=AlarmListResponse)
def get_alarms(train: str = "train", eval: str = "test", grade: str | None = None) -> dict[str, Any]:
    """알람 판정 GBDT 전환 (spec §A) + 사전 알람 로그 전면 개편 (spec §B-1) --
    부트스트랩 앙상블로 예측한 최종 수율(Y) 신뢰구간 상한(pred_hi)이 목표
    수율 - 민감도 오프셋*σ 아래인 wafer만 알람으로 낸다. 이 라우트는 목표
    수율/민감도를 직접 조절하는 UI가 없는 호출자(원인 분석 탭의 알람
    삼각형 마커)가 쓰므로 기본값(85.0/0.5)을 쓴다 -- 사전 알람 로그
    화면 자체는 `/alarms/predictions`에서 원시 예측치를 받아 클라이언트가
    실시간으로 재분류한다.

    알람 신뢰도 게이트 (spec 알람 신뢰도 게이트 §A-2) -- train→eval 전이
    AUC 하한이 0.65 미만이면 알람을 아예 내지 않는다.
    """
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)

    scored, auc_lo, gate_passed = _scored_wafers(train, eval, train_df, eval_df)
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
        }
    )


def compute_alarm_notification_items(train: str, eval: str) -> list[dict[str, Any]] | None:
    """알림 발송(src.notifications.dispatch)이 쓰는 알람 목록 -- `get_alarms`와
    같은 파이프라인(게이트 + 기본 목표/민감도)을 그대로 재사용한다.
    데이터셋을 찾을 수 없으면 예외 대신 None을 반환한다 -- 알림 발송은
    best-effort라 404로 스케줄러 잡 전체를 죽이면 안 된다.
    """
    try:
        train_df = _dataframe_or_404(train)
        eval_df = _dataframe_or_404(eval)
    except HTTPException:
        return None

    scored, _auc_lo, _gate_passed = _scored_wafers(train, eval, train_df, eval_df)
    alarm_scored = [s for s in scored if s.grade in ("심각", "위험", "주의")]

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
            }
        )
    return items


def _factor_band_dict(band) -> dict[str, Any]:
    return {
        "feature": band.feature,
        "target": band.target,
        "kind": band.kind,
        "x_min": band.x_min,
        "x_max": band.x_max,
        "lcl": band.lcl,
        "ucl": band.ucl,
        "recommended_lo": band.recommended_lo,
        "recommended_hi": band.recommended_hi,
        "out_of_control": {"count": band.out_of_control.count, "mean_defect_rate": band.out_of_control.mean_defect_rate},
        "out_of_recommended": {
            "count": band.out_of_recommended.count,
            "mean_defect_rate": band.out_of_recommended.mean_defect_rate,
        },
        "in_recommended": {"count": band.in_recommended.count, "mean_defect_rate": band.in_recommended.mean_defect_rate},
    }


@router.get("/alarms/predictions", response_model=AlertsDataResponse)
def get_alarms_predictions(train: str = "train", eval: str = "test") -> dict[str, Any]:
    """사전 알람 로그 전면 개편 (spec §A-3) -- 등급을 서버가 매겨 내려주지
    않고, wafer별 원시 예측치(pred_mean/pred_lo/pred_hi)와 목표 수율 조정에
    필요한 학습 Y 통계(sigma·분위수)를 내려준다. 목표 수율·민감도를 조절할
    때마다 이 응답을 다시 받아올 필요가 없다 -- frontend의 classifyWafer가
    `src.analysis.alarm_gbdt.classify_wafer`와 동일한 공식으로 클라이언트
    에서 즉시 재분류한다(§A-3: "API를 재호출하지 마라").
    """
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)

    auc_lo, gate_passed = _auc_gate(train, eval)
    prediction = _cached_bootstrap_prediction(train, eval)

    # "미계측" 정의는 그대로 둔다 (spec §B-2) -- 선정 인자가 하나도
    # 계측되지 않은 wafer는 예측이 있어도 신뢰할 수 없어 등급을 매기지
    # 않는다.
    measured_id_set = _measured_ids_for_alarm_factors(train_df, eval_df, schema)

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
                }
            )

    # 정밀도·재현율 실시간 추정용 홀드아웃 (spec §A-4) -- train을 LOT
    # 기준으로 5-fold 잘라 얻은 out-of-fold 예측치. 표본이 부족하면 None
    # (frontend가 "추정 불가"로 표시한다).
    features = alarm_gbdt.feature_columns(schema)
    holdout_result = alarm_gbdt.compute_holdout_predictions(train_df, features) if features else None
    holdout = (
        {
            "actual_y": holdout_result.actual_y.tolist(),
            "pred_point": holdout_result.pred_point.tolist(),
            "residual_std": holdout_result.residual_std,
        }
        if holdout_result is not None
        else None
    )

    y_train = pd.to_numeric(train_df[alarm_gbdt.FINAL_YIELD_COLUMN], errors="coerce").dropna()
    has_y = len(y_train) > 0
    sigma = float(y_train.std()) if has_y else 0.0

    # Per-factor 인자별 불량률 breakdown (§D-1, 그대로 유지) -- 강함·보통
    # 등급 인자 전부. `_cached_ranked_rows`는 Pareto 화면이 이미 조회한
    # 타깃이면 즉시 반환되는 프로세스 전역 캐시라 여기서 다시 스코어링하지
    # 않는다.
    factor_bands_with_eps2: list[tuple[float, dict[str, Any]]] = []
    qualifying_factors: list[ParetoFactor] = []
    for t in schema.target_cols:
        for row in _cached_ranked_rows(train, t):
            tier = confidence_tier(row["eps2"], row["p_value"])
            if tier not in ("strong", "moderate"):
                continue
            factor = _row_to_factor(train_df, t, row)
            qualifying_factors.append(factor)
            control_range = compute_control_range(train_df, factor)
            band = compute_factor_band(train_df, eval_df, factor, control_range)
            if band is not None:
                factor_bands_with_eps2.append((row["eps2"], {**_factor_band_dict(band), "confidence_tier": tier}))
    factor_bands_with_eps2.sort(key=lambda pair: pair[0], reverse=True)
    factor_bands = [item for _eps2, item in factor_bands_with_eps2]

    # 계측 편향 재검토 (spec 문구 전수 검토 §A-7) -- 위 강함·보통 인자
    # 전체를 대상으로 한다 (드롭다운과 같은 범위).
    measurement_bias = summarize_measurement_bias(per_factor_measurement_bias(train_df, qualifying_factors))

    return round_floats(
        {
            "train_dataset_id": train,
            "eval_dataset_id": eval,
            "total_wafers": len(eval_df),
            "sigma": sigma,
            "train_y_min": float(y_train.min()) if has_y else 0.0,
            "train_y_max": float(y_train.max()) if has_y else 0.0,
            "train_y_median": float(y_train.median()) if has_y else 0.0,
            "train_y_p1": float(y_train.quantile(0.01)) if has_y else 0.0,
            "train_y_p99": float(y_train.quantile(0.99)) if has_y else 0.0,
            "predictions": predictions,
            "holdout": holdout,
            "auc_lower_bound": auc_lo,
            "auc_gate_passed": gate_passed,
            "auc_gate_threshold": alarm_gbdt.AUC_GATE,
            "factor_bands": factor_bands,
            "measurement_bias": measurement_bias,
        }
    )


def _build_report_payload(dataset: str) -> dict[str, Any]:
    """The one function backing both /api/analysis/report (JSON download)
    and /api/analysis/context (SUNI chatbot context) -- same dict, so the
    chatbot never narrates a number the download button wouldn't also show.
    """
    registry = get_dataset_registry()
    train_df = _dataframe_or_404(dataset)
    eval_df = _dataframe_or_404(REPORT_EVAL_DATASET_ID)
    train_meta = registry.get_summary(dataset) or {}
    eval_meta = registry.get_summary(REPORT_EVAL_DATASET_ID) or {}
    return build_analysis_report(
        train_df,
        eval_df,
        train_dataset_id=dataset,
        eval_dataset_id=REPORT_EVAL_DATASET_ID,
        train_meta=train_meta,
        eval_meta=eval_meta,
        app_version=APP_VERSION,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


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
    구성이 아예 다른 다른 데이터셋(예: mentorship_dataset_final의
    Step20_D1은 test.csv에 없는 컬럼이다)을 판정 기준으로 쓰면 그 데이터셋의
    실제 계측률과 무관하게 전량 "판정불가"로 나와 §B-6 축소 조건이 항상
    빗나간다 -- 원인 분석 탭 자체가 데이터셋 선택기 하나뿐이라는 점과도
    맞는다.
    """
    train_df = _dataframe_or_404(dataset)
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
        target: list(_cached_ranked_rows(dataset, target)) for target in schema.target_cols
    }

    # "판정 가능 여부"(조치 불가/추가 판정)는 알람 목록과 동일한 FDR-유의
    # 인자 전체 집합으로 판단한다 -- get_alarms_predictions가 쓰는 것과
    # 같은 개념(_alarm_factors). 타깃마다 1위 인자 하나만 쓰면(select_primary_factor)
    # 여러 타깃이 같은 인자를 1위로 뽑는 데이터셋(예:
    # mentorship_dataset_final은 5개 타깃 모두 Step20_D1이 1위)에서 사실상
    # 서로 다른 컬럼 1개만 보는 셈이 되어, 그 인자 하나의 계측률만으로
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
    # mentorship_dataset_final 등)에서 아래 권장구간 계산(SPC/ML 부트스트랩,
    # 인자 수가 많으면 수십 초)까지 돌리는 건 낭비다.
    show_full_card = total_wafers > 0 and (len(unmeasured_ids) / total_wafers) >= MIN_ACTION_BLOCKED_SHARE

    if not show_full_card:
        summary = compute_measurement_expansion(
            train_df, eval_df, {}, {}, {}, classify_measured_bands(
                train_df, eval_df, alarm_ids, normal_ids, unmeasured_ids, [], []
            ), [], total_wafers=total_wafers,
        )
    else:
        bands = classify_measured_bands(
            train_df, eval_df, alarm_ids, normal_ids, unmeasured_ids, judgment_factors, control_ranges
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
            factor_summary = compute_factor_recommendation(train_df, factor, control_range)
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
        }
    )


RELIABILITY_CACHE_PAIRS = 8


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
    return {
        "model_id": metadata.get("model_id"),
        "trained_at": metadata.get("created_at"),
        "source_filename": metadata.get("source_filename"),
        "targets": targets,
        "final_yield": final_yield,
    }
