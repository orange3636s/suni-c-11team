from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status

from api.routes.datasets import get_dataset_registry
from api.schemas.analysis import (
    AlarmListResponse,
    AlarmSummaryResponse,
    AnalysisContextResponse,
    AnalysisReportResponse,
    CategoricalScatterResponse,
    ControlRangeListResponse,
    HeatmapResponse,
    ModelPerformanceResponse,
    ParetoRankingResponse,
    RecommendationListResponse,
    ScreeningScatterResponse,
)
from api.settings import APP_VERSION, settings
from src.analysis.alarm_bands import classify_measured_bands, compute_factor_band
from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.llm_stats import measurement_bias_p
from src.analysis.recommendations import compute_recommendations
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
    select_fdr_significant_factors,
    select_primary_factor,
)
from src.analysis.screening.selector import _ranked_rows_with_contribution as _ranked_rows
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

    data = build_scatter_data(df, df, factor)
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
    return {
        "dataset_id": dataset_id,
        "target": target,
        "total_factor_count": len(ranked),
        "n80": n80,
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


@router.get("/alarms", response_model=AlarmListResponse)
def get_alarms(train: str = "train", eval: str = "test", severity: str | None = None) -> dict[str, Any]:
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)
    factors, _ = _alarm_factors(train_df, schema)

    items: list[dict[str, Any]] = []
    for factor in factors:
        control_range = compute_control_range(train_df, factor)
        for alarm in evaluate_alarms(eval_df, control_range):
            if severity and alarm.severity != severity:
                continue
            items.append(
                {
                    "lot_wafer_id": alarm.lot_wafer_id,
                    "lot_id": alarm.lot_id,
                    "wafer_slot": alarm.wafer_slot,
                    "step": factor.step,
                    "feature": alarm.feature,
                    "kind": alarm.kind,
                    "target": alarm.target,
                    "value": alarm.value,
                    "normal_range": [alarm.lower, alarm.upper],
                    "deviation": alarm.deviation,
                    "direction": alarm.direction,
                    "severity": alarm.severity,
                    "actual_y": alarm.actual_y,
                }
            )
    items.sort(key=lambda item: item["lot_wafer_id"])
    return {"train_dataset_id": train, "eval_dataset_id": eval, "items": items, "total": len(items)}


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


@router.get("/alarms/summary", response_model=AlarmSummaryResponse)
def get_alarm_summary(train: str = "train", eval: str = "test") -> dict[str, Any]:
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)
    factors, _ = _alarm_factors(train_df, schema)

    control_ranges = [compute_control_range(train_df, factor) for factor in factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)

    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]

    lot_counts: dict[str, int] = {}
    for v in verdicts:
        if v.status == "alarm" and v.lot_id:
            lot_counts[v.lot_id] = lot_counts.get(v.lot_id, 0) + 1
    top_lots = sorted(
        ({"lot_id": lot_id, "alarm_count": count} for lot_id, count in lot_counts.items()),
        key=lambda item: item["alarm_count"],
        reverse=True,
    )

    # 3-way band split for the redesigned 사전 알람 로그 summary (spec PART B):
    # 관리한계 이탈 (alarm, already computed above) / 권장구간 밖 / 권장구간 내,
    # reusing the same alarm-eligible factor set and its recommended windows
    # so these numbers never disagree with /api/alarms or /api/recommendations.
    bands = classify_measured_bands(
        train_df, eval_df, alarm_ids, normal_ids, unmeasured_ids, factors, control_ranges
    )

    # Per-factor 인자별 불량률 breakdown (카드②): the 5 primary (1위) factors,
    # one per target -- the same "1위 인자" concept used everywhere else
    # (training summary, 개선 권장 목록), not the full alarm-eligible set.
    factor_bands: list[dict[str, Any]] = []
    for target in schema.target_cols:
        primary = select_primary_factor(train_df, schema, target)
        if primary is None:
            continue
        primary_control_range = compute_control_range(train_df, primary)
        band = compute_factor_band(train_df, eval_df, primary, primary_control_range)
        if band is not None:
            factor_bands.append(_factor_band_dict(band))

    bias_p = measurement_bias_p(train_df, schema)

    return round_floats(
        {
            "train_dataset_id": train,
            "eval_dataset_id": eval,
            "total_wafers": len(verdicts),
            "measured_wafers": len(alarm_ids) + len(normal_ids),
            "counts": {
                "alarm": len(alarm_ids),
                "out_of_recommended": bands.out_of_recommended.count,
                "in_recommended": bands.in_recommended.count,
                "unmeasured": len(unmeasured_ids),
            },
            "band_yield": {
                "alarm": bands.alarm.mean_yield,
                "out_of_recommended": bands.out_of_recommended.mean_yield,
                "in_recommended": bands.in_recommended.mean_yield,
                "unmeasured": bands.unmeasured.mean_yield,
            },
            "top_lots": top_lots,
            "measurement_bias_p": bias_p,
            "factor_bands": factor_bands,
        }
    )


@router.get("/recommendations", response_model=RecommendationListResponse)
def get_recommendations(train: str = "train", eval: str = "test") -> dict[str, Any]:
    """개선 권장 목록: wafers outside the recommended range of each
    target's primary (1위) factor -- the same factor already shown on
    the training/root-cause screens, not the full R+D+Config pool. Rows
    already caught by that same factor's alarm (LCL/UCL) are excluded
    (spec §3-1: "관리한계 이탈로 이미 알람에 잡힌 건은 개선 권장 목록에서
    제외한다").
    """
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)

    primary_factors: dict[str, ParetoFactor] = {}
    for target in schema.target_cols:
        factor = select_primary_factor(train_df, schema, target)
        if factor is not None:
            primary_factors[target] = factor

    rows, factor_summaries = compute_recommendations(train_df, eval_df, schema, primary_factors=primary_factors)

    excluded_alarm_count = 0
    for target, factor in primary_factors.items():
        summary = factor_summaries.get(target)
        if summary is None or factor.feature not in eval_df.columns:
            continue
        control_range = compute_control_range(train_df, factor)
        already_alarmed = {a.lot_wafer_id for a in evaluate_alarms(eval_df, control_range)}
        ex = pd.to_numeric(eval_df[factor.feature], errors="coerce")
        id_col = eval_df["Lot_Wafer_ID"] if "Lot_Wafer_ID" in eval_df.columns else None
        for position, value in ex.items():
            if pd.isna(value) or summary.recommended_lo <= value <= summary.recommended_hi:
                continue
            wafer_id = str(id_col.loc[position]) if id_col is not None else str(position)
            if wafer_id in already_alarmed:
                excluded_alarm_count += 1

    items = [
        {
            "lot_wafer_id": row.lot_wafer_id,
            "lot_id": row.lot_id,
            "step": row.step,
            "feature": row.feature,
            "kind": row.kind,
            "target": row.target,
            "value": row.value,
            "recommended_range": [row.recommended_lo, row.recommended_hi],
            "direction": row.direction,
            "expected_improvement_pct": row.expected_improvement_pct,
            "tag": row.tag,
        }
        for row in rows
    ]
    items.sort(key=lambda item: item["lot_wafer_id"])
    return {
        "train_dataset_id": train,
        "eval_dataset_id": eval,
        "items": items,
        "total": len(items),
        "excluded_alarm_count": excluded_alarm_count,
    }


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


@router.get("/analysis/context", response_model=AnalysisContextResponse)
def get_analysis_context(dataset: str = "train", *, response: Response) -> dict[str, Any]:
    """The SUNI chatbot's grounding context -- same underlying report as
    /api/analysis/report, but `alarms`/`recommendations` are reshaped into
    `{summary, records}` (see build_chat_context's docstring) so the
    chatbot can answer a question about one specific wafer's alarm, not
    just the aggregate counts. No Content-Disposition, and explicitly
    non-cached since the frontend calls this right before /api/chat.
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
