from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status

from api.routes.datasets import get_dataset_registry
from api.schemas.analysis import (
    AlarmListResponse,
    AlarmSummaryResponse,
    AnalysisReportResponse,
    CategoricalScatterResponse,
    ControlRangeListResponse,
    HeatmapResponse,
    ModelPerformanceResponse,
    ParetoRankingResponse,
    ScreeningResponse,
    ScreeningScatterResponse,
)
from api.settings import APP_VERSION, settings
from src.analysis.control_range import (
    compute_control_range,
    evaluate_alarms,
    summarize_wafer_status,
)
from src.analysis.report import build_analysis_report
from src.analysis.scatter import build_categorical_data, build_scatter_data
from src.analysis.screening.heatmap import HeatmapData, build_heatmap
from src.analysis.screening.schema import parse_schema
from src.analysis.screening.selector import (
    confidence_tier,
    find_factor,
    score_all_factors,
    select_pareto_factors_all_targets,
)
from src.ml.inference import get_latest_model_metadata
from src.runtime.datasets import DatasetNotFoundError
from src.runtime.store import RuntimeStore

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


@router.get("/screening", response_model=ScreeningResponse)
def get_screening(dataset: str = "train") -> dict[str, Any]:
    df = _dataframe_or_404(dataset)
    schema = parse_schema(df)
    results = select_pareto_factors_all_targets(df, schema)
    return {
        "dataset_id": dataset,
        "schema_warnings": [f"파싱하지 못한 컬럼: {column}" for column in schema.unmapped],
        "targets": [
            {
                "target": target,
                "factors": [vars(factor) for factor in result.factors],
                "reference_only": [vars(factor) for factor in result.reference_only],
                "excluded_count": result.excluded_count,
                "no_significant_factor": result.no_significant_factor,
            }
            for target, result in results.items()
        ],
    }


@router.get("/screening/scatter", response_model=ScreeningScatterResponse)
def get_screening_scatter(dataset: str, target: str, feature: str) -> dict[str, Any]:
    df = _dataframe_or_404(dataset)
    schema = parse_schema(df)
    results = select_pareto_factors_all_targets(df, schema)
    target_result = results.get(target)
    if target_result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃 결과가 없습니다.")
    factor = next(
        (f for f in [*target_result.factors, *target_result.reference_only] if f.feature == feature),
        None,
    )
    if factor is None:
        # Not among the selected/top-10-reference factors -- still resolve it
        # so a heatmap cell for any of the 58 R/D factors can open a scatter,
        # even when it never passed FDR (see get_screening_heatmap).
        factor = find_factor(df, schema, target, feature)
    if factor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{feature}' 인자를 찾을 수 없습니다.")
    if factor.kind == "Config":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{feature}'은(는) Config(범주형) 인자입니다. /api/screening/scatter/categorical을 사용하세요.",
        )

    data = build_scatter_data(df, df, factor)
    return {
        "points": data.points,
        "reference_lines": data.reference_lines,
        "normal_range": data.normal_range,
        "bins": data.bins,
        "optimal_center": data.optimal_center,
        "eps2": data.eps2,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "n": data.n,
        "axis": data.axis,
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
        "groups": [vars(group) for group in data.groups],
        "eps2": data.eps2,
        "p_value": data.p_value,
        "q_value": data.q_value,
        "significant": data.significant,
        "confidence_tier": data.confidence_tier,
        "n": data.n,
        "axis": data.axis,
    }


@lru_cache(maxsize=128)
def _cached_heatmap(dataset_id: str, metric: str, kind: str) -> HeatmapData:
    # Cached per (dataset_id, metric, kind): dataset content is immutable
    # once a dataset_id exists (uploads mint a fresh uuid; bundled files
    # are static).
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    return build_heatmap(df, schema, metric=metric, kind=kind)  # type: ignore[arg-type]


@router.get("/screening/heatmap", response_model=HeatmapResponse)
def get_screening_heatmap(
    dataset: str = "train",
    metric: Literal["spearman", "eps2"] = "spearman",
    kind: Literal["all", "R", "D", "Config"] = "all",
) -> dict[str, Any]:
    heatmap = _cached_heatmap(dataset, metric, kind)
    # Config has no rho -- build_heatmap silently forces eps2 internally;
    # echo the metric it actually used, not necessarily what was asked for.
    effective_metric = "eps2" if kind == "Config" else metric
    return {
        "dataset_id": dataset,
        "metric": effective_metric,
        "kind": kind,
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


@lru_cache(maxsize=256)
def _cached_pareto_items(dataset_id: str, target: str, kind: str) -> tuple[dict, ...]:
    # Cached per (dataset_id, target, kind): the root-cause tab's "실행"
    # flow fetches all 5 targets x 4 kinds up front (20 combos), and each
    # kind filter otherwise re-runs the same score_all_factors(target)
    # pass redundantly. Dataset content is immutable once a dataset_id
    # exists (see the heatmap cache's docstring for why that's safe).
    df = _dataframe_or_404(dataset_id)
    schema = parse_schema(df)
    if target not in schema.target_cols:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}' 타깃을 찾을 수 없습니다.")
    rows = score_all_factors(df, schema, target)
    if kind != "all":
        rows = [row for row in rows if row["kind"] == kind]
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{target}'/{kind} 결과가 없습니다.")

    rows.sort(key=lambda r: r["eps2"], reverse=True)
    total_eps2 = sum(r["eps2"] for r in rows)
    cumulative = 0.0
    items = []
    for row in rows:
        pct = (row["eps2"] / total_eps2 * 100.0) if total_eps2 > 0 else 0.0
        cumulative += pct
        items.append(
            {
                "feature": row["feature"],
                "kind": row["kind"],
                "step": row["step"],
                "eps2": row["eps2"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "significant": row["significant"],
                "confidence_tier": confidence_tier(row["p_value"]),
                "n_observed": row["n_observed"],
                "contribution_pct": pct,
                "cumulative_pct": cumulative,
            }
        )
    return tuple(items)


@router.get("/screening/pareto", response_model=ParetoRankingResponse)
def get_screening_pareto(
    dataset: str = "train",
    target: str = "Y1",
    kind: Literal["all", "R", "D", "Config"] = "all",
) -> dict[str, Any]:
    """Every factor of the requested kind for one target, ranked by eps2,
    with contribution/cumulative denominated by that kind's pool only --
    the data behind the root-cause tab's Pareto chart (top-10/20/all is a
    client slice of this one list, so switching that toggle needs no
    refetch). Not gated by FDR significance: every candidate is included,
    tiered by confidence_tier instead of filtered out.
    """
    items = list(_cached_pareto_items(dataset, target, kind))
    return {
        "dataset_id": dataset,
        "target": target,
        "kind": kind,
        "total_factor_count": len(items),
        "items": items,
    }


def _selected_factors(train_df, schema):
    results = select_pareto_factors_all_targets(train_df, schema)
    factors = []
    no_significant: list[str] = []
    for target, result in results.items():
        if result.no_significant_factor or not result.factors:
            no_significant.append(target)
            continue
        factors.extend(result.factors)
    return factors, no_significant


def _control_range_dict(control_range) -> dict[str, Any]:
    data = vars(control_range).copy()
    data["reference_lines"] = [vars(line) for line in control_range.reference_lines]
    return data


@router.get("/control-ranges", response_model=ControlRangeListResponse)
def get_control_ranges(dataset: str = "train") -> dict[str, Any]:
    train_df = _dataframe_or_404(dataset)
    schema = parse_schema(train_df)
    factors, no_significant = _selected_factors(train_df, schema)
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
    factors, _ = _selected_factors(train_df, schema)

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


@router.get("/alarms/summary", response_model=AlarmSummaryResponse)
def get_alarm_summary(train: str = "train", eval: str = "test") -> dict[str, Any]:
    train_df = _dataframe_or_404(train)
    eval_df = _dataframe_or_404(eval)
    schema = parse_schema(train_df)
    factors, _ = _selected_factors(train_df, schema)

    control_ranges = [compute_control_range(train_df, factor) for factor in factors]
    alarms_by_feature = {cr.feature: evaluate_alarms(eval_df, cr) for cr in control_ranges}
    verdicts = summarize_wafer_status(eval_df, control_ranges, alarms_by_feature)

    alarm_ids = [v.lot_wafer_id for v in verdicts if v.status == "alarm"]
    normal_ids = [v.lot_wafer_id for v in verdicts if v.status == "normal"]
    unmeasured_ids = [v.lot_wafer_id for v in verdicts if v.status == "unmeasured"]

    indexed = eval_df.set_index("Lot_Wafer_ID") if "Lot_Wafer_ID" in eval_df.columns else None
    alarm_avg = None
    no_alarm_avg = None
    if indexed is not None and "Y" in eval_df.columns:
        if alarm_ids:
            alarm_avg = float(indexed.loc[alarm_ids, "Y"].mean())
        no_alarm_group = normal_ids + unmeasured_ids
        if no_alarm_group:
            no_alarm_avg = float(indexed.loc[no_alarm_group, "Y"].mean())

    lot_counts: dict[str, int] = {}
    for v in verdicts:
        if v.status == "alarm" and v.lot_id:
            lot_counts[v.lot_id] = lot_counts.get(v.lot_id, 0) + 1
    top_lots = sorted(
        ({"lot_id": lot_id, "alarm_count": count} for lot_id, count in lot_counts.items()),
        key=lambda item: item["alarm_count"],
        reverse=True,
    )

    return {
        "train_dataset_id": train,
        "eval_dataset_id": eval,
        "counts": {"alarm": len(alarm_ids), "normal": len(normal_ids), "unmeasured": len(unmeasured_ids)},
        "alarm_group_yield_avg": alarm_avg,
        "no_alarm_group_yield_avg": no_alarm_avg,
        "yield_gap": (alarm_avg - no_alarm_avg) if alarm_avg is not None and no_alarm_avg is not None else None,
        "top_lots": top_lots,
    }


@router.get("/analysis/report", response_model=AnalysisReportResponse)
def get_analysis_report(dataset: str = "train") -> dict[str, Any]:
    """Full JSON analysis report -- always denominated by the full
    R+D+Config pool and always "전체" kind, regardless of which kind tab
    the caller happens to be viewing (see build_analysis_report's
    docstring for why the factor list and the alarm list use different,
    deliberately non-interchangeable factor sets).
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
            "no_significant_factor": bool(detail.get("no_significant_factor")),
            "feature": detail.get("feature"),
            "kind": detail.get("kind"),
            "eps2": detail.get("eps2"),
            "relation_shape": detail.get("relation_shape"),
            "optimal_center": detail.get("optimal_center"),
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
            "no_significant_factor": False,
            "feature": None,
            "kind": None,
            "eps2": None,
            "relation_shape": None,
            "optimal_center": None,
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
