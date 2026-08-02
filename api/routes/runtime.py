from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from api.settings import settings
from api.schemas.runtime import AnalysisHistoryListResponse, AnalysisOverviewResponse
from src.runtime.store import RuntimeStore


router = APIRouter(prefix="/api", tags=["runtime-dashboard"])


def get_runtime_store() -> RuntimeStore:
    return RuntimeStore(settings.runtime_db_path, settings.runtime_artifact_dir)


def _history_filters(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class AlertPatch(BaseModel):
    status: str


@router.get("/alerts")
def get_alerts(
    risk_level: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    model_id: str | None = None,
    lot_id: str | None = None,
    wafer_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    sort: str = "date",
) -> dict[str, Any]:
    return get_runtime_store().list_alerts({
        "risk_level": risk_level, "status": status_filter, "model_id": model_id,
        "lot_id": lot_id, "wafer_id": wafer_id, "limit": limit, "offset": offset, "sort": sort,
    })


@router.get("/alerts/summary")
def get_alert_summary() -> dict[str, int]:
    return get_runtime_store().alert_summary()


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: str) -> dict[str, Any]:
    alert = get_runtime_store().get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="알람을 찾을 수 없습니다.")
    return alert


@router.patch("/alerts/{alert_id}")
def patch_alert(alert_id: str, patch: AlertPatch) -> dict[str, Any]:
    try:
        alert = get_runtime_store().update_alert(alert_id, patch.status)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="알람을 찾을 수 없습니다.")
    return alert


@router.get("/predictions/history")
def get_prediction_history(
    model_id: str | None = None,
    filename: str | None = None,
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = 50,
    offset: int = 0,
    sort: str = "newest",
) -> dict[str, Any]:
    return get_runtime_store().list_predictions(_history_filters(
        model_id=model_id, filename=filename, search=search, date_from=date_from, date_to=date_to, status=status_filter,
        limit=limit, offset=offset, sort=sort,
    ))


@router.get("/predictions/history/{prediction_id}")
def get_prediction_history_detail(prediction_id: str) -> dict[str, Any]:
    detail = get_runtime_store().get_prediction(prediction_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="예측 이력을 찾을 수 없습니다.")
    return detail


def delete_prediction_history(prediction_id: str) -> dict[str, Any]:
    if not get_runtime_store().delete_prediction(prediction_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="예측 이력을 찾을 수 없습니다.")
    return {"success": True, "prediction_id": prediction_id, "linked_analyses_preserved": True}


@router.get("/analyses/history")
def get_analysis_history(
    model_id: str | None = None,
    prediction_id: str | None = None,
    filename: str | None = None,
    search: str | None = None,
    target: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = 50,
    offset: int = 0,
    sort: str = "newest",
) -> dict[str, Any]:
    return get_runtime_store().list_analyses(_history_filters(
        model_id=model_id, prediction_id=prediction_id, filename=filename, search=search,
        target=target, date_from=date_from, date_to=date_to,
        status=status_filter, limit=limit, offset=offset, sort=sort,
    ))


@router.get("/analyses/history/{analysis_id}")
def get_analysis_history_detail(analysis_id: str) -> dict[str, Any]:
    detail = get_runtime_store().get_analysis(analysis_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="분석 이력을 찾을 수 없습니다.")
    return _normalize_analysis_history_detail(detail)


def delete_analysis_history(analysis_id: str) -> dict[str, Any]:
    if not get_runtime_store().delete_analysis(analysis_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="분석 이력을 찾을 수 없습니다.")
    return {"success": True, "analysis_id": analysis_id, "linked_prediction_preserved": True}


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _normalize_analysis_history_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Return analysis data without exposing removed Report snapshots."""
    artifact = dict(_record(detail.get("artifact")))
    artifact.pop("report_snapshot", None)
    if not artifact:
        return detail
    response = dict(_record(artifact.get("response")))
    response.pop("report_snapshot", None)
    analysis = dict(_record(artifact.get("analysis_result")) or _record(
        response.get("analysis_result")
    ))
    analysis.pop("report", None)
    if response or analysis:
        response["analysis_result"] = analysis or None
        lot_analysis = (
            _record(response.get("lot_analysis"))
            or _record(artifact.get("lot_analysis"))
            or _record(analysis.get("lot_analysis"))
        )
        response["lot_analysis"] = lot_analysis
        response["relationship_paths"] = _records(
            response.get("relationship_paths")
        ) or _records(analysis.get("relationships"))
        response["available_steps"] = [
            int(value)
            for value in response.get("available_steps", [])
            if isinstance(value, int) and not isinstance(value, bool)
        ] if isinstance(response.get("available_steps"), list) else []
        response["caveats"] = _strings(response.get("caveats"))
        warnings = response.get("selection_bias_warnings")
        if not isinstance(warnings, list):
            data_quality = _record(analysis.get("data_quality"))
            warnings = data_quality.get("selection_bias_warnings")
        response["selection_bias_warnings"] = _strings(warnings)

        statistics = _record(response.get("statistics"))
        if not statistics:
            candidate = _record(analysis.get("statistics"))
            if any(
                key in candidate
                for key in (
                    "methods", "numeric", "categorical", "scatter_data",
                    "boxplot_data", "categorical_relationships",
                )
            ):
                statistics = candidate
        numeric_statistics = _records(statistics.get("numeric"))
        categorical_statistics = _records(statistics.get("categorical"))
        if isinstance(statistics.get("scatter_data"), list):
            scatter_data = _records(statistics.get("scatter_data"))
        else:
            scatter_data = [
                point
                for row in numeric_statistics
                for point in _records(row.get("scatter_data"))
            ]
        if isinstance(statistics.get("boxplot_data"), list):
            boxplot_data = _records(statistics.get("boxplot_data"))
        else:
            boxplot_data = [
                summary
                for row in categorical_statistics
                for summary in (
                    _records(row.get("boxplot_data"))
                    or _records(row.get("category_summary"))
                )
            ]
        if isinstance(statistics.get("categorical_relationships"), list):
            categorical_relationships = _records(
                statistics.get("categorical_relationships")
            )
        else:
            categorical_relationships = categorical_statistics
        response["statistics"] = {
            "methods": _strings(statistics.get("methods")),
            "numeric": numeric_statistics,
            "categorical": categorical_statistics,
            "scatter_data": scatter_data,
            "boxplot_data": boxplot_data,
            "categorical_relationships": categorical_relationships,
        }

        rankings = _record(response.get("rankings"))
        normalized_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for mode in ("shap", "correlation"):
            groups = _record(rankings.get(mode))
            normalized_rankings[mode] = {
                key: _records(groups.get(key))
                for key in (
                    "all", "r", "d", "config", "overall", "R", "D",
                    "EQ", "eq", "model", "equipment", "chamber",
                    "measurement", "missing", "indicator", "observed",
                )
            }
        response["rankings"] = normalized_rankings
        artifact = {**artifact, "response": response}
        if analysis:
            artifact["analysis_result"] = analysis
        if lot_analysis:
            artifact["lot_analysis"] = lot_analysis
    return {**detail, "artifact": artifact}


def _integer(value: Any) -> int | None:
    numeric = _number(value)
    return int(numeric) if numeric is not None else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        numeric = _number(value)
        if numeric is not None:
            return numeric
    return None


def _first_integer(*values: Any) -> int | None:
    for value in values:
        numeric = _integer(value)
        if numeric is not None:
            return numeric
    return None


def _number_map(value: Any) -> dict[str, float | None]:
    return {
        str(key): _number(item)
        for key, item in _record(value).items()
    }


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    evaluation = _record(metrics.get("evaluation_summary"))
    cv = _record(metrics.get("cv"))
    containers = [
        _record(metrics.get("test")),
        _record(evaluation.get("metric_summary")),
        _record(evaluation.get("aggregate_metrics")),
        _record(cv.get("aggregate_metrics")),
        metrics,
    ]
    for container in containers:
        value = container.get(name)
        if isinstance(value, dict):
            value = value.get("mean", value.get("value"))
        numeric = _number(value)
        if numeric is not None:
            return numeric
    return None


def _empty_overview() -> dict[str, Any]:
    summary = {
        "wafer_count": None, "lot_count": None,
        "average_predicted_yield": None, "minimum_predicted_yield": None,
        "critical_count": None, "warning_count": None, "normal_count": None,
        "low_confidence_count": None, "risk_lot_count": None,
    }
    model_metrics = {"r2": None, "rmse": None, "mae": None}
    multi_y = {
        "predicted_y_mean": None,
        "failure_rates": {}, "fail_bit_counts": {},
    }
    causes = {
        "top_failure_target": None, "top_features": [], "top_steps": [],
        "top_equipment": [], "top_chambers": [],
    }
    availability = {
        "summary": False, "model_metrics": False, "multi_y": False,
        "causes": False, "risk_lots": False, "risk_wafers": False,
        "pareto": False, "relationships": False,
    }
    return {
        "source": {
            "type": "empty", "analysis_id": None, "created_at": None,
            "completed_at": None, "status": "empty", "source_filename": None,
            "model_id": None, "model_name": None, "artifact_available": False,
            "artifact_status": "not_applicable",
        },
        "summary": summary, "model_metrics": model_metrics, "multi_y": multi_y,
        "causes": causes, "risk_lots": [], "risk_wafers": [], "pareto": [],
        "relationships": [], "warnings": [], "availability": availability,
        "source_type": "empty", "source_id": None, "created_at": None,
        "source_label": "저장된 원인 분석 결과 없음", "filename": None,
        "model": None, "data_quality": {},
    }


def _overview_analysis(
    detail: dict[str, Any],
    *,
    explicitly_selected: bool = False,
) -> dict[str, Any]:
    metadata = _record(detail.get("metadata"))
    artifact = _record(detail.get("artifact"))
    response = _record(artifact.get("response")) or artifact
    analysis = (
        _record(response.get("analysis_result"))
        or _record(artifact.get("analysis_result"))
    )
    if not analysis and any(key in artifact for key in ("risk", "multi_y", "feature_importance")):
        analysis = artifact

    explanation = _record(response.get("explanation"))
    stored_summary = _record(metadata.get("summary"))
    model = _record(analysis.get("model")) or _record(explanation.get("model"))
    metrics = _record(analysis.get("metrics"))
    risk = _record(analysis.get("risk"))
    confidence = _record(analysis.get("confidence"))
    multi_y_source = _record(analysis.get("multi_y"))
    importance = _record(analysis.get("feature_importance"))

    risk_lots = _records(analysis.get("lot_summary"))
    risk_wafers = _records(analysis.get("risk_wafers"))
    top_features = (
        _records(importance.get("global"))
        or _records(explanation.get("global_importance"))
    )
    top_steps = (
        _records(importance.get("steps"))
        or _records(importance.get("top_steps"))
        or _records(explanation.get("step_summary"))
    )
    top_equipment = (
        _records(explanation.get("equipment_summary"))
        or _records(analysis.get("equipment_summary"))
    )
    path_rows = _records(analysis.get("relationships")) or _records(response.get("relationship_paths"))
    statistics = _record(analysis.get("statistics")) or _record(response.get("statistics"))
    statistic_rows = [*_records(statistics.get("numeric")), *_records(statistics.get("categorical"))]
    relationships = statistic_rows[:20] if statistic_rows else path_rows[:10]

    pareto_source = response.get("pareto", artifact.get("pareto", analysis.get("pareto")))
    pareto = _records(_record(pareto_source).get("features")) if isinstance(pareto_source, dict) else _records(pareto_source)

    chambers: list[dict[str, Any]] = []
    seen_chambers: set[str] = set()
    for item in path_rows:
        chamber = item.get("chamber")
        if not isinstance(chamber, str) or not chamber.strip() or chamber in seen_chambers:
            continue
        seen_chambers.add(chamber)
        chambers.append({
            "chamber": chamber,
            "rank": item.get("rank"),
            "path_score": item.get("path_score"),
        })
        if len(chambers) == 5:
            break

    failure_rates = _number_map(
        multi_y_source.get("failure_rate_averages", multi_y_source.get("failure_rates"))
    )
    fail_bit_counts = _number_map(
        multi_y_source.get("fail_bit_count_averages", multi_y_source.get("fail_bit_counts"))
    )
    top_failure_target = stored_summary.get("top_failure_target")
    if not isinstance(top_failure_target, str):
        available_failure_rates = {
            key: value for key, value in failure_rates.items() if value is not None
        }
        top_failure_target = (
            max(available_failure_rates, key=available_failure_rates.get)
            if available_failure_rates else None
        )

    risk_lot_count = _first_integer(stored_summary.get("risk_lot_count"))
    if risk_lot_count is None and risk_lots:
        risk_lot_count = sum(
            1 for lot in risk_lots
            if (_number(lot.get("danger_count")) or 0) > 0
            or (_number(lot.get("warning_count")) or 0) > 0
        )

    summary = {
        "wafer_count": _first_integer(
            metadata.get("row_count"), _record(analysis.get("dataset")).get("row_count"),
        ),
        "lot_count": _first_integer(metadata.get("lot_count")),
        "average_predicted_yield": _first_number(
            stored_summary.get("average_predicted_yield"),
        ),
        "minimum_predicted_yield": _first_number(
            stored_summary.get("minimum_predicted_yield"),
        ),
        "critical_count": _first_integer(
            risk.get("critical_count"), stored_summary.get("critical_count"),
        ),
        "warning_count": _first_integer(
            risk.get("warning_count"), stored_summary.get("warning_count"),
        ),
        "normal_count": _first_integer(
            risk.get("normal_count"), stored_summary.get("normal_count"),
        ),
        "low_confidence_count": _first_integer(
            confidence.get("low_confidence_count"), stored_summary.get("low_confidence_count"),
        ),
        "risk_lot_count": risk_lot_count,
    }
    model_metrics = {
        "r2": _metric_value(metrics, "r2"),
        "rmse": _metric_value(metrics, "rmse"),
        "mae": _metric_value(metrics, "mae"),
    }
    multi_y = {
        "predicted_y_mean": _first_number(
            multi_y_source.get("average_predicted_y"),
            stored_summary.get("predicted_y_mean"),
            stored_summary.get("average_predicted_yield"),
        ),
        "failure_rates": failure_rates or _number_map(stored_summary.get("failure_rates")),
        "fail_bit_counts": fail_bit_counts or _number_map(stored_summary.get("fail_bit_counts")),
    }
    causes = {
        "top_failure_target": top_failure_target,
        "top_features": top_features[:5], "top_steps": top_steps[:5],
        "top_equipment": top_equipment[:5], "top_chambers": chambers,
    }

    artifact_status = metadata.get("artifact_status")
    if artifact_status not in {"available", "missing", "corrupted"}:
        artifact_status = "available" if artifact else "missing"
    status_value = metadata.get("status")
    source_status = status_value if isinstance(status_value, str) else "partial"
    warnings = _strings(analysis.get("warnings"))
    if artifact_status == "missing":
        warnings.append("분석 결과 Artifact가 없어 저장된 메타데이터만 표시합니다.")
    elif artifact_status == "corrupted":
        warnings.append("분석 결과 Artifact가 손상되어 저장된 메타데이터만 표시합니다.")
    if metadata.get("metadata_decode_errors"):
        warnings.append("일부 Legacy 메타데이터를 읽을 수 없어 제공 가능한 항목만 표시합니다.")
    warnings = list(dict.fromkeys(warnings))

    availability = {
        "summary": any(value is not None for value in summary.values()),
        "model_metrics": any(value is not None for value in model_metrics.values()),
        "multi_y": any(value is not None for key, value in multi_y.items() if key.endswith("_mean"))
        or bool(multi_y["failure_rates"] or multi_y["fail_bit_counts"]),
        "causes": bool(top_failure_target or top_features or top_steps or top_equipment or chambers),
        "risk_lots": bool(risk_lots), "risk_wafers": bool(risk_wafers),
        "pareto": bool(pareto), "relationships": bool(relationships),
    }
    source = {
        "type": "analysis", "analysis_id": metadata.get("analysis_id"),
        "created_at": metadata.get("created_at") or analysis.get("created_at"),
        "completed_at": metadata.get("completed_at"), "status": source_status,
        "source_filename": metadata.get("source_filename") or _record(analysis.get("dataset")).get("filename"),
        "model_id": metadata.get("model_id") or model.get("model_id"),
        "model_name": metadata.get("model_name_snapshot") or model.get("model_name"),
        "artifact_available": artifact_status == "available",
        "artifact_status": artifact_status,
    }
    legacy_model = {
        "model_id": source["model_id"], "model_name": source["model_name"],
        "model_type": metadata.get("model_type_snapshot"),
        "compatibility": model.get("compatibility", "snapshot"),
        "cv_r2_mean": model_metrics["r2"], "cv_r2_std": None,
        "cv_rmse_mean": model_metrics["rmse"], "mae": model_metrics["mae"],
    }
    return {
        "source": source, "summary": summary, "model_metrics": model_metrics,
        "multi_y": multi_y, "causes": causes, "risk_lots": risk_lots[:5],
        "risk_wafers": risk_wafers[:5], "pareto": pareto[:10],
        "relationships": relationships, "warnings": warnings,
        "availability": availability,
        "source_type": "analysis", "source_id": source["analysis_id"],
        "created_at": source["created_at"],
        "source_label": "선택한 원인 분석" if explicitly_selected else "최근 원인 분석",
        "filename": source["source_filename"], "model": legacy_model,
        "data_quality": _record(analysis.get("data_quality")),
    }


@router.get("/dashboard/overview", response_model=AnalysisOverviewResponse)
def get_dashboard_overview(analysis_id: str | None = None) -> dict[str, Any]:
    store = get_runtime_store()
    if analysis_id:
        selected = store.get_analysis(analysis_id)
        if selected is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="선택한 분석 이력을 찾을 수 없습니다.")
        return _overview_analysis(selected, explicitly_selected=True)

    listing = store.list_analyses({"limit": 100, "offset": 0, "sort": "newest"})
    items = listing.get("items", []) if isinstance(listing, dict) else []
    selected_item = next((item for item in items if item.get("status") == "completed"), None)
    if selected_item is None:
        selected_item = next((item for item in items if item.get("status") in {"partial", "running"}), None)
    if selected_item and selected_item.get("analysis_id"):
        detail = store.get_analysis(str(selected_item["analysis_id"]))
        if detail is not None:
            return _overview_analysis(detail)

    snapshot = store.latest_analysis_snapshot()
    if snapshot is not None:
        # Snapshot is complete overview material and never needs a model/artifact load.
        overview = _empty_overview()
        overview["has_analysis"] = True
        overview["source"] = {**overview["source"], "generated_at": snapshot.get("analyzed_at"), "model_id": snapshot.get("active_model_id"), "dataset_version": snapshot.get("dataset_version"), "stale": bool(snapshot.get("stale")), "stale_reason": snapshot.get("stale_reason")}
        overview["analysis"] = snapshot
        return overview
    return _empty_overview()
