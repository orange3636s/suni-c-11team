from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from src.ml.model_io import to_json_safe


def _alert(
    danger_count: int,
    warning_count: int,
) -> dict[str, Any]:
    if danger_count > 0:
        return {
            "required": True,
            "severity": "danger",
            "reason": f"위험 Wafer {danger_count:,}개 탐지",
            "danger_count": danger_count,
            "warning_count": warning_count,
        }
    if warning_count > 0:
        return {
            "required": True,
            "severity": "warning",
            "reason": f"주의 Wafer {warning_count:,}개 탐지",
            "danger_count": 0,
            "warning_count": warning_count,
        }
    return {
        "required": False,
        "severity": "normal",
        "reason": "위험 또는 주의 Wafer가 탐지되지 않음",
        "danger_count": 0,
        "warning_count": 0,
    }


def build_automation_response(
    report: dict[str, Any],
    *,
    include_report: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now().astimezone()
    executive = report["executive_summary"]
    risk_wafers = report["top_risk_wafers"]
    predicted_values = [
        float(item["predicted_value"])
        for item in risk_wafers
        if item.get("predicted_value") is not None
    ]
    minimum_prediction = min(predicted_values) if predicted_values else None
    danger_count = int(executive["danger_count"])
    warning_count = int(executive["warning_count"])
    risk_count = danger_count + warning_count
    alert = _alert(danger_count, warning_count)
    severity_label = {
        "danger": "위험",
        "warning": "주의",
        "normal": "정상",
    }[alert["severity"]]
    top_feature = (
        report["top_features"][0]["feature"]
        if report["top_features"]
        else None
    )
    summary = {
        "total_wafers": executive["total_wafers"],
        "average_predicted_yield": executive[
            "average_predicted_yield"
        ],
        "normal_count": executive["normal_count"],
        "warning_count": warning_count,
        "danger_count": danger_count,
        "risk_count": risk_count,
        "risk_ratio": executive["risk_ratio"],
        "minimum_predicted_yield": minimum_prediction,
    }
    automation_message = {
        "title": f"[{severity_label}] 제조 공정 수율 분석 알림",
        "summary": (
            f"총 {summary['total_wafers']:,}개 Wafer 중 위험 "
            f"{danger_count:,}개, 주의 {warning_count:,}개가 "
            "탐지되었습니다."
        ),
        "detail": (
            f"평균 예측 수율 "
            f"{float(summary['average_predicted_yield']):.2f}%, "
            + (
                f"최저 예측 수율 {minimum_prediction:.2f}%"
                if minimum_prediction is not None
                else "최저 예측 수율 정보 없음"
            )
        ),
        "top_cause": (
            f"주요 모델 기반 원인 후보: {top_feature}"
            if top_feature
            else "탐지된 모델 기반 원인 후보가 없습니다."
        ),
    }
    response = {
        "success": True,
        "analysis_id": (
            f"analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid4().hex[:6]}"
        ),
        "created_at": timestamp.isoformat(),
        "filename": report["filename"],
        "model": {
            "model_id": report["model"]["model_id"],
            "target": report["model"]["target"],
            "model_name": report["model"]["model_name"],
            "test_r2": report["model"]["test_metrics"].get("r2"),
            "test_rmse": report["model"]["test_metrics"].get("rmse"),
        },
        "summary": summary,
        "alert": alert,
        "automation_message": automation_message,
        "top_findings": report["key_findings"],
        "top_risk_wafers": risk_wafers,
        "top_features": report["top_features"],
        "top_steps": report["top_steps"],
        "parameter_type_summary": report["parameter_type_summary"],
        "model_quality_warnings": report["model_quality_warnings"],
        "report": {
            "included": include_report,
            "report_id": report["report_id"] if include_report else None,
            "download_endpoint": (
                "/api/report/download" if include_report else None
            ),
        },
        "warnings": report["warnings"],
    }
    return to_json_safe(response)
