from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from src.ml.explainability import ExplainResult
from src.ml.inference import LoadedPredictionModel, PredictionResult
from src.ml.model_io import to_json_safe
from src.reporting.summaries import (
    build_executive_summary,
    build_key_findings,
    build_lot_summaries,
    build_recommendations,
    build_top_risk_wafers,
    parameter_type_with_ratios,
)


METHODOLOGY_NOTES = [
    "본 결과는 머신러닝 모델의 예측 결과입니다.",
    "SHAP은 모델 예측 기여도를 설명하며 실제 공정 인과관계를 확정하지 않습니다.",
    "모델 성능이 낮으면 원인 후보 해석의 신뢰도가 제한될 수 있습니다.",
    "랜덤 또는 합성 데이터로 학습한 모델은 실제 공정 성능이 낮을 수 있습니다.",
    "공정 조건을 변경하기 전에 엔지니어 검토와 현장 검증이 필요합니다.",
]


def build_report(
    filename: str,
    loaded: LoadedPredictionModel,
    prediction: PredictionResult,
    explanation: ExplainResult,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now().astimezone()
    executive = build_executive_summary(prediction, explanation)
    lot_summaries, lot_warnings = build_lot_summaries(
        prediction,
        explanation,
    )
    parameter_summary = parameter_type_with_ratios(explanation)
    model_warnings = list(explanation.model_quality_warnings)
    if explanation.is_fallback:
        model_warnings.append(
            "SHAP을 사용할 수 없어 모델 독립형 fallback 설명을 사용했습니다."
        )
    if explanation.sampling_used:
        model_warnings.append(
            "SHAP 분석은 전체 행이 아닌 위험도 우선 표본에 대해 수행했습니다."
        )
    report = {
        "success": True,
        "report_id": (
            f"report_{timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid4().hex[:8]}"
        ),
        "created_at": timestamp.isoformat(),
        "filename": filename,
        "model": {
            "model_id": loaded.model_id,
            "target": prediction.target,
            "model_name": prediction.model_name,
            "test_metrics": loaded.metadata.get("metrics", {}).get(
                "test",
                {"r2": None, "rmse": None, "mae": None},
            ),
        },
        "executive_summary": executive,
        "key_findings": build_key_findings(
            executive,
            explanation,
            lot_summaries,
            parameter_summary,
        ),
        "top_risk_wafers": build_top_risk_wafers(
            prediction,
            explanation,
        ),
        "lot_summary": lot_summaries,
        "top_features": explanation.global_importance,
        "top_steps": explanation.step_summary,
        "parameter_type_summary": parameter_summary,
        "recommendations": build_recommendations(
            executive,
            explanation,
            lot_summaries,
            parameter_summary,
            loaded.metadata,
        ),
        "model_quality_warnings": list(dict.fromkeys(model_warnings)),
        "methodology_notes": METHODOLOGY_NOTES,
        "explanation_method": explanation.explanation_method,
        "is_fallback": explanation.is_fallback,
        "warnings": list(
            dict.fromkeys(
                [
                    *prediction.warnings,
                    *explanation.warnings,
                    *lot_warnings,
                ]
            )
        ),
    }
    return to_json_safe(report)
