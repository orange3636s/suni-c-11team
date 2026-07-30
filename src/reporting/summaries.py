from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from src.ml.explainability import ExplainResult
from src.ml.inference import PredictionResult
from src.preprocessing import _extract_lot_values


RISK_ORDER = {"danger": 0, "warning": 1, "normal": 2, None: 3}


def build_executive_summary(
    prediction: PredictionResult,
    explanation: ExplainResult,
) -> dict[str, Any]:
    risk_count = prediction.warning_count + prediction.danger_count
    risk_ratio = (
        risk_count / prediction.total_rows
        if prediction.total_rows
        else 0.0
    )
    return {
        "total_wafers": prediction.total_rows,
        "average_predicted_yield": prediction.average_prediction,
        "normal_count": prediction.normal_count,
        "warning_count": prediction.warning_count,
        "danger_count": prediction.danger_count,
        "risk_ratio": risk_ratio,
        "analyzed_rows": explanation.analyzed_rows,
        "shap_sampling_used": explanation.sampling_used,
        "sampling_strategy": explanation.sampling_strategy,
    }


def _explanation_by_identifier(
    explanation: ExplainResult,
) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("identifier")): item
        for item in explanation.wafer_explanations
    }


def build_top_risk_wafers(
    prediction: PredictionResult,
    explanation: ExplainResult,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    target = prediction.target
    prediction_column = f"predicted_{target}"
    actual_column = f"actual_{target}"
    local_by_id = _explanation_by_identifier(explanation)
    ordered = sorted(
        prediction.predictions,
        key=lambda row: (
            RISK_ORDER.get(row.get("risk_level"), 3),
            float(row.get(prediction_column, float("inf"))),
        ),
    )
    results: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        identifier = row.get(prediction.identifier_column)
        local = local_by_id.get(str(identifier), {})
        contributors = local.get("top_negative_contributors", [])
        results.append(
            {
                "identifier": identifier,
                "predicted_value": row.get(prediction_column),
                "risk_level": row.get("risk_level"),
                "actual_value": row.get(actual_column),
                "absolute_error": row.get("absolute_error"),
                "top_harmful_features": [
                    item["feature"] for item in contributors
                ],
                "top_step": (
                    contributors[0].get("step") if contributors else None
                ),
                "top_parameter_type": (
                    contributors[0].get("parameter_type")
                    if contributors
                    else None
                ),
            }
        )
    return results


def build_lot_summaries(
    prediction: PredictionResult,
    explanation: ExplainResult,
) -> tuple[list[dict[str, Any]], list[str]]:
    identifiers = [
        row.get(prediction.identifier_column)
        for row in prediction.predictions
    ]
    lots = _extract_lot_values(pd.Series(identifiers, dtype="string"))
    if lots.isna().any():
        return [], [
            "일부 Wafer 식별자에서 Lot 정보를 추출할 수 없어 "
            "LOT별 위험 요약을 생략했습니다."
        ]

    local_by_id = _explanation_by_identifier(explanation)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lot, row in zip(lots.tolist(), prediction.predictions, strict=True):
        groups[str(lot)].append(row)

    target_column = f"predicted_{prediction.target}"
    summaries: list[dict[str, Any]] = []
    for lot_id, rows in groups.items():
        feature_scores: Counter[str] = Counter()
        step_scores: Counter[str] = Counter()
        for row in rows:
            identifier = row.get(prediction.identifier_column)
            local = local_by_id.get(str(identifier), {})
            for item in local.get("top_negative_contributors", []):
                score = float(item.get("harmful_contribution", 0.0))
                feature_scores[str(item.get("feature"))] += score
                step_scores[str(item.get("step"))] += score
        danger_count = sum(
            row.get("risk_level") == "danger" for row in rows
        )
        warning_count = sum(
            row.get("risk_level") == "warning" for row in rows
        )
        normal_count = sum(
            row.get("risk_level") == "normal" for row in rows
        )
        summaries.append(
            {
                "lot_id": lot_id,
                "wafer_count": len(rows),
                "average_predicted_yield": sum(
                    float(row[target_column]) for row in rows
                )
                / len(rows),
                "danger_count": danger_count,
                "warning_count": warning_count,
                "normal_count": normal_count,
                "danger_ratio": danger_count / len(rows),
                "top_harmful_feature": (
                    feature_scores.most_common(1)[0][0]
                    if feature_scores
                    else None
                ),
                "top_harmful_step": (
                    step_scores.most_common(1)[0][0]
                    if step_scores
                    else None
                ),
            }
        )
    summaries.sort(
        key=lambda item: (
            item["danger_ratio"],
            item["danger_count"],
            -item["average_predicted_yield"],
        ),
        reverse=True,
    )
    return summaries, []


def parameter_type_with_ratios(
    explanation: ExplainResult,
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in explanation.parameter_type_summary]
    total = sum(float(item["harmful_contribution"]) for item in rows)
    for item in rows:
        item["ratio"] = (
            float(item["harmful_contribution"]) / total
            if total > 0
            else None
        )
    return rows


def build_key_findings(
    executive: dict[str, Any],
    explanation: ExplainResult,
    lot_summaries: list[dict[str, Any]],
    parameter_summary: list[dict[str, Any]],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    findings.append(
        {
            "severity": (
                "danger" if executive["danger_count"] else "info"
            ),
            "title": "Wafer 위험 분류",
            "description": (
                f"총 {executive['total_wafers']:,}개 Wafer 중 "
                f"{executive['danger_count']:,}개가 위험 구간으로 "
                "분류되었습니다."
            ),
            "evidence": (
                f"주의 또는 위험 비율 "
                f"{executive['risk_ratio'] * 100:.1f}%"
            ),
        }
    )
    findings.append(
        {
            "severity": "info",
            "title": "평균 예측값",
            "description": (
                f"평균 예측 수율은 "
                f"{executive['average_predicted_yield']:.2f}%입니다."
            ),
            "evidence": (
                f"분석 대상 {executive['total_wafers']:,}개 Wafer"
            ),
        }
    )
    if explanation.step_summary:
        top = explanation.step_summary[0]
        findings.append(
            {
                "severity": "warning",
                "title": "주요 공정 단계 후보",
                "description": (
                    f"{top['step']}이 모델 기반 주요 원인 후보 중 "
                    "가장 높은 예측 기여도를 보였습니다."
                ),
                "evidence": (
                    f"harmful contribution "
                    f"{float(top['harmful_contribution']):.4f}"
                ),
            }
        )
    if parameter_summary:
        top = parameter_summary[0]
        ratio = top.get("ratio")
        findings.append(
            {
                "severity": "warning",
                "title": "주요 파라미터 유형 후보",
                "description": (
                    f"{top['parameter_type']} 계열 파라미터의 "
                    "예측 기여도가 가장 높게 나타났습니다."
                ),
                "evidence": (
                    f"전체 harmful contribution의 {ratio * 100:.1f}%"
                    if ratio is not None
                    else "전체 harmful contribution이 0입니다."
                ),
            }
        )
    if explanation.global_importance:
        top = explanation.global_importance[0]
        findings.append(
            {
                "severity": "warning",
                "title": "상위 Feature 후보",
                "description": (
                    f"{top['feature']}가 모델 기반 주요 원인 후보로 "
                    "상위에 탐지되었습니다."
                ),
                "evidence": (
                    f"평균 위험 기여도 "
                    f"{float(top['mean_harmful_contribution']):.4f}"
                ),
            }
        )
    if lot_summaries and lot_summaries[0]["danger_count"] > 0:
        top = lot_summaries[0]
        findings.append(
            {
                "severity": "danger",
                "title": "위험 LOT 우선순위",
                "description": (
                    f"{top['lot_id']}에서 위험 예측이 상대적으로 "
                    "집중되었습니다."
                ),
                "evidence": (
                    f"위험 Wafer {top['danger_count']}개 / "
                    f"전체 {top['wafer_count']}개"
                ),
            }
        )
    if explanation.sampling_used:
        findings.append(
            {
                "severity": "info",
                "title": "SHAP 표본 분석",
                "description": (
                    "위험도가 높은 Wafer를 우선하여 SHAP 분석을 "
                    "수행했습니다."
                ),
                "evidence": (
                    f"{explanation.total_rows:,}개 중 "
                    f"{explanation.analyzed_rows:,}개 분석"
                ),
            }
        )
    return findings[:7]


def build_recommendations(
    executive: dict[str, Any],
    explanation: ExplainResult,
    lot_summaries: list[dict[str, Any]],
    parameter_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if executive["danger_count"] > 0:
        scope = (
            f" 특히 {lot_summaries[0]['lot_id']}을 우선 확인하세요."
            if lot_summaries
            else ""
        )
        recommendations.append(
            {
                "priority": "high",
                "title": "위험 Wafer 우선 검토",
                "description": (
                    "위험 구간 Wafer의 공정 이력과 측정 결과를 "
                    f"엔지니어가 우선 검토하세요.{scope}"
                ),
            }
        )
    if explanation.step_summary:
        step = explanation.step_summary[0]["step"]
        recommendations.append(
            {
                "priority": "medium",
                "title": f"{step} 공정 로그 검토",
                "description": (
                    f"{step} 관련 로그, 장비 이력 및 레시피 변경 "
                    "내역을 우선 비교하세요."
                ),
            }
        )
    if parameter_summary:
        parameter_type = parameter_summary[0]["parameter_type"]
        recommendations.append(
            {
                "priority": "medium",
                "title": f"{parameter_type} 계열 파라미터 확인",
                "description": (
                    f"반복 탐지된 {parameter_type} 계열 값의 분포와 "
                    "공정 기록을 확인하세요."
                ),
            }
        )
    test_r2 = metadata.get("metrics", {}).get("test", {}).get("r2")
    if test_r2 is None or float(test_r2) <= 0:
        recommendations.append(
            {
                "priority": "high",
                "title": "모델 개선 우선",
                "description": (
                    "모델 성능이 제한적이므로 추가 데이터 확보와 "
                    "재학습 후 결과를 다시 검증하세요."
                ),
            }
        )
    if metadata.get("model_name") == "DummyRegressor":
        recommendations.append(
            {
                "priority": "high",
                "title": "기준 모델 교체 검토",
                "description": (
                    "DummyRegressor 결과이므로 원인 후보 활용 전에 "
                    "설명력이 있는 모델을 재학습하세요."
                ),
            }
        )
    if explanation.is_fallback:
        recommendations.append(
            {
                "priority": "medium",
                "title": "설명 방식 한계 확인",
                "description": (
                    "SHAP 대신 모델 독립형 fallback이 사용되었으므로 "
                    "후보 순위를 보조 근거로만 활용하세요."
                ),
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": "low",
                "title": "정기 검토 유지",
                "description": (
                    "상위 원인 후보와 신규 Wafer 예측 변화를 정기적으로 "
                    "엔지니어가 검토하세요."
                ),
            }
        )
    return recommendations
