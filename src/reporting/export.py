from __future__ import annotations

from html import escape
from typing import Any


def _value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return escape(str(value))


def _table(
    headers: list[str],
    rows: list[list[Any]],
) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_value(value)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def render_report_html(report: dict[str, Any]) -> str:
    executive = report["executive_summary"]
    model = report["model"]
    metrics = model["test_metrics"]
    kpis = [
        ("분석 Wafer", f"{executive['total_wafers']:,}"),
        ("평균 예측 수율", f"{executive['average_predicted_yield']:.2f}%"),
        ("위험 Wafer", f"{executive['danger_count']:,}"),
        ("주의 Wafer", f"{executive['warning_count']:,}"),
        ("주의·위험 비율", f"{executive['risk_ratio'] * 100:.1f}%"),
        ("SHAP 분석 행", f"{executive['analyzed_rows']:,}"),
    ]
    finding_html = "".join(
        (
            f'<article class="finding {escape(item["severity"])}">'
            f'<h3>{escape(item["title"])}</h3>'
            f'<p>{escape(item["description"])}</p>'
            f'<small>{escape(item["evidence"])}</small></article>'
        )
        for item in report["key_findings"]
    )
    recommendation_html = "".join(
        (
            f"<li><strong>{escape(item['title'])}</strong>"
            f"<span>{escape(item['description'])}</span></li>"
        )
        for item in report["recommendations"]
    )
    warning_html = "".join(
        f"<li>{escape(item)}</li>"
        for item in [
            *report["model_quality_warnings"],
            *report["warnings"],
        ]
    ) or "<li>추가 경고가 없습니다.</li>"
    methodology_html = "".join(
        f"<li>{escape(item)}</li>"
        for item in report["methodology_notes"]
    )
    risk_table = _table(
        [
            "Wafer",
            "예측값",
            "위험도",
            "실제값",
            "절대오차",
            "상위 후보",
        ],
        [
            [
                item["identifier"],
                item["predicted_value"],
                item["risk_level"],
                item["actual_value"],
                item["absolute_error"],
                ", ".join(item["top_harmful_features"]),
            ]
            for item in report["top_risk_wafers"]
        ],
    )
    lot_table = _table(
        [
            "LOT",
            "Wafer 수",
            "평균 예측",
            "위험",
            "주의",
            "정상",
            "위험 비율",
            "상위 후보",
        ],
        [
            [
                item["lot_id"],
                item["wafer_count"],
                item["average_predicted_yield"],
                item["danger_count"],
                item["warning_count"],
                item["normal_count"],
                f"{item['danger_ratio'] * 100:.1f}%",
                item["top_harmful_feature"],
            ]
            for item in report["lot_summary"]
        ],
    )
    feature_table = _table(
        ["순위", "Feature", "Step", "유형", "전체 중요도", "위험 기여도"],
        [
            [
                item["rank"],
                item["feature"],
                item["step"],
                item["parameter_type"],
                item["mean_abs_shap"],
                item["mean_harmful_contribution"],
            ]
            for item in report["top_features"]
        ],
    )
    step_table = _table(
        ["순위", "Step", "전체 중요도", "위험 기여도", "Feature 수"],
        [
            [
                item["rank"],
                item["step"],
                item["mean_abs_shap"],
                item["harmful_contribution"],
                item["feature_count"],
            ]
            for item in report["top_steps"][:10]
        ],
    )
    parameter_table = _table(
        ["순위", "유형", "전체 중요도", "위험 기여도", "비율"],
        [
            [
                item["rank"],
                item["parameter_type"],
                item["mean_abs_shap"],
                item["harmful_contribution"],
                (
                    f"{item['ratio'] * 100:.1f}%"
                    if item["ratio"] is not None
                    else "-"
                ),
            ]
            for item in report["parameter_type_summary"]
        ],
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>제조 AI 분석 보고서</title>
<style>
:root{{--navy:#17243a;--muted:#657187;--border:#dfe5ed;--red:#bd344c;
--orange:#c8750c;--green:#16775a}}*{{box-sizing:border-box}}
body{{margin:0;background:#f4f6f9;color:var(--navy);font:14px/1.6 Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:40px 24px}}header,.card{{background:#fff;
border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px}}
h1{{margin:0 0 8px;font-size:28px}}h2{{margin:0 0 16px;font-size:19px}}
h3,p{{margin-top:0}}.muted,small{{color:var(--muted)}}.kpis{{display:grid;
grid-template-columns:repeat(3,1fr);gap:12px}}.kpi{{border:1px solid var(--border);
border-radius:12px;padding:16px}}.kpi strong{{display:block;font-size:22px}}
.findings{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
.finding{{border-left:4px solid #6a7b94;background:#f8fafc;padding:14px}}
.finding.danger{{border-color:var(--red)}}.finding.warning{{border-color:var(--orange)}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{border-bottom:1px solid var(--border);padding:9px;text-align:left}}
th{{background:#f4f6f9}}ul{{padding-left:20px}}li span{{display:block;color:var(--muted)}}
.danger-text{{color:var(--red)}}@media(max-width:700px){{.kpis,.findings{{grid-template-columns:1fr}}}}
@media print{{body{{background:#fff}}main{{max-width:none;padding:0}}.card,header{{break-inside:avoid}}}}
</style></head><body><main>
<header><h1>제조 공정 AI 자동 분석 보고서</h1>
<p class="muted">생성 시각: {escape(report["created_at"])} · 보고서 ID:
{escape(report["report_id"])}</p><p>파일: {escape(report["filename"])}</p></header>
<section class="card"><h2>모델 정보</h2><p>{escape(model["model_name"])} ·
Target {escape(model["target"])} · Test R² {_value(metrics.get("r2"), 4)} ·
RMSE {_value(metrics.get("rmse"), 4)} · MAE {_value(metrics.get("mae"), 4)}</p></section>
<section class="card"><h2>Executive Summary</h2><div class="kpis">
{''.join(f'<div class="kpi"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label, value in kpis)}
</div></section>
<section class="card"><h2>Key Findings</h2><div class="findings">{finding_html}</div></section>
<section class="card"><h2>위험 Wafer Top 목록</h2>{risk_table}</section>
<section class="card"><h2>LOT별 위험 현황</h2>{lot_table}</section>
<section class="card"><h2>Top Feature</h2>{feature_table}</section>
<section class="card"><h2>Top Step</h2>{step_table}</section>
<section class="card"><h2>R/D/EQ 유형별 기여도</h2>{parameter_table}</section>
<section class="card"><h2>엔지니어 검토 권고</h2><ul>{recommendation_html}</ul></section>
<section class="card"><h2>모델 성능 및 주의사항</h2><ul>{warning_html}</ul></section>
<section class="card"><h2>분석 방법론 및 한계</h2><ul>{methodology_html}</ul></section>
</main></body></html>"""
