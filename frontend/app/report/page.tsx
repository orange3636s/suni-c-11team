"use client";

import type { ChangeEvent } from "react";
import { useEffect, useState } from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import {
  downloadReport,
  generateReport,
  getModels,
} from "@/lib/api";
import type {
  ModelSummary,
  ReportOptions,
  ReportResponse,
} from "@/types/data";

const DEFAULT_OPTIONS: ReportOptions = {
  warning_threshold: 95,
  danger_threshold: 90,
  max_rows: 500,
  top_n: 20,
};

function number(value: number | null, digits = 2): string {
  return value === null
    ? "-"
    : value.toLocaleString("ko-KR", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
}

function riskLabel(value: string | null): string {
  if (value === "danger") return "위험";
  if (value === "warning") return "주의";
  if (value === "normal") return "정상";
  return "-";
}

export default function ReportPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] =
    useState<ReportOptions>(DEFAULT_OPTIONS);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    void getModels()
      .then((response) => {
        setModels(response.models);
        setModelId(
          response.models.find((model) => model.target === "Y")
            ?.model_id ??
            response.models[0]?.model_id ??
            "",
        );
      })
      .catch((requestError: unknown) =>
        setError(
          requestError instanceof Error
            ? requestError.message
            : "모델 목록을 불러오지 못했습니다.",
        ),
      );
  }, []);

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setReport(null);
    setError("");
    if (selected && !selected.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    if (selected && selected.size > 20 * 1024 * 1024) {
      setFile(null);
      setError("파일 크기는 20MB 이하여야 합니다.");
      return;
    }
    setFile(selected);
  }

  async function createReport() {
    if (!file || !modelId || loading) return;
    if (options.warning_threshold <= options.danger_threshold) {
      setError("주의 기준값은 위험 기준값보다 커야 합니다.");
      return;
    }
    setLoading(true);
    setError("");
    setReport(null);
    try {
      setReport(await generateReport(file, modelId, options));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "분석 보고서 생성 중 오류가 발생했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function saveHtml() {
    if (!file || !modelId || downloading) return;
    setDownloading(true);
    setError("");
    try {
      const blob = await downloadReport(file, modelId, options);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `manufacturing_ai_report_${Date.now()}.html`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "HTML 보고서 다운로드 중 오류가 발생했습니다.",
      );
    } finally {
      setDownloading(false);
    }
  }

  const qualityWarnings = report
    ? [...report.model_quality_warnings, ...report.warnings]
    : [];

  return (
    <div className="appShell">
      <Sidebar activeItem="분석 보고서" />
      <div className="contentShell">
        <Header />
        <main className="mainContent reportPage">
          <section className="intro">
            <div>
              <span className="eyebrow">Engineering Report</span>
              <h1>자동 분석 보고서</h1>
              <p>
                수율 예측과 SHAP 기반 원인 후보를 한 번에 분석해 엔지니어용
                보고서로 정리합니다.
              </p>
            </div>
          </section>

          <section className="resultCard reportControls">
            <div className="fieldGroup">
              <label htmlFor="report-model">학습 모델</label>
              <select
                id="report-model"
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
              >
                <option value="">모델 선택</option>
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.target} · {model.model_name} · {model.created_at}
                  </option>
                ))}
              </select>
            </div>
            <div className="fieldGroup">
              <label htmlFor="report-file">분석 CSV</label>
              <input
                id="report-file"
                type="file"
                accept=".csv,text/csv"
                onChange={handleFile}
              />
            </div>
            <div className="reportOptionGrid">
              {(
                [
                  ["warning_threshold", "주의 기준", 0, 100],
                  ["danger_threshold", "위험 기준", 0, 100],
                  ["max_rows", "최대 SHAP 행", 1, 1000],
                  ["top_n", "원인 후보 Top N", 1, 100],
                ] as const
              ).map(([key, label, min, max]) => (
                <div className="fieldGroup" key={key}>
                  <label htmlFor={`report-${key}`}>{label}</label>
                  <input
                    id={`report-${key}`}
                    type="number"
                    min={min}
                    max={max}
                    value={options[key]}
                    onChange={(event) =>
                      setOptions({
                        ...options,
                        [key]: Number(event.target.value),
                      })
                    }
                  />
                </div>
              ))}
            </div>
            <div className="uploadActions">
              <button
                className="button primary"
                type="button"
                disabled={!file || !modelId || loading}
                data-loading={loading}
                aria-busy={loading}
                onClick={() => void createReport()}
              >
                {loading ? "보고서 생성 중..." : "분석 보고서 생성"}
              </button>
              <button
                className="button secondary"
                type="button"
                disabled={!report || downloading}
                data-loading={downloading}
                aria-busy={downloading}
                onClick={() => void saveHtml()}
              >
                {downloading ? "다운로드 중..." : "HTML 보고서 다운로드"}
              </button>
            </div>
            {!loading && !report && (
              <p className="emptyMessage">
                CSV와 학습 모델을 선택한 뒤 분석 보고서를 생성하세요.
              </p>
            )}
            {loading && (
              <div className="reportProgress" role="status">
                <strong>
                  검증, 전처리, 예측 및 원인 분석을 수행하고 있습니다.
                </strong>
                <span>
                  데이터 검증 → 데이터 전처리 → 수율 예측 → 원인 분석 →
                  보고서 생성
                </span>
              </div>
            )}
            {!models.length && (
              <p className="warningMessage">
                저장 모델이 없습니다. 먼저 모델 학습을 진행해 주세요.
              </p>
            )}
            {error && <p className="errorMessage">{error}</p>}
          </section>

          {report && (
            <>
              {qualityWarnings.length > 0 && (
                <section className="reportWarningBanner">
                  <strong>모델 및 분석 주의사항</strong>
                  <ul>
                    {qualityWarnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </section>
              )}

              <section className="resultCard">
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">Executive Summary</span>
                    <h2>운영 요약</h2>
                  </div>
                  <p>
                    {report.model.model_name} · Test R²{" "}
                    {number(report.model.test_metrics.r2, 4)} ·{" "}
                    {report.explanation_method}
                  </p>
                </div>
                <div className="reportKpiGrid">
                  {[
                    ["분석 Wafer", report.executive_summary.total_wafers],
                    [
                      "평균 예측 수율",
                      `${number(
                        report.executive_summary.average_predicted_yield,
                      )}%`,
                    ],
                    ["정상", report.executive_summary.normal_count],
                    ["주의", report.executive_summary.warning_count],
                    ["위험", report.executive_summary.danger_count],
                    [
                      "주의·위험 비율",
                      `${number(
                        report.executive_summary.risk_ratio * 100,
                        1,
                      )}%`,
                    ],
                    ["SHAP 분석 행", report.executive_summary.analyzed_rows],
                    [
                      "SHAP 샘플링",
                      report.executive_summary.shap_sampling_used
                        ? "사용"
                        : "미사용",
                    ],
                  ].map(([label, value]) => (
                    <div className="metricCard" key={label}>
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
              </section>

              <section className="resultCard">
                <span className="sectionLabel">Key Findings</span>
                <h2>핵심 분석 결과</h2>
                <ol className="findingGrid">
                  {report.key_findings.map((finding) => (
                    <li
                      className={`findingCard ${finding.severity}`}
                      key={`${finding.title}-${finding.evidence}`}
                    >
                      <strong>{finding.title}</strong>
                      <p>{finding.description}</p>
                      <small>{finding.evidence}</small>
                    </li>
                  ))}
                </ol>
              </section>

              <ReportTables report={report} />

              <section className="reportTwoColumn">
                <article className="resultCard">
                  <span className="sectionLabel">Recommendations</span>
                  <h2>엔지니어 검토 권고</h2>
                  <ul className="recommendationList">
                    {report.recommendations.map((item) => (
                      <li key={item.title}>
                        <strong>{item.title}</strong>
                        <span>{item.description}</span>
                      </li>
                    ))}
                  </ul>
                </article>
                <article className="resultCard">
                  <span className="sectionLabel">Methodology</span>
                  <h2>분석 방법론과 한계</h2>
                  <ul className="methodologyList">
                    {report.methodology_notes.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </article>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function ReportTables({ report }: { report: ReportResponse }) {
  return (
    <>
      <section className="resultCard">
        <h2>위험 Wafer Top 목록</h2>
        <div className="tableWrapper">
          <table className="dataTable">
            <thead>
              <tr>
                <th>Wafer</th>
                <th>예측값</th>
                <th>위험도</th>
                <th>실제값</th>
                <th>절대오차</th>
                <th>상위 후보</th>
              </tr>
            </thead>
            <tbody>
              {report.top_risk_wafers.map((wafer, index) => (
                <tr key={`${String(wafer.identifier)}-${index}`}>
                  <td>{String(wafer.identifier)}</td>
                  <td>{number(wafer.predicted_value)}</td>
                  <td>
                    <span
                      className={`riskBadge ${wafer.risk_level ?? "normal"}`}
                    >
                      {riskLabel(wafer.risk_level)}
                    </span>
                  </td>
                  <td>{number(wafer.actual_value)}</td>
                  <td>{number(wafer.absolute_error)}</td>
                  <td>{wafer.top_harmful_features.join(", ") || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {report.lot_summary.length > 0 && (
        <section className="resultCard">
          <h2>LOT별 위험 요약</h2>
          <div className="tableWrapper">
            <table className="dataTable">
              <thead>
                <tr>
                  <th>LOT</th>
                  <th>Wafer 수</th>
                  <th>평균 예측</th>
                  <th>위험</th>
                  <th>주의</th>
                  <th>위험 비율</th>
                  <th>상위 후보</th>
                </tr>
              </thead>
              <tbody>
                {report.lot_summary.map((lot) => (
                  <tr key={lot.lot_id}>
                    <td>{lot.lot_id}</td>
                    <td>{lot.wafer_count}</td>
                    <td>{number(lot.average_predicted_yield)}</td>
                    <td>{lot.danger_count}</td>
                    <td>{lot.warning_count}</td>
                    <td>{number(lot.danger_ratio * 100, 1)}%</td>
                    <td>{lot.top_harmful_feature ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="reportThreeColumn">
        <RankList
          title="주요 Feature"
          rows={report.top_features.slice(0, 10).map((item) => ({
            label: item.feature,
            value: item.mean_harmful_contribution,
          }))}
        />
        <RankList
          title="주요 Step"
          rows={report.top_steps.slice(0, 10).map((item) => ({
            label: item.step,
            value: item.harmful_contribution,
          }))}
        />
        <RankList
          title="R/D/EQ 기여도"
          rows={report.parameter_type_summary.map((item) => ({
            label: `${item.parameter_type} ${
              item.ratio === null
                ? ""
                : `(${number(item.ratio * 100, 1)}%)`
            }`,
            value: item.harmful_contribution,
          }))}
        />
      </section>
    </>
  );
}

function RankList({
  title,
  rows,
}: {
  title: string;
  rows: { label: string; value: number }[];
}) {
  return (
    <article className="resultCard">
      <h2>{title}</h2>
      <ol className="rankList">
        {rows.map((row, index) => (
          <li key={`${row.label}-${index}`}>
            <span>
              <b>{index + 1}</b>
              {row.label}
            </span>
            <strong>{number(row.value, 4)}</strong>
          </li>
        ))}
      </ol>
    </article>
  );
}
