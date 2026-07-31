"use client";

import type { ChangeEvent } from "react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import {
  downloadExplanation,
  explainCsv,
  getModels,
} from "@/lib/api";
import type {
  ExplainOptions,
  ExplainResponse,
  ModelSummary,
} from "@/types/data";

const DEFAULT_OPTIONS: ExplainOptions = {
  max_rows: 500,
  top_n: 20,
  per_wafer_top_n: 5,
};

function formatNumber(value: number): string {
  return value.toLocaleString("ko-KR", {
    maximumFractionDigits: 5,
  });
}

function formatFeatureLabel(value: string): string {
  const match = value.match(/^Step_?(\d+)_?(R|D|EQ)(?:_(.+))?$/i);
  if (!match) {
    return value.length > 24 ? `${value.slice(0, 22)}…` : value;
  }
  const [, step, rawType, parameter] = match;
  const typeLabel =
    rawType.toUpperCase() === "R"
      ? "Response"
      : rawType.toUpperCase() === "D"
        ? "Delta"
        : "Equipment";
  const suffix = parameter
    ? ` · ${parameter.replaceAll("_", " ")}`
    : "";
  const formatted = `Step ${step} · ${typeLabel}${suffix}`;
  return formatted.length > 30 ? `${formatted.slice(0, 28)}…` : formatted;
}

function riskLabel(risk: string | null): string {
  if (risk === "danger") return "위험";
  if (risk === "warning") return "주의";
  return "정상";
}

export default function RootCausePage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] =
    useState<ExplainOptions>(DEFAULT_OPTIONS);
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [selectedWafer, setSelectedWafer] = useState(0);
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
      .catch((requestError: unknown) => {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "모델 목록을 불러오지 못했습니다.",
        );
      });
  }, []);

  const selected = result?.wafer_explanations[selectedWafer];
  const globalChart = useMemo(
    () =>
      (result?.global_importance ?? []).slice(0, 15).map((item) => ({
        name: item.feature,
        value: item.mean_harmful_contribution,
        step: item.step,
        parameterType: item.parameter_type,
        meanAbsShap: item.mean_abs_shap,
      })),
    [result],
  );
  const stepChart = useMemo(
    () =>
      (result?.step_summary ?? []).slice(0, 10).map((item) => ({
        name: item.step,
        value: item.harmful_contribution,
      })),
    [result],
  );
  const typeChart = useMemo(
    () =>
      (result?.parameter_type_summary ?? []).map((item) => ({
        name: item.parameter_type,
        value: item.harmful_contribution,
      })),
    [result],
  );

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setResult(null);
    setError("");
    if (selectedFile && !selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    setFile(selectedFile);
  }

  async function runAnalysis() {
    if (!file || !modelId || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await explainCsv(file, modelId, options));
      setSelectedWafer(0);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "원인 분석 중 오류가 발생했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv() {
    if (!file || !modelId || downloading) return;
    setDownloading(true);
    setError("");
    try {
      const blob = await downloadExplanation(file, modelId, options);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `explanation_${modelId}.csv`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "결과 다운로드 중 오류가 발생했습니다.",
      );
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar activeItem="원인 분석" />
      <div className="contentShell">
        <Header />
        <main className="mainContent rootCausePage">
          <section className="intro">
            <div>
              <span className="eyebrow">Explainable AI</span>
              <h1>SHAP 기반 원인 후보 분석</h1>
              <p>
                저장된 모델의 예측을 공정 단계·파라미터·Wafer별로
                분해합니다.
              </p>
            </div>
          </section>

          <section className="resultCard analysisControls">
            <div className="fieldGroup">
              <label htmlFor="analysis-model">학습 모델</label>
              <select
                id="analysis-model"
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
              <label htmlFor="analysis-file">분석 CSV</label>
              <input
                id="analysis-file"
                type="file"
                accept=".csv,text/csv"
                onChange={handleFile}
              />
            </div>
            <div className="analysisOptionGrid">
              {(
                [
                  ["max_rows", "최대 분석 행", 1, 1000],
                  ["top_n", "전체 Top N", 1, 100],
                  ["per_wafer_top_n", "Wafer별 Top N", 1, 20],
                ] as const
              ).map(([key, label, min, max]) => (
                <div className="fieldGroup" key={key}>
                  <label htmlFor={key}>{label}</label>
                  <input
                    id={key}
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
                onClick={() => void runAnalysis()}
              >
                {loading ? "SHAP 계산 중..." : "원인 분석"}
              </button>
              <button
                className="button secondary"
                type="button"
                disabled={!result || downloading}
                data-loading={downloading}
                aria-busy={downloading}
                onClick={() => void downloadCsv()}
              >
                {downloading ? "다운로드 중..." : "전체 중요도 CSV"}
              </button>
            </div>
            <p className="emptyMessage">
              SHAP 계산은 데이터와 모델 크기에 따라 시간이 걸릴 수 있습니다.
              기본 분석 한도는 위험도 우선 500행입니다.
            </p>
            {!models.length && (
              <p className="emptyMessage">
                저장된 모델이 없습니다. 먼저 <a href="/training">모델 학습</a>을
                진행해 주세요.
              </p>
            )}
            {error && <p className="errorMessage">{error}</p>}
          </section>

          {result && (
            <>
              <section className="resultCard">
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">분석 요약</span>
                    <h2>
                      {result.analysis_summary.analyzed_rows.toLocaleString()}
                      개 행 설명
                    </h2>
                  </div>
                  <p>
                    {result.model.model_name} ·{" "}
                    {result.analysis_summary.explanation_method}
                    {result.analysis_summary.is_fallback
                      ? " (모델 독립 대체 방식)"
                      : ""}
                  </p>
                </div>
                <p className="analysisDisclaimer">
                  본 분석은 머신러닝 모델의 예측 기여도를 설명합니다. SHAP
                  값이 높은 feature가 실제 불량의 직접 원인임을 확정하는 것은
                  아닙니다.
                </p>
                {[...result.model_quality_warnings, ...result.warnings].map(
                  (warning) => (
                    <p className="warningMessage" key={warning}>
                      {warning}
                    </p>
                  ),
                )}
              </section>

              <section className="analysisChartGrid">
                <ChartCard
                  title="전체 위험 기여도 Top 15"
                  description="값이 클수록 모델 예측을 위험 방향으로 더 크게 이동시킨 feature입니다."
                  data={globalChart}
                  color="#b33a46"
                />
                <ChartCard
                  title="공정 Step별 위험 기여도 Top 10"
                  description="동일 Step에 속한 feature의 위험 기여도를 합산해 비교합니다."
                  data={stepChart}
                  color="#1769aa"
                />
                <ChartCard
                  title="파라미터 유형별 위험 기여도"
                  description="Response · Delta · Equipment 유형별 영향 크기를 비교합니다."
                  data={typeChart}
                  color="#a96208"
                />
              </section>

              <section className="resultCard">
                <div className="sectionHeading compact">
                  <div>
                    <span className="sectionLabel">개별 설명</span>
                    <h2>Wafer별 기여 변수</h2>
                  </div>
                </div>
                <div className="tableWrapper">
                  <table className="dataTable">
                    <thead>
                      <tr>
                        <th>{result.identifier_column}</th>
                        <th>예측값</th>
                        <th>위험도</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.wafer_explanations.map((wafer, index) => (
                        <tr
                          className={
                            selectedWafer === index ? "selectedRow" : ""
                          }
                          key={`${String(wafer.identifier)}-${index}`}
                          onClick={() => setSelectedWafer(index)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedWafer(index);
                            }
                          }}
                          tabIndex={0}
                          role="button"
                          aria-label={`${String(wafer.identifier)} 상세 기여도 ${
                            selectedWafer === index ? "선택됨" : "보기"
                          }`}
                        >
                          <td>{String(wafer.identifier)}</td>
                          <td>{formatNumber(wafer.prediction)}</td>
                          <td>
                            <span
                              className={`riskBadge ${
                                wafer.risk_level ?? "normal"
                              }`}
                            >
                              {riskLabel(wafer.risk_level)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {selected && (
                  <>
                    <p className="waferDetailSummary">
                      {result.identifier_column}:{" "}
                      <strong>{String(selected.identifier)}</strong> · 예측값{" "}
                      <strong>{formatNumber(selected.prediction)}</strong> ·
                      기준값 <strong>{formatNumber(selected.base_value)}</strong>{" "}
                      · 위험도{" "}
                      <strong>{riskLabel(selected.risk_level)}</strong>
                    </p>
                    <div className="contributionGrid">
                      <ContributionList
                        title="수율 악화 기여"
                        rows={selected.top_negative_contributors}
                        field="harmful_contribution"
                      />
                      <ContributionList
                        title="수율 개선 기여"
                        rows={selected.top_positive_contributors}
                        field="beneficial_contribution"
                      />
                    </div>
                  </>
                )}
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  description,
  data,
  color,
}: {
  title: string;
  description: string;
  data: {
    name: string;
    value: number;
    step?: string;
    parameterType?: string;
    meanAbsShap?: number;
  }[];
  color: string;
}) {
  return (
    <article className="resultCard analysisChartCard">
      <h3>{title}</h3>
      <p className="chartDescription">{description}</p>
      <div
        className="chartCanvas"
        role="img"
        aria-label={`${title}. ${description}`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 18, bottom: 4, left: 20 }}
          >
            <CartesianGrid
              stroke="rgba(0, 0, 0, 0.07)"
              strokeDasharray="3 5"
              horizontal={false}
            />
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#86868b" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={120}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#6e6e73" }}
              tickFormatter={formatFeatureLabel}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const item = payload[0].payload as (typeof data)[number];
                return (
                  <div className="chartTooltip">
                    <strong>{item.name}</strong>
                    {item.step && <span>Step: {item.step}</span>}
                    {item.parameterType && (
                      <span>유형: {item.parameterType}</span>
                    )}
                    {item.meanAbsShap !== undefined && (
                      <span>
                        전체 중요도: {formatNumber(item.meanAbsShap)}
                      </span>
                    )}
                    <span>위험 기여도: {formatNumber(item.value)}</span>
                  </div>
                );
              }}
            />
            <Bar
              dataKey="value"
              radius={[0, 6, 6, 0]}
              animationDuration={220}
            >
              {data.map((item) => (
                <Cell key={item.name} fill={color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function ContributionList({
  title,
  rows,
  field,
}: {
  title: string;
  rows: ExplainResponse["wafer_explanations"][number]["top_negative_contributors"];
  field: "harmful_contribution" | "beneficial_contribution";
}) {
  return (
    <div>
      <h3>{title}</h3>
      <p className="chartDescription">
        {field === "harmful_contribution"
          ? "값이 클수록 해당 Wafer의 수율을 낮추는 방향입니다."
          : "값이 클수록 해당 Wafer의 수율을 높이는 방향입니다."}
      </p>
      <div
        className="localChart"
        role="img"
        aria-label={`${title} feature 영향도 막대 차트`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows.map((row) => ({
              name: row.feature,
              value: row[field],
            }))}
            layout="vertical"
            margin={{ left: 12 }}
          >
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#86868b" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={112}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#6e6e73" }}
              tickFormatter={formatFeatureLabel}
            />
            <Tooltip
              formatter={(value) => [
                formatNumber(Number(value)),
                "기여도",
              ]}
              contentStyle={{
                border: "1px solid rgba(0, 0, 0, 0.08)",
                borderRadius: 12,
                boxShadow: "0 12px 30px rgba(35, 42, 52, 0.08)",
              }}
            />
            <Bar
              dataKey="value"
              fill={
                field === "harmful_contribution" ? "#b33a46" : "#287a5b"
              }
              radius={[0, 6, 6, 0]}
              animationDuration={220}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ol className="contributionList">
        {rows.map((row) => (
          <li key={row.feature}>
            <div>
              <strong title={row.feature}>{formatFeatureLabel(row.feature)}</strong>
              <span>
                {row.step} · {row.parameter_type} · 값{" "}
                {String(row.value ?? "-")}
              </span>
            </div>
            <b>{formatNumber(row[field])}</b>
          </li>
        ))}
      </ol>
    </div>
  );
}
