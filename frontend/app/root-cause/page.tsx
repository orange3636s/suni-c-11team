"use client";

import type { ChangeEvent } from "react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import {
  analyzeRelationships,
  downloadExplanation,
  getModels,
} from "@/lib/api";
import type {
  ExplainOptions,
  ExplainResponse,
  ModelSummary,
  RelationshipAnalysisResponse,
  RelationshipFeature,
  RelationshipPath,
} from "@/types/data";

const DEFAULT_OPTIONS: ExplainOptions = {
  max_rows: 500,
  top_n: 10,
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
  const [relationships, setRelationships] =
    useState<RelationshipAnalysisResponse | null>(null);
  const [rankingMode, setRankingMode] =
    useState<"shap" | "correlation">("shap");
  const [rankingGroup, setRankingGroup] =
    useState<"overall" | "R" | "D" | "EQ">("overall");
  const [correlationMethod, setCorrelationMethod] =
    useState<"pearson" | "spearman">("pearson");
  const [selectedPath, setSelectedPath] = useState(0);
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
  const rankingData = useMemo(
    () => relationships?.rankings[rankingMode][rankingGroup] ?? [],
    [rankingGroup, rankingMode, relationships],
  );

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setResult(null);
    setRelationships(null);
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
      const response = await analyzeRelationships(
        file,
        modelId,
        options,
        correlationMethod,
      );
      setRelationships(response);
      setResult(response.explanation);
      setSelectedWafer(0);
      setSelectedPath(0);
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
              <div className="fieldGroup">
                <label htmlFor="top_n">Top N</label>
                <select
                  id="top_n"
                  value={options.top_n}
                  onChange={(event) =>
                    setOptions({
                      ...options,
                      top_n: Number(event.target.value),
                    })
                  }
                >
                  {[5, 10, 15, 20].map((value) => (
                    <option key={value} value={value}>
                      Top {value}
                    </option>
                  ))}
                </select>
              </div>
              <div className="fieldGroup">
                <label htmlFor="correlation-method">상관 방식</label>
                <select
                  id="correlation-method"
                  value={correlationMethod}
                  onChange={(event) =>
                    setCorrelationMethod(
                      event.target.value as "pearson" | "spearman",
                    )
                  }
                >
                  <option value="pearson">Pearson</option>
                  <option value="spearman">Spearman</option>
                </select>
              </div>
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

              {relationships && (
                <>
                  <section className="resultCard relationshipSection">
                    <div className="sectionHeading compact">
                      <div>
                        <span className="sectionLabel">1단계 · 영향 변수 순위</span>
                        <h2>전체 및 그룹별 Top N</h2>
                      </div>
                      <p>
                        Ranking basis:{" "}
                        {rankingData[0]?.ranking_basis ?? "데이터 없음"}
                      </p>
                    </div>
                    <div className="relationshipToolbar">
                      <SegmentedControl
                        options={[
                          ["shap", "SHAP"],
                          ["correlation", "Correlation"],
                        ]}
                        value={rankingMode}
                        onChange={(value) =>
                          setRankingMode(value as "shap" | "correlation")
                        }
                      />
                      <SegmentedControl
                        options={[
                          ["overall", "전체"],
                          ["R", "R"],
                          ["D", "D"],
                          ["EQ", "EQ"],
                        ]}
                        value={rankingGroup}
                        onChange={(value) =>
                          setRankingGroup(
                            value as "overall" | "R" | "D" | "EQ",
                          )
                        }
                      />
                    </div>
                    {rankingData.length ? (
                      <RankingChart data={rankingData} />
                    ) : (
                      <p className="emptyMessage">
                        선택한 기준의 순위 데이터가 없습니다.
                      </p>
                    )}
                  </section>

                  <ParetoSection analysis={relationships} />

                  <PathSection
                    paths={relationships.relationship_paths}
                    selectedIndex={selectedPath}
                    onSelect={setSelectedPath}
                    confidenceCriteria={relationships.confidence_criteria}
                  />
                </>
              )}

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

function SegmentedControl({
  options,
  value,
  onChange,
}: {
  options: [string, string][];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="segmentedControl">
      {options.map(([option, label]) => (
        <button
          key={option}
          type="button"
          className={value === option ? "active" : ""}
          aria-pressed={value === option}
          onClick={() => onChange(option)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function groupColor(group: string): string {
  if (group === "R") return "#1769aa";
  if (group === "D") return "#a96208";
  return "#647185";
}

function RankingChart({ data }: { data: RelationshipFeature[] }) {
  return (
    <div className="rankingLayout">
      <div className="chartCanvas relationshipChart" role="img">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 18, bottom: 4, left: 28 }}
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
              dataKey="display_name"
              width={138}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "#6e6e73" }}
              tickFormatter={formatFeatureLabel}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const item = payload[0].payload as RelationshipFeature;
                return (
                  <div className="chartTooltip">
                    <strong>{item.feature}</strong>
                    <span>유형: {item.group}</span>
                    <span>점수: {formatNumber(item.score ?? 0)}</span>
                    <span>방향: {item.direction}</span>
                    <span>
                      유효 표본: {item.valid_count?.toLocaleString() ?? "SHAP 기준"}
                    </span>
                  </div>
                );
              }}
            />
            <Bar dataKey="score" radius={[0, 6, 6, 0]} animationDuration={220}>
              {data.map((item) => (
                <Cell key={item.feature} fill={groupColor(item.group)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="tableWrapper rankingTable">
        <table className="dataTable">
          <thead>
            <tr>
              <th>순위</th><th>변수</th><th>유형</th><th>점수</th>
              <th>방향</th><th>유효 표본</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={item.feature}>
                <td>{index + 1}</td>
                <td title={item.feature}>{item.display_name}</td>
                <td>{item.group}</td>
                <td>{item.score === null ? "-" : formatNumber(item.score)}</td>
                <td>{item.direction}</td>
                <td>{item.valid_count?.toLocaleString() ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ParetoSection({
  analysis,
}: {
  analysis: RelationshipAnalysisResponse;
}) {
  const { pareto } = analysis;
  return (
    <section className="resultCard relationshipSection">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">2단계 · 검토 우선순위</span>
          <h2>누적 영향도 80% Pareto</h2>
        </div>
        <p>Ranking basis: {pareto.ranking_basis}</p>
      </div>
      <div className="paretoSummary">
        <div><span>우선 검토 변수</span><strong>{pareto.required_feature_count}개</strong></div>
        <div><span>누적 영향도</span><strong>{(pareto.cumulative_contribution * 100).toFixed(1)}%</strong></div>
        <div><span>전체 변수</span><strong>{pareto.total_feature_count}개</strong></div>
        <div>
          <span>상위 그룹 구성</span>
          <strong>R {pareto.group_counts.R} · D {pareto.group_counts.D} · EQ {pareto.group_counts.EQ}</strong>
        </div>
      </div>
      {pareto.features.length ? (
        <div className="paretoScroll">
          <div
            className="paretoChart"
            style={{ width: Math.max(760, pareto.features.length * 52) }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={pareto.features}>
                <CartesianGrid stroke="rgba(0,0,0,.07)" vertical={false} />
                <XAxis
                  dataKey="display_name"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) => String(value).replace("Step ", "S")}
                />
                <YAxis yAxisId="impact" tick={{ fontSize: 10 }} />
                <YAxis
                  yAxisId="share"
                  orientation="right"
                  domain={[0, 1]}
                  tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  formatter={(value, name) => [
                    name === "cumulative_share"
                      ? `${(Number(value) * 100).toFixed(1)}%`
                      : formatNumber(Number(value)),
                    name === "cumulative_share" ? "누적 영향도" : "영향도",
                  ]}
                />
                <ReferenceLine yAxisId="share" y={0.8} stroke="#b33a46" strokeDasharray="5 5" />
                <Bar yAxisId="impact" dataKey="impact" radius={[5, 5, 0, 0]}>
                  {pareto.features.map((item) => (
                    <Cell
                      key={item.feature}
                      fill={item.within_threshold ? groupColor(item.group) : "#c7c7cc"}
                    />
                  ))}
                </Bar>
                <Line
                  yAxisId="share"
                  type="monotone"
                  dataKey="cumulative_share"
                  stroke="#b33a46"
                  dot={false}
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <p className="emptyMessage">누적 영향도를 계산할 데이터가 없습니다.</p>
      )}
      <p className="analysisDisclaimer">{pareto.caveat}</p>
    </section>
  );
}

function associationValue(value: AssociationSummaryLike): string {
  if (!value) return "-";
  const metric = value.eta_squared ?? value.pearson;
  return metric === null || metric === undefined ? "-" : metric.toFixed(3);
}

type AssociationSummaryLike = RelationshipPath["r_d"];

function PathSection({
  paths,
  selectedIndex,
  onSelect,
  confidenceCriteria,
}: {
  paths: RelationshipPath[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  confidenceCriteria: Record<string, string>;
}) {
  const selected = paths[selectedIndex];
  return (
    <>
      <section className="resultCard relationshipSection">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">3단계 · 연관 경로</span>
            <h2>R·EQ → D → Y 우선순위</h2>
          </div>
          <p>Path relevance score · 인과 효과가 아닌 탐색 점수</p>
        </div>
        {paths.length ? (
          <div className="tableWrapper">
            <table className="dataTable pathTable">
              <thead><tr>
                <th>Rank</th><th>Step</th><th>Response</th><th>Equipment</th>
                <th>Defect</th><th>R→D</th><th>EQ→D</th><th>D→Y</th>
                <th>SHAP</th><th>Score</th><th>Sample</th><th>Confidence</th>
              </tr></thead>
              <tbody>
                {paths.map((path, index) => (
                  <tr
                    key={path.step}
                    className={selectedIndex === index ? "selectedRow" : ""}
                    onClick={() => onSelect(index)}
                  >
                    <td>{path.rank}</td><td>Step {path.step}</td>
                    <td>{path.response ?? "-"}</td><td>{path.equipment ?? "-"}</td>
                    <td>{path.defect}</td><td>{associationValue(path.r_d)}</td>
                    <td>{associationValue(path.eq_d)}</td><td>{associationValue(path.d_y)}</td>
                    <td>{formatNumber(path.shap_importance)}</td>
                    <td>{path.path_score.toFixed(3)}</td>
                    <td>{path.valid_count.toLocaleString()}</td>
                    <td><span className={`confidenceBadge ${path.confidence}`}>{path.confidence}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="emptyMessage">동일 Step의 연관 경로를 구성할 데이터가 없습니다.</p>}
        <p className="chartDescription">
          Confidence 기준: 충분({confidenceCriteria.sufficient}) · 주의({confidenceCriteria.caution}) · 부족({confidenceCriteria.insufficient})
        </p>
      </section>
      {selected && <PathDetail path={selected} />}
    </>
  );
}

function PathDetail({ path }: { path: RelationshipPath }) {
  return (
    <section className="resultCard relationshipSection">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">4단계 · 선택 경로 상세</span>
          <h2>Step {path.step} 관계 패널</h2>
        </div>
      </div>
      <div className="pathDetailGrid">
        <ScatterPanel title={`${path.response ?? "R"} vs ${path.defect}`} data={path.r_vs_d} xLabel={path.response ?? "R"} yLabel={path.defect} />
        <EquipmentPanel title={`${path.equipment ?? "EQ"} vs ${path.defect}`} data={path.eq_vs_d} />
        <ScatterPanel title={`${path.defect} vs Y`} data={path.d_vs_y} xLabel={path.defect} yLabel="Final Yield Y" />
      </div>
      <div className="interpretationPanel">
        <strong>5단계 · 엔지니어 해석</strong>
        <p>{path.interpretation}</p>
        <span>Correlation does not imply causation · 공식 공정 Spec이 아닌 데이터 기반 분석 결과입니다.</span>
      </div>
    </section>
  );
}

function ScatterPanel({ title, data, xLabel, yLabel }: {
  title: string; data: { x: number; y: number }[]; xLabel: string; yLabel: string;
}) {
  return (
    <article className="relationshipPanel">
      <h3>{title}</h3>
      {data.length ? (
        <div className="detailChart">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 12, bottom: 18, left: 5 }}>
              <CartesianGrid stroke="rgba(0,0,0,.07)" />
              <XAxis type="number" dataKey="x" name={xLabel} tick={{ fontSize: 10 }} />
              <YAxis type="number" dataKey="y" name={yLabel} tick={{ fontSize: 10 }} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={data} fill="#1769aa" fillOpacity={0.62} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="emptyMessage">표시할 유효 데이터가 없습니다.</p>}
    </article>
  );
}

function EquipmentPanel({ title, data }: {
  title: string; data: RelationshipPath["eq_vs_d"];
}) {
  return (
    <article className="relationshipPanel">
      <h3>{title}</h3>
      {data.length ? (
        <div className="detailChart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 12, bottom: 18, left: 5 }}>
              <CartesianGrid stroke="rgba(0,0,0,.07)" vertical={false} />
              <XAxis dataKey="equipment" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="q3" fill="#d8e8f5" name="Q3" />
              <Bar dataKey="median" fill="#647185" name="중앙값" />
              <Line dataKey="mean" stroke="#a96208" name="평균" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ) : <p className="emptyMessage">범주형 Equipment 분포 데이터가 없습니다.</p>}
      {data.some((item) => item.sample_warning) && (
        <p className="warningMessage">표본 10개 미만 Equipment는 해석에 주의가 필요합니다.</p>
      )}
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
