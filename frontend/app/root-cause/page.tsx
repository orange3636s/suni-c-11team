"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Header from "@/components/Header";
import CsvUploadPanel from "@/components/CsvUploadPanel";
import Sidebar from "@/components/Sidebar";
import {
  analyzeRelationships,
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

type WaferSort =
  | "risk-desc"
  | "risk-asc"
  | "id-asc"
  | "id-desc"
  | "prediction-desc"
  | "prediction-asc";

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
  const [waferSearch, setWaferSearch] = useState("");
  const [waferSort, setWaferSort] = useState<WaferSort>("risk-desc");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const waferRowRefs = useRef(new Map<number, HTMLTableRowElement>());

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
  const sortedWafers = useMemo(() => {
    const wafers = result?.wafer_explanations.map((wafer, index) => ({
      wafer,
      originalIndex: index,
    })) ?? [];
    const riskScore = (risk: string | null) =>
      risk === "danger" ? 2 : risk === "warning" ? 1 : 0;

    return wafers.sort((left, right) => {
      if (waferSort === "risk-desc") {
        return (
          riskScore(right.wafer.risk_level) -
          riskScore(left.wafer.risk_level)
        );
      }
      if (waferSort === "risk-asc") {
        return (
          riskScore(left.wafer.risk_level) -
          riskScore(right.wafer.risk_level)
        );
      }
      if (waferSort === "prediction-desc") {
        return right.wafer.prediction - left.wafer.prediction;
      }
      if (waferSort === "prediction-asc") {
        return left.wafer.prediction - right.wafer.prediction;
      }

      const comparison = String(left.wafer.identifier).localeCompare(
        String(right.wafer.identifier),
        "ko",
        { numeric: true, sensitivity: "base" },
      );
      return waferSort === "id-desc" ? -comparison : comparison;
    });
  }, [result, waferSort]);

  function handleFile(selectedFile?: File) {
    setResult(null);
    setRelationships(null);
    setError("");
    if (selectedFile && !selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    setFile(selectedFile ?? null);
  }

  function handleWaferSearch(value: string) {
    setWaferSearch(value);
    const query = value.trim().toLocaleLowerCase("ko");
    if (!query) return;

    const match = sortedWafers.find(({ wafer }) =>
      String(wafer.identifier).toLocaleLowerCase("ko").includes(query),
    );
    if (!match) return;

    selectWafer(match.originalIndex);
  }

  function selectWafer(index: number, scroll = true) {
    setSelectedWafer(index);
    const identifier = result?.wafer_explanations[index]?.identifier;
    if (identifier !== undefined) {
      setWaferSearch(String(identifier));
      localStorage.setItem("root-cause-recent-wafer", String(identifier));
    }
    if (!scroll) return;
    requestAnimationFrame(() => {
      waferRowRefs.current.get(index)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
  }

  function handleQuickWaferSelect(value: string) {
    setWaferSearch(value);
    const matchIndex =
      result?.wafer_explanations.findIndex(
        (wafer) => String(wafer.identifier) === value,
      ) ?? -1;
    if (matchIndex >= 0) selectWafer(matchIndex);
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
      const recentWafer = localStorage.getItem("root-cause-recent-wafer");
      const recentIndex = response.explanation.wafer_explanations.findIndex(
        (wafer) => String(wafer.identifier) === recentWafer,
      );
      setSelectedWafer(recentIndex >= 0 ? recentIndex : 0);
      setSelectedPath(0);
      setWaferSearch("");
      setWaferSort("risk-desc");
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
            <div className="fieldGroup analysisFileField">
              <label htmlFor="analysis-file">분석 CSV</label>
              <CsvUploadPanel
                id="analysis-file"
                file={file}
                onFileSelect={handleFile}
                disabled={loading}
                compact
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
                        <span className="sectionLabel">Feature Importance</span>
                        <h2>Top Yield Loss Drivers</h2>
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
                  <label className="waferQuickSelector">
                    <span>LOT_WAFER_ID</span>
                    <input
                      type="search"
                      list="root-cause-wafer-options"
                      value={waferSearch || String(selected?.identifier ?? "")}
                      onChange={(event) =>
                        handleQuickWaferSelect(event.target.value)
                      }
                      placeholder="Wafer 검색"
                    />
                    <datalist id="root-cause-wafer-options">
                      {result.wafer_explanations.map((wafer, index) => (
                        <option
                          key={`${String(wafer.identifier)}-${index}`}
                          value={String(wafer.identifier)}
                        />
                      ))}
                    </datalist>
                  </label>
                </div>
                {selected && (
                  <>
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
                    <p className="waferDetailSummary">
                      {result.identifier_column}:{" "}
                      <strong>{String(selected.identifier)}</strong> · 예측값{" "}
                      <strong>{formatNumber(selected.prediction)}</strong> ·
                      기준값 <strong>{formatNumber(selected.base_value)}</strong>{" "}
                      · 위험도{" "}
                      <strong>{riskLabel(selected.risk_level)}</strong>
                    </p>
                  </>
                )}
                <div className="waferListTools">
                  <label className="waferSearch">
                    <span className="visuallyHidden">
                      LOT_WAFER_ID 검색
                    </span>
                    <input
                      type="search"
                      placeholder="LOT_WAFER_ID 검색"
                      value={waferSearch}
                      onChange={(event) =>
                        handleWaferSearch(event.target.value)
                      }
                    />
                  </label>
                  <label className="waferSort">
                    <span className="visuallyHidden">LOT 목록 정렬</span>
                    <select
                      value={waferSort}
                      onChange={(event) =>
                        setWaferSort(event.target.value as WaferSort)
                      }
                    >
                      <option value="risk-desc">위험도 높은 순</option>
                      <option value="risk-asc">위험도 낮은 순</option>
                      <option value="id-asc">LOT_WAFER_ID 오름차순</option>
                      <option value="id-desc">LOT_WAFER_ID 내림차순</option>
                      <option value="prediction-desc">예측값 높은 순</option>
                      <option value="prediction-asc">예측값 낮은 순</option>
                    </select>
                  </label>
                </div>
                <div
                  className="tableWrapper waferListScroll"
                  aria-label={`${result.identifier_column} 목록`}
                >
                  <table className="dataTable">
                    <thead>
                      <tr>
                        <th>{result.identifier_column}</th>
                        <th>예측값</th>
                        <th>위험도</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedWafers.map(({ wafer, originalIndex }) => (
                        <tr
                          ref={(node) => {
                            if (node) {
                              waferRowRefs.current.set(originalIndex, node);
                            } else {
                              waferRowRefs.current.delete(originalIndex);
                            }
                          }}
                          className={
                            selectedWafer === originalIndex ? "selectedRow" : ""
                          }
                          key={`${String(wafer.identifier)}-${originalIndex}`}
                          onClick={() => selectWafer(originalIndex, false)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              selectWafer(originalIndex, false);
                            }
                          }}
                          tabIndex={0}
                          role="button"
                          aria-label={`${String(wafer.identifier)} 상세 기여도 ${
                            selectedWafer === originalIndex
                              ? "선택됨"
                              : "보기"
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
  if (group === "R") return "var(--chart-primary)";
  if (group === "D") return "var(--warning)";
  return "var(--chart-secondary)";
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
              stroke="var(--chart-grid)"
              strokeDasharray="3 5"
              horizontal={false}
            />
            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
            />
            <YAxis
              type="category"
              dataKey="display_name"
              width={138}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
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
          <span className="sectionLabel">Cumulative Impact</span>
          <h2>Pareto Analysis</h2>
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
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
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
                      fill={item.within_threshold ? groupColor(item.group) : "var(--chart-muted)"}
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
  const [pathSearch, setPathSearch] = useState("");
  const selected = paths[selectedIndex];
  const visiblePaths = useMemo(
    () =>
      paths
        .map((path, index) => ({ path, index }))
        .filter(({ path }) => {
          const query = pathSearch.trim().toLocaleLowerCase("ko");
          if (!query) return true;
          return [
            `Step ${path.step}`,
            path.response,
            path.equipment,
            path.defect,
          ].some((value) =>
            String(value ?? "").toLocaleLowerCase("ko").includes(query),
          );
        })
        .sort((left, right) => right.path.path_score - left.path.path_score),
    [pathSearch, paths],
  );
  return (
    <section className="resultCard relationshipSection pathUnifiedCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">Relationship Analysis</span>
            <h2>Relationship Path</h2>
          </div>
          <p>Path relevance score · 인과 효과가 아닌 탐색 점수</p>
        </div>
        <div className="relationshipToolbar pathToolbar">
          <label className="waferSearch">
            <span className="visuallyHidden">Relationship Path 검색</span>
            <input
              type="search"
              value={pathSearch}
              onChange={(event) => setPathSearch(event.target.value)}
              placeholder="Step, Response, Equipment, Defect 검색"
            />
          </label>
          <span className="pathSortLabel">Score 높은 순</span>
        </div>
        {paths.length ? (
          <div className="tableWrapper relationshipPathScroll">
            <table className="dataTable pathTable">
              <thead><tr>
                <th>Rank</th><th>Step</th><th>Response</th><th>Equipment</th>
                <th>Defect</th><th>R→D</th><th>EQ→D</th><th>D→Y</th>
                <th>SHAP</th><th>Score</th><th>Sample</th><th>Confidence</th>
              </tr></thead>
              <tbody>
                {visiblePaths.map(({ path, index }) => (
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
        {paths.length > 0 && visiblePaths.length === 0 && (
          <p className="emptyMessage">검색 조건과 일치하는 경로가 없습니다.</p>
        )}
        <p className="chartDescription">
          Confidence 기준: 충분({confidenceCriteria.sufficient}) · 주의({confidenceCriteria.caution}) · 부족({confidenceCriteria.insufficient})
        </p>
        {selected && <PathDetail path={selected} />}
    </section>
  );
}

function PathDetail({ path }: { path: RelationshipPath }) {
  return (
    <div className="relationshipDetailBlock">
      <div className="sectionHeading compact">
        <div>
          <span className="sectionLabel">Selected Path</span>
          <h2>Step {path.step} 관계 패널</h2>
        </div>
      </div>
      <div className="pathDetailGrid">
        <ScatterPanel title={`${path.response ?? "R"} vs ${path.defect}`} data={path.r_vs_d} xLabel={path.response ?? "R"} yLabel={path.defect} xDescription="Response Value" yDescription="Defect Measurement" />
        <EquipmentPanel title={`${path.equipment ?? "EQ"} vs ${path.defect}`} data={path.eq_vs_d} />
        <ScatterPanel title={`${path.defect} vs Y`} data={path.d_vs_y} xLabel={path.defect} yLabel="Final Yield Y" xDescription="Defect Value" yDescription="Final Yield (%)" />
      </div>
      <YieldBoxPlot path={path} />
    </div>
  );
}

type BoxSummary = {
  label: string;
  count: number;
  median: number;
  q1: number;
  q3: number;
  whiskerMin: number;
  whiskerMax: number;
  outliers: number[];
  outlierCount: number;
};

function quantile(values: number[], ratio: number): number {
  if (values.length === 1) return values[0];
  const position = (values.length - 1) * ratio;
  const lower = Math.floor(position);
  const remainder = position - lower;
  return values[lower] + (values[lower + 1] - values[lower]) * remainder;
}

function summarizeValues(label: string, rawValues: number[]): BoxSummary | null {
  const values = rawValues.filter(Number.isFinite).sort((left, right) => left - right);
  if (!values.length) return null;
  const q1 = quantile(values, 0.25);
  const median = quantile(values, 0.5);
  const q3 = quantile(values, 0.75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;
  const inliers = values.filter((value) => value >= lowerFence && value <= upperFence);
  const outliers = values.filter((value) => value < lowerFence || value > upperFence);
  return {
    label,
    count: values.length,
    median,
    q1,
    q3,
    whiskerMin: inliers[0] ?? values[0],
    whiskerMax: inliers[inliers.length - 1] ?? values[values.length - 1],
    outliers,
    outlierCount: outliers.length,
  };
}

function numericBoxGroups(
  points: { x: number; y: number }[] | undefined,
): BoxSummary[] {
  const sorted = (points ?? [])
    .filter(({ x, y }) => Number.isFinite(x) && Number.isFinite(y))
    .sort((left, right) => left.x - right.x);
  if (!sorted.length) return [];
  const groupCount = Math.min(4, sorted.length);
  return Array.from({ length: groupCount }, (_, groupIndex) => {
    const start = Math.floor((groupIndex * sorted.length) / groupCount);
    const end = Math.floor(((groupIndex + 1) * sorted.length) / groupCount);
    const group = sorted.slice(start, end);
    const first = group[0]?.x ?? 0;
    const last = group[group.length - 1]?.x ?? first;
    return summarizeValues(
      first === last
        ? formatNumber(first)
        : `${formatNumber(first)}–${formatNumber(last)}`,
      group.map(({ y }) => y),
    );
  }).filter((summary): summary is BoxSummary => summary !== null);
}

function YieldBoxPlot({ path }: { path: RelationshipPath }) {
  const [tab, setTab] = useState<"R" | "D" | "EQ">("R");
  const groups = useMemo(() => {
    if (tab === "R") return numericBoxGroups(path.r_vs_y);
    if (tab === "D") return numericBoxGroups(path.d_vs_y);
    return (path.eq_vs_y ?? []).map((item) => ({
      label: item.equipment,
      count: item.count,
      median: item.median,
      q1: item.q1,
      q3: item.q3,
      whiskerMin: item.whisker_min ?? item.minimum,
      whiskerMax: item.whisker_max ?? item.maximum,
      outliers: item.outliers ?? [],
      outlierCount: item.outlier_count ?? item.outliers?.length ?? 0,
    }));
  }, [path, tab]);
  const variable =
    tab === "R"
      ? path.response ?? "Response"
      : tab === "D"
        ? path.defect
        : path.equipment ?? "Equipment";

  return (
    <section className="yieldBoxPlotSection">
      <div className="boxPlotHeading">
        <div>
          <span className="sectionLabel">Yield Distribution</span>
          <h3>{variable} · Yield Box Plot</h3>
          <p>Median, IQR, 1.5×IQR whisker, outlier와 표본 수를 실제 분석 데이터로 계산합니다.</p>
        </div>
        <SegmentedControl
          options={[["R", "R"], ["D", "D"], ["EQ", "EQ"]]}
          value={tab}
          onChange={(value) => setTab(value as "R" | "D" | "EQ")}
        />
      </div>
      {groups.length ? (
        <>
          <BoxPlotGraphic groups={groups} variable={variable} />
          <div className="boxStatGrid">
            {groups.map((group) => (
              <article key={group.label} className="boxStatItem">
                <strong title={group.label}>{group.label}</strong>
                <span>Median {formatNumber(group.median)}</span>
                <span>IQR {formatNumber(group.q3 - group.q1)}</span>
                <span>Outlier {group.outlierCount.toLocaleString()} · n={group.count.toLocaleString()}</span>
              </article>
            ))}
          </div>
        </>
      ) : (
        <p className="emptyMessage">
          선택 경로의 {tab} 변수와 Yield를 함께 비교할 유효 데이터가 없습니다.
        </p>
      )}
    </section>
  );
}

function BoxPlotGraphic({
  groups,
  variable,
}: {
  groups: BoxSummary[];
  variable: string;
}) {
  const allValues = groups.flatMap((group) => [
    group.whiskerMin,
    group.whiskerMax,
    ...group.outliers,
  ]);
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = rawMax === rawMin ? Math.max(Math.abs(rawMax) * 0.05, 1) : (rawMax - rawMin) * 0.08;
  const min = rawMin - padding;
  const max = rawMax + padding;
  const width = Math.max(720, groups.length * 110);
  const height = 330;
  const plotTop = 24;
  const plotBottom = 270;
  const scaleY = (value: number) =>
    plotBottom - ((value - min) / (max - min)) * (plotBottom - plotTop);
  const ticks = Array.from({ length: 5 }, (_, index) => min + ((max - min) * index) / 4);

  return (
    <div className="boxPlotScroll" role="img" aria-label={`${variable}별 Yield box plot`}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ minWidth: width }}>
        <text className="boxAxisTitle" x="16" y="18">Final Yield Y</text>
        {ticks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={tick}>
              <line className="boxGridLine" x1="66" x2={width - 16} y1={y} y2={y} />
              <text className="boxTick" x="58" y={y + 4} textAnchor="end">{formatNumber(tick)}</text>
            </g>
          );
        })}
        {groups.map((group, index) => {
          const slot = (width - 90) / groups.length;
          const x = 74 + slot * index + slot / 2;
          const boxWidth = Math.min(52, slot * 0.58);
          const tooltip = `${group.label}\nMedian: ${formatNumber(group.median)}\nIQR: ${formatNumber(group.q3 - group.q1)}\nOutlier: ${group.outlierCount}\nn: ${group.count}`;
          return (
            <g key={`${group.label}-${index}`}>
              <title>{tooltip}</title>
              <line className="boxWhisker" x1={x} x2={x} y1={scaleY(group.whiskerMax)} y2={scaleY(group.whiskerMin)} />
              <line className="boxWhisker" x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={scaleY(group.whiskerMax)} y2={scaleY(group.whiskerMax)} />
              <line className="boxWhisker" x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={scaleY(group.whiskerMin)} y2={scaleY(group.whiskerMin)} />
              <rect
                className="boxShape"
                x={x - boxWidth / 2}
                y={scaleY(group.q3)}
                width={boxWidth}
                height={Math.max(2, scaleY(group.q1) - scaleY(group.q3))}
                rx="5"
              />
              <line className="boxMedian" x1={x - boxWidth / 2} x2={x + boxWidth / 2} y1={scaleY(group.median)} y2={scaleY(group.median)} />
              {group.outliers.slice(0, 30).map((outlier, outlierIndex) => (
                <circle className="boxOutlier" key={`${outlier}-${outlierIndex}`} cx={x} cy={scaleY(outlier)} r="3" />
              ))}
              <text className="boxCategory" x={x} y="294" textAnchor="middle">{group.label.length > 14 ? `${group.label.slice(0, 12)}…` : group.label}</text>
              <text className="boxCount" x={x} y="312" textAnchor="middle">n={group.count}</text>
            </g>
          );
        })}
        <text className="boxAxisTitle" x={width / 2} y="328" textAnchor="middle">{variable}</text>
      </svg>
    </div>
  );
}

function ScatterPanel({ title, data, xLabel, yLabel, xDescription, yDescription }: {
  title: string;
  data: { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  xDescription: string;
  yDescription: string;
}) {
  return (
    <article className="relationshipPanel">
      <h3>{title}</h3>
      {data.length ? (
        <div className="detailChart">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 18, bottom: 42, left: 28 }}>
              <CartesianGrid stroke="var(--chart-grid)" />
              <XAxis
                type="number"
                dataKey="x"
                name={xLabel}
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: `${xLabel} (${xDescription})`, position: "insideBottom", offset: -24, fill: "var(--chart-axis)" }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={yLabel}
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: `${yLabel} (${yDescription})`, angle: -90, position: "insideLeft", offset: -16, fill: "var(--chart-axis)" }}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                formatter={(value, name) => [formatNumber(Number(value)), String(name)]}
              />
              <Legend verticalAlign="top" height={28} />
              <Scatter name={`${yLabel} by ${xLabel}`} data={data} fill="var(--chart-primary)" fillOpacity={0.72} />
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
            <ComposedChart data={data} margin={{ top: 10, right: 18, bottom: 42, left: 28 }}>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="equipment"
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: "Equipment category", position: "insideBottom", offset: -24, fill: "var(--chart-axis)" }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
                label={{ value: "Defect value", angle: -90, position: "insideLeft", offset: -16, fill: "var(--chart-axis)" }}
              />
              <Tooltip formatter={(value, name) => [formatNumber(Number(value)), String(name)]} />
              <Legend verticalAlign="top" height={28} />
              <Bar dataKey="q3" fill="var(--chart-primary-soft)" name="Q3" />
              <Bar dataKey="median" fill="var(--chart-secondary)" name="중앙값" />
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
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={112}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "var(--chart-axis)" }}
              tickFormatter={formatFeatureLabel}
            />
            <Tooltip
              formatter={(value) => [
                formatNumber(Number(value)),
                "기여도",
              ]}
              contentStyle={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                background: "var(--chart-tooltip)",
                color: "var(--text-primary)",
                boxShadow: "var(--shadow-elevated)",
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
