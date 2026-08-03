"use client";

import { useCallback, useEffect, useState } from "react";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import PlotlyChart from "@/components/PlotlyChart";
import {
  createTrainingJob,
  downloadDatasetFile,
  getModelPerformance,
  getScreening,
  getTrainingJob,
} from "@/lib/api";
import type { DatasetSummary, ModelPerformanceResponse, ScreeningResponse, TargetScreeningResult } from "@/types/data";

const BUNDLED_TRAIN_ID = "train";
const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

const showMetric = (value?: number | null, digits = 3) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
const showDate = (value?: string | null) =>
  value ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "-";

const SHAPE_LABEL: Record<string, string> = {
  monotonic_increasing: "단조 증가",
  monotonic_decreasing: "단조 감소",
  u_shape: "U자형",
  unclear: "불명확",
};
const KIND_LABEL: Record<string, string> = { R: "계측값", D: "결함수", Config: "장비 설정" };

const BENCHMARK_REFERENCE = [
  { name: "A. 중앙값 대체 + 클리핑 (현행)", y: 0.114 },
  { name: "B. 전체 인자 + NaN 보존", y: 0.146 },
  { name: "C. 선정 인자 + dev + 마스크", y: 0.177 },
];

function paretoChartSpec(result: TargetScreeningResult) {
  const factors = result.factors;
  return {
    data: [
      {
        type: "bar",
        x: factors.map((f) => f.feature),
        y: factors.map((f) => f.contribution_pct),
        name: "개별 기여율(%)",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        x: factors.map((f) => f.feature),
        y: factors.map((f) => f.cumulative_pct),
        name: "누적 기여율(%)",
        yaxis: "y2",
      },
    ],
    layout: {
      title: { text: `${result.target} Pareto 차트` },
      yaxis: { title: { text: "개별 기여율(%)" } },
      yaxis2: { title: { text: "누적 기여율(%)" }, overlaying: "y", side: "right", range: [0, 105] },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 80, y1: 80, line: { dash: "dash", color: "gray" } }],
    },
  };
}

export default function TrainingPage() {
  const [datasetId, setDatasetId] = useState(BUNDLED_TRAIN_ID);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [performance, setPerformance] = useState<ModelPerformanceResponse | null>(null);
  const [screening, setScreening] = useState<ScreeningResponse | null>(null);
  const [screeningLoading, setScreeningLoading] = useState(false);
  const [showReferenceOnly, setShowReferenceOnly] = useState<Record<string, boolean>>({});

  const loadPerformance = useCallback(async () => {
    try {
      setPerformance(await getModelPerformance());
    } catch {
      setPerformance(null);
    }
  }, []);

  const loadScreening = useCallback(async (id: string) => {
    setScreeningLoading(true);
    try {
      setScreening(await getScreening(id));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "인자 스크리닝 결과를 불러오지 못했습니다.");
    } finally {
      setScreeningLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadPerformance(), 0);
    return () => window.clearTimeout(timer);
  }, [loadPerformance]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadScreening(datasetId), 0);
    return () => window.clearTimeout(timer);
  }, [datasetId, loadScreening]);

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const job = await getTrainingJob(jobId);
        setStage(job.stage);
        setProgress(job.progress);
        if (job.status === "completed") {
          window.clearInterval(timer);
          setJobId(null);
          setMessage("스크리닝 기반 Y1~Y5 GBDT 학습이 완료되었습니다.");
          await loadPerformance();
        } else if (job.status === "failed" || job.status === "interrupted") {
          window.clearInterval(timer);
          setJobId(null);
          setError(job.error || "모델 학습 중 서버 오류가 발생했습니다.");
        }
      } catch (pollError) {
        window.clearInterval(timer);
        setJobId(null);
        setError(pollError instanceof Error ? pollError.message : "학습 상태를 확인하지 못했습니다.");
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId, loadPerformance]);

  async function train() {
    if (jobId) return;
    setError("");
    setMessage("");
    setStage("학습 데이터셋을 불러오는 중입니다.");
    setProgress(0);
    try {
      const selected = datasets.find((item) => item.dataset_id === datasetId);
      const file = await downloadDatasetFile(datasetId, selected?.original_filename ?? "dataset.csv");
      setJobId((await createTrainingJob(file)).job_id);
    } catch (trainingError) {
      setError(trainingError instanceof Error ? trainingError.message : "모델 학습을 시작하지 못했습니다.");
    }
  }

  const selectedDataset = datasets.find((item) => item.dataset_id === datasetId);
  const isBundledTrain = datasetId === BUNDLED_TRAIN_ID;
  const noConfigSelected = screening?.targets.every((t) => t.factors.every((f) => f.kind !== "Config")) ?? true;

  return (
    <DashboardShell activeItem="모델 학습">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">MACHINE LEARNING</span>
        <h1>모델 학습</h1>
        <p>선정 인자(ε² + BH-FDR) 기반 Y1~Y5 GBDT를 학습하고, 인자 스크리닝 결과를 함께 확인합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">DATASET</span>
            <h2>학습 데이터셋</h2>
          </div>
        </div>
        <DatasetSelector label="학습용 데이터셋" value={datasetId} onChange={setDatasetId} onDatasetsLoaded={setDatasets} />
        {selectedDataset && selectedDataset.warnings.length > 0 && (
          <div className="datasetWarningBanner">
            <strong>주의</strong>
            <ul>
              {selectedDataset.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="uploadActions">
          <button className="button primary" type="button" disabled={Boolean(jobId)} onClick={() => void train()}>
            {jobId ? "모델 학습 중…" : "학습 실행"}
          </button>
        </div>
        {jobId && (
          <p className="trainingProgress" role="status">
            {stage} · {progress}%
          </p>
        )}
        {message && <p className="messageBox success" role="status">{message}</p>}
        {error && <p className="errorMessage" role="alert">{error}</p>}
      </section>

      {!isBundledTrain && (
        <section className="messageBox">기준값은 내장 데이터셋(train.CSV → test.CSV)에만 적용됩니다.</section>
      )}

      <section className="rcGrid">
        {TARGETS.map((target) => {
          const detail = performance?.targets.find((t) => t.target === target);
          return (
            <article className="resultCard" key={target}>
              <div className="sectionHeading compact">
                <div>
                  <span className="sectionLabel">{target}</span>
                  <h2>{detail?.no_significant_factor ? "유의 인자 없음" : detail?.feature ?? "-"}</h2>
                </div>
              </div>
              {detail && !detail.no_significant_factor ? (
                <div className="secomKpiGrid">
                  <div><span>R²</span><strong>{showMetric(detail.r2)}</strong></div>
                  <div><span>MAE</span><strong>{showMetric(detail.mae)}</strong></div>
                  <div><span>ε²</span><strong>{showMetric(detail.eps2)}</strong></div>
                  <div><span>관계형태</span><strong>{SHAPE_LABEL[detail.relation_shape ?? ""] ?? "-"}</strong></div>
                </div>
              ) : (
                <p className="emptyMessage">통계적으로 유의한 인자가 없습니다.</p>
              )}
            </article>
          );
        })}
        <article className="resultCard">
          <div className="sectionHeading compact">
            <div>
              <span className="sectionLabel">Y (최종)</span>
              <h2>최종 수율</h2>
            </div>
          </div>
          <div className="secomKpiGrid">
            <div><span>R²</span><strong>{showMetric(performance?.final_yield?.r2)}</strong></div>
            <div><span>MAE</span><strong>{showMetric(performance?.final_yield?.mae)}</strong></div>
            <div><span>학습 시각</span><strong>{showDate(performance?.trained_at)}</strong></div>
          </div>
        </article>
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">BENCHMARK</span>
            <h2>전처리 방식별 CV 성능 비교 (train.CSV, GroupKFold(5))</h2>
          </div>
        </div>
        <div className="tableWrap">
          <table>
            <thead><tr><th>방식</th><th>Y R²</th></tr></thead>
            <tbody>
              {BENCHMARK_REFERENCE.map((row) => (
                <tr key={row.name}><td>{row.name}</td><td>{row.y.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="emptyMessage">scripts/benchmark.py 실행 결과. 억지로 인자를 늘리거나 전처리를 바꿔 이 값을 올리지 않습니다.</p>
      </section>

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">SCREENING</span>
            <h2>인자 스크리닝 결과</h2>
          </div>
        </div>
        {noConfigSelected && (
          <p className="messageBox">Config(장비 설정) 30개 중 통계적으로 유의한 인자는 0개입니다.</p>
        )}
        {screeningLoading && <p className="emptyMessage">불러오는 중…</p>}
        {screening?.targets.map((result) => (
          <div key={result.target} style={{ marginBottom: 24 }}>
            <h3 style={{ margin: "12px 0" }}>{result.target}</h3>
            {result.no_significant_factor ? (
              <p className="emptyMessage">통계적으로 유의한 인자가 없습니다.</p>
            ) : (
              <>
                <PlotlyChart spec={paretoChartSpec(result)} height={280} />
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th>인자명</th><th>종류</th><th>ε²</th><th>기여율</th><th>누적%</th><th>관측수</th><th>q값</th><th>관계형태</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.factors.map((factor) => (
                        <tr key={factor.feature}>
                          <td>{factor.feature}</td>
                          <td>{KIND_LABEL[factor.kind]}</td>
                          <td>{showMetric(factor.eps2)}</td>
                          <td>{showMetric(factor.contribution_pct, 1)}%</td>
                          <td>{showMetric(factor.cumulative_pct, 1)}%</td>
                          <td>{factor.n_observed}</td>
                          <td>{factor.q_value < 0.001 ? factor.q_value.toExponential(2) : showMetric(factor.q_value, 4)}</td>
                          <td>
                            {SHAPE_LABEL[factor.relation_shape]}
                            {factor.optimal_center != null && ` (중심 ${showMetric(factor.optimal_center, 1)})`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            {result.reference_only.length > 0 && (
              <>
                <button
                  type="button"
                  className="referenceOnlyToggle"
                  onClick={() =>
                    setShowReferenceOnly((current) => ({ ...current, [result.target]: !current[result.target] }))
                  }
                >
                  {showReferenceOnly[result.target] ? "접기" : `FDR 탈락 인자 펼치기 (${result.reference_only.length})`}
                </button>
                {showReferenceOnly[result.target] && (
                  <div className="tableWrap">
                    <table>
                      <thead><tr><th>인자명</th><th>종류</th><th>ε²</th><th>q값</th><th>상태</th></tr></thead>
                      <tbody>
                        {result.reference_only.map((factor) => (
                          <tr key={factor.feature} className="referenceOnlyRow">
                            <td>{factor.feature}</td>
                            <td>{KIND_LABEL[factor.kind]}</td>
                            <td>{showMetric(factor.eps2)}</td>
                            <td>{showMetric(factor.q_value, 4)}</td>
                            <td><span className="referenceOnlyBadge">통계적 유의성 미달</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </section>

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>이 분석은 해당 인자가 계측된 wafer만 대상으로 합니다. R은 전체의 15%, D는 5%입니다. 미계측 wafer로의 일반화는 보장되지 않습니다.</li>
          <li>ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행/후행 관계나 교락 인자는 반영되지 않았습니다.</li>
          <li>Config(장비)에서 유의 인자가 검출되지 않은 것은 &quot;장비 영향이 없다&quot;가 아니라 &quot;현재 표본으로는 검출되지 않는다&quot;는 뜻입니다. 장비당 표본이 278장 수준이라 ε² 0.01 미만의 효과는 검출력이 부족합니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}
