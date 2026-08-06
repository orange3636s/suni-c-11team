"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import DashboardShell from "@/components/DashboardShell";
import DatasetSelector from "@/components/DatasetSelector";
import { DatasetMismatchWarning, LastRunNote } from "@/components/LastRunNote";
import type { HeatmapCellSelection } from "@/components/CorrelationHeatmap";
import HeatmapParetoSection from "@/components/HeatmapParetoSection";
import {
  createTrainingJob,
  downloadDatasetFile,
  getModelPerformance,
  getScreeningHeatmap,
  getScreeningPareto,
  getTrainingJob,
  saveTrainingState,
} from "@/lib/api";
import { kindLabel } from "@/lib/kindLabels";
import { formatQValue } from "@/lib/numberFormat";
import { formatLastRun } from "@/lib/timeFormat";
import type { DatasetSummary, ModelPerformanceResponse, ParetoRankingItem, ParetoRankingResponse } from "@/types/data";

const BUNDLED_TRAIN_ID = "train";
const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

const showMetric = (value?: number | null, digits = 3) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";

/** A completed job with no usable payload is an error, not "not yet run" --
 * silently falling back to the empty state would hide a real failure
 * (spec §1-4). */
function hasUsableResult(performance: ModelPerformanceResponse | null): boolean {
  return Boolean(performance && performance.model_id && performance.targets.length > 0);
}

const TIER_LABEL: Record<string, string> = { strong: "강함", moderate: "보통", weak: "약함", reference: "참고" };

const BENCHMARK_REFERENCE = [
  { name: "A. 중앙값 대체 + 클리핑 (현행)", y: 0.114, adopted: false },
  { name: "B. 전체 인자 + NaN 보존", y: 0.146, adopted: false },
  { name: "C. 선정 인자 + dev + 마스크", y: 0.177, adopted: true },
];

export default function TrainingPage() {
  const router = useRouter();
  // 학습 결과 상태 유지 (spec: 학습·분석 결과 상태 유지) -- performance/
  // paretoByTarget/analysisReady/activeTarget all live in the shared
  // AnalysisStateProvider context now, not local useState, so switching
  // away to another tab and back renders instantly from memory with no
  // network call (checklist §탭 이동 #1/#4).
  const { training, setTraining, hydrated } = useAnalysisState();
  const [datasetId, setDatasetId] = useState(BUNDLED_TRAIN_ID);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [stage, setStage] = useState("");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [performanceLoading, setPerformanceLoading] = useState(true);
  const [performanceError, setPerformanceError] = useState("");

  const performance = training?.performance ?? null;
  const paretoByTarget = training?.paretoByTarget ?? {};
  const analysisReady = training?.analysisReady ?? false;
  const activeTarget = training?.activeTarget ?? "Y1";

  function setActiveTarget(target: string) {
    setTraining((previous) => (previous ? { ...previous, activeTarget: target } : previous));
  }

  // 재접속/새로고침으로 복원된 결과의 데이터셋을 셀렉터에도 반영한다 (spec
  // §4-3) -- 하이드레이션이 끝난 뒤 한 번만, 사용자가 이미 고른 값을 덮어쓰지
  // 않도록 이 컴포넌트 마운트당 정확히 한 번만 실행된다.
  const trainedDatasetRef = useRef(datasetId);
  const syncedFromRestore = useRef(false);
  useEffect(() => {
    if (!hydrated || syncedFromRestore.current) return;
    syncedFromRestore.current = true;
    if (!training) return;
    const timer = window.setTimeout(() => setDatasetId(training.dataset), 0);
    return () => window.clearTimeout(timer);
  }, [hydrated, training]);

  // `training` is read through a ref inside loadPerformance below so the
  // callback never closes over a stale snapshot without needing it in the
  // dependency array (which would otherwise recreate it, and the polling
  // effect below, on every context update).
  const trainingRef = useRef(training);
  useEffect(() => {
    trainingRef.current = training;
  }, [training]);

  // `requireResult` marks a load that follows a job the UI itself just
  // watched finish -- only then do we know for certain a result *should*
  // exist, so only then does an empty/missing payload count as an error
  // instead of the plain "not run yet" empty state.
  const loadPerformance = useCallback(
    async (options?: { requireResult?: boolean; forDataset?: string }) => {
      setPerformanceLoading(true);
      try {
        const result = await getModelPerformance();
        if (options?.requireResult && !hasUsableResult(result)) {
          setPerformanceError("학습은 완료되었지만 결과를 불러오지 못했습니다.");
        } else {
          setPerformanceError("");
        }
        if (hasUsableResult(result)) {
          const forDataset = options?.forDataset ?? trainingRef.current?.dataset ?? datasetId;
          setTraining((previous) =>
            previous && previous.dataset === forDataset
              ? { ...previous, performance: result }
              : { dataset: forDataset, createdAt: result.trained_at ?? new Date().toISOString(), performance: result, paretoByTarget: {}, analysisReady: false, activeTarget: "Y1" },
          );
        }
      } catch (loadError) {
        if (options?.requireResult) {
          setPerformanceError(loadError instanceof Error ? loadError.message : "결과를 불러오지 못했습니다.");
        }
      } finally {
        setPerformanceLoading(false);
      }
    },
    [datasetId, setTraining],
  );

  // Fallback only: if hydration has finished and there is still nothing
  // in context (a fresh server session's app_state table is empty, or a
  // model was promoted by some path that predates this feature), fall
  // back to the old self-healing lookup -- exactly once, never on every
  // tab visit (checklist §재접속 #4/#10, §탭 이동 #4).
  useEffect(() => {
    if (!hydrated) return;
    const timer = window.setTimeout(() => {
      if (training) setPerformanceLoading(false);
      else void loadPerformance();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [hydrated, training, loadPerformance]);

  useEffect(() => {
    if (!jobId) return;
    const trainedDataset = trainedDatasetRef.current;
    const timer = window.setInterval(async () => {
      try {
        const job = await getTrainingJob(jobId);
        setStage(job.stage);
        setProgress(job.progress);
        if (job.status === "completed") {
          window.clearInterval(timer);
          setJobId(null);
          setStage("히트맵 집계 중");
          setProgress(99);
          try {
            const paretoResults = await Promise.all(
              TARGETS.map((t) => getScreeningPareto(trainedDataset, t).then((response) => [t, response] as const)),
            );
            const paretoMap = Object.fromEntries(paretoResults) as Record<string, ParetoRankingResponse>;
            await getScreeningHeatmap(trainedDataset, "spearman").catch(() => {});
            setTraining((previous) => ({
              dataset: trainedDataset,
              createdAt: new Date().toISOString(),
              performance: previous?.performance ?? { model_id: null, trained_at: null, source_filename: null, targets: [], final_yield: null },
              paretoByTarget: paretoMap,
              analysisReady: true,
              activeTarget: previous?.activeTarget ?? "Y1",
            }));
          } catch {
            setTraining((previous) => (previous ? { ...previous, analysisReady: false } : previous));
          }
          setMessage("스크리닝 기반 Y1~Y5 GBDT 학습이 완료되었습니다.");
          await loadPerformance({ requireResult: true, forDataset: trainedDataset });
          // 학습 완료 직후 저장 (spec §3-4) -- 실패해도 학습 자체는 이미
          // 성공했으므로 조용히 무시한다 (spec §3-2).
          try {
            const latestPerformance = await getModelPerformance();
            if (hasUsableResult(latestPerformance)) {
              void saveTrainingState(trainedDataset, { performance: latestPerformance }).catch(() => {});
            }
          } catch {
            // best-effort only
          }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, loadPerformance]);

  async function train() {
    if (jobId) return;
    setError("");
    setMessage("");
    setStage("학습 데이터셋을 불러오는 중입니다.");
    setProgress(0);
    // Snapshot which dataset this run is actually for -- the selector
    // stays interactive during a run, so the polling effect must not
    // trust `datasetId` at completion time, only at submission time.
    trainedDatasetRef.current = datasetId;
    try {
      const selected = datasets.find((item) => item.dataset_id === datasetId);
      const file = await downloadDatasetFile(datasetId, selected?.original_filename ?? "dataset.csv");
      setJobId((await createTrainingJob(file)).job_id);
    } catch (trainingError) {
      setError(trainingError instanceof Error ? trainingError.message : "모델 학습을 시작하지 못했습니다.");
    }
  }

  function handleHeatmapSelect(selection: HeatmapCellSelection) {
    router.push(`/root-cause?target=${encodeURIComponent(selection.target)}&feature=${encodeURIComponent(selection.feature)}`);
  }

  function handleParetoBarClick(item: ParetoRankingItem) {
    router.push(`/root-cause?target=${encodeURIComponent(activeTarget)}&feature=${encodeURIComponent(item.feature)}`);
  }

  const selectedDataset = datasets.find((item) => item.dataset_id === datasetId);
  const isBundledTrain = datasetId === BUNDLED_TRAIN_ID;

  const isRunning = Boolean(jobId);
  const hasResult = hasUsableResult(performance);
  // 셀렉터를 바꿨는데 화면은 이전 데이터셋 결과인 경우 (spec §5-3) -- 결과를
  // 지우지 않고 경고만 띄운다.
  const datasetMismatch = Boolean(training && training.dataset !== datasetId);

  return (
    <DashboardShell activeItem="모델 학습">
      <section className="uploadIntro pageHeading">
        <span className="eyebrow">MACHINE LEARNING</span>
        <h1>모델 학습</h1>
        <p>선정 인자(ε²) 기반 Y1~Y5 GBDT를 학습하고, 전체 상관관계 히트맵과 Pareto 인자 스크리닝을 함께 확인합니다.</p>
      </section>

      <section className="uploadCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">DATASET</span>
            <h2>학습 데이터셋</h2>
          </div>
          <LastRunNote createdAt={training?.createdAt} />
        </div>
        <DatasetSelector label="학습용 데이터셋" value={datasetId} onChange={setDatasetId} onDatasetsLoaded={setDatasets} />
        <DatasetMismatchWarning mismatch={datasetMismatch} />
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
          <button className="button primary" type="button" disabled={isRunning} onClick={() => void train()}>
            {isRunning ? "모델 학습 중…" : "학습 실행"}
          </button>
        </div>
        {isRunning && (
          <p className="trainingProgress" role="status">
            {stage} · {progress}%
          </p>
        )}
        {!isRunning && stage === "히트맵 집계 중" && !analysisReady && (
          <p className="trainingProgress" role="status">히트맵 집계 중 · {progress}%</p>
        )}
        {message && <p className="messageBox success" role="status">{message}</p>}
        {error && (
          <p className="errorMessage" role="alert">
            {error}{" "}
            <button type="button" className="button" style={{ marginLeft: 8 }} onClick={() => void train()}>
              재시도
            </button>
          </p>
        )}
      </section>

      {!isBundledTrain && (
        <section className="messageBox">기준값은 내장 데이터셋(train.CSV → test.CSV)에만 적용됩니다.</section>
      )}

      {/* 학습 모델 요약바 (§1-2) + 타깃별 통합 테이블 (§1-3) */}
      {isRunning || (performanceLoading && !hydrated) ? (
        <section className="trainingSummarySkeleton" role="status" aria-label="학습 결과 불러오는 중">
          <span className="trainingSummarySkeletonBar label" />
          <span className="trainingSummarySkeletonBar metrics" />
        </section>
      ) : performanceError ? (
        <section className="trainingResultError" role="alert">
          <span>{performanceError}</span>
          <button type="button" className="button" onClick={() => void loadPerformance({ requireResult: true })}>
            재시도
          </button>
        </section>
      ) : hasResult ? (
        <section className="trainingSummaryBar">
          <span className="trainingSummaryLabel">학습 모델</span>
          <span className="trainingSummaryMetrics">
            <span>R² {showMetric(performance?.final_yield?.r2)}</span>
            <span>MAE {showMetric(performance?.final_yield?.mae)}</span>
            <span>학습 시각 {formatLastRun(performance?.trained_at)}</span>
          </span>
        </section>
      ) : (
        <section className="trainingSummarySkeleton">
          <span className="trainingSummaryLabel">학습 모델</span>
          <p className="emptyMessage" style={{ margin: 0 }}>학습을 실행하면 표시됩니다.</p>
        </section>
      )}

      {hasResult && !isRunning && !performanceError && (
        <section className="resultCard">
          <div className="sectionHeading compact">
            <div>
              <span className="sectionLabel">TARGETS</span>
              <h2>타깃별 성능</h2>
            </div>
          </div>
          <div className="tableWrap">
            <table className="trainingTargetTable">
              <thead>
                <tr>
                  <th style={{ width: "10%" }}>타깃</th>
                  <th style={{ width: "28%" }}>1위 인자</th>
                  <th className="numCol" style={{ width: "13%" }}>ε²</th>
                  <th className="numCol" style={{ width: "13%" }}>기여율</th>
                  <th style={{ width: "12%" }}>등급</th>
                  <th className="numCol" style={{ width: "12%" }}>R²</th>
                  <th className="numCol" style={{ width: "12%" }}>MAE</th>
                </tr>
              </thead>
              <tbody>
                {(performance?.targets ?? []).map((detail) => (
                  <tr key={detail.target}>
                    <td>{detail.target}</td>
                    <td>{detail.no_factor_available ? "분석 불가" : detail.feature ?? "-"}</td>
                    <td className="numCol">{showMetric(detail.eps2)}</td>
                    <td className="numCol">{detail.contribution_pct != null ? `${detail.contribution_pct.toFixed(1)}%` : "-"}</td>
                    <td>
                      {!detail.no_factor_available && detail.confidence_tier ? (
                        <span className={`confidenceBadge tier-${detail.confidence_tier}`} style={{ marginLeft: 0 }}>
                          {TIER_LABEL[detail.confidence_tier]}
                        </span>
                      ) : "-"}
                    </td>
                    <td className="numCol">{showMetric(detail.r2)}</td>
                    <td className="numCol">{showMetric(detail.mae)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="resultCard">
        <div className="sectionHeading compact">
          <div>
            <span className="sectionLabel">PREPROCESSING</span>
            <h2>데이터 전처리</h2>
          </div>
        </div>
        <div className="tableWrap benchmarkTableWrap">
          <table className="benchmarkTable">
            <thead>
              <tr>
                <th>방식</th>
                <th className="numCol">R²</th>
                <th aria-hidden="true" />
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_REFERENCE.map((row) => (
                <tr key={row.name}>
                  <td>{row.name}</td>
                  <td className="numCol">{row.y.toFixed(3)}</td>
                  <td>{row.adopted && <span className="confidenceBadge adopted">채택</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <HeatmapParetoSection
        // The displayed result's own dataset (not the live selector value)
        // -- keeps the heatmap in sync with paretoByTarget even while the
        // selector points elsewhere (spec §5-3: don't auto-clear on a
        // selector change, just warn; a stale heatmap fetch here would be
        // exactly the "다른 데이터셋의 결과를 보게 된다" bug that guards against).
        datasetId={training?.dataset ?? datasetId}
        enabled={analysisReady}
        paretoByTarget={paretoByTarget}
        activeTarget={activeTarget}
        onActiveTargetChange={setActiveTarget}
        onBarClick={handleParetoBarClick}
        onHeatmapCellSelect={handleHeatmapSelect}
      />

      {analysisReady && (
        <section className="resultCard">
          <div className="sectionHeading compact">
            <div>
              <span className="sectionLabel">SCREENING</span>
              <h2>{activeTarget} 인자 스크리닝 (상위 5개)</h2>
            </div>
          </div>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th style={{ width: "22%" }}>인자명</th>
                  <th style={{ width: "12%" }}>종류</th>
                  <th className="numCol" style={{ width: "11%" }}>ε²</th>
                  <th className="numCol" style={{ width: "11%" }}>기여율</th>
                  <th className="numCol" style={{ width: "11%" }}>누적%</th>
                  <th className="numCol" style={{ width: "11%" }}>관측수</th>
                  <th className="numCol" style={{ width: "12%" }}>q값</th>
                  <th style={{ width: "10%" }}>신뢰도</th>
                </tr>
              </thead>
              <tbody>
                {(paretoByTarget[activeTarget]?.items ?? []).map((factor) => (
                  <tr key={factor.feature}>
                    <td>{factor.feature}</td>
                    <td>{kindLabel(factor.kind)}</td>
                    <td className="numCol">{showMetric(factor.eps2)}</td>
                    <td className="numCol">{showMetric(factor.contribution_pct, 1)}%</td>
                    <td className="numCol">{showMetric(factor.cumulative_pct, 1)}%</td>
                    <td className="numCol">{factor.n_observed}</td>
                    <td className="numCol">{formatQValue(factor.q_value)}</td>
                    <td><span className={`confidenceBadge tier-${factor.confidence_tier}`} style={{ marginLeft: 0 }}>{TIER_LABEL[factor.confidence_tier]}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="analysisDisclaimers">
        <strong>해석 시 한계</strong>
        <ul>
          <li>이 분석은 해당 인자가 계측된 wafer만 대상으로 합니다. R은 전체의 15%, D는 5%입니다. 미계측 wafer로의 일반화는 보장되지 않습니다.</li>
          <li>ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행/후행 관계나 교락 인자는 반영되지 않았습니다.</li>
          <li>Eq.(장비)에서 유의 인자가 검출되지 않은 것은 &quot;장비 영향이 없다&quot;가 아니라 &quot;현재 표본으로는 검출되지 않는다&quot;는 뜻입니다. 장비당 표본이 278장 수준이라 ε² 0.01 미만의 효과는 검출력이 부족합니다.</li>
        </ul>
      </section>
    </DashboardShell>
  );
}
