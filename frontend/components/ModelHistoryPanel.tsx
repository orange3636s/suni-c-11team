"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import EmptyState from "@/components/EmptyState";
import StatusBadge from "@/components/StatusBadge";
import { deleteModel, getModelDetail, getModels } from "@/lib/api";
import type {
  ModelDetail,
  ModelDetailMetrics,
  ModelSummary,
  TargetEnsembleConfig,
} from "@/types/data";

type ModelSort =
  | "newest"
  | "oldest"
  | "r2-desc"
  | "rmse-asc"
  | "name-asc"
  | "name-desc";

function displayValue(value: unknown, suffix = ""): string {
  if (value === null || value === undefined || value === "") return "미기록";
  if (typeof value === "number") {
    return `${value.toLocaleString("ko-KR", {
      maximumFractionDigits: 4,
    })}${suffix}`;
  }
  return `${String(value)}${suffix}`;
}

function metricValue(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "미기록";
}

function dateValue(value: string | null | undefined): string {
  if (!value) return "미기록";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("ko-KR");
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function normalizedMetrics(value: unknown): Record<string, ModelDetailMetrics> {
  const source = recordValue(value);
  return Object.fromEntries(
    Object.entries(source).filter((entry): entry is [string, ModelDetailMetrics] =>
      Boolean(entry[1]) && typeof entry[1] === "object" && !Array.isArray(entry[1]),
    ),
  );
}

function normalizedTargetConfigs(value: unknown): Record<string, TargetEnsembleConfig> {
  const source = recordValue(value);
  return Object.fromEntries(
    Object.entries(source).filter((entry): entry is [string, TargetEnsembleConfig] =>
      Boolean(entry[1]) && typeof entry[1] === "object" && !Array.isArray(entry[1]),
    ),
  );
}

function DetailItem({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function ModelHistoryPanel() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<ModelSort>("newest");
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteCandidate, setDeleteCandidate] = useState<ModelDetail | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const [notice, setNotice] = useState("");
  const [detailError, setDetailError] = useState("");
  const detailRequest = useRef<AbortController | null>(null);
  const deletionInFlight = useRef(false);
  const deleteTrigger = useRef<HTMLButtonElement | null>(null);
  const deleteDialog = useRef<HTMLElement | null>(null);

  function openDeleteDialog() {
    if (!detail || deleting) return;
    setDeleteCandidate(detail);
    setDeleteError("");
    setNotice("");
  }

  function closeDeleteDialog() {
    if (deleting) return;
    setDeleteCandidate(null);
    setDeleteError("");
  }

  function updateModelIdQuery(modelId?: string) {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("view", "history");
      if (modelId) url.searchParams.set("model_id", modelId);
      else url.searchParams.delete("model_id");
      window.history.replaceState({}, "", url);
    } catch (queryError) {
      console.warn("모델 선택 URL을 갱신하지 못했습니다.", queryError);
    }
  }

  function clearPersistedModelSelection() {
    try {
      window.sessionStorage.removeItem("semiconductor-ai:last-model-id");
      window.localStorage.removeItem("semiconductor-ai:last-model-id");
    } catch (storageError) {
      console.warn("삭제된 모델 선택 상태를 저장소에서 제거하지 못했습니다.", storageError);
    }
  }

  async function handleDelete() {
    const candidate = deleteCandidate;
    if (!candidate || deleting || deletionInFlight.current) return;
    deletionInFlight.current = true;
    setDeleting(true);
    setDeleteError("");
    setError("");
    let deleted: Awaited<ReturnType<typeof deleteModel>>;
    try {
      deleted = await deleteModel(candidate.model_id);
    } catch (requestError) {
      const detailMessage = requestError instanceof Error
        ? requestError.message
        : "알 수 없는 오류가 발생했습니다.";
      setDeleteError(`모델 삭제에 실패했습니다. ${detailMessage}`);
      return;
    } finally {
      deletionInFlight.current = false;
      setDeleting(false);
    }

    const remainingModels = models.filter(
      (model) => model.model_id !== candidate.model_id,
    );
    const deletedIndex = models.findIndex(
      (model) => model.model_id === candidate.model_id,
    );
    const nextModel = remainingModels[
      Math.min(Math.max(deletedIndex, 0), remainingModels.length - 1)
    ] ?? remainingModels[0];

    detailRequest.current?.abort();
    detailRequest.current = null;
    clearPersistedModelSelection();
    updateModelIdQuery();
    setSearch("");
    setModels(remainingModels);
    setSelectedModelId("");
    setDetail(null);
    setDetailError("");
    setDetailLoading(false);
    setDeleteCandidate(null);
    const predictionHistory = typeof deleted.prediction_history_count === "number"
      ? `Prediction History ${deleted.prediction_history_count}건`
      : "Prediction History";
    const analysisHistory = typeof deleted.analysis_history_count === "number"
      ? `Analysis History ${deleted.analysis_history_count}건`
      : "Analysis History";
    setNotice(
      `모델이 삭제되었습니다. ${predictionHistory}와 ${analysisHistory}는 유지됩니다.`,
    );
    const refreshed = await loadModels(nextModel?.model_id);
    if (!refreshed && nextModel) {
      setSelectedModelId(nextModel.model_id);
      updateModelIdQuery(nextModel.model_id);
      setDetailError(
        "모델은 삭제됐지만 남은 모델 정보를 다시 불러오지 못했습니다.",
      );
    }
    // 삭제된 model_id가 새로고침 뒤 다시 선택되지 않도록 이력 탭만 URL에 남긴다.
    updateModelIdQuery();
  }

  async function loadDetail(modelId: string) {
    detailRequest.current?.abort();
    const controller = new AbortController();
    detailRequest.current = controller;
    setSelectedModelId(modelId);
    updateModelIdQuery(modelId);
    setDetailLoading(true);
    setDetailError("");
    try {
      const response = await getModelDetail(modelId, controller.signal);
      if (!controller.signal.aborted) setDetail(response);
    } catch (requestError) {
      if (controller.signal.aborted) return;
      setDetail(null);
      setDetailError(
        requestError instanceof Error && requestError.name !== "AbortError"
          ? requestError.message
          : "모델 상세 정보를 불러오지 못했습니다.",
      );
    } finally {
      if (detailRequest.current === controller) {
        detailRequest.current = null;
        setDetailLoading(false);
      }
    }
  }

  async function loadModels(preferredModelId?: string): Promise<boolean> {
    setLoading(true);
    setError("");
    try {
      const response = await getModels();
      const nextModels = [...response.models].sort(
        (left, right) =>
          new Date(right.created_at).getTime() -
          new Date(left.created_at).getTime(),
      );
      setModels(nextModels);
      setLoaded(true);
      if (nextModels.length) {
        const requestedId = preferredModelId ??
          new URLSearchParams(window.location.search).get("model_id");
        const initialModel = nextModels.find((model) => model.model_id === requestedId) ?? nextModels[0];
        await loadDetail(initialModel.model_id);
      } else {
        setSelectedModelId("");
        setDetail(null);
        setDetailError("");
        updateModelIdQuery();
      }
      return true;
    } catch (requestError) {
      setLoaded(true);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "저장 모델 목록을 불러오지 못했습니다.",
      );
      return false;
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadModels();
    }, 0);
    return () => window.clearTimeout(timer);
    // The history panel performs one initial read; retry remains explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => detailRequest.current?.abort(), []);

  useEffect(() => {
    if (!deleteCandidate) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const trigger = deleteTrigger.current;
    const handleDialogKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !deletionInFlight.current) {
        event.preventDefault();
        setDeleteCandidate(null);
        setDeleteError("");
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(
        deleteDialog.current?.querySelectorAll<HTMLElement>(
          "button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
        ) ?? [],
      );
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleDialogKeyDown);
    return () => {
      document.removeEventListener("keydown", handleDialogKeyDown);
      document.body.style.overflow = previousOverflow;
      trigger?.focus();
    };
  }, [deleteCandidate]);

  const visibleModels = useMemo(() => {
    const normalizedSearch = search.trim().toLocaleLowerCase("ko");
    return models
      .filter((model) =>
        `${model.model_name} ${model.model_id} ${model.target}`
          .toLocaleLowerCase("ko")
          .includes(normalizedSearch),
      )
      .sort((left, right) => {
        if (sort === "newest" || sort === "oldest") {
          const difference =
            new Date(right.created_at).getTime() -
            new Date(left.created_at).getTime();
          return sort === "oldest" ? -difference : difference;
        }
        if (sort === "r2-desc") {
          return (
            (right.test_metrics.r2 ?? Number.NEGATIVE_INFINITY) -
            (left.test_metrics.r2 ?? Number.NEGATIVE_INFINITY)
          );
        }
        if (sort === "rmse-asc") {
          return (
            (left.test_metrics.rmse ?? Number.POSITIVE_INFINITY) -
            (right.test_metrics.rmse ?? Number.POSITIVE_INFINITY)
          );
        }
        const comparison = left.model_name.localeCompare(
          right.model_name,
          "ko",
          { numeric: true, sensitivity: "base" },
        );
        return sort === "name-desc" ? -comparison : comparison;
      });
  }, [models, search, sort]);

  const safeMetrics = useMemo(() => normalizedMetrics(detail?.metrics), [detail]);
  const targetEnsembleConfigs = useMemo(
    () => normalizedTargetConfigs(detail?.target_ensemble_configs),
    [detail],
  );
  const selectedMetrics = useMemo(
    () =>
      (["train", "validation", "test"] as const).map((name) => ({
        name: name === "validation" ? "Validation" : `${name[0].toUpperCase()}${name.slice(1)}`,
        ...(safeMetrics[name] ?? {
          r2: null,
          rmse: null,
          mse: null,
          mae: null,
        }),
      })),
    [safeMetrics],
  );

  if (!loaded && !loading) {
    return (
      <div className="modelHistoryInitial">
        <EmptyState
          title="저장 모델 이력을 조회하세요."
          description="새 CSV나 재학습 없이 기존 모델의 성능과 메타데이터를 확인합니다."
        />
        <button className="button primary" type="button" onClick={() => void loadModels()}>
          저장 모델 불러오기
        </button>
      </div>
    );
  }

  return (
    <div className="modelHistory">
      <div className="modelHistoryHeader">
        <div>
          <span className="sectionLabel">Saved models</span>
          <h2>저장된 모델 이력</h2>
        </div>
        <div className="modelHistoryStatus">
          <StatusBadge
            label={error ? "API 오류" : loading ? "불러오는 중" : "연결됨"}
            tone={error ? "danger" : loading ? "neutral" : "success"}
          />
          <button
            className="button secondary"
            type="button"
            disabled={loading}
            onClick={() => void loadModels()}
          >
            재시도
          </button>
        </div>
      </div>

      {error && <div className="messageBox error" role="alert">{error}</div>}
      {notice && <div className="messageBox success" role="status">{notice}</div>}

      {!models.length && !loading ? (
        <EmptyState
          title="저장된 모델이 없습니다."
          description="새 모델을 학습하면 이곳에서 이력을 확인할 수 있습니다."
        />
      ) : (
        <>
          <div className="modelHistoryLayout">
            <aside className="savedModelBrowser" aria-label="저장 모델 목록">
              <div className="modelHistoryTools">
                <input
                  type="search"
                  placeholder="모델명 또는 버전 검색"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
                <select
                  aria-label="저장 모델 정렬"
                  value={sort}
                  onChange={(event) => setSort(event.target.value as ModelSort)}
                >
                  <option value="newest">최신 학습 순</option>
                  <option value="oldest">오래된 학습 순</option>
                  <option value="r2-desc">R² 높은 순</option>
                  <option value="rmse-asc">RMSE 낮은 순</option>
                  <option value="name-asc">모델명 오름차순</option>
                  <option value="name-desc">모델명 내림차순</option>
                </select>
              </div>
              <div className="savedModelList">
                {visibleModels.map((model) => (
                  <button
                    className={
                      selectedModelId === model.model_id ? "active" : ""
                    }
                    type="button"
                    key={model.model_id}
                    onClick={() => void loadDetail(model.model_id)}
                  >
                    <span>
                      <strong>{model.model_name}</strong>
                      <small>{model.target} · {dateValue(model.created_at)}</small>
                      <small>Schema · {model.compatibility}</small>
                    </span>
                    <span>
                      <small>Test R²</small>
                      <b>{metricValue(model.test_metrics.r2)}</b>
                    </span>
                  </button>
                ))}
                {!visibleModels.length && (
                  <p className="emptyListMessage">검색 결과가 없습니다.</p>
                )}
              </div>
            </aside>

            <section className="modelDetailPanel" aria-live="polite">
              {detailLoading ? (
                <div className="modelDetailLoading" role="status">
                  모델 상세 정보를 불러오고 있습니다.
                </div>
              ) : detailError ? (
                <div className="modelDetailError messageBox error retryMessage" role="alert">
                  <span><strong>모델 상세 정보를 불러오지 못했습니다.</strong><small>{detailError}</small></span>
                  <button className="button secondary" type="button" onClick={() => void loadDetail(selectedModelId)}>다시 시도</button>
                </div>
              ) : detail ? (
                <>
                  <div className="modelDetailHeading">
                    <div>
                      <span className="sectionLabel">Model detail</span>
                      <h3 title={detail.model_name ?? detail.model_id}>{displayValue(detail.model_name)}</h3>
                    </div>
                    <div className="modelDetailActions">
                    {detail.compatibility === "compatible" && (
                      <a
                        className="button primary linkButton"
                        href={`/prediction?model_id=${encodeURIComponent(detail.model_id)}`}
                      >
                        이 모델로 예측
                      </a>
                    )}
                    <button ref={deleteTrigger} type="button" className="button danger compact" disabled={deleting || Boolean(deleteCandidate)} onClick={openDeleteDialog} aria-label={`${detail.model_name ?? detail.model_id} 모델 삭제`} title="모델 삭제">모델 삭제</button>
                    </div>
                  </div>

                  <div className="modelMetadataGrid">
                    <DetailItem label="Model ID" value={detail.model_id} />
                    <DetailItem label="Model Type" value={displayValue(detail.model_type)} />
                    <DetailItem label="Model Version" value={displayValue(detail.model_version)} />
                    <DetailItem label="Created At" value={dateValue(detail.created_at)} />
                    <DetailItem label="Target" value={displayValue(detail.target)} />
                    <DetailItem label="Schema Version" value={displayValue(detail.schema_version)} />
                    <DetailItem label="Compatibility" value={detail.compatibility} />
                    <DetailItem label="Config Parser" value={displayValue(detail.config_parser_version)} />
                    <DetailItem label="Missing Indicator" value={detail.missing_indicator_used === null ? "미기록" : detail.missing_indicator_used ? "사용" : "미사용"} />
                    <DetailItem label="Outlier Policy" value={displayValue(detail.outlier_policy)} />
                    <DetailItem label="Group Column" value={displayValue(detail.group_column)} />
                    <DetailItem label="Feature Count" value={displayValue(detail.feature_count)} />
                    <DetailItem label="Train Rows" value={displayValue(detail.dataset_rows?.train)} />
                    <DetailItem label="Validation Rows" value={displayValue(detail.dataset_rows?.validation)} />
                    <DetailItem label="Test Rows" value={displayValue(detail.dataset_rows?.test)} />
                    <DetailItem label="Train Ratio" value={displayValue(detail.dataset_split?.train === undefined ? null : detail.dataset_split.train * 100, "%")} />
                    <DetailItem label="Validation Ratio" value={displayValue(detail.dataset_split?.validation === undefined ? null : detail.dataset_split.validation * 100, "%")} />
                    <DetailItem label="Test Ratio" value={displayValue(detail.dataset_split?.test === undefined ? null : detail.dataset_split.test * 100, "%")} />
                    <DetailItem label="Random Seed" value={displayValue(detail.random_seed)} />
                    <DetailItem label="Split Method" value={displayValue(detail.split_method)} />
                    <DetailItem label="Preprocessing" value={displayValue(detail.preprocessing_version)} />
                    <DetailItem label="Training Time" value={displayValue(detail.training_time_seconds, "초")} />
                    <DetailItem label="Source CSV" value={displayValue(detail.source_filename)} />
                    <DetailItem label="Model File" value={displayValue(detail.model_file)} />
                    <DetailItem label="저장 상태" value={detail.storage_status === "available" ? "저장됨" : "모델 파일 없음"} />
                    <DetailItem label="Champion" value={detail.champion === null ? "미기록" : detail.champion ? "Yes" : "No"} />
                    <DetailItem label="Ensemble" value={detail.ensemble_enabled === null ? "Single Model" : detail.ensemble_enabled ? "사용" : "미사용"} />
                    <DetailItem label="Ensemble Size" value={displayValue(detail.ensemble_mode)} />
                    <DetailItem label="Ensemble Method" value={displayValue(detail.ensemble_method)} />
                    <DetailItem label="Production Retrained" value={detail.production_ensemble_retrained === null ? "미기록" : detail.production_ensemble_retrained ? "완료" : "미완료"} />
                  </div>

                  {Object.keys(targetEnsembleConfigs).length > 0 && (
                    <div className="tableWrap modelMetricsTable">
                      <table><thead><tr><th>Target</th><th>유형</th><th>Method</th><th>Base Models</th><th>Weight</th><th>Single 대비 개선</th></tr></thead>
                      <tbody>{Object.entries(targetEnsembleConfigs).map(([target, config]) => (
                        <tr key={target}><th>{target}</th><td>{config.selected_type ?? "-"}</td><td>{config.method ?? "-"}</td><td>{Array.isArray(config.base_models) ? config.base_models.join(" / ") : "-"}</td><td>{Object.entries(recordValue(config.weights)).map(([name, weight]) => `${name} ${metricValue(weight)}`).join(" · ") || "-"}</td><td>{typeof config.improvement_over_single?.rmse_relative === "number" && Number.isFinite(config.improvement_over_single.rmse_relative) ? `${(config.improvement_over_single.rmse_relative * 100).toFixed(2)}%` : "-"}</td></tr>
                      ))}</tbody></table>
                    </div>
                  )}

                  <div className="tableWrap modelMetricsTable">
                    <table>
                      <thead>
                        <tr>
                          <th>Dataset</th>
                          <th>R²</th>
                          <th>RMSE</th>
                          <th>MSE</th>
                          <th>MAE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedMetrics.map((metric) => (
                          <tr key={metric.name}>
                            <th>{metric.name}</th>
                            <td>{metricValue(metric.r2)}</td>
                            <td>{metricValue(metric.rmse)}</td>
                            <td>{metricValue(metric.mse)}</td>
                            <td>{metricValue(metric.mae)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <SelectedModelCharts metrics={safeMetrics} />
                </>
              ) : (
                <EmptyState
                  title="모델을 선택하세요."
                  description="목록에서 저장 모델을 선택하면 상세 정보를 표시합니다."
                />
              )}
            </section>
          </div>
        </>
      )}
      {deleteCandidate && (
        <div
          className="modelDeleteBackdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeDeleteDialog();
          }}
        >
          <section
            ref={deleteDialog}
            className="modelDeleteDialog"
            role="dialog"
            aria-modal="true"
            aria-busy={deleting}
            aria-labelledby="model-delete-title"
            aria-describedby="model-delete-description"
          >
            <header>
              <span className="sectionLabel">Delete model</span>
              <h3 id="model-delete-title">모델을 삭제할까요?</h3>
            </header>
            <p id="model-delete-description">
              모델 Bundle과 Metadata는 삭제되며 복구할 수 없습니다.
              Prediction History와 Analysis History는 유지됩니다.
            </p>
            <dl className="modelDeleteSummary">
              <div><dt>모델명</dt><dd>{displayValue(deleteCandidate.model_name)}</dd></div>
              <div><dt>Model ID</dt><dd>{deleteCandidate.model_id}</dd></div>
              <div><dt>생성 시각</dt><dd>{dateValue(deleteCandidate.created_at)}</dd></div>
              <div><dt>모델 유형</dt><dd>{displayValue(deleteCandidate.model_type)}</dd></div>
            </dl>
            {deleteError && (
              <div className="messageBox error" role="alert">{deleteError}</div>
            )}
            <div className="modelDeleteActions">
              <button
                className="button secondary"
                type="button"
                disabled={deleting}
                onClick={closeDeleteDialog}
                autoFocus
              >
                취소
              </button>
              <button
                className="button danger"
                type="button"
                disabled={deleting}
                onClick={() => void handleDelete()}
              >
                {deleting ? "삭제 중…" : "삭제 확인"}
              </button>
            </div>
            <span className="srOnly" role="status" aria-live="polite">
              {deleting ? "모델을 삭제하고 있습니다." : ""}
            </span>
          </section>
        </div>
      )}
    </div>
  );
}

function SelectedModelCharts({
  metrics,
}: {
  metrics?: Record<string, ModelDetailMetrics> | null;
}) {
  const safeMetrics = normalizedMetrics(metrics);
  const data = (["train", "validation", "test"] as const).map((name) => ({
    dataset:
      name === "validation"
        ? "Validation"
        : `${name[0].toUpperCase()}${name.slice(1)}`,
    r2: safeMetrics[name]?.r2,
    rmse: safeMetrics[name]?.rmse,
  }));
  const hasR2 = data.some((item) => typeof item.r2 === "number");
  const hasRmse = data.some((item) => typeof item.rmse === "number");

  if (!hasR2 && !hasRmse) return null;
  return (
    <div className="modelMetricCharts">
      {(
        [
          ["r2", "Train / Validation / Test R²"],
          ["rmse", "Train / Validation / Test RMSE"],
        ] as const
      ).map(([metric, title]) =>
        data.some((item) => typeof item[metric] === "number") ? (
          <article key={metric}>
            <h4>{title}</h4>
            <div className="modelMetricChart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                  <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                  <XAxis dataKey="dataset" tick={{ fill: "var(--chart-axis)", fontSize: 10 }} />
                  <YAxis tick={{ fill: "var(--chart-axis)", fontSize: 10 }} />
                  <Tooltip
                    formatter={(value) => [
                      metricValue(value),
                      metric.toUpperCase(),
                    ]}
                  />
                  <Bar dataKey={metric} fill="var(--chart-primary)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>
        ) : null,
      )}
    </div>
  );
}
