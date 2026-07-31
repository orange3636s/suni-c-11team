"use client";

import { useEffect, useMemo, useState } from "react";
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
import { getModelDetail, getModels } from "@/lib/api";
import type {
  ModelDetail,
  ModelDetailMetrics,
  ModelSummary,
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

function metricValue(value: number | null | undefined): string {
  return value === null || value === undefined ? "미기록" : value.toFixed(4);
}

function dateValue(value: string | null): string {
  if (!value) return "미기록";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("ko-KR");
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

  async function loadDetail(modelId: string) {
    setSelectedModelId(modelId);
    setDetailLoading(true);
    setError("");
    try {
      setDetail(await getModelDetail(modelId));
    } catch (requestError) {
      setDetail(null);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "모델 상세 정보를 불러오지 못했습니다.",
      );
    } finally {
      setDetailLoading(false);
    }
  }

  async function loadModels() {
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
        await loadDetail(nextModels[0].model_id);
      } else {
        setSelectedModelId("");
        setDetail(null);
      }
    } catch (requestError) {
      setLoaded(true);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "저장 모델 목록을 불러오지 못했습니다.",
      );
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

  const selectedMetrics = useMemo(
    () =>
      (["train", "validation", "test"] as const).map((name) => ({
        name: name === "validation" ? "Validation" : `${name[0].toUpperCase()}${name.slice(1)}`,
        ...(detail?.metrics[name] ?? {
          r2: null,
          rmse: null,
          mse: null,
          mae: null,
        }),
      })),
    [detail],
  );

  if (!loaded && !loading) {
    return (
      <div className="modelHistoryInitial">
        <EmptyState
          title="저장 모델 이력을 조회하세요."
          description="새 CSV나 재학습 없이 기존 모델의 성능과 메타데이터를 확인합니다."
        />
        <button className="button primary" type="button" onClick={loadModels}>
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
            onClick={loadModels}
          >
            재시도
          </button>
        </div>
      </div>

      {error && <div className="messageBox error" role="alert">{error}</div>}

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
              ) : detail ? (
                <>
                  <div className="modelDetailHeading">
                    <div>
                      <span className="sectionLabel">Model detail</span>
                      <h3>{displayValue(detail.model_name)}</h3>
                    </div>
                    <a
                      className="button primary linkButton"
                      href={`/prediction?model_id=${encodeURIComponent(detail.model_id)}`}
                    >
                      이 모델로 예측
                    </a>
                  </div>

                  <div className="modelMetadataGrid">
                    <DetailItem label="Model ID" value={detail.model_id} />
                    <DetailItem label="Model Type" value={displayValue(detail.model_type)} />
                    <DetailItem label="Model Version" value={displayValue(detail.model_version)} />
                    <DetailItem label="Created At" value={dateValue(detail.created_at)} />
                    <DetailItem label="Target" value={displayValue(detail.target)} />
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
                  </div>

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

                  <SelectedModelCharts metrics={detail.metrics} />
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
    </div>
  );
}

function SelectedModelCharts({
  metrics,
}: {
  metrics: Record<string, ModelDetailMetrics>;
}) {
  const data = (["train", "validation", "test"] as const).map((name) => ({
    dataset:
      name === "validation"
        ? "Validation"
        : `${name[0].toUpperCase()}${name.slice(1)}`,
    r2: metrics[name]?.r2,
    rmse: metrics[name]?.rmse,
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
                      Number(value).toFixed(4),
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
