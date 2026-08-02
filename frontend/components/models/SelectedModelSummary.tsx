import type { MetricAggregate, ModelSummary } from "@/types/data";

type SelectedModelSummaryProps = {
  model: ModelSummary | null | undefined;
};

function metric(value: unknown): string | null {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : null;
}

function aggregateMetric(label: string, value: MetricAggregate | undefined): string | null {
  const mean = metric(value?.mean);
  if (mean === null) return null;
  const standardDeviation = metric(value?.std);
  return standardDeviation === null ? `${label} ${mean}` : `${label} ${mean} ± ${standardDeviation}`;
}

function compatibilityLabel(model: ModelSummary): string {
  if (!model.available || !model.loadable) return "사용 불가";
  if (model.compatibility === "incompatible") return "호환되지 않음";
  if (model.compatibility === "legacy") return "이전 모델";
  if (model.compatibility === "unknown_schema") return "스키마 확인 필요";
  return "호환 가능";
}

function createdAtLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("ko-KR", {
        year: "numeric",
        month: "numeric",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
}

export default function SelectedModelSummary({ model }: SelectedModelSummaryProps) {
  if (!model) return null;
  const cv = model.cv_summary;
  const modelType = model.model_type === "hybrid_multi_y"
    ? "Hybrid Multi-Y"
    : model.model_type ?? "Single Model";
  const foldParts: string[] = [];
  if (typeof cv?.outer_folds === "number" && Number.isFinite(cv.outer_folds)) foldParts.push(`Outer ${cv.outer_folds}`);
  if (typeof cv?.inner_folds === "number" && Number.isFinite(cv.inner_folds)) foldParts.push(`Inner ${cv.inner_folds}`);
  const protocol = cv
    ? [modelType, cv.name ?? "Nested CV", ...foldParts].join(" · ")
    : modelType;
  const cvMetrics = cv?.metric_summary ?? cv?.aggregate_metrics;
  const metricParts = cvMetrics
    ? [aggregateMetric("R²", cvMetrics.r2), aggregateMetric("RMSE", cvMetrics.rmse)].filter((item): item is string => item !== null)
    : [];
  if (!metricParts.length) {
    const testR2 = metric(model.test_metrics.r2);
    const testRmse = metric(model.test_metrics.rmse);
    if (testR2 !== null) metricParts.push(`R² ${testR2}`);
    if (testRmse !== null) metricParts.push(`RMSE ${testRmse}`);
  }
  const metrics = metricParts.length ? metricParts.join(" · ") : "평가 지표 없음";

  return (
    <div className="selectedModelSummary" aria-label="선택 모델 정보">
      <span className="selectedModelIcon" aria-hidden="true">
        <svg viewBox="0 0 20 20" width="17" height="17">
          <path d="M4 5.5 10 2l6 3.5v7L10 16l-6-3.5v-7Z" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="m4.5 5.8 5.5 3.1 5.5-3.1M10 9v6.4" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </span>
      <span className="selectedModelCopy">
        <span className="selectedModelTitleRow">
          <strong title={model.model_name}>{model.model_name}</strong>
          <span className={`modelCompatibility ${model.available && model.loadable ? model.compatibility : "unavailable"}`}>
            {compatibilityLabel(model)}
          </span>
        </span>
        <small title={model.model_id}>Model ID · {model.model_id}</small>
        <small>{protocol}</small>
        <small>{metrics} · {createdAtLabel(model.created_at)}</small>
        {(!model.available || !model.loadable) && (
          <small className="modelUnavailableReason">
            {model.incompatibility_reason ?? "현재 서버에서 이 모델을 사용할 수 없습니다."}
          </small>
        )}
      </span>
    </div>
  );
}
