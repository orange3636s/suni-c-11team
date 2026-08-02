import type { MetricAggregate, MetricSummary } from "@/types/data";

const METRIC_KEYS = ["r2", "rmse", "mae", "mse"] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeMetricAggregate(value: unknown): MetricAggregate | undefined {
  if (!isRecord(value)) return undefined;
  const { mean, std } = value;
  if (typeof mean !== "number" || !Number.isFinite(mean)) return undefined;
  if (typeof std !== "number" || !Number.isFinite(std)) return undefined;
  return { mean, std };
}

function metricSummaryFrom(value: unknown): MetricSummary | undefined {
  if (!isRecord(value)) return undefined;
  const summary: MetricSummary = {};
  for (const key of METRIC_KEYS) {
    const aggregate = normalizeMetricAggregate(value[key]);
    if (aggregate) summary[key] = aggregate;
  }
  return Object.keys(summary).length > 0 ? summary : undefined;
}

function nestedSummary(container: unknown): MetricSummary | undefined {
  if (!isRecord(container)) return undefined;
  return metricSummaryFrom(container.metric_summary)
    ?? metricSummaryFrom(container.aggregate_metrics);
}

export function normalizeMetricSummary(result: unknown): MetricSummary | undefined {
  if (!isRecord(result)) return undefined;
  return nestedSummary(result.evaluation_summary)
    ?? nestedSummary(result.cv)
    ?? nestedSummary(result.cv_summary)
    ?? metricSummaryFrom(result.metric_summary)
    ?? metricSummaryFrom(result.aggregate_metrics);
}
