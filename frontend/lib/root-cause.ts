import type {
  AnalysisHistoryDetail,
  AnalysisHistorySummary,
  AnalysisResult,
  ExplainResponse,
  LotCauseAnalysis,
  RelationshipAnalysisResponse,
  RelationshipFeature,
  RelationshipStatistics,
} from "@/types/data";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function nullableText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function list<T>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : [];
}

export function normalizeExplainResponse(value: unknown): ExplainResponse {
  const payload = record(value);
  const model = record(payload?.model);
  const summary = record(payload?.analysis_summary);
  if (!payload || !model) {
    throw new Error("원인 분석 응답에 필수 explanation 데이터가 없습니다.");
  }

  return {
    ...payload,
    success: payload.success !== false,
    filename: text(payload.filename),
    model: {
      model_id: text(model.model_id),
      target: text(model.target, "Y1"),
      model_name: text(model.model_name, "Y1~Y5 자동 모델"),
    },
    analysis_summary: {
      total_rows: finite(summary?.total_rows),
      analyzed_rows: finite(summary?.analyzed_rows),
      sampling_used: summary?.sampling_used === true,
      sampling_strategy: text(summary?.sampling_strategy, "none"),
      explanation_method: text(summary?.explanation_method, "SHAP"),
      is_fallback: summary?.is_fallback === true,
    },
    explanation_method: text(payload.explanation_method, "SHAP"),
    is_fallback: payload.is_fallback === true,
    identifier_column: text(payload.identifier_column, "Lot_Wafer_ID"),
    global_importance: list(payload.global_importance),
    step_summary: list(payload.step_summary),
    parameter_type_summary: list(payload.parameter_type_summary),
    equipment_summary: list(payload.equipment_summary),
    wafer_explanations: list(payload.wafer_explanations),
    model_quality_warnings: list(payload.model_quality_warnings),
    warnings: list(payload.warnings),
  } as ExplainResponse;
}

function normalizeRankingGroups(value: unknown): Record<string, RelationshipFeature[]> {
  const source = record(value) ?? {};
  const config = list<RelationshipFeature>(source.Config ?? source.EQ ?? source.config);
  return {
    all: list(source.all),
    R: list(source.R ?? source.r),
    D: list(source.D ?? source.d),
    Config: config,
  };
}

function emptyStatistics(): RelationshipStatistics {
  return {
    methods: [],
    numeric: [],
    categorical: [],
    scatter_data: [],
    boxplot_data: [],
    categorical_relationships: [],
  };
}

function normalizeLotAnalysis(value: unknown): LotCauseAnalysis | null {
  const source = record(value);
  if (!source) return null;
  return {
    target: nullableText(source.target),
    aggregation: nullableText(source.aggregation),
    sampling_used: typeof source.sampling_used === "boolean" ? source.sampling_used : null,
    total_lot_count: typeof source.total_lot_count === "number" ? source.total_lot_count : null,
    excluded_row_count: typeof source.excluded_row_count === "number" ? source.excluded_row_count : null,
    lots: list(source.lots),
  };
}

export function normalizeRelationshipResponse(value: unknown): RelationshipAnalysisResponse {
  const payload = record(value);
  if (!payload) throw new Error("원인 분석 응답 형식이 올바르지 않습니다.");
  const rankings = record(payload.rankings) ?? {};
  const paretoSource = record(payload.pareto) ?? {};
  const groupCounts = record(paretoSource.group_counts) ?? {};
  const correlationMethod = payload.correlation_method === "spearman" ? "spearman" : "pearson";
  const analysisUnit = payload.analysis_unit === "lot_aggregated"
    ? "lot_aggregated"
    : "wafer_observed_only";

  return {
    ...payload,
    success: payload.success !== false,
    filename: text(payload.filename),
    explanation: payload.explanation ? normalizeExplainResponse(payload.explanation) : null,
    target: text(payload.target, "Y1"),
    correlation_method: correlationMethod,
    rankings: {
      shap: normalizeRankingGroups(rankings.shap),
      correlation: normalizeRankingGroups(rankings.correlation),
    },
    pareto: {
      threshold: finite(paretoSource.threshold, 0.8),
      required_feature_count: finite(paretoSource.required_feature_count),
      cumulative_contribution: finite(paretoSource.cumulative_contribution),
      total_feature_count: finite(paretoSource.total_feature_count),
      total_impact: finite(paretoSource.total_impact),
      group_counts: {
        R: finite(groupCounts.R ?? groupCounts.r),
        D: finite(groupCounts.D ?? groupCounts.d),
        Config: finite(groupCounts.Config ?? groupCounts.EQ ?? groupCounts.config),
      },
      ranking_basis: text(paretoSource.ranking_basis),
      caveat: text(paretoSource.caveat),
      features: list(paretoSource.features),
    },
    relationship_paths: list(payload.relationship_paths),
    statistics: (record(payload.statistics) as RelationshipStatistics | null) ?? emptyStatistics(),
    available_steps: list(payload.available_steps),
    confidence_criteria: (record(payload.confidence_criteria) as Record<string, string> | null) ?? {},
    caveats: list(payload.caveats),
    analysis_unit: analysisUnit,
    config_summary: record(payload.config_summary) ?? {},
    selection_bias_warnings: list(payload.selection_bias_warnings),
    analysis_result: (record(payload.analysis_result) as AnalysisResult | null),
    lot_analysis: normalizeLotAnalysis(payload.lot_analysis),
    analysis_id: nullableText(payload.analysis_id),
    prediction_id: nullableText(payload.prediction_id),
    history_saved: typeof payload.history_saved === "boolean" ? payload.history_saved : undefined,
    history_warning: nullableText(payload.history_warning),
    artifact_available: typeof payload.artifact_available === "boolean" ? payload.artifact_available : null,
  } as RelationshipAnalysisResponse;
}

export function normalizeAnalysisHistoryDetail(
  value: unknown,
  fallbackAnalysisId: string,
): AnalysisHistoryDetail {
  const payload = record(value);
  if (!payload) throw new Error("원인 분석 이력 응답 형식이 올바르지 않습니다.");
  const metadataSource = record(payload.metadata) ?? payload;
  const artifactSource = record(payload.artifact);
  const responseSource = record(artifactSource?.response);
  const metadata: AnalysisHistorySummary & Record<string, unknown> = {
    ...metadataSource,
    analysis_id: text(metadataSource.analysis_id, fallbackAnalysisId),
    prediction_id: nullableText(metadataSource.prediction_id),
    created_at: text(metadataSource.created_at),
    completed_at: nullableText(metadataSource.completed_at),
    status: text(metadataSource.status, "completed") as AnalysisHistorySummary["status"],
    source_filename: nullableText(metadataSource.source_filename),
    model_id: nullableText(metadataSource.model_id),
    model_name_snapshot: nullableText(metadataSource.model_name_snapshot),
    row_count: typeof metadataSource.row_count === "number" ? metadataSource.row_count : null,
    lot_count: typeof metadataSource.lot_count === "number" ? metadataSource.lot_count : null,
    warning_count: typeof metadataSource.warning_count === "number" ? metadataSource.warning_count : null,
  };

  return {
    metadata,
    artifact: artifactSource
      ? {
          ...artifactSource,
          ...(responseSource ? { response: normalizeRelationshipResponse(responseSource) } : {}),
          ...(record(artifactSource.analysis_result)
            ? { analysis_result: artifactSource.analysis_result as AnalysisResult }
            : {}),
          ...(record(artifactSource.lot_analysis)
            ? { lot_analysis: normalizeLotAnalysis(artifactSource.lot_analysis)! }
            : {}),
        }
      : null,
    source_prediction_deleted: typeof payload.source_prediction_deleted === "boolean"
      ? payload.source_prediction_deleted
      : null,
    linked_analysis_count: typeof payload.linked_analysis_count === "number"
      ? payload.linked_analysis_count
      : null,
  };
}
