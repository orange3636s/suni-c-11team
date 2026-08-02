import type {
  AnalysisHistorySummary,
  AnalysisOverviewResponse,
  AnalysisSourceMetadata,
  OverviewAvailability,
  OverviewCauseItem,
  OverviewCauseSummary,
  OverviewModelMetrics,
  OverviewMultiY,
  OverviewRelationship,
  OverviewRiskLot,
  OverviewRiskWafer,
  OverviewSummary,
} from "@/types/data";


function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function integer(value: unknown): number | null {
  const number = finite(value);
  return number === null ? null : Math.trunc(number);
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function boolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    : [];
}

function numberMap(value: unknown): Record<string, number | null> {
  return Object.fromEntries(
    Object.entries(record(value)).map(([key, item]) => [key, finite(item)]),
  );
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    const number = finite(value);
    if (number !== null) return number;
  }
  return null;
}

function firstInteger(...values: unknown[]): number | null {
  for (const value of values) {
    const number = integer(value);
    if (number !== null) return number;
  }
  return null;
}

function stringOrNumber(value: unknown): string | number | null {
  return text(value) ?? finite(value);
}

function normalizeCauseItem(value: Record<string, unknown>): OverviewCauseItem {
  return {
    rank: integer(value.rank),
    feature: text(value.feature),
    display_name: text(value.display_name),
    step: stringOrNumber(value.step),
    equipment: text(value.equipment),
    chamber: text(value.chamber),
    mean_abs_shap: finite(value.mean_abs_shap),
    impact: finite(value.impact),
    score: finite(value.score),
    cumulative_share: finite(value.cumulative_share),
    path_score: finite(value.path_score),
  };
}

function normalizeRiskLot(value: Record<string, unknown>): OverviewRiskLot {
  return {
    lot_id: text(value.lot_id),
    wafer_count: integer(value.wafer_count),
    average_predicted_yield: finite(value.average_predicted_yield),
    danger_count: integer(value.danger_count),
    warning_count: integer(value.warning_count),
    normal_count: integer(value.normal_count),
    danger_ratio: finite(value.danger_ratio),
    top_harmful_feature: text(value.top_harmful_feature),
    top_harmful_step: text(value.top_harmful_step),
  };
}

function normalizeRiskWafer(value: Record<string, unknown>): OverviewRiskWafer {
  return {
    identifier: stringOrNumber(value.identifier),
    predicted_value: finite(value.predicted_value),
    prediction: finite(value.prediction),
    risk_level: text(value.risk_level),
    top_harmful_features: strings(value.top_harmful_features),
    top_step: text(value.top_step),
  };
}

function normalizeRelationship(value: Record<string, unknown>): OverviewRelationship {
  return {
    relation: text(value.relation),
    feature: text(value.feature),
    target: text(value.target),
    response: text(value.response),
    defect: text(value.defect),
    equipment: text(value.equipment),
    pearson: finite(value.pearson),
    spearman: finite(value.spearman),
    pearson_p_value: finite(value.pearson_p_value),
    spearman_p_value: finite(value.spearman_p_value),
    pearson_fdr_p_value: finite(value.pearson_fdr_p_value),
    spearman_fdr_p_value: finite(value.spearman_fdr_p_value),
    effect_size: finite(value.effect_size),
    path_score: finite(value.path_score),
    valid_count: integer(value.valid_count),
    direction: text(value.direction),
    interpretation: text(value.interpretation),
  };
}

function emptySummary(): OverviewSummary {
  return {
    wafer_count: null,
    lot_count: null,
    average_predicted_yield: null,
    minimum_predicted_yield: null,
    critical_count: null,
    warning_count: null,
    normal_count: null,
    low_confidence_count: null,
    risk_lot_count: null,
  };
}

function emptyMetrics(): OverviewModelMetrics {
  return { r2: null, rmse: null, mae: null };
}

function emptyMultiY(): OverviewMultiY {
  return {
    predicted_y_mean: null,
    failure_rates: {},
    fail_bit_counts: {},
  };
}

function emptyCauses(): OverviewCauseSummary {
  return {
    top_failure_target: null,
    top_features: [],
    top_steps: [],
    top_equipment: [],
    top_chambers: [],
  };
}

function emptyAvailability(): OverviewAvailability {
  return {
    summary: false,
    model_metrics: false,
    multi_y: false,
    causes: false,
    risk_lots: false,
    risk_wafers: false,
    pareto: false,
    relationships: false,
  };
}

export function createEmptyOverview(): AnalysisOverviewResponse {
  const source: AnalysisSourceMetadata = {
    type: "empty",
    analysis_id: null,
    created_at: null,
    completed_at: null,
    status: "empty",
    source_filename: null,
    model_id: null,
    model_name: null,
    artifact_available: false,
    artifact_status: "not_applicable",
  };
  return {
    source,
    summary: emptySummary(),
    model_metrics: emptyMetrics(),
    multi_y: emptyMultiY(),
    causes: emptyCauses(),
    risk_lots: [],
    risk_wafers: [],
    pareto: [],
    relationships: [],
    warnings: [],
    availability: emptyAvailability(),
    source_type: "empty",
    source_id: null,
    created_at: null,
    source_label: "저장된 원인 분석 결과 없음",
    filename: null,
    model: null,
    data_quality: {},
  };
}

function artifactStatus(value: unknown, available: boolean): AnalysisSourceMetadata["artifact_status"] {
  if (value === "available" || value === "missing" || value === "corrupted" || value === "not_applicable") {
    return value;
  }
  return available ? "available" : "missing";
}

export function normalizeOverviewAnalysis(payload: unknown): AnalysisOverviewResponse {
  if (!isRecord(payload)) {
    throw new Error("Dashboard API 응답이 JSON 객체가 아닙니다.");
  }
  const nestedSource = record(payload.source);
  const legacyType = text(payload.source_type);
  const inferredType = text(nestedSource.type) ?? legacyType;
  if (inferredType === "empty") return createEmptyOverview();
  if (inferredType !== "analysis") {
    throw new Error("Dashboard API가 원인 분석 이력이 아닌 Source를 반환했습니다.");
  }

  const summarySource = record(payload.summary);
  const modelSource = record(payload.model);
  const metricSource = record(payload.model_metrics);
  const multiYSource = record(payload.multi_y);
  const causeSource = record(payload.causes);
  const availabilitySource = record(payload.availability);

  const legacySourceId = text(payload.source_id);
  const available = boolean(nestedSource.artifact_available)
    ?? boolean(payload.artifact_available)
    ?? true;
  const source: AnalysisSourceMetadata = {
    type: "analysis",
    analysis_id: text(nestedSource.analysis_id) ?? legacySourceId,
    created_at: text(nestedSource.created_at) ?? text(payload.created_at),
    completed_at: text(nestedSource.completed_at),
    status: text(nestedSource.status) ?? "completed",
    source_filename: text(nestedSource.source_filename) ?? text(payload.filename),
    model_id: text(nestedSource.model_id) ?? text(modelSource.model_id),
    model_name: text(nestedSource.model_name) ?? text(modelSource.model_name),
    artifact_available: available,
    artifact_status: artifactStatus(nestedSource.artifact_status, available),
  };
  if (!source.analysis_id) {
    throw new Error("Dashboard API 응답에 analysis_id가 없습니다.");
  }

  const summary: OverviewSummary = {
    wafer_count: firstInteger(summarySource.wafer_count, summarySource.row_count),
    lot_count: firstInteger(summarySource.lot_count),
    average_predicted_yield: firstNumber(summarySource.average_predicted_yield),
    minimum_predicted_yield: firstNumber(summarySource.minimum_predicted_yield),
    critical_count: firstInteger(summarySource.critical_count, summarySource.danger_count),
    warning_count: firstInteger(summarySource.warning_count),
    normal_count: firstInteger(summarySource.normal_count),
    low_confidence_count: firstInteger(summarySource.low_confidence_count),
    risk_lot_count: firstInteger(summarySource.risk_lot_count),
  };
  const modelMetrics: OverviewModelMetrics = {
    r2: firstNumber(metricSource.r2, modelSource.cv_r2_mean, modelSource.r2),
    rmse: firstNumber(metricSource.rmse, modelSource.cv_rmse_mean, modelSource.rmse),
    mae: firstNumber(metricSource.mae, modelSource.mae),
  };
  const multiY: OverviewMultiY = {
    predicted_y_mean: firstNumber(multiYSource.predicted_y_mean),
    failure_rates: numberMap(multiYSource.failure_rates),
    fail_bit_counts: numberMap(multiYSource.fail_bit_counts),
  };
  const causes: OverviewCauseSummary = {
    top_failure_target: text(causeSource.top_failure_target),
    top_features: records(causeSource.top_features).map(normalizeCauseItem),
    top_steps: records(causeSource.top_steps).map(normalizeCauseItem),
    top_equipment: records(causeSource.top_equipment).map(normalizeCauseItem),
    top_chambers: records(causeSource.top_chambers).map(normalizeCauseItem),
  };
  const riskLots = records(payload.risk_lots).map(normalizeRiskLot);
  const riskWafers = records(payload.risk_wafers).map(normalizeRiskWafer);
  const pareto = records(payload.pareto).map(normalizeCauseItem);
  const relationships = records(payload.relationships).map(normalizeRelationship);

  const derivedAvailability: OverviewAvailability = {
    summary: Object.values(summary).some((value) => value !== null),
    model_metrics: Object.values(modelMetrics).some((value) => value !== null),
    multi_y: multiY.predicted_y_mean !== null
      || Object.keys(multiY.failure_rates).length > 0
      || Object.keys(multiY.fail_bit_counts).length > 0,
    causes: Boolean(
      causes.top_failure_target
      || causes.top_features.length
      || causes.top_steps.length
      || causes.top_equipment.length
      || causes.top_chambers.length
    ),
    risk_lots: riskLots.length > 0,
    risk_wafers: riskWafers.length > 0,
    pareto: pareto.length > 0,
    relationships: relationships.length > 0,
  };
  const availability: OverviewAvailability = {
    summary: boolean(availabilitySource.summary) ?? derivedAvailability.summary,
    model_metrics: boolean(availabilitySource.model_metrics) ?? derivedAvailability.model_metrics,
    multi_y: boolean(availabilitySource.multi_y) ?? derivedAvailability.multi_y,
    causes: boolean(availabilitySource.causes) ?? derivedAvailability.causes,
    risk_lots: boolean(availabilitySource.risk_lots) ?? derivedAvailability.risk_lots,
    risk_wafers: boolean(availabilitySource.risk_wafers) ?? derivedAvailability.risk_wafers,
    pareto: boolean(availabilitySource.pareto) ?? derivedAvailability.pareto,
    relationships: boolean(availabilitySource.relationships) ?? derivedAvailability.relationships,
  };

  return {
    source,
    summary,
    model_metrics: modelMetrics,
    multi_y: multiY,
    causes,
    risk_lots: riskLots,
    risk_wafers: riskWafers,
    pareto,
    relationships,
    warnings: strings(payload.warnings),
    availability,
    source_type: "analysis",
    source_id: source.analysis_id,
    created_at: source.created_at,
    source_label: text(payload.source_label) ?? "원인 분석 이력",
    filename: source.source_filename,
    model: Object.keys(modelSource).length ? modelSource : null,
    data_quality: record(payload.data_quality),
  };
}

export function overviewFromHistory(item: AnalysisHistorySummary): AnalysisOverviewResponse {
  const empty = createEmptyOverview();
  const summarySource = record(item.summary);
  const artifactAvailable = item.artifact_available ?? false;
  const artifactState = item.status === "artifact_corrupted"
    ? "corrupted"
    : artifactAvailable
      ? "available"
      : "missing";
  const summary: OverviewSummary = {
    ...empty.summary,
    wafer_count: item.row_count,
    lot_count: item.lot_count,
    average_predicted_yield: firstNumber(item.average_predicted_yield, summarySource.average_predicted_yield),
    minimum_predicted_yield: firstNumber(summarySource.minimum_predicted_yield),
    critical_count: firstInteger(item.critical_count, summarySource.critical_count),
    warning_count: firstInteger(item.warning_wafer_count, summarySource.warning_count),
    normal_count: firstInteger(summarySource.normal_count),
    low_confidence_count: firstInteger(summarySource.low_confidence_count),
    risk_lot_count: firstInteger(summarySource.risk_lot_count),
  };
  const source: AnalysisSourceMetadata = {
    type: "analysis",
    analysis_id: item.analysis_id,
    created_at: item.created_at,
    completed_at: item.completed_at ?? null,
    status: item.status,
    source_filename: item.source_filename,
    model_id: item.model_id,
    model_name: item.model_name ?? item.model_name_snapshot,
    artifact_available: artifactAvailable,
    artifact_status: artifactState,
  };
  return {
    ...empty,
    source,
    summary,
    availability: {
      ...empty.availability,
      summary: Object.values(summary).some((value) => value !== null),
    },
    source_type: "analysis",
    source_id: item.analysis_id,
    created_at: item.created_at,
    source_label: "선택한 원인 분석",
    filename: item.source_filename,
    model: {
      model_id: item.model_id,
      model_name: item.model_name ?? item.model_name_snapshot,
    },
  };
}

type SelectionOptions = {
  urlAnalysisId: string | null;
  currentAnalysisId: string | null;
  sessionAnalysisId: string | null;
};

export function resolveOverviewSelection(
  items: AnalysisHistorySummary[],
  options: SelectionOptions,
): { analysisId: string | null; invalidUrlAnalysisId: string | null } {
  const byId = (analysisId: string | null) => (
    analysisId ? items.find((item) => item.analysis_id === analysisId) : undefined
  );
  const completed = items.find((item) => (
    item.status === "completed"
    || item.status === "artifact_missing"
    || item.status === "artifact_corrupted"
  ));
  const partial = items.find((item) => item.status === "partial");

  if (options.urlAnalysisId) {
    const urlItem = byId(options.urlAnalysisId);
    if (urlItem) {
      return { analysisId: urlItem.analysis_id, invalidUrlAnalysisId: null };
    }
    return {
      analysisId: completed?.analysis_id ?? partial?.analysis_id ?? null,
      invalidUrlAnalysisId: options.urlAnalysisId,
    };
  }
  const current = byId(options.currentAnalysisId);
  if (current) return { analysisId: current.analysis_id, invalidUrlAnalysisId: null };
  const session = byId(options.sessionAnalysisId);
  if (session && session.status !== "failed" && session.status !== "running") {
    return { analysisId: session.analysis_id, invalidUrlAnalysisId: null };
  }
  return {
    analysisId: completed?.analysis_id ?? partial?.analysis_id ?? null,
    invalidUrlAnalysisId: null,
  };
}
