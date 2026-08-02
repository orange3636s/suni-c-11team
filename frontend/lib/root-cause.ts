import type {
  AnalysisHistoryDetail,
  AnalysisHistorySummary,
  AnalysisResult,
  AssociationSummary,
  CategoricalStatistic,
  CategoryTargetSummary,
  EquipmentDistribution,
  ExplainResponse,
  GlobalImportanceItem,
  HistoryStatus,
  LocalContributionItem,
  LotCauseAnalysis,
  LotCauseItem,
  LotFeatureImportanceItem,
  LotParetoItem,
  LotWaferItem,
  NumericStatistic,
  RelationshipAnalysisResponse,
  RelationshipFeature,
  RelationshipPath,
  RelationshipStatistics,
  ReportLotSummary,
  ReportResponse,
  ReportRiskWafer,
  StatisticalTestResult,
  WaferExplanation,
} from "@/types/data";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
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

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    : [];
}

function finiteNumbers(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const number = finite(item);
    return number === null ? [] : [number];
  });
}

function nullableNumberMap(value: unknown): Record<string, number | null> {
  const output: Record<string, number | null> = {};
  for (const [key, item] of Object.entries(record(value))) {
    if (item === null) output[key] = null;
    else {
      const number = finite(item);
      if (number !== null) output[key] = number;
    }
  }
  return output;
}

function numberMap(value: unknown): Record<string, number> {
  const output: Record<string, number> = {};
  for (const [key, item] of Object.entries(record(value))) {
    const number = finite(item);
    if (number !== null) output[key] = number;
  }
  return output;
}

function stringMap(value: unknown): Record<string, string> {
  const output: Record<string, string> = {};
  for (const [key, item] of Object.entries(record(value))) {
    const normalized = text(item);
    if (normalized !== null) output[key] = normalized;
  }
  return output;
}

function hasOwn(value: JsonRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function riskLevel(value: unknown): "normal" | "warning" | "danger" | null {
  if (value === "normal") return "normal";
  if (value === "warning") return "warning";
  if (value === "danger") return "danger";
  return null;
}

function historyStatus(value: unknown): HistoryStatus | null {
  if (value === "running") return "running";
  if (value === "completed") return "completed";
  if (value === "failed") return "failed";
  if (value === "partial") return "partial";
  if (value === "artifact_missing") return "artifact_missing";
  if (value === "artifact_corrupted") return "artifact_corrupted";
  if (value === "model_deleted") return "model_deleted";
  if (value === "prediction_deleted") return "prediction_deleted";
  return null;
}

function relationshipConfidence(value: unknown): RelationshipPath["confidence"] | null {
  if (value === "sufficient") return "sufficient";
  if (value === "caution") return "caution";
  if (value === "insufficient") return "insufficient";
  return null;
}

function findingSeverity(value: unknown): ReportResponse["key_findings"][number]["severity"] | null {
  if (value === "info") return "info";
  if (value === "warning") return "warning";
  if (value === "danger") return "danger";
  return null;
}

function recommendationPriority(value: unknown): ReportResponse["recommendations"][number]["priority"] | null {
  if (value === "high") return "high";
  if (value === "medium") return "medium";
  if (value === "low") return "low";
  return null;
}

function normalizeLocalContribution(value: JsonRecord): LocalContributionItem | null {
  const feature = text(value.feature);
  const shapValue = finite(value.shap_value);
  const harmful = finite(value.harmful_contribution);
  const beneficial = finite(value.beneficial_contribution);
  const step = text(value.step);
  const parameterType = text(value.parameter_type);
  if (
    feature === null || shapValue === null || harmful === null || beneficial === null ||
    step === null || parameterType === null
  ) return null;
  return {
    feature,
    value: value.value,
    shap_value: shapValue,
    harmful_contribution: harmful,
    beneficial_contribution: beneficial,
    step,
    parameter_type: parameterType,
  };
}

function normalizeWaferExplanation(value: JsonRecord): WaferExplanation | null {
  const identifier = hasOwn(value, "identifier") ? value.identifier : value.wafer_id;
  const prediction = finite(value.prediction);
  const baseValue = finite(value.base_value);
  if (identifier === undefined || prediction === null || baseValue === null) return null;
  const slotNumber = finite(value.slot ?? value.wafer_slot);
  const slotText = text(value.slot ?? value.wafer_slot);
  const waferSlot = integer(value.wafer_slot ?? value.slot);
  return {
    identifier,
    lot_id: text(value.lot_id),
    wafer_id: text(value.wafer_id ?? value.lot_wafer_id),
    wafer_slot: waferSlot,
    slot: slotNumber ?? slotText,
    prediction,
    risk_level: riskLevel(value.risk_level),
    base_value: baseValue,
    top_negative_contributors: records(value.top_negative_contributors)
      .map(normalizeLocalContribution)
      .filter((item): item is LocalContributionItem => item !== null),
    top_positive_contributors: records(value.top_positive_contributors)
      .map(normalizeLocalContribution)
      .filter((item): item is LocalContributionItem => item !== null),
  };
}

function normalizeGlobalImportance(value: unknown): GlobalImportanceItem[] {
  return records(value).flatMap((item, index) => {
    const feature = text(item.feature);
    const meanAbsolute = finite(item.mean_abs_shap);
    const meanHarmful = finite(item.mean_harmful_contribution);
    if (feature === null || meanAbsolute === null || meanHarmful === null) return [];
    return [{
      rank: integer(item.rank) ?? index + 1,
      feature,
      step: text(item.step) ?? "",
      parameter_type: text(item.parameter_type) ?? "",
      parameter_name: text(item.parameter_name) ?? "",
      mean_abs_shap: meanAbsolute,
      mean_harmful_contribution: meanHarmful,
      direction: text(item.direction) ?? "",
    }];
  });
}

function normalizeStepSummary(value: unknown): ExplainResponse["step_summary"] {
  return records(value).flatMap((item, index) => {
    const step = text(item.step);
    const meanAbsolute = finite(item.mean_abs_shap);
    const harmful = finite(item.harmful_contribution);
    const featureCount = integer(item.feature_count);
    if (step === null || meanAbsolute === null || harmful === null || featureCount === null) return [];
    return [{ rank: integer(item.rank) ?? index + 1, step, mean_abs_shap: meanAbsolute, harmful_contribution: harmful, feature_count: featureCount }];
  });
}

function normalizeParameterSummary(value: unknown): ExplainResponse["parameter_type_summary"] {
  return records(value).flatMap((item, index) => {
    const parameterType = text(item.parameter_type);
    const meanAbsolute = finite(item.mean_abs_shap);
    const harmful = finite(item.harmful_contribution);
    const featureCount = integer(item.feature_count);
    if (parameterType === null || meanAbsolute === null || harmful === null || featureCount === null) return [];
    return [{ rank: integer(item.rank) ?? index + 1, parameter_type: parameterType, mean_abs_shap: meanAbsolute, harmful_contribution: harmful, feature_count: featureCount }];
  });
}

function normalizeEquipmentSummary(value: unknown): ExplainResponse["equipment_summary"] {
  return records(value).flatMap((item, index) => {
    const equipment = text(item.equipment);
    const meanAbsolute = finite(item.mean_abs_shap);
    const harmful = finite(item.harmful_contribution);
    if (equipment === null || meanAbsolute === null || harmful === null) return [];
    return [{ rank: integer(item.rank) ?? index + 1, equipment, mean_abs_shap: meanAbsolute, harmful_contribution: harmful }];
  });
}

function parseExplainResponse(value: unknown): ExplainResponse | null {
  if (!isRecord(value)) return null;
  const model = record(value.model);
  const summary = record(value.analysis_summary);
  const filename = text(value.filename);
  const modelId = text(model.model_id);
  const target = text(model.target);
  const modelName = text(model.model_name);
  const totalRows = integer(summary.total_rows);
  const analyzedRows = integer(summary.analyzed_rows);
  const samplingUsed = boolean(summary.sampling_used);
  const samplingStrategy = text(summary.sampling_strategy);
  const summaryMethod = text(summary.explanation_method);
  const summaryFallback = boolean(summary.is_fallback);
  const explanationMethod = text(value.explanation_method) ?? summaryMethod;
  const fallback = boolean(value.is_fallback) ?? summaryFallback;
  const identifierColumn = text(value.identifier_column);
  const success = boolean(value.success);
  if (
    filename === null || modelId === null || target === null || modelName === null ||
    totalRows === null || analyzedRows === null || samplingUsed === null ||
    samplingStrategy === null || summaryMethod === null || summaryFallback === null ||
    explanationMethod === null || fallback === null || identifierColumn === null || success === null
  ) return null;
  return {
    success,
    filename,
    model: { model_id: modelId, target, model_name: modelName },
    analysis_summary: {
      total_rows: totalRows,
      analyzed_rows: analyzedRows,
      sampling_used: samplingUsed,
      sampling_strategy: samplingStrategy,
      explanation_method: summaryMethod,
      is_fallback: summaryFallback,
    },
    explanation_method: explanationMethod,
    is_fallback: fallback,
    identifier_column: identifierColumn,
    global_importance: normalizeGlobalImportance(value.global_importance),
    step_summary: normalizeStepSummary(value.step_summary),
    parameter_type_summary: normalizeParameterSummary(value.parameter_type_summary),
    equipment_summary: normalizeEquipmentSummary(value.equipment_summary),
    wafer_explanations: records(value.wafer_explanations)
      .map(normalizeWaferExplanation)
      .filter((item): item is WaferExplanation => item !== null),
    model_quality_warnings: strings(value.model_quality_warnings),
    warnings: strings(value.warnings),
  };
}

export function normalizeExplainResponse(value: unknown): ExplainResponse {
  const response = parseExplainResponse(value);
  if (!response) {
    throw new Error("원인 분석 응답에 필수 explanation 데이터가 없습니다.");
  }
  return response;
}

function inferredGroup(feature: string): string {
  const match = feature.split("__").at(-1)?.match(/^Step\d+_(R|D|Config|Model|Equipment|EQ|Chamber)/i);
  return match?.[1] ?? "unknown";
}

function normalizeRelationshipFeature(value: JsonRecord): RelationshipFeature | null {
  const feature = text(value.feature);
  if (feature === null) return null;
  const group = text(value.group ?? value.parameter_type) ?? inferredGroup(feature);
  return {
    rank: integer(value.rank) ?? undefined,
    feature,
    display_name: text(value.display_name) ?? feature,
    step: integer(value.step),
    group,
    ranking_basis: text(value.ranking_basis) ?? "",
    score: finite(value.score ?? value.mean_abs_shap),
    signed_association: finite(value.signed_association),
    direction: text(value.direction) ?? "",
    valid_count: integer(value.valid_count),
    missing_count: integer(value.missing_count),
    missing_rate: finite(value.missing_rate),
    category_count: integer(value.category_count),
    is_categorical: boolean(value.is_categorical) ?? /config|model|equipment|eq|chamber/i.test(group),
    p_value: finite(value.p_value),
    fdr_p_value: finite(value.fdr_p_value),
    effect_size: finite(value.effect_size),
    mean_abs_shap: finite(value.mean_abs_shap ?? value.score),
    pearson: finite(value.pearson),
    spearman: finite(value.spearman),
    coverage: finite(value.coverage),
    reason: text(value.reason),
    source_features: strings(value.source_features),
    source_feature_count: integer(value.source_feature_count),
  };
}

function uniqueFeatures(rows: RelationshipFeature[]): RelationshipFeature[] {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = `${row.feature}\u0000${row.group}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function normalizeRankingGroups(value: unknown): Record<string, RelationshipFeature[]> {
  const source = record(value);
  const groups: Record<string, RelationshipFeature[]> = {};
  for (const [key, rows] of Object.entries(source)) {
    groups[key] = records(rows)
      .map(normalizeRelationshipFeature)
      .filter((item): item is RelationshipFeature => item !== null);
  }
  const first = (...keys: string[]): RelationshipFeature[] => {
    for (const key of keys) if (groups[key]?.length) return groups[key];
    return [];
  };
  const r = first("r", "R");
  const d = first("d", "D");
  const model = first("model", "MODEL");
  const equipment = first("equipment", "eq", "EQ");
  const chamber = first("chamber", "CHAMBER");
  const measurement = first("measurement", "missing", "indicator", "observed");
  const config = first("config", "Config").length
    ? first("config", "Config")
    : uniqueFeatures([...model, ...equipment, ...chamber]);
  const all = first("all", "overall").length
    ? first("all", "overall")
    : uniqueFeatures([...r, ...d, ...config, ...measurement]);
  return {
    ...groups,
    all,
    overall: all,
    r,
    R: r,
    d,
    D: d,
    config,
    model,
    equipment,
    eq: equipment,
    EQ: equipment,
    chamber,
    measurement,
    missing: measurement,
    indicator: measurement,
    observed: measurement,
  };
}

function normalizeRankings(value: unknown): RelationshipAnalysisResponse["rankings"] {
  const source = record(value);
  return {
    shap: normalizeRankingGroups(source.shap),
    correlation: normalizeRankingGroups(source.correlation),
  };
}

function normalizeTestResult(value: unknown): StatisticalTestResult {
  const source = record(value);
  return {
    statistic: finite(source.statistic),
    p_value: finite(source.p_value),
    fdr_p_value: finite(source.fdr_p_value),
  };
}

function normalizeScatterPoints(value: unknown): { x: number; y: number }[] {
  return records(value).flatMap((item) => {
    const x = finite(item.x);
    const y = finite(item.y);
    return x === null || y === null ? [] : [{ x, y }];
  });
}

function normalizeCategorySummary(value: unknown): CategoryTargetSummary[] {
  return records(value).flatMap((item) => {
    const category = text(item.category ?? item.equipment);
    const count = integer(item.count);
    if (category === null || count === null) return [];
    const outliers = finiteNumbers(item.outliers);
    return [{
      category,
      count,
      coverage: finite(item.coverage),
      mean: finite(item.mean),
      median: finite(item.median),
      q1: finite(item.q1),
      q3: finite(item.q3),
      minimum: finite(item.minimum),
      maximum: finite(item.maximum),
      whisker_min: finite(item.whisker_min),
      whisker_max: finite(item.whisker_max),
      outliers,
      outlier_count: integer(item.outlier_count) ?? outliers.length,
      sample_warning: boolean(item.sample_warning),
    }];
  });
}

function normalizeNumericStatistic(value: JsonRecord): NumericStatistic | null {
  const feature = text(value.feature);
  const target = text(value.target);
  const validCount = integer(value.valid_count);
  const excludedCount = integer(value.excluded_count);
  const coverage = finite(value.coverage);
  if (feature === null || target === null || validCount === null || excludedCount === null || coverage === null) return null;
  return {
    relation: text(value.relation) ?? "",
    feature,
    target,
    pearson: finite(value.pearson),
    spearman: finite(value.spearman),
    pearson_p_value: finite(value.pearson_p_value),
    spearman_p_value: finite(value.spearman_p_value),
    pearson_fdr_p_value: finite(value.pearson_fdr_p_value),
    spearman_fdr_p_value: finite(value.spearman_fdr_p_value),
    effect_size: finite(value.effect_size),
    valid_count: validCount,
    excluded_count: excludedCount,
    coverage,
    direction: text(value.direction) ?? "",
    strength: text(value.strength) ?? "",
    group: text(value.group) ?? undefined,
    scatter_data: normalizeScatterPoints(value.scatter_data),
    scatter_sampled: boolean(value.scatter_sampled) ?? undefined,
    reason: text(value.reason),
  };
}

function normalizeCategoricalStatistic(value: JsonRecord): CategoricalStatistic | null {
  const feature = text(value.feature);
  const target = text(value.target);
  const validCount = integer(value.valid_count);
  const excludedCount = integer(value.excluded_count);
  const coverage = finite(value.coverage);
  const categoryCount = integer(value.category_count);
  if (
    feature === null || target === null || validCount === null || excludedCount === null ||
    coverage === null || categoryCount === null
  ) return null;
  const summaries = normalizeCategorySummary(value.category_summary ?? value.boxplot_data);
  return {
    relation: text(value.relation) ?? "",
    feature,
    target,
    valid_count: validCount,
    excluded_count: excludedCount,
    coverage,
    category_count: categoryCount,
    effect_size: finite(value.effect_size),
    anova: normalizeTestResult(value.anova),
    welch_anova: normalizeTestResult(value.welch_anova),
    kruskal: normalizeTestResult(value.kruskal),
    group: text(value.group) ?? undefined,
    source_type: text(value.source_type) ?? undefined,
    category_summary: summaries,
    boxplot_data: summaries,
    reason: text(value.reason),
  };
}

function normalizeStatistics(value: unknown): RelationshipStatistics {
  const source = record(value);
  const numeric = records(source.numeric)
    .map(normalizeNumericStatistic)
    .filter((item): item is NumericStatistic => item !== null);
  const categorical = records(source.categorical)
    .map(normalizeCategoricalStatistic)
    .filter((item): item is CategoricalStatistic => item !== null);
  const categoricalAliases = records(source.categorical_relationships)
    .map(normalizeCategoricalStatistic)
    .filter((item): item is CategoricalStatistic => item !== null);
  const explicitScatter = normalizeScatterPoints(source.scatter_data);
  const explicitBoxes = normalizeCategorySummary(source.boxplot_data);
  return {
    methods: strings(source.methods),
    numeric,
    categorical,
    scatter_data: explicitScatter.length
      ? explicitScatter
      : numeric.flatMap((item) => item.scatter_data ?? []),
    boxplot_data: explicitBoxes.length
      ? explicitBoxes
      : categorical.flatMap((item) => item.boxplot_data ?? []),
    categorical_relationships: categoricalAliases.length ? categoricalAliases : categorical,
  };
}

function normalizeAssociation(value: unknown): AssociationSummary | null {
  if (!isRecord(value)) return null;
  const validCount = integer(value.valid_count);
  const excludedCount = integer(value.excluded_count);
  if (validCount === null || excludedCount === null) return null;
  return {
    pearson: finite(value.pearson),
    spearman: finite(value.spearman),
    eta_squared: finite(value.eta_squared),
    valid_count: validCount,
    excluded_count: excludedCount,
    category_count: integer(value.category_count) ?? undefined,
    direction: text(value.direction) ?? undefined,
    strength: text(value.strength) ?? undefined,
  };
}

function normalizeEquipmentDistribution(value: JsonRecord): EquipmentDistribution | null {
  const equipment = text(value.equipment ?? value.category);
  const count = integer(value.count);
  const mean = finite(value.mean);
  const median = finite(value.median);
  const q1 = finite(value.q1);
  const q3 = finite(value.q3);
  const minimum = finite(value.minimum);
  const maximum = finite(value.maximum);
  if (
    equipment === null || count === null || mean === null || median === null ||
    q1 === null || q3 === null || minimum === null || maximum === null
  ) return null;
  const outliers = finiteNumbers(value.outliers);
  return {
    equipment,
    count,
    mean,
    median,
    q1,
    q3,
    minimum,
    maximum,
    whisker_min: finite(value.whisker_min) ?? undefined,
    whisker_max: finite(value.whisker_max) ?? undefined,
    outliers,
    outlier_count: integer(value.outlier_count) ?? outliers.length,
    sample_warning: boolean(value.sample_warning),
  };
}

function normalizeRelationshipPath(value: JsonRecord, index: number): RelationshipPath | null {
  const step = integer(value.step);
  const defect = text(value.defect);
  const shapImportance = finite(value.shap_importance);
  const validCount = integer(value.valid_count);
  const missingRate = finite(value.missing_rate);
  const pathScore = finite(value.path_score);
  const confidence = relationshipConfidence(value.confidence);
  if (
    step === null || defect === null || shapImportance === null || validCount === null ||
    missingRate === null || pathScore === null || confidence === null
  ) return null;
  return {
    rank: integer(value.rank) ?? index + 1,
    step,
    response: text(value.response),
    defect,
    equipment: text(value.equipment),
    model: text(value.model),
    chamber: text(value.chamber),
    r_d: normalizeAssociation(value.r_d),
    eq_d: normalizeAssociation(value.eq_d),
    d_y: normalizeAssociation(value.d_y),
    r_y: normalizeAssociation(value.r_y),
    eq_y: normalizeAssociation(value.eq_y),
    shap_importance: shapImportance,
    valid_count: validCount,
    missing_rate: missingRate,
    confidence,
    path_score: pathScore,
    path_status: text(value.path_status) ?? "",
    interpretation: text(value.interpretation) ?? "",
    r_vs_d: normalizeScatterPoints(value.r_vs_d),
    r_vs_y: normalizeScatterPoints(value.r_vs_y),
    eq_vs_d: records(value.eq_vs_d)
      .map(normalizeEquipmentDistribution)
      .filter((item): item is EquipmentDistribution => item !== null),
    eq_vs_y: records(value.eq_vs_y)
      .map(normalizeEquipmentDistribution)
      .filter((item): item is EquipmentDistribution => item !== null),
    d_vs_y: normalizeScatterPoints(value.d_vs_y),
  };
}

function normalizePareto(value: unknown): RelationshipAnalysisResponse["pareto"] {
  const source = record(value);
  const threshold = finite(source.threshold) ?? 0.8;
  const features = records(source.features).flatMap((item) => {
    const feature = normalizeRelationshipFeature(item);
    const impact = finite(item.impact);
    const share = finite(item.share);
    const cumulativeShare = finite(item.cumulative_share);
    if (feature === null || impact === null || share === null || cumulativeShare === null) return [];
    return [{
      ...feature,
      impact,
      share,
      cumulative_share: cumulativeShare,
      within_threshold: boolean(item.within_threshold) ?? cumulativeShare <= threshold,
    }];
  });
  const directCounts = record(source.group_counts);
  const within = features.filter((item) => item.within_threshold);
  const countGroup = (group: string): number => within.filter((item) => item.group.toLowerCase() === group).length;
  const requiredCount = integer(source.required_feature_count) ?? within.length;
  return {
    threshold,
    required_feature_count: requiredCount,
    cumulative_contribution: finite(source.cumulative_contribution)
      ?? features[Math.max(requiredCount - 1, 0)]?.cumulative_share
      ?? 0,
    total_feature_count: integer(source.total_feature_count) ?? features.length,
    total_impact: finite(source.total_impact) ?? features.reduce((sum, item) => sum + item.impact, 0),
    group_counts: {
      R: integer(directCounts.R) ?? countGroup("r"),
      D: integer(directCounts.D) ?? countGroup("d"),
      EQ: integer(directCounts.EQ) ?? countGroup("eq") + countGroup("equipment"),
    },
    ranking_basis: text(source.ranking_basis) ?? "",
    caveat: text(source.caveat) ?? "",
    features,
  };
}

function normalizeReportRiskWafer(value: JsonRecord): ReportRiskWafer | null {
  const identifier = hasOwn(value, "identifier") ? value.identifier : value.lot_wafer_id;
  const predicted = finite(value.predicted_value ?? value.prediction);
  if (identifier === undefined || predicted === null) return null;
  return {
    identifier,
    predicted_value: predicted,
    risk_level: riskLevel(value.risk_level),
    actual_value: finite(value.actual_value),
    absolute_error: finite(value.absolute_error),
    top_harmful_features: strings(value.top_harmful_features),
    top_step: text(value.top_step),
    top_parameter_type: text(value.top_parameter_type),
  };
}

function normalizeReportLotSummary(value: JsonRecord): ReportLotSummary | null {
  const lotId = text(value.lot_id);
  const waferCount = integer(value.wafer_count);
  const average = finite(value.average_predicted_yield);
  const dangerCount = integer(value.danger_count);
  const warningCount = integer(value.warning_count);
  const normalCount = integer(value.normal_count);
  const dangerRatio = finite(value.danger_ratio);
  if (
    lotId === null || waferCount === null || average === null || dangerCount === null ||
    warningCount === null || normalCount === null || dangerRatio === null
  ) return null;
  return {
    lot_id: lotId,
    wafer_count: waferCount,
    average_predicted_yield: average,
    danger_count: dangerCount,
    warning_count: warningCount,
    normal_count: normalCount,
    danger_ratio: dangerRatio,
    top_harmful_feature: text(value.top_harmful_feature),
    top_harmful_step: text(value.top_harmful_step),
  };
}

function normalizeMultiYWafer(value: JsonRecord): AnalysisResult["multi_y"]["wafer_results"][number] | null {
  const identifier = hasOwn(value, "identifier") ? value.identifier : value.wafer_id;
  if (identifier === undefined) return null;
  return {
    identifier,
    direct_y: finite(value.direct_y),
    derived_y: finite(value.derived_y),
    ensemble_y: finite(value.ensemble_y),
    direct_derived_gap: finite(value.direct_derived_gap),
    failure_rates: numberMap(value.failure_rates),
    fail_bit_counts: numberMap(value.fail_bit_counts),
  };
}

function normalizeAnalysisResult(value: unknown): AnalysisResult | null {
  if (!isRecord(value)) return null;
  const model = record(value.model);
  const dataset = record(value.dataset);
  const target = record(value.target);
  const multiY = record(value.multi_y);
  const risk = record(value.risk);
  const quality = record(value.data_quality);
  const analysisId = text(value.analysis_id);
  const analysisVersion = text(value.analysis_version);
  const createdAt = text(value.created_at);
  const modelId = text(model.model_id);
  const filename = text(dataset.filename);
  const rowCount = integer(dataset.row_count);
  const targetName = text(target.name);
  const warningThreshold = finite(risk.warning_threshold);
  const criticalThreshold = finite(risk.critical_threshold);
  const normalCount = integer(risk.normal_count);
  const warningCount = integer(risk.warning_count);
  const criticalCount = integer(risk.critical_count);
  const rCoverage = finite(quality.r_measurement_coverage);
  const dCoverage = finite(quality.d_measurement_coverage);
  const configCompleteness = finite(quality.config_completeness_rate);
  const configParseErrors = integer(quality.config_parse_error_count);
  if (
    analysisId === null || analysisVersion === null || createdAt === null || modelId === null ||
    filename === null || rowCount === null || targetName === null || warningThreshold === null ||
    criticalThreshold === null || normalCount === null || warningCount === null || criticalCount === null ||
    rCoverage === null || dCoverage === null || configCompleteness === null || configParseErrors === null
  ) return null;
  const methodology = record(value.methodology);
  const report = record(value.report);
  return {
    analysis_id: analysisId,
    analysis_version: analysisVersion,
    created_at: createdAt,
    model: {
      model_id: modelId,
      model_name: text(model.model_name),
      model_version: text(model.model_version),
      schema_version: text(model.schema_version),
      compatibility: text(model.compatibility),
      structure: text(model.structure),
    },
    dataset: {
      filename,
      fingerprint: text(dataset.fingerprint),
      row_count: rowCount,
      identifier_column: text(dataset.identifier_column),
    },
    target: {
      name: targetName,
      label: text(target.label) ?? targetName,
      type: text(target.type),
      unit: text(target.unit),
    },
    metrics: record(value.metrics),
    multi_y: {
      average_direct_y: finite(multiY.average_direct_y),
      average_derived_y: finite(multiY.average_derived_y),
      average_ensemble_y: finite(multiY.average_ensemble_y),
      ensemble_weight: finite(multiY.ensemble_weight),
      failure_rate_averages: nullableNumberMap(multiY.failure_rate_averages),
      fail_bit_count_averages: nullableNumberMap(multiY.fail_bit_count_averages),
      wafer_results: records(multiY.wafer_results)
        .map(normalizeMultiYWafer)
        .filter((item): item is AnalysisResult["multi_y"]["wafer_results"][number] => item !== null),
    },
    risk: {
      warning_threshold: warningThreshold,
      critical_threshold: criticalThreshold,
      normal_count: normalCount,
      warning_count: warningCount,
      critical_count: criticalCount,
      risk_probability: finite(risk.risk_probability),
    },
    confidence: record(value.confidence),
    feature_importance: record(value.feature_importance),
    shap: record(value.shap),
    wafer_explanations: records(value.wafer_explanations)
      .map(normalizeWaferExplanation)
      .filter((item): item is WaferExplanation => item !== null),
    relationships: records(value.relationships)
      .map(normalizeRelationshipPath)
      .filter((item): item is RelationshipPath => item !== null),
    statistics: normalizeStatistics(value.statistics),
    lot_analysis: normalizeLotAnalysis(value.lot_analysis),
    risk_wafers: records(value.risk_wafers)
      .map(normalizeReportRiskWafer)
      .filter((item): item is ReportRiskWafer => item !== null),
    lot_summary: records(value.lot_summary)
      .map(normalizeReportLotSummary)
      .filter((item): item is ReportLotSummary => item !== null),
    data_quality: {
      r_measurement_coverage: rCoverage,
      d_measurement_coverage: dCoverage,
      config_completeness_rate: configCompleteness,
      target_consistency_rate: finite(quality.target_consistency_rate),
      config_parse_error_count: configParseErrors,
      missing_indicator_used: boolean(quality.missing_indicator_used),
      outlier_policy: text(quality.outlier_policy),
      selection_bias_warnings: strings(quality.selection_bias_warnings),
    },
    methodology: {
      ...methodology,
      ...(Array.isArray(methodology.notes) ? { notes: strings(methodology.notes) } : {}),
    },
    report: {
      report_id: text(report.report_id),
      report_version: text(report.report_version),
    },
    warnings: strings(value.warnings),
  };
}

function normalizeReportTargetAnalysis(
  value: unknown,
): NonNullable<ReportResponse["target_analysis"]> | null {
  if (!isRecord(value)) return null;
  return {
    target: text(value.target),
    rankings: normalizeRankings(value.rankings),
    pareto: normalizePareto(value.pareto),
    statistics: normalizeStatistics(value.statistics),
  };
}

function normalizeReportRelationshipAnalysis(
  value: unknown,
): NonNullable<ReportResponse["relationship_analysis"]> | null {
  if (!isRecord(value)) return null;
  return {
    relationship_paths: records(value.relationship_paths)
      .map(normalizeRelationshipPath)
      .filter((item): item is RelationshipPath => item !== null),
    statistics: normalizeStatistics(value.statistics),
  };
}

function normalizeReportResponse(value: unknown): ReportResponse | null {
  if (!isRecord(value)) return null;
  const model = record(value.model);
  const metrics = record(model.test_metrics);
  const summary = record(value.executive_summary);
  const success = boolean(value.success);
  const reportId = text(value.report_id);
  const createdAt = text(value.created_at);
  const filename = text(value.filename);
  const modelId = text(model.model_id);
  const target = text(model.target);
  const modelName = text(model.model_name);
  const average = finite(summary.average_predicted_yield);
  const normalCount = integer(summary.normal_count);
  const warningCount = integer(summary.warning_count);
  const dangerCount = integer(summary.danger_count);
  const riskRatio = finite(summary.risk_ratio);
  const explanationMethod = text(value.explanation_method);
  const fallback = boolean(value.is_fallback);
  if (
    success === null || reportId === null || createdAt === null || filename === null ||
    modelId === null || target === null || modelName === null || average === null ||
    normalCount === null || warningCount === null || dangerCount === null || riskRatio === null ||
    explanationMethod === null || fallback === null
  ) return null;
  return {
    success,
    report_id: reportId,
    created_at: createdAt,
    filename,
    model: {
      model_id: modelId,
      target,
      model_name: modelName,
      test_metrics: {
        r2: finite(metrics.r2),
        rmse: finite(metrics.rmse),
        mae: finite(metrics.mae),
        mse: finite(metrics.mse),
      },
    },
    executive_summary: {
      total_wafers: integer(summary.total_wafers),
      average_predicted_yield: average,
      normal_count: normalCount,
      warning_count: warningCount,
      danger_count: dangerCount,
      risk_ratio: riskRatio,
      analyzed_rows: integer(summary.analyzed_rows),
      shap_sampling_used: boolean(summary.shap_sampling_used),
      sampling_strategy: text(summary.sampling_strategy),
    },
    key_findings: records(value.key_findings).flatMap((item) => {
      const severity = findingSeverity(item.severity);
      const title = text(item.title);
      const description = text(item.description);
      const evidence = text(item.evidence);
      if (!severity || !title || !description || !evidence) return [];
      return [{ severity, title, description, evidence }];
    }),
    top_risk_wafers: records(value.top_risk_wafers)
      .map(normalizeReportRiskWafer)
      .filter((item): item is ReportRiskWafer => item !== null),
    lot_summary: records(value.lot_summary)
      .map(normalizeReportLotSummary)
      .filter((item): item is ReportLotSummary => item !== null),
    top_features: normalizeGlobalImportance(value.top_features),
    top_steps: normalizeStepSummary(value.top_steps),
    parameter_type_summary: records(value.parameter_type_summary).flatMap((item, index) => {
      const normalized = normalizeParameterSummary([item])[0];
      return normalized ? [{ ...normalized, rank: integer(item.rank) ?? index + 1, ratio: finite(item.ratio) }] : [];
    }),
    recommendations: records(value.recommendations).flatMap((item) => {
      const priority = recommendationPriority(item.priority);
      const title = text(item.title);
      const description = text(item.description);
      if (!priority || !title || !description) return [];
      return [{ priority, title, description }];
    }),
    model_quality_warnings: strings(value.model_quality_warnings),
    methodology_notes: strings(value.methodology_notes),
    explanation_method: explanationMethod,
    is_fallback: fallback,
    warnings: strings(value.warnings),
    analysis_id: text(value.analysis_id),
    snapshot_metadata: isRecord(value.snapshot_metadata) ? value.snapshot_metadata : null,
    lot_analysis: normalizeLotAnalysis(value.lot_analysis),
    target_analysis: normalizeReportTargetAnalysis(value.target_analysis),
    relationship_analysis: normalizeReportRelationshipAnalysis(value.relationship_analysis),
  };
}

function normalizeLotFeature(value: JsonRecord, fallbackGroup: string): LotFeatureImportanceItem | null {
  const feature = text(value.feature);
  const numericStep = integer(value.step);
  const step = text(value.step) ?? (numericStep === null ? null : `Step${numericStep}`);
  const meanSignedShap = finite(value.mean_signed_shap);
  const meanAbsoluteShap = finite(value.mean_abs_shap);
  const adverseContribution = finite(value.adverse_contribution);
  const improvementContribution = finite(value.improvement_contribution);
  const sampleCount = integer(value.sample_count);
  const coverage = finite(value.coverage);
  if (
    feature === null || step === null || meanSignedShap === null || meanAbsoluteShap === null ||
    adverseContribution === null || improvementContribution === null || sampleCount === null || coverage === null
  ) return null;
  return {
    rank: integer(value.rank) ?? undefined,
    feature,
    display_name: text(value.display_name) ?? feature,
    step,
    group: text(value.group) ?? fallbackGroup,
    mean_signed_shap: meanSignedShap,
    mean_abs_shap: meanAbsoluteShap,
    adverse_contribution: adverseContribution,
    improvement_contribution: improvementContribution,
    sample_count: sampleCount,
    coverage,
    source_features: strings(value.source_features),
  };
}

function normalizeLotFeatureGroups(value: unknown): LotCauseItem["feature_importance"] {
  const source = record(value);
  const group = (name: "all" | "r" | "d" | "config") => records(source[name])
    .map((item) => normalizeLotFeature(item, name))
    .filter((item): item is LotFeatureImportanceItem => item !== null);
  return { all: group("all"), r: group("r"), d: group("d"), config: group("config") };
}

function normalizeLotParetoGroups(value: unknown): LotCauseItem["pareto"] {
  const source = record(value);
  const group = (name: "all" | "r" | "d" | "config") => records(source[name]).flatMap((item) => {
    const feature = text(item.feature);
    const adverseContribution = finite(item.adverse_contribution ?? item.impact);
    const share = finite(item.share);
    const cumulativeShare = finite(item.cumulative_share);
    const sampleCount = integer(item.sample_count);
    const coverage = finite(item.coverage);
    if (
      feature === null || adverseContribution === null || share === null ||
      cumulativeShare === null || sampleCount === null || coverage === null
    ) return [];
    const impact = finite(item.impact);
    const withinThreshold = boolean(item.within_threshold);
    const output: LotParetoItem = {
      rank: integer(item.rank) ?? undefined,
      feature,
      display_name: text(item.display_name) ?? feature,
      group: text(item.group) ?? name,
      adverse_contribution: adverseContribution,
      ...(impact === null ? {} : { impact }),
      share,
      cumulative_share: cumulativeShare,
      ...(withinThreshold === null ? {} : { within_threshold: withinThreshold }),
      sample_count: sampleCount,
      coverage,
    };
    return [output];
  });
  return { all: group("all"), r: group("r"), d: group("d"), config: group("config") };
}

function normalizeLotWafer(value: JsonRecord, fallbackLotId: string): LotWaferItem | null {
  const identifier = hasOwn(value, "identifier")
    ? value.identifier
    : value.lot_wafer_id ?? value.wafer_id;
  if (identifier === undefined) return null;
  const waferIdNumber = finite(value.wafer_id);
  const waferIdText = text(value.wafer_id);
  const prediction = finite(value.prediction ?? value.predicted_value);
  return {
    identifier,
    lot_id: text(value.lot_id) ?? fallbackLotId,
    wafer_id: waferIdNumber ?? waferIdText,
    wafer_slot: integer(value.wafer_slot),
    prediction,
    predicted_value: finite(value.predicted_value ?? value.prediction),
    predicted_yield: finite(value.predicted_yield),
    risk_level: riskLevel(value.risk_level),
    confidence: finite(value.confidence),
    top_feature: text(value.top_feature),
    top_step: text(value.top_step),
    top_config: text(value.top_config),
    shap_available: boolean(value.shap_available),
  };
}

function normalizeLotCauseItem(value: JsonRecord): LotCauseItem | null {
  const lotId = text(value.lot_id);
  if (lotId === null) return null;
  const causes = record(value.top_causes);
  return {
    lot_id: lotId,
    wafer_count: integer(value.wafer_count),
    analyzed_wafer_count: integer(value.analyzed_wafer_count),
    shap_coverage: finite(value.shap_coverage),
    average_predicted_value: finite(value.average_predicted_value),
    average_predicted_yield: finite(value.average_predicted_yield),
    minimum_predicted_value: finite(value.minimum_predicted_value),
    maximum_predicted_value: finite(value.maximum_predicted_value),
    risk_extreme_predicted_value: finite(value.risk_extreme_predicted_value),
    risk_extreme_direction: value.risk_extreme_direction === "minimum" || value.risk_extreme_direction === "maximum"
      ? value.risk_extreme_direction
      : null,
    critical_wafer_count: integer(value.critical_wafer_count),
    warning_wafer_count: integer(value.warning_wafer_count),
    normal_wafer_count: integer(value.normal_wafer_count),
    average_confidence: finite(value.average_confidence),
    top_failure_target: text(value.top_failure_target),
    top_failure_rate_target: text(value.top_failure_rate_target),
    top_failure_rate_average: finite(value.top_failure_rate_average),
    top_fail_bit_count_target: text(value.top_fail_bit_count_target),
    top_fail_bit_count_average: finite(value.top_fail_bit_count_average),
    feature_importance: normalizeLotFeatureGroups(value.feature_importance),
    pareto: normalizeLotParetoGroups(value.pareto),
    wafer_list: records(value.wafer_list)
      .map((item) => normalizeLotWafer(item, lotId))
      .filter((item): item is LotWaferItem => item !== null),
    top_causes: {
      feature: text(causes.feature),
      step: text(causes.step) ?? (integer(causes.step) === null ? null : String(integer(causes.step))),
      config: text(causes.config),
      failure_target: text(causes.failure_target),
    },
  };
}

function normalizeLotAnalysis(value: unknown): LotCauseAnalysis | null {
  if (!isRecord(value)) return null;
  return {
    target: text(value.target),
    aggregation: text(value.aggregation),
    sampling_used: boolean(value.sampling_used),
    total_lot_count: integer(value.total_lot_count),
    excluded_row_count: integer(value.excluded_row_count),
    lots: records(value.lots)
      .map(normalizeLotCauseItem)
      .filter((item): item is LotCauseItem => item !== null),
  };
}

function parseRelationshipResponse(value: unknown): RelationshipAnalysisResponse | null {
  if (!isRecord(value)) return null;
  const explanation = parseExplainResponse(value.explanation);
  if (!explanation) return null;
  const correlation = text(value.correlation_method);
  const unit = text(value.analysis_unit);
  const relationshipPaths = records(value.relationship_paths)
    .map(normalizeRelationshipPath)
    .filter((item): item is RelationshipPath => item !== null);
  return {
    success: boolean(value.success) ?? explanation.success,
    filename: text(value.filename) ?? explanation.filename,
    explanation,
    target: text(value.target) ?? explanation.model.target,
    correlation_method: correlation === "pearson" || correlation === "spearman" ? correlation : null,
    rankings: normalizeRankings(value.rankings),
    pareto: normalizePareto(value.pareto),
    relationship_paths: relationshipPaths,
    statistics: normalizeStatistics(value.statistics),
    available_steps: Array.isArray(value.available_steps)
      ? value.available_steps.flatMap((item) => {
          const step = integer(item);
          return step === null ? [] : [step];
        })
      : [],
    confidence_criteria: stringMap(value.confidence_criteria),
    caveats: strings(value.caveats),
    analysis_unit: unit === "wafer_observed_only" || unit === "lot_aggregated" ? unit : null,
    config_summary: record(value.config_summary),
    selection_bias_warnings: strings(value.selection_bias_warnings),
    analysis_result: normalizeAnalysisResult(value.analysis_result),
    report_snapshot: normalizeReportResponse(value.report_snapshot),
    lot_analysis: normalizeLotAnalysis(value.lot_analysis),
    analysis_id: text(value.analysis_id),
    prediction_id: text(value.prediction_id),
    history_saved: boolean(value.history_saved) ?? undefined,
    history_warning: text(value.history_warning),
  };
}

export function normalizeRelationshipResponse(value: unknown): RelationshipAnalysisResponse {
  const response = parseRelationshipResponse(value);
  if (!response) {
    throw new Error("관계 분석 응답에 필수 explanation 데이터가 없습니다.");
  }
  return response;
}

function normalizeHistorySummary(value: unknown, expectedAnalysisId?: string): (AnalysisHistorySummary & JsonRecord) | null {
  if (!isRecord(value)) return null;
  const analysisId = text(value.analysis_id) ?? text(expectedAnalysisId);
  const createdAt = text(value.created_at);
  const status = historyStatus(value.status);
  if (analysisId === null || createdAt === null || status === null) return null;
  return {
    ...value,
    analysis_id: analysisId,
    prediction_id: text(value.prediction_id),
    created_at: createdAt,
    completed_at: text(value.completed_at),
    status,
    source_filename: text(value.source_filename),
    model_id: text(value.model_id),
    model_name: text(value.model_name),
    model_name_snapshot: text(value.model_name_snapshot),
    row_count: integer(value.row_count),
    lot_count: integer(value.lot_count),
    average_predicted_yield: finite(value.average_predicted_yield),
    critical_count: integer(value.critical_count),
    warning_wafer_count: integer(value.warning_wafer_count),
    top_failure_target: text(value.top_failure_target),
    artifact_available: boolean(value.artifact_available) ?? undefined,
    summary: isRecord(value.summary) ? value.summary : null,
    default_target: text(value.default_target),
    warning_count: integer(value.warning_count),
  };
}

function normalizeHistoryArtifact(value: unknown): NonNullable<AnalysisHistoryDetail["artifact"]> | null {
  if (!isRecord(value)) return null;
  const output: NonNullable<AnalysisHistoryDetail["artifact"]> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!["response", "analysis_result", "report_snapshot", "lot_analysis"].includes(key)) output[key] = item;
  }
  const response = parseRelationshipResponse(value.response);
  const analysisResult = normalizeAnalysisResult(value.analysis_result);
  const reportSnapshot = normalizeReportResponse(value.report_snapshot);
  const lotAnalysis = normalizeLotAnalysis(value.lot_analysis);
  if (response) output.response = response;
  if (analysisResult) output.analysis_result = analysisResult;
  if (reportSnapshot) output.report_snapshot = reportSnapshot;
  if (lotAnalysis) output.lot_analysis = lotAnalysis;
  return output;
}

export function normalizeAnalysisHistoryDetail(
  value: unknown,
  expectedAnalysisId?: string,
): AnalysisHistoryDetail {
  if (!isRecord(value)) throw new Error("분석 이력 상세 응답 형식이 올바르지 않습니다.");
  const metadata = normalizeHistorySummary(value.metadata, expectedAnalysisId);
  if (!metadata) throw new Error("분석 이력 상세 응답에 필수 metadata가 없습니다.");
  return {
    metadata,
    artifact: value.artifact === null ? null : normalizeHistoryArtifact(value.artifact),
    source_prediction_deleted: boolean(value.source_prediction_deleted),
    linked_analysis_count: integer(value.linked_analysis_count),
  };
}
