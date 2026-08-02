import type { ProcessingSummary } from "@/types/data";

type PreprocessingSummaryProps = { summary: ProcessingSummary | null | undefined };

function show(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "사용" : "미사용";
  if (typeof value === "number") return `${(value * 100).toFixed(2)}%`;
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function missingLabel(value: ProcessingSummary["missing_strategy"]): string {
  if (value === "native") return "Native Missing 적용";
  if (value === "median") return "Median Imputation 적용";
  if (value === "model_specific") return "모델별 Native/Median 적용";
  return value ? `결측 처리: ${value}` : "결측 처리 정보 없음";
}

export default function PreprocessingSummary({ summary }: PreprocessingSummaryProps) {
  if (!summary) return null;
  if (summary.pipeline_version === "auto_multi_y_hgbr_v1") {
    const removedColumns = [
      ...(summary.removed_all_missing_columns ?? []),
      ...(summary.removed_constant_columns ?? []),
      ...(summary.removed_near_constant_columns ?? []),
    ];
    const items = [
      ["숫자형 Feature", `${summary.numeric_feature_count ?? 0}개`],
      ["범주형 Config", `${summary.categorical_config_count ?? 0}개`],
      ["제거된 상수·준상수 열", `${removedColumns.length}개`],
      ["결측치 대체 열", `${summary.missing_imputed_columns ?? 0}개`],
      ["이상치 완화 열", `${summary.winsorized_columns ?? 0}개`],
      ["학습 행", (summary.training_row_count ?? 0).toLocaleString()],
      ["Lot", (summary.lot_count ?? 0).toLocaleString()],
      ["분할 방식", summary.split_method ?? "자동 분할"],
      ["Pipeline", summary.pipeline_version],
    ];
    return <section className="preprocessingSummaryCard" aria-labelledby="preprocessing-summary-title"><div><span className="sectionLabel">Auto Pipeline</span><h3 id="preprocessing-summary-title">데이터 처리 요약</h3></div><ul>{items.map(([label, value]) => <li key={label}><span aria-hidden="true">✓</span><strong>{label}</strong><small>{value}</small></li>)}</ul></section>;
  }
  const coverage = summary.measurement_coverage ?? {};
  const missingStrategy = summary.missing_strategy ?? summary.missing_handling;
  const outlierStrategy = summary.outlier_strategy ?? summary.outlier_policy;
  const rawItems: Array<[string, unknown]> = [
    [missingLabel(missingStrategy), true],
    [outlierStrategy === "flag_only" ? `Outlier Indicator 생성 · 원본값 유지${summary.outlier_indicator_count != null ? ` (${summary.outlier_indicator_count}개)` : ""}` : outlierStrategy === "iqr" ? "IQR Clip 적용" : outlierStrategy === "model_specific" ? "모델별 Flag Only/IQR Clip 적용" : `이상치 처리: ${outlierStrategy ?? "정보 없음"}`, true],
    [`Missing Indicator ${summary.missing_indicator ? "생성" : "미생성"}${summary.missing_indicator_count != null ? ` (${summary.missing_indicator_count}개)` : ""}`, true],
    [`R Feature ${summary.r_column_count ?? 0}개 처리 완료`, summary.r_column_count != null],
    [`D Feature ${summary.d_column_count ?? 0}개 처리 완료`, summary.d_column_count != null],
    [`Step Feature ${summary.step_feature_count ?? 0}개 생성 완료`, summary.step_feature_count != null],
    [`Config Parsing ${summary.config_parsed === false ? "오류 확인 필요" : "완료"}${summary.config_column_count != null ? ` (${summary.config_column_count}개)` : ""}`, summary.config_parsed !== undefined],
    [summary.fallback_used ? "Fallback 적용" : "Fallback 미사용", summary.fallback_used !== undefined],
    ["R 측정 커버리지", coverage.r],
    ["D 측정 커버리지", coverage.d],
  ];
  const items = rawItems.filter(([, value]) => value !== null && value !== undefined && value !== "" && value !== false);
  if (!items.length) return null;
  return <section className="preprocessingSummaryCard" aria-labelledby="preprocessing-summary-title"><div><span className="sectionLabel">Pipeline</span><h3 id="preprocessing-summary-title">데이터 처리 요약</h3></div><ul>{items.map(([label, value]) => <li key={String(label)}><span aria-hidden="true">✓</span><strong>{label}</strong>{value !== true && !String(label).includes("커버리지") ? <small>{show(value)}</small> : String(label).includes("커버리지") ? <small>{show(value)}</small> : null}</li>)}</ul></section>;
}
