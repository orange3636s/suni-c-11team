export type ModelMetrics = {
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  mse?: number | null;
};

export type TrainingJobState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export type TrainingJobResult = {
  model_id: string;
  target: string;
  best_model: string;
  test_metrics: ModelMetrics | null;
  feature_count: number;
  warning_count: number;
};

export type TrainingJobCreateResponse = {
  job_id: string;
  status: "queued";
};

export type TrainingJobStatusResponse = {
  job_id: string;
  status: TrainingJobState;
  stage: string;
  progress: number;
  elapsed_seconds: number;
  result: TrainingJobResult | null;
  error: string | null;
};

export type DatasetSummary = {
  dataset_id: string;
  kind: "bundled" | "uploaded";
  original_filename: string;
  uploaded_at: string | null;
  row_count: number;
  column_count: number;
  lot_min: string | null;
  lot_max: string | null;
  lot_count: number | null;
  warnings: string[];
  unmapped_columns: string[];
  deletable: boolean;
};

export type DatasetListResponse = { items: DatasetSummary[] };

export type DatasetUploadResponse = {
  success: boolean;
  dataset_id: string | null;
  blocking_errors: string[];
  warnings: string[];
  unmapped_columns: string[];
  row_count?: number | null;
  column_count?: number | null;
  lot_min?: string | null;
  lot_max?: string | null;
  lot_count?: number | null;
  // Y 계열 감지 여부 -- true면 "학습에 사용하려면 모델 학습·
  // 자동화에서 실행하세요" 안내를 띄운다(자동 학습은 절대 걸지 않는다).
  has_target_columns?: boolean;
  target_status?: TargetStatus | null;
};

export type TargetStatus = {
  state: "missing_columns" | "all_missing" | "partial" | "complete";
  present_columns: string[];
  missing_columns: string[];
  valid_cell_count: number;
  total_cell_count: number;
  message: string;
};

export type TargetProvenance = {
  dataset_id: string;
  dataset_version: string;
  hydration_version: string;
  model_id: string | null;
  model_version: string | null;
  predicted_at: string | null;
  measured_rows: number;
  predicted_rows: number;
  mixed_rows: number;
  measured_target_cells: number;
  predicted_target_cells: number;
  derived_y_rows: number;
  uses_predictions: boolean;
  cache_hit: boolean;
  warnings: string[];
  warning_counts: Record<string, number>;
  source_status: TargetStatus;
  feature_coverage: Record<string, unknown>;
};

export type DatasetSchemaResponse = {
  dataset_id: string;
  steps_present: number[];
  config_steps: number[];
  max_step: number;
  r_columns: string[];
  d_columns: string[];
  config_columns: string[];
  target_columns: string[];
  unmapped_columns: string[];
  missing_rates: Record<string, number>;
  r_measurement_rate: number | null;
  d_measurement_rate: number | null;
  target_status: TargetStatus;
};

export type RelationShape = "monotonic_increasing" | "monotonic_decreasing" | "u_shape" | "unclear";

export type ScatterPoint = {
  x: number;
  y: number;
  lot_wafer_id: string | null;
  lot_id: string | null;
  in_range: boolean;
  config: string | null;
};

export type NormalRange = {
  lo: number | null;
  hi: number | null;
  one_sided: boolean;
  fallback_applied: boolean;
};

export type ReferenceLineKey =
  | "mean"
  | "q1"
  | "q3"
  | "iqr_lo"
  | "iqr_hi"
  | "s3_lo"
  | "s3_hi"
  | "s6_lo"
  | "s6_hi";

export type ReferenceLine = {
  key: ReferenceLineKey;
  value: number;
  drawable: boolean;
  alarm_relevant: boolean;
  formula: string;
  outside_count: number;
};

export type BinProfile = {
  x_mean: number;
  y_mean: number;
  y_lo: number;
  y_hi: number;
  n: number;
  x_lo: number;
  x_hi: number;
  bin_span_ratio: number;
  /** Outlier-widened bin (its own [x_lo, x_hi] spans an outsized share of
   * the factor's overall range) -- rendered as a dashed curve segment. */
  sparse: boolean;
};

/** One method's (SPC or ML) 권장구간 fit + the F2 x 안정성 score it was
 * judged on, behind the "SPC / ML 방식 전환" toggle.
 * `window`/`optimal_center` drive
 * the chart band when this method is selected; the rest backs the
 * comparison card's numbers only. */
export type MethodWindow = {
  window: [number, number];
  optimal_center: number;
  recall: number;
  precision: number;
  f2: number;
  width_sd: number;
  stability: number;
  score: number;
  /** Whether the control range actually cut into this method's raw
   * (unclamped) window -- drives the "(관리한계에 맞춰 조정됨)" tooltip. */
  clamped: boolean;
};

export type WindowMethod = "spc" | "ml";

export type MethodComparison = {
  spc: MethodWindow | null;
  ml: MethodWindow | null;
  /** Which method's window backs the alarm log / 개선 권장 목록 everywhere
   * else in the app -- the SPC/ML toggle only changes what's *displayed*
   * on this chart, never this. */
  adopted: WindowMethod;
  adopted_reason: string;
};

export type ScreeningScatterResponse = {
  points: ScatterPoint[];
  reference_lines: ReferenceLine[];
  normal_range: NormalRange;
  bins: BinProfile[];
  optimal_center: number | null;
  /** Set only when a classified optimal center existed but was dropped
   * (fell outside its own recommended window, or was picked from a
   * sparse bin) -- shown as the disabled-toggle tooltip reason instead
   * of the generic "단조 관계라..." message. */
  optimal_center_dropped_reason: string | null;
  /** 설명력 -- 히트맵 셀/파레토 막대가 같은 (인자, 타깃) 쌍에 대해
   * 보여주는 것과 정확히 같은 Adjusted R² 값이다. */
  adj_r2: number;
  /** adj_r2를 낸 다항 적합 차수(1 | 2). Config 인자는 null. */
  degree: number | null;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  /** 하한(30) 이상이지만 종류별(R/D) 정상 판정 임계 미만 -- 배제
   * 대신 confidence_tier를 이미 한 단계 낮춘 상태로 내려온다. 프론트는
   * 이 값으로 "표본 부족" 배지만 덧붙이면 된다. */
  under_sampled: boolean;
  relation_shape: RelationShape;
  n: number;
  axis: { x_label: string; y_label: string };
  /** null for Config factors (no numeric x to fit either method on). */
  methods: MethodComparison | null;
  target_provenance: TargetProvenance | null;
};

export type CategoricalGroup = {
  category: string;
  n: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  values: number[];
};

export type CategoricalScatterResponse = {
  groups: CategoricalGroup[];
  adj_r2: number;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  n: number;
  axis: { x_label: string; y_label: string };
  target_provenance: TargetProvenance | null;
};

// -- 알림 연동 -----------------------------------------------------------

export type NotificationGrade = "심각" | "위험" | "주의";

export type SlackChannelSummary = {
  connected: boolean;
  target: string | null;
  webhook_masked: string | null;
  verified_at: string | null;
};

export type TelegramChannelSummary = {
  connected: boolean;
  target: string | null;
  chat_id_masked: string | null;
  verified_at: string | null;
};

export type GmailChannelSummary = {
  connected: boolean;
  pending: boolean;
  email: string | null;
  verified_at: string | null;
};

// 발송 조건은 등급뿐이다 -- 자동 발송은 Refresh Time 주기 자동화가
// 수행하므로 사용자가 발송 시각을 고르는 설정은 없다.
export type NotificationConditions = {
  grades: NotificationGrade[];
};

// "알림·자동화 설정" 팝업의 자동화 섹션. 비밀번호 필드는 없다 --
// 환경변수(DB_PASSWORD)로만 받는다.
export type AutomationSettingsSummary = {
  enabled: boolean;
  sql_host: string;
  sql_port: string;
  sql_db: string;
  sql_user: string;
  refresh_interval_minutes: number;
  last_run_at: string | null;
  last_run_status: string | null;
  last_run_sent_count: number | null;
};

export type NotificationSettingsSummary = {
  slack: SlackChannelSummary;
  telegram: TelegramChannelSummary;
  gmail: GmailChannelSummary;
  conditions: NotificationConditions;
  automation: AutomationSettingsSummary;
  // 텔레그램 봇 username -- 백엔드가 단일 소스다. 프런트는 자체
  // 환경변수나 하드코딩 폴백을 두지 않는다. 미설정이면 null.
  telegram_bot_username: string | null;
};

export type AutomationSaveRequest = {
  enabled: boolean;
  sql_host: string;
  sql_port: string;
  sql_db: string;
  sql_user: string;
  refresh_interval_minutes: number;
};

export type AutomationTestResponse = { ok: boolean; error: string | null };

// 알림 기록 -- 발송/건너뜀 이력과 발송 당시의 메시지 전문(재계산
// 없이 그대로 보관).
export type NotifyHistoryItem = {
  id: number;
  sent_at: string;
  trigger: string;
  channels: string[];
  dataset_label: string | null;
  model_version: string | null;
  status: "sent" | "skipped";
  skip_reason: string | null;
  message_text: string | null;
  sent_count: number;
};

export type NotifyHistoryListResponse = {
  items: NotifyHistoryItem[];
};

export type SendTestResponse = { ok: boolean; error: string | null };

export type DispatchResponse = {
  skipped: boolean;
  reason: string | null;
  sent_count: number | null;
  results: Record<string, { ok: boolean; error: string | null }> | null;
};

// -- 수율 예측 ------------------------------------------------------------
// y(=100 − Σ Y1~Y5) 오름차순 상위 N건. 정렬 기준은 y 하나뿐이다.

// y1~y5 셀 하나의 색상 메타데이터. direction은 방향(악화/개선),
// shade는 예측 근거(파레토 기여율) 농도 -- 실측값은 shade="measured"로
// 내려와 무채색으로 렌더한다("하지 말 것: 실측값 셀에 색을 입히지 마라").
export type AlertCellColor = {
  direction: "red" | "blue" | null;
  shade: "dark" | "medium" | "light" | "gray" | "measured";
  feature: string | null;
  contribution_pct: number | null;
  factor_value: number | null;
  optimal_center: number | null;
};

// 웨이퍼·타깃별로 실제 쓰인(폴백 포함) 핵심 인자. rank_used가
// 1보다 크면 폴백이 일어났다는 뜻이고, contribution_pct는 그 폴백된
// 인자 자신의 기여율이다(1위보다 낮게 표시돼 근거 강도가 바로 드러난다).
// measured가 false면 1~5위가 전부 미계측이라는 뜻이다 -- feature/
// contribution_pct는 이때도 1위 인자를 참고용으로 채운다.
export type YieldCoreFactorCell = {
  feature: string | null;
  contribution_pct: number | null;
  rank_used: number | null;
  factor_value: number | null;
  measured: boolean;
};

// n/5 신뢰도와 툴팁용 계측/미계측 타깃 상세.
export type YieldReliabilityDetailItem = { target: string; feature: string };

export type YieldReliabilityInfo = {
  count: number;
  measured: YieldReliabilityDetailItem[];
  unmeasured: YieldReliabilityDetailItem[];
};

// 두 갈래(구간 조정/계측 추가)로 조립된 권장사항 문장.
export type YieldRecommendation = {
  text: string;
  adjustable_targets: string[];
  measurement_gap_targets: string[];
};

export type YieldCandidate = {
  lot_wafer_id: string;
  lot_id: string | null;
  y: number;
  y_components: Record<string, number>;
  cells: Record<string, AlertCellColor>;
  core_factors: Record<string, YieldCoreFactorCell>;
  reliability: YieldReliabilityInfo;
  recommendation: YieldRecommendation;
};

// 폴백 순위 분포("58%가 전부 미계측이다" 같은 통계). rank_counts의
// 키는 "1".."5"(JSON은 정수 키를 지원하지 않는다).
export type YieldFallbackSummary = {
  rank_counts: Record<string, number>;
  none_count: number;
  total_combinations: number;
};

export type YieldPredictionResponse = {
  train_dataset_id: string;
  eval_dataset_id: string;
  total_wafers: number;
  candidates: YieldCandidate[];
  unmeasured_wafer_ids: string[];
  unmeasured_count: number;
  fallback_summary: YieldFallbackSummary;
  target_provenance: TargetProvenance | null;
  // "모델 분석" 스냅샷 캐시에서 왔으면 그 analysis_id, 즉석
  // 계산이면 null -- 모니터링/원인분석과 같은 분석 회차를 보고 있는지
  // 프런트가 구분할 수 있게 한다.
  analysis_id: string | null;
};

export type ConfidenceTier = "strong" | "moderate" | "weak" | "reference";

export type ParetoRankingItem = {
  feature: string;
  kind: string;
  step: number;
  adj_r2: number;
  /** 다항 적합 차수(1 | 2). Config 인자는 null. */
  degree: number | null;
  p_value: number;
  q_value: number;
  significant: boolean;
  confidence_tier: ConfidenceTier;
  /** 표본 부족(하한 30 이상, 종류별 정상 판정 임계 미만) 배지용. */
  under_sampled: boolean;
  n_observed: number;
  contribution_pct: number;
  cumulative_pct: number;
};

// 20,000행을 넘는 데이터셋은 인자 순위/히트맵/트리맵 계산에
// 로트 단위 표본을 쓴다 -- null이면 전량 기준, 아니면 화면에 "N행 중
// M행 표본" 고지를 띄운다(숫자를 조용히 표본으로 바꾸고 말 안 하는 걸
// 막기 위해 반드시 표시한다, SampleNotice 컴포넌트).
export type SampleInfo = {
  is_sampled: boolean;
  original_rows: number;
  sampled_rows: number;
  lot_count: number | null;
  seed: number;
};

export type ParetoRankingResponse = {
  dataset_id: string;
  target: string;
  total_factor_count: number;
  n80: number | null;
  fdr_pass_count: number;
  effect_size_pass_count: number;
  max_adj_r2: number | null;
  items: ParetoRankingItem[];
  analyzable_target_samples?: number;
  model_available?: boolean;
  factor_measurement_insufficient?: boolean;
  target_provenance: TargetProvenance | null;
  sample_info?: SampleInfo | null;
};

// -- 모니터링 홈 블록③(데이터 한계) 원천 ---------------------------------
// 백엔드가 스냅샷에 담아 보내는 값을 프런트는 표시만 한다.

// (타깃, 인자)별 전체 계측률 vs 최악 10% wafer 계측률·배수.
export type MnarRateRow = {
  target: string;
  feature: string;
  overall_rate_pct: number;
  worst_decile_rate_pct: number;
  ratio: number;
};

// 랏 간/랏 내 분산 분해 + 무효과 기대값(1/랏당wafer수).
export type VarianceDecomposition = {
  lot_count: number;
  wafers_per_lot: number;
  between_lot_pct: number;
  within_lot_pct: number;
  no_effect_expected_pct: number;
  icc: number;
};

// 블록⑤ 「핵심 인자 커버리지」 -- wafer 한 장이 핵심 인자 몇 개로
// 판정되는가. eval(지금 분석 중인 배치, 보통 test_remove_y.CSV) 기준이다
// -- train과 달리 "지금 배치의 계측 상태"를 묻는 질문이라서다.
export type CoreFactorCoverageRow = {
  measured_count: number;
  wafer_count: number;
  pct: number;
};

export type CoreFactorCoverage = {
  dataset_id: string;
  core_features: string[];
  total_wafers: number;
  rows: CoreFactorCoverageRow[];
};

// 블록⑥ 「불량 원인 독립성」 -- 두 원인이 동시에 상위 10%인 wafer 비율
// (%). 독립이면 기댓값 1.00%. train.CSV 기준. matrix[i][j]는
// targets[i]/targets[j] 쌍의 값이고 대각선은 null.
export type DefectCooccurrence = {
  targets: string[];
  matrix: (number | null)[][];
};

// 모니터링 홈 블록③~⑥(데이터 한계 진단)이 유일한 소비처. 자동 갱신이
// 아직 한 번도 돌지 않았거나 계산이 실패하면 이 페이로드 자체가
// null로 내려온다 -- 화면이 그 상태를 별도로 안내한다.
export type FmeaTablePayload = {
  dataset_id: string;
  train_total_wafers: number;
  mnar_rate_report: MnarRateRow[];
  variance_decomposition: VarianceDecomposition | null;
  core_factor_coverage: CoreFactorCoverage | null;
  defect_cooccurrence: DefectCooccurrence | null;
  target_provenance: TargetProvenance | null;
};

// 모니터링 홈 블록①(조치 우선순위)·블록②(조치 가능 범위)의
// 공통 원천 -- 항상 train.CSV 기준이다. 한 행이 (타깃,
// 인자) 하나를 나타내며, 두 블록이 같은 행 배열을 서로 다른 필드로
// 나눠 그린다(회수 폭·비중·기대 회수는 블록①, 계측 카운트는 블록②).
export type ActionPriorityRow = {
  target: string;
  feature: string;
  relation_shape: RelationShape;
  factor_value: number | null;
  range_lo: number | null;
  range_hi: number | null;
  measured_count: number;
  out_of_range_count: number;
  total_wafers: number;
  recovery_width_pp: number | null; // 회수 폭 = 구간 밖 평균 손실 − 구간 안 평균 손실
  share_pct: number; // 비중 = 이 타깃의 평균 손실 / 5개 타깃 평균 손실 합계
  expected_recovery_pp: number | null; // 기대 회수 = 회수 폭 × 비중
  contribution_pct: number;
  dimmed: boolean; // 기대 회수 < 0.1%p
  dim_reason: string | null;
};

export type ActionPriorityNoQualifyingTarget = {
  target: string;
  max_contribution_pct: number;
};

// 모니터링 홈 블록③(데이터 한계) 하단 -- 랏 간/랏 내로 쪼갠 분산을
// 불량모드(Y1~Y5)로 한 번 더 쪼갠 값. cov(Y_i, L)/var(L)로 정의해 합이
// 정확히 100%가 된다. _action_priority_payload가 내려보내므로 항상
// train.CSV 기준(블록③의 나머지는 eval 기준인 것과 다르다).
export type ModeVarianceShareRow = {
  target: string;
  mean_loss_pp: number;
  mean_share_pct: number;
  variance_share_pct: number;
};

export type ActionPriorityPayload = {
  dataset_id: string;
  total_wafers: number;
  estimated_additional_action_wafers: number; // 캡션의 "계측 +10%p 시 추가 조치 대상" 추정치
  no_qualifying_factor: ActionPriorityNoQualifyingTarget[];
  rows: ActionPriorityRow[]; // 기대 회수 내림차순
  mode_variance_share: ModeVarianceShareRow[] | null; // 변동 기여 내림차순
  target_provenance: TargetProvenance | null;
};

// 히트맵 보기는 numeric 하나뿐이다(Config는 히트맵으로 그리지 않는다).
export type HeatmapKind = "numeric";

export type HeatmapResponse = {
  dataset_id: string;
  /** 항상 "adj_r2" -- 지표 토글은 없다. */
  metric: "adj_r2";
  kind: HeatmapKind;
  features: string[];
  targets: string[];
  /** 셀 농도/표시 숫자 -- Adjusted R²(설명력, 부호 없음). 산점도의 R²,
   * 파레토 막대와 같은 값이다. */
  values: Array<Array<number | null>>;
  /** 셀별 다항 적합 차수(1 | 2). 표본 부족 셀은 null. */
  degree: Array<Array<number | null>>;
  /** 셀별 관계 형태 -- 방향은 부호 있는 상관 대신 이 값과
   * optimal_center로만 전달한다(핵심 인자 다수가 U자라 전체구간 부호는
   * 표본 절반에 대해 반대로 읽힌다). */
  shape: Array<Array<RelationShape | null>>;
  /** U자/2차 셀의 꼭짓점 x값 -- 그 외에는 null. */
  optimal_center: Array<Array<number | null>>;
  n: number[][];
  q: Array<Array<number | null>>;
  significant: boolean[][];
  tier: Array<Array<ConfidenceTier | null>>;
  /** 상관계수는 그려지지만(n>=30) 종류별 표본 게이트 미달로 유의 인자
   * 목록에는 없는 셀 -- 히트맵과 유의 인자 판정이 어긋나는 지점을
   * 표시한다. 캐시된 응답에는 이 키가 없을 수 있어 항상 옵셔널로 읽는다. */
  gate_excluded?: boolean[][];
  scale: { min: number; max: number };
  excluded_configs: number;
  /** 셀 툴팁 "계측률 %"의 분모(= 이 그리드를 센 프레임의 행 수). 그리드의
   * 최대 n으로 대신할 수 없다 -- R 인자는 전체의 15%만 계측되므로 최대
   * n을 분모로 쓰면 계측률이 항상 100%로 나온다. */
  total_rows?: number;
  target_provenance: TargetProvenance | null;
  sample_info?: SampleInfo | null;
};

export type TargetPerformance = {
  target: string;
  no_factor_available: boolean;
  feature: string | null;
  kind: string | null;
  adj_r2: number | null;
  degree: number | null;
  contribution_pct: number | null;
  relation_shape: RelationShape | null;
  optimal_center: number | null;
  p_value: number | null;
  confidence_tier: ConfidenceTier | null;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  n: number | null;
  /** 아래 셋은 전체 Y(final_yield)에만 채워진다 -- 타깃별 항목은 null.
   * `adj_r2`(인자 스크리닝 효과 크기)와 `r2_adjusted`(모델 회귀 성능의
   * 자유도 보정 R²)는 이름만 비슷할 뿐 완전히 다른 수다. */
  mse: number | null;
  r2_adjusted: number | null;
  auc: number | null;
};

export type ModelPerformanceResponse = {
  model_id: string | null;
  trained_at: string | null;
  source_filename: string | null;
  targets: TargetPerformance[];
  final_yield: TargetPerformance | null;
  row_count: number | null;
  feature_count: number | null;
};

// 승격 여부와 무관한 학습 시도 이력.
export type PromotionEvent = {
  created_at: string;
  candidate_model_id: string;
  promoted: number; // SQLite에서 0/1로 내려온다.
  reason: string;
  candidate_metric: number | null;
  active_metric: number | null;
  previous_model_id: string | null;
};

export type PromotionHistoryResponse = {
  items: PromotionEvent[];
};

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Server-persisted "latest result" for each of the 3 long-running tabs.
// `points` is never present on a restored ScreeningScatterResponse --
// see GET /api/state/latest's docstring (src/runtime/app_state.py) --
// so a restored entry is always `points: []` until the page independently
// refetches it.

// 모델 학습 팝업의 SQL 연결 정보(비밀번호 제외)·Refresh 주기를
// 서버에 저장한다 -- 최근 학습 결과(performance)와 같은 레코드에 함께
// 실어 GET /api/state/latest 한 번으로 둘 다 복원한다.
export type LatestTrainingPayload = {
  performance: ModelPerformanceResponse;
  sqlHost?: string;
  sqlPort?: string;
  sqlDb?: string;
  sqlUser?: string;
  refreshIntervalMinutes?: number | null;
};

export type LatestTrainingRecord = {
  schema_version: number;
  created_at: string;
  dataset: string;
  payload: LatestTrainingPayload;
};

// Deliberately narrow -- the three saved records together must stay under
// 100KB. Per-factor
// scatter/categorical detail (관리한계·권장구간·최적중심 등) is NOT part of
// what gets persisted, even with points stripped: 25 factors' worth of
// reference lines/bins/methods comparison alone measured ~105KB against
// train.CSV, well over budget on its own. `paretoByTarget` is what the
// screening table/Pareto bars render from and is cheap (a few KB); the
// scatter detail is refetched via the same per-factor calls a live run
// already makes (frontend's fetchAllScatterData), same as points always were.
export type LatestAnalysisPayload = {
  activeTarget: string;
  paretoByTarget: Record<string, ParetoRankingResponse>;
  targetProvenance?: TargetProvenance | null;
  // 프런트는 이 필드를 절대 채워 보내지 않는다 -- 서버가
  // POST /api/state/analysis 저장 시점에 채운다(api/routes/state.py의
  // `_with_fmea`). 그래서 요청 바디에는 나타나지 않고, hydrate() 쪽에서는
  // (FMEA가 채워지기 전에 저장된 레코드가 아닌 한) 항상 값이 있다.
  fmea?: FmeaTablePayload | null;
  fmeaError?: string | null;
  // 모니터링 홈 블록①·② -- fmea와 같은 방식(서버가 저장 시점에
  // 채운다, `_with_action_priority`)이지만 항상 train.CSV 기준이라 eval
  // 데이터셋과 무관하다.
  actionPriority?: ActionPriorityPayload | null;
  actionPriorityError?: string | null;
  // 저장된 스냅샷의 응답 형태·내용 규칙이 여전히 유효한지 프론트가 직접
  // 검사하는 버전 -- frontend/lib/snapshotVersion.ts의
  // ANALYSIS_SNAPSHOT_VERSION과 다르면 복원하지 않는다. 백엔드 봉투의
  // schema_version(app_state.py)과는 별개다 -- 그쪽이 필터링해 버리면
  // 프론트는 "저장된 적 없음"과 "낡아서 못 쓴다"를 구분할 수 없어 조용히
  // 빈 화면이 된다.
  snapshotVersion: number;
};

export type LatestAnalysisRecord = {
  schema_version: number;
  created_at: string;
  dataset: string;
  payload: LatestAnalysisPayload;
};

// 이 레코드에서 의미가 있는 것은 train/eval 데이터셋과 시각뿐이다
// (모니터링 홈의 "판정 결과와 조회 중인 데이터셋이 다르다" 경고에 쓰임).
// payload는 레코드마다 내용이 다를 수 있어 형태를 고정하지 않는다.
export type LatestAlarmsRecord = {
  schema_version: number;
  created_at: string;
  train_dataset: string;
  eval_dataset: string;
  payload: Record<string, unknown>;
};

export type LatestStateResponse = {
  training: LatestTrainingRecord | null;
  analysis: LatestAnalysisRecord | null;
  alarms: LatestAlarmsRecord | null;
  notifications: NotificationSettingsSummary;
  // 저장된 레코드 중 하나 이상이 더 이상 존재하지 않는 데이터셋을 가리켜
  // 서버가 통째로 버렸으면 true (dataset을 "train"으로 바꿔치기하지
  // 않는다 -- 저장된 payload가 잘못된 라벨을 달고 뜨는 게 더 나쁘다).
  // 프론트는 이 신호로만 재실행 안내를 띄운다.
  dataset_fallback_applied: boolean;
  // 복원 자체(GET /api/state/latest)가 실패했다는 신호(DB 손상 등) --
  // "저장된 결과 없음"과 구분해야 사용자가 결과가 사라진 줄 알고
  // 재분석을 다시 돌리지 않는다.
  degraded: boolean;
};

// -- 자동 갱신 파이프라인 스냅샷 -----------------------------------------

export type RefreshSnapshotSource = {
  // "manual"은 원인 분석·수율 예측에서 업로드해 활성화한 평가
  // 데이터셋 -- "자동 갱신으로 복귀"를 누르기 전까지 주기 잡도 이 값을
  // 그대로 쓴다.
  mode: "sql" | "fallback" | "manual";
  train_dataset: string;
  eval_dataset: string;
  eval_dataset_filename: string | null;
  row_count: number;
};

export type RefreshSnapshotModel = {
  champion_version: string | null;
  trained_at: string | null;
  promoted: boolean | null;
  skipped_reason: string | null;
  training_job_submitted?: string;
};

export type RefreshSnapshotMonitoring = {
  predicted_yield: { point: number; lo: number; hi: number } | null;
  gap: { lo: number; hi: number } | null;
  gap_pareto: Array<Record<string, unknown>>;
  treemap: { step: number; cells: unknown[] } | null;
};

export type RefreshSnapshot = {
  schema_version: number;
  created_at: string;
  // 모니터링·원인분석·알림기록 3종이 같은 계산 결과에서 나왔음을
  // 확인하는 공유 id -- created_at을 그대로 쓴다(스냅샷 자체가 이미
  // 원자적으로 저장되므로 별도 채번이 필요 없다).
  analysis_id: string;
  source: RefreshSnapshotSource;
  model: RefreshSnapshotModel;
  analysis: {
    paretoByTarget: Record<string, unknown>;
    fmea: FmeaTablePayload | null;
    fmeaError: string | null;
    actionPriority: ActionPriorityPayload | null;
    actionPriorityError: string | null;
    target_provenance: TargetProvenance | null;
    // 이 회차의 수율 예측 표 캐시 -- GET /api/alerts/ranking과
    // 같은 모양(YieldPredictionResponse에서 train/eval id·analysis_id만
    // 뺀 것)이다.
    yieldPrediction: Omit<YieldPredictionResponse, "train_dataset_id" | "eval_dataset_id" | "analysis_id"> | null;
  };
  monitoring: RefreshSnapshotMonitoring;
  errors: string[];
};

export type SnapshotResponse = {
  snapshot: RefreshSnapshot | null;
  stale_version: boolean;
  stale_model: boolean;
};

// 첫 기동 부트스트랩(스냅샷이 아직 없을 때 1회 학습+분석) 진행 상태.
// 부트스트랩이 한 번도 시작되지 않았으면 null.
export type BootstrapStatus = {
  status: "running" | "done" | "failed";
  // 큰 단계 이름만 온다(예: "학습 중") -- 실제로 측정할 수 없는 세부
  // 진행률(0~99%)을 지어내지 않는다. 알 수 없으면 null이고, 화면은
  // "첫 분석 진행 중"만 보여준다.
  stage: string | null;
  error: string | null;
  // status가 "failed"일 때만 의미가 있다 -- "bundled_train_data_missing"
  // 이면 재시도해도 소용없는 진짜 복구 불가능 케이스(내장 학습 데이터
  // 자체가 없음)다. 그 외(null)는 일시적 실패로 취급한다 -- 다음 런타임
  // 요청이 자동으로 재시도한다(api/main.py::ensure_usable_champion).
  // 저장된 레코드에 이 키가 아예 없을 수 있어 옵셔널이다.
  reason?: string | null;
  updated_at: string;
};

// 업로드로 활성화된 수동 평가 데이터셋 -- 있으면 헤더/화면에
// "수동 · {filename}" 배지와 "자동 갱신으로 복귀" 버튼을 보여준다.
export type ManualEvalOverride = {
  dataset_id: string;
  filename: string;
  set_at: string;
};

// 네 화면(모니터링/Config별 트리맵/원인분석/수율예측)이 공유하는
// "분석 시작" 진행 표시 -- 실행 중이 아니면 null.
export type AnalysisProgress = {
  stage: string;
  index: number;
  total: number;
  analysis_id: string;
  // 단계 전환마다(그리고 오래 걸리는 단계 안에서는 주기적으로) 갱신된다
  // -- 프런트는 이 값이 오래되면(60초) 서버 프로세스가 죽었다고 판단해
  // "중단되었습니다" 배너로 전환한다. 응답에 이 필드가 없을 수 있어
  // 옵셔널로 둔다.
  heartbeat_at?: string | null;
  // 배너에 "· 100,000행 · 약 2분 예상"을 붙이는 데 쓴다.
  // 대략적인 추정치이지 정밀한 SLA가 아니다.
  row_count?: number | null;
  estimated_seconds?: number | null;
};

// "분석 시작"의 최근 실행 결과 -- `analysis_progress`(위)와 달리 실행이
// 끝난 뒤에도 남는다.
// 백그라운드 실행이 조용히 실패했을 때(`triggered: true`는 받았는데
// 스냅샷이 안 생기는 상태) 화면이 원인을 보여줄 수 있게 한다.
export type LastRunStatus = {
  status: "running" | "succeeded" | "failed";
  analysis_id: string | null;
  failed_stage: string | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type SnapshotMetaResponse = {
  created_at: string | null;
  bootstrap: BootstrapStatus | null;
  // "모델 분석" 팝업의 [분석 시작] 버튼이든 서버 기동 부트스트랩/
  // 학습 후 자동 복구든, 파이프라인이 지금 실행 중이면 true.
  refresh_running: boolean;
  analysis_progress: AnalysisProgress | null;
  manual_eval_override: ManualEvalOverride | null;
  // 실행이 한 번도 없었으면 null.
  last_run: LastRunStatus | null;
};

// -- 즐겨찾기 -------------------------------------------------------------
// 점 데이터는 절대 담지 않는다 -- 열 때 이 파라미터로 API를 다시 불러
// 렌더한다.
export type FavoriteViewType = "scatter" | "box";

export type FavoriteSnapshot = {
  dataset: string;
  target: string;
  feature: string;
  viewType: FavoriteViewType;
  colorBy?: string;
  method?: string | null;
  isConfig: boolean;
  // 해석 문구는 저장 시점 값을 문자열로 그대로 보관한다 -- 카드를
  // 열 때마다 재계산하지 않는다(목록이 느려진다). 해당 인자가 "보통"
  // 등급이 아니어서 해석 문구 자체가 없었으면 빈 문자열.
  interpretation: string;
  // 저장 시점의 활성 모델 id(챔피언) -- 이후 재학습/재승격으로 현재
  // 챔피언과 달라지면 카드에 "이전 분석 기준" 배지를 붙인다. 학습된
  // 모델이 아예 없던 시점에 저장됐으면 null.
  championVersion: string | null;
};

export type FavoriteRecord = {
  favorite_id: string;
  created_at: string;
  snapshot: FavoriteSnapshot;
};

export type FavoriteListResponse = {
  items: FavoriteRecord[];
};

// 모니터링 홈 트리맵 -- Model/EQ/Chamber 분해는 서버의 공통 Config
// 파서가 담당한다. 면적은 n, 색은 선택한 불량률 평균이다.
export type ConfigTreemapGroup = {
  config: string;
  model: string;
  equipment: string;
  chamber: string;
  n: number;
  mean: number;
  median: number;
  p5: number;
  p95: number;
};

export type ConfigTreemapResponse = {
  dataset_id: string;
  step: number;
  target: "Y" | "Y1" | "Y2" | "Y3" | "Y4" | "Y5";
  target_label: string;
  deprecated_target: boolean;
  overall_mean: number;
  groups: ConfigTreemapGroup[];
  // 이 스텝 Config가 최종 수율과의 Adjusted R²(Config는 더미 회귀 기반)
  // + BH-FDR 게이트를 통과했는지 -- 통과 못 하면 타일 채색을 끈다
  // (ConfigTreemap.tsx).
  significant: boolean;
  empty_reason: string | null;
  target_provenance: TargetProvenance | null;
  // "모델 분석"이 마지막으로 저장한 스냅샷의 analysis_id --
  // 모니터링/원인분석/수율예측과 같은 값이면 같은 분석 회차다. 스냅샷이
  // 없으면 null.
  analysis_id: string | null;
  sample_info?: SampleInfo | null;
};

