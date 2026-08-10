import type {
  AlarmListResponse,
  AlertsDataResponse,
  CategoricalScatterResponse,
  ConfigTreemapResponse,
  DatasetListResponse,
  DatasetSchemaResponse,
  DatasetUploadResponse,
  DispatchResponse,
  FavoriteListResponse,
  FavoriteRecord,
  FavoriteSnapshot,
  HeatmapKind,
  HeatmapResponse,
  LatestAlarmsPayload,
  LatestAnalysisPayload,
  LatestStateResponse,
  LatestTrainingPayload,
  MeasurementExpansionResponse,
  ModelPerformanceResponse,
  NotificationConditions,
  NotificationSettingsSummary,
  ParetoRankingResponse,
  PromotionHistoryResponse,
  ReliabilityResponse,
  ScreeningScatterResponse,
  SendTestResponse,
  SnapshotMetaResponse,
  SnapshotResponse,
  TrainingJobCreateResponse,
  TrainingJobStatusResponse,
  YieldPredictionResponse,
} from "@/types/data";

export class ApiResponseError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiResponseError";
    this.status = status;
  }
}

export class ApiTimeoutError extends Error {
  constructor(message = "요청이 시간 내에 끝나지 않았습니다.") {
    super(message);
    this.name = "ApiTimeoutError";
  }
}

export class ApiNetworkError extends Error {
  constructor(message = "서버에 연결할 수 없습니다.") {
    super(message);
    this.name = "ApiNetworkError";
  }
}

// 원인 분석 실행은 여러 타깃 x 인자 조합을 한 번에 조회해 12초 안팎이 걸릴
// 수 있다 -- 이전에는 명시적인 타임아웃이 없어 연결이 멎으면 스피너가
// 무한정 돌았다. 90초는 그 요청들이 정상적으로 끝나는 시간보다 넉넉히
// 길게 잡은 상한선이다.
const DEFAULT_TIMEOUT_MS = 90_000;
// E-3: 파일 업로드(CSV)는 대개 더 오래 걸릴 수 있어 여유를 더 둔다 --
// 여전히 상한선이 있는 것과 없는 것의 차이가 핵심이지, 정확한 값은
// 중요하지 않다(연결이 끊기면 결국 이 시간 안에 끝난다는 게 중요하다).
const UPLOAD_TIMEOUT_MS = 5 * 60_000;

// E-3: getJson/postJson에만 있던 AbortController+타임아웃을 업로드·삭제·
// 즐겨찾기 등 나머지 raw fetch 호출에도 공통으로 적용한다 -- 이게 없으면
// 연결이 멎었을 때(서버 다운, 네트워크 끊김) "업로드 중…"/"삭제 중…"이
// 무한정 떠 있는다. 에러 메시지 문구는 호출부마다 다르므로 그건 그대로
// 두고, fetch 자체에 타임아웃을 거는 부분만 공유한다.
async function timedFetch(url: string, init: RequestInit = {}, timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

// E-2: useApiStatus.ts가 이 함수 대신 자기만의 apiBaseUrl()을 따로
// 두고 있었다 -- 그쪽은 프로덕션에서 미설정이어도 조용히 127.0.0.1로
// 폴백해, 실제 사고(NEXT_PUBLIC_API_BASE_URL 미설정) 때 상태 배지는
// "연결 끊김"만 보여주고 원인(요청은 여기서 곧장 에러를 던진다)을
// 감춘다. 하나만 남기고 공유한다.
export function getApiBaseUrl(): string {
  const configuredUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim().replace(/\/$/, "");
  if (configuredUrl) {
    return configuredUrl;
  }
  if (process.env.NODE_ENV === "development") {
    return "http://127.0.0.1:8000";
  }
  throw new Error(
    "NEXT_PUBLIC_API_BASE_URL이 설정되지 않았습니다. " +
      "Vercel 프로젝트 환경변수에 Render API URL을 설정한 뒤 재배포해 주세요.",
  );
}

function rethrowApiConfigurationError(error: unknown): void {
  if (
    error instanceof Error &&
    error.message.startsWith("NEXT_PUBLIC_API_BASE_URL")
  ) {
    throw error;
  }
}

async function getErrorMessage(response: Response): Promise<string> {
  const fallback = `요청 처리에 실패했습니다. (${response.status})`;

  try {
    const body = (await response.json()) as {
      detail?: string | { message?: string; errors?: string[] };
    };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (body.detail && typeof body.detail.message === "string") {
      const errors = body.detail.errors?.join(" ");
      return errors ? `${body.detail.message} ${errors}` : body.detail.message;
    }
  } catch {
    return fallback;
  }

  return fallback;
}

export async function createTrainingJob(
  file: File,
): Promise<TrainingJobCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await timedFetch(`${getApiBaseUrl()}/api/train/jobs`, { method: "POST", body: formData }, UPLOAD_TIMEOUT_MS);
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "학습 Job을 생성할 수 없습니다. 백엔드 실행 상태를 확인해 주세요.",
    );
  }

  if (!response.ok) {
    throw new ApiResponseError(response.status, await getErrorMessage(response));
  }
  return response.json() as Promise<TrainingJobCreateResponse>;
}

export async function getTrainingJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<TrainingJobStatusResponse> {
  let response: Response;
  try {
    response = await fetch(
      `${getApiBaseUrl()}/api/train/jobs/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        cache: "no-store",
        signal,
      },
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    rethrowApiConfigurationError(error);
    throw new Error(
      "학습 상태를 확인할 수 없습니다. 잠시 후 다시 시도해 주세요.",
    );
  }

  if (!response.ok) {
    throw new ApiResponseError(response.status, await getErrorMessage(response));
  }
  return response.json() as Promise<TrainingJobStatusResponse>;
}

async function getJson<T>(path: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, { method: "GET", cache: "no-store", signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiTimeoutError();
    }
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  } finally {
    window.clearTimeout(timer);
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiTimeoutError();
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  } finally {
    window.clearTimeout(timer);
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<T>;
}

export function getDatasets(): Promise<DatasetListResponse> {
  return getJson("/api/datasets");
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  let response: Response;
  try {
    response = await timedFetch(`${getApiBaseUrl()}/api/datasets`, { method: "POST", body: formData }, UPLOAD_TIMEOUT_MS);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiTimeoutError();
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<DatasetUploadResponse>;
}

export async function deleteDataset(datasetId: string): Promise<{ success: boolean; dataset_id: string }> {
  let response: Response;
  try {
    response = await timedFetch(`${getApiBaseUrl()}/api/datasets/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiTimeoutError();
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json();
}

export function getDatasetSchema(datasetId: string): Promise<DatasetSchemaResponse> {
  return getJson(`/api/datasets/${encodeURIComponent(datasetId)}/schema`);
}

export function getScreeningScatter(dataset: string, target: string, feature: string): Promise<ScreeningScatterResponse> {
  return getJson(`/api/screening/scatter?${new URLSearchParams({ dataset, target, feature }).toString()}`);
}

export function getScreeningScatterCategorical(dataset: string, target: string, feature: string): Promise<CategoricalScatterResponse> {
  return getJson(`/api/screening/scatter/categorical?${new URLSearchParams({ dataset, target, feature }).toString()}`);
}

export function getAlarms(
  trainDataset: string,
  evalDataset: string,
  options?: { grade?: string; target?: number; sensitivity?: number },
): Promise<AlarmListResponse> {
  const params = new URLSearchParams({ train: trainDataset, eval: evalDataset });
  if (options?.grade) params.set("grade", options.grade);
  // 지시서: 원인 분석 탭의 알람 삼각형이 수율 예측에서 저장한 목표
  // 수율·민감도를 그대로 넘겨 두 화면의 판정 기준을 일치시킨다.
  // 생략하면(최초 실행 등 저장된 값이 없을 때) 백엔드 기본값을 쓴다.
  if (options?.target != null) params.set("target", String(options.target));
  if (options?.sensitivity != null) params.set("sensitivity", String(options.sensitivity));
  return getJson(`/api/alarms?${params.toString()}`);
}

// 지시서 작업 2(특정 스텝까지의 정보만으로 예측) -- maxStep을 생략하면
// 기존 동작과 완전히 같다(전체 스텝 기준).
export function getAlertsData(
  trainDataset: string,
  evalDataset: string,
  maxStep?: number | null,
): Promise<AlertsDataResponse> {
  const params = new URLSearchParams({ train: trainDataset, eval: evalDataset });
  if (maxStep != null) params.set("max_step", String(maxStep));
  return getJson(`/api/alarms/predictions?${params.toString()}`);
}

// VA~VD: y(=100 − Σ Y1~Y5) 오름차순 전체 목록(신뢰도==0 웨이퍼 제외) --
// 수율 예측 화면이 쓰는 유일한 판정 엔드포인트다(위 getAlertsData/구
// 5분류 체계는 더 이상 수율 예측에서 호출하지 않는다). 상위 10/전체
// 보기·검색·정렬은 프런트가 이 전체 목록 위에서 수행한다(VB-4: 검색
// 중에는 상위 10 제한을 해제해야 하므로 서버가 미리 자르면 안 된다).
export function getYieldPrediction(trainDataset: string, evalDataset: string): Promise<YieldPredictionResponse> {
  const params = new URLSearchParams({ train: trainDataset, eval: evalDataset });
  return getJson(`/api/alerts/ranking?${params.toString()}`);
}

export function getReliability(dataset: string, evalDataset: string): Promise<ReliabilityResponse> {
  return getJson(`/api/analysis/reliability?${new URLSearchParams({ dataset, eval: evalDataset }).toString()}`);
}

export function getModelPerformance(): Promise<ModelPerformanceResponse> {
  return getJson("/api/models/performance");
}

// 자동 수집 파이프라인 §2-2 -- 승격 여부와 무관한 학습 시도 이력. 모델
// 학습·자동화 팝업이 "게이트 미달로 교체되지 않음"을 보여주는 데 쓴다.
export function getPromotionHistory(limit = 5): Promise<PromotionHistoryResponse> {
  return getJson(`/api/models/promotion-history?${new URLSearchParams({ limit: String(limit) }).toString()}`);
}

// TC-4: metric 토글이 사라졌다 -- numeric 보기는 항상 ε²(농도)+rho(방향)를
// 함께 받는다. TC-3: categorical 보기도 계층(config_level) 선택 없이 항상
// 원본 Config 조합 그대로다.
export function getScreeningHeatmap(dataset: string, kind?: HeatmapKind): Promise<HeatmapResponse> {
  const params = new URLSearchParams({ dataset });
  if (kind) params.set("kind", kind);
  return getJson(`/api/screening/heatmap?${params.toString()}`);
}

// `topN`은 백엔드의 PARETO_TOP_N(10) 기본값을 그대로 쓴다 -- 원인 분석 탭은
// 생략해서 그 기본값을 받고, 학습 탭 스크리닝 표는 5로 고정해 넘긴다(같은
// 엔드포인트를 두 곳이 서로 다른 개수로 쓰므로 기본값에 암묵적으로 기대면
// 한쪽이 조용히 따라 바뀐다).
export function getScreeningPareto(dataset: string, target: string, topN?: number): Promise<ParetoRankingResponse> {
  const params = new URLSearchParams({ dataset, target });
  if (topN != null) params.set("top_n", String(topN));
  return getJson(`/api/screening/pareto?${params.toString()}`);
}

export function getMeasurementExpansion(dataset: string): Promise<MeasurementExpansionResponse> {
  return getJson(`/api/analysis/measurement-expansion?${new URLSearchParams({ dataset }).toString()}`);
}

export function getConfigTreemap(dataset: string, step: number, target = "Y1"): Promise<ConfigTreemapResponse> {
  return getJson(`/api/monitoring/config-treemap?${new URLSearchParams({ dataset, step: String(step), target }).toString()}`);
}

// -- 알림 연동 (설정 패널 신설 §C/§D) -----------------------------------
// state/latest가 마운트 시 1번에 알림 설정을 함께 실어 온다 (spec §D-3) --
// 이 함수들은 사용자가 설정 패널에서 실제로 무언가를 바꿀 때만 호출된다.

export function connectSlack(webhookUrl: string, channel: string | null): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/slack", { webhook_url: webhookUrl, channel });
}

// D-3: webhookUrl을 생략하면(이미 연결된 채널을 테스트할 때) 서버가
// 저장된 값을 쓴다 -- 연결 요약에는 마스킹된 값만 있어 프런트가 원본을
// 다시 보낼 수 없다.
export function testSlack(webhookUrl?: string): Promise<SendTestResponse> {
  return postJson("/api/notify/slack/test", { webhook_url: webhookUrl ?? null });
}

export function verifyTelegramCode(code: string): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/telegram/verify", { code });
}

export function testTelegram(): Promise<SendTestResponse> {
  return postJson("/api/notify/telegram/test", {});
}

export function connectGmail(email: string): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/gmail", { email });
}

export function testGmail(): Promise<SendTestResponse> {
  return postJson("/api/notify/gmail/test", {});
}

export function saveNotificationConditions(conditions: NotificationConditions): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/conditions", conditions);
}

export async function disconnectNotificationChannel(channel: "slack" | "telegram" | "gmail"): Promise<NotificationSettingsSummary> {
  let response: Response;
  try {
    response = await timedFetch(`${getApiBaseUrl()}/api/notify/${channel}`, { method: "DELETE" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiTimeoutError();
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<NotificationSettingsSummary>;
}

// 원인 분석 실행 직후 fire-and-forget으로 호출된다 (spec §C-4 "분석 실행
// 직후") -- 실패해도 분석 결과 표시를 막으면 안 되지만, 발송 실패
// 자체는 사용자에게 보여야 한다(A-1). 호출부는 실패 시 오류 배너를
// 띄우고, 이미 렌더된 분석 결과는 그대로 둔다.
export function dispatchAlarmNotifications(trainDataset: string, evalDataset: string): Promise<DispatchResponse> {
  return postJson("/api/notify/dispatch", { train_dataset: trainDataset, eval_dataset: evalDataset, dashboard_url: null });
}

// YD: 수율 예측 화면 "알림 전송" 버튼 -- 확인 다이얼로그에서 연결된
// 채널을 보여주려면 먼저 현재 설정을 읽어야 한다.
export function getNotificationSettings(): Promise<NotificationSettingsSummary> {
  return getJson("/api/notify/settings");
}

// YD: 버튼을 눌러 지금 바로 발송한다 -- `/dispatch`(옛 알람)와 별도
// 엔드포인트라 AUC 게이트·목표 수율 설정과 무관하게 동작하지만,
// 억제 규칙(신규분만·시간당 예산·최소 간격 10분)은 그대로 적용된다.
export function dispatchYieldUpdateNotification(trainDataset: string, evalDataset: string): Promise<DispatchResponse> {
  return postJson("/api/notify/yield-update/dispatch", {
    train_dataset: trainDataset,
    eval_dataset: evalDataset,
    dashboard_url: null,
  });
}

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Called once on app mount by AnalysisStateProvider (spec §4-2/§6) --
// never per tab-switch. A short timeout keeps a slow/unreachable API from
// stalling first paint; the provider treats any failure the same as "no
// saved result yet" (spec: "복원 실패가 앱을 막으면 안 된다").
export function getLatestState(): Promise<LatestStateResponse> {
  return getJson("/api/state/latest", 15_000);
}

// J-4: 자동 갱신 스냅샷. `getSnapshotMeta`는 60초 폴링·포커스 복귀 때
// 가볍게 부르는 용도(created_at만 온다) -- 그 값이 캐시보다 최신일
// 때만 `getSnapshot`으로 전체를 다시 받는다. 매 폴링마다 전체를 다시
// 받지 않는다(지시서: "같으면 아무것도 하지 않는다").
export function getSnapshotMeta(): Promise<SnapshotMetaResponse> {
  return getJson("/api/state/snapshot/meta", 10_000);
}

export function getSnapshot(): Promise<SnapshotResponse> {
  return getJson("/api/state/snapshot", 15_000);
}

// AF: 모니터링의 "최신화" 버튼 -- 주기 잡과 같은 파이프라인을 백그라운드로
// 1회 실행한다(응답은 즉시 온다, 완료는 getSnapshotMeta 폴링이 감지).
// 이미 실행 중이면 409(ApiResponseError.status === 409).
export function triggerRefresh(): Promise<{ triggered: boolean }> {
  return postJson("/api/state/refresh", {}, 10_000);
}

// AG-1: 원인 분석·수율 예측에서 새 파일을 업로드하면 부른다 -- 그
// 데이터셋을 활성 평가 데이터셋으로 바꾸고 스냅샷 파이프라인을 1회
// 실행한다. 화면별 개별 재분석은 만들지 않는다.
export function activateDataset(datasetId: string): Promise<{ activated: boolean; dataset_id: string }> {
  return postJson("/api/state/activate-dataset", { dataset_id: datasetId }, 10_000);
}

// AG-3: "자동 갱신으로 복귀" -- 수동 override를 지우고 원래 소스(SQL/폴백)로
// 되돌린다.
export function deactivateDataset(): Promise<{ deactivated: boolean; triggered: boolean }> {
  return postJson("/api/state/deactivate-dataset", {}, 10_000);
}

// Fire-and-forget from the caller's point of view (spec §3-2: a save
// failure must never surface as an analysis/training failure) -- these
// still return a Promise so a caller that wants to log a failure can,
// but every call site here is expected to `.catch(() => {})`.
// H-3⑤: `schedule_applied`가 false면 상태 저장은 됐지만 자동 수집 주기
// 반영(스케줄러 reschedule/pause)은 실패한 것 -- 호출부가 구분해서
// 안내할 수 있게 응답에 그대로 실어 보낸다.
export function saveTrainingState(
  dataset: string,
  payload: LatestTrainingPayload,
): Promise<{ saved: boolean; schedule_applied: boolean }> {
  return postJson("/api/state/training", { dataset, payload }, 15_000);
}

export function saveAnalysisState(dataset: string, payload: LatestAnalysisPayload): Promise<{ saved: boolean }> {
  return postJson("/api/state/analysis", { dataset, payload }, 15_000);
}

export function saveAlarmsState(
  trainDataset: string,
  evalDataset: string,
  payload: LatestAlarmsPayload,
): Promise<{ saved: boolean }> {
  return postJson("/api/state/alarms", { train_dataset: trainDataset, eval_dataset: evalDataset, payload }, 15_000);
}

// -- 즐겨찾기 (지시서 J) -- 서버에 저장한다(브라우저 저장소 금지, J-4). 점
// 데이터는 스냅샷에 절대 담지 않는다 -- 열 때 스냅샷 파라미터로 API를
// 다시 호출해 렌더한다.
export function getFavorites(): Promise<FavoriteListResponse> {
  return getJson("/api/favorites");
}

export function createFavorite(snapshot: FavoriteSnapshot): Promise<FavoriteRecord> {
  return postJson("/api/favorites", { snapshot });
}

export async function deleteFavorite(favoriteId: string): Promise<{ deleted: boolean }> {
  let response: Response;
  try {
    response = await timedFetch(`${getApiBaseUrl()}/api/favorites/${encodeURIComponent(favoriteId)}`, { method: "DELETE" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new ApiTimeoutError();
    rethrowApiConfigurationError(error);
    throw new ApiNetworkError();
  }
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json();
}

export type ChatMode = "report" | "chat";
export type ChatHistoryTurn = { role: "user" | "assistant"; content: string };

// "no_llm"/"no_analysis" are terminal states a retry can't fix (spec §5-5:
// only "timeout" and "other" get a 재시도 button).
export type ChatErrorKind = "no_llm" | "no_analysis" | "timeout" | "other";

export type ChatStreamHandlers = {
  onDelta: (text: string) => void;
  onDone: () => void;
  onError: (message: string, kind: ChatErrorKind) => void;
};

// D-4: 총소요 기준이 아니라 idle(무수신) 기준이다 -- SUNI 보고서 응답은
// 20~60초+ 걸릴 수 있는데(정상 범위), 총소요로 재면 느린 날 이미 잘
// 받고 있던 스트림이 90초를 넘긴 순간 통째로 끊긴다. 대신 이 시간
// 동안 새 바이트가 하나도 안 오면(연결이 실제로 멎은 것) 끊는다 --
// 스트림 자체가 살아 있는 한(계속 델타가 도착하는 한) 아무리 길어도
// 끊지 않는다.
const CHAT_STREAM_IDLE_TIMEOUT_MS = 30_000;

/** Streams /api/chat's SSE body, decoding `data: {...}\n\n` frames as they
 * arrive. Returns a handle to cancel the in-flight request (component
 * unmount, user navigates away mid-stream). */
export function streamChat(
  params: { message: string; mode: ChatMode; dataset: string; history?: ChatHistoryTurn[] },
  handlers: ChatStreamHandlers,
): { cancel: () => void } {
  const controller = new AbortController();
  // D-4: 매 청크(응답 시작 포함)마다 다시 설정한다 -- 이게 idle 타임아웃의
  // 전부다. 새 데이터가 도착하는 한 이 setTimeout은 항상 발동 전에
  // clearTimeout으로 취소되고 다시 걸린다.
  let timer = window.setTimeout(() => controller.abort(), CHAT_STREAM_IDLE_TIMEOUT_MS);
  function resetIdleTimer() {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => controller.abort(), CHAT_STREAM_IDLE_TIMEOUT_MS);
  }
  let settled = false;

  function finishError(message: string, kind: ChatErrorKind = "other") {
    if (settled) return;
    settled = true;
    window.clearTimeout(timer);
    handlers.onError(message, kind);
  }

  function finishDone() {
    if (settled) return;
    settled = true;
    window.clearTimeout(timer);
    handlers.onDone();
  }

  (async () => {
    let response: Response;
    try {
      response = await fetch(`${getApiBaseUrl()}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ history: [], ...params }),
        signal: controller.signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        finishError("응답 시간이 초과되었습니다. 다시 시도해 주세요.", "timeout");
        return;
      }
      finishError("답변을 생성하지 못했습니다. 다시 시도해 주세요.");
      return;
    }

    if (!response.ok) {
      if (response.status === 503) {
        finishError("LLM이 연결되지 않았습니다. 관리자에게 문의해 주세요.", "no_llm");
      } else if (response.status === 400) {
        finishError(await getErrorMessage(response), "no_analysis");
      } else {
        finishError("답변을 생성하지 못했습니다. 다시 시도해 주세요.");
      }
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      finishError("답변을 생성하지 못했습니다. 다시 시도해 주세요.");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        resetIdleTimer(); // D-4: 바이트가 도착했다 -- idle이 아니다.
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const jsonText = line.slice(5).trim();
          if (!jsonText) continue;
          let payload: { delta?: string; done?: boolean; error?: string };
          try {
            payload = JSON.parse(jsonText) as typeof payload;
          } catch {
            continue;
          }
          if (payload.error) {
            finishError(payload.error);
            return;
          }
          if (payload.delta) handlers.onDelta(payload.delta);
          if (payload.done) {
            finishDone();
            return;
          }
        }
      }
      finishDone();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        finishError("응답 시간이 초과되었습니다. 다시 시도해 주세요.", "timeout");
      } else {
        finishError("답변을 생성하지 못했습니다. 다시 시도해 주세요.");
      }
    }
  })();

  return { cancel: () => controller.abort() };
}
