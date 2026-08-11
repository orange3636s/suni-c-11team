import type {
  AutomationSaveRequest,
  AutomationTestResponse,
  CategoricalScatterResponse,
  ConfigTreemapResponse,
  DatasetListResponse,
  DatasetSchemaResponse,
  DatasetUploadResponse,
  DispatchResponse,
  FavoriteListResponse,
  FavoriteRecord,
  FavoriteSnapshot,
  HeatmapResponse,
  LatestAnalysisPayload,
  LatestStateResponse,
  LatestTrainingPayload,
  ModelPerformanceResponse,
  NotificationConditions,
  NotificationSettingsSummary,
  NotifyHistoryListResponse,
  ParetoRankingResponse,
  PromotionHistoryResponse,
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
// 수 있다. 90초는 그 요청들이 정상적으로 끝나는 시간보다 넉넉히 길게 잡은
// 상한선이다 -- 상한이 없으면 연결이 멎었을 때 스피너가 무한정 돈다.
const DEFAULT_TIMEOUT_MS = 90_000;
// 파일 업로드(CSV)는 대개 더 오래 걸릴 수 있어 여유를 더 둔다 -- 상한선이
// 있다는 것 자체가 핵심이지, 정확한 값은 중요하지 않다(연결이 끊겨도 결국
// 이 시간 안에 끝난다는 게 중요하다).
const UPLOAD_TIMEOUT_MS = 5 * 60_000;
// 대용량(최대 200,000행) 데이터셋에서 원인 분석/수율 예측
// 조회(스크리닝 히트맵·산점도·순위표)가 90초를 넘을 수 있다 -- 이 세
// 엔드포인트에만 넉넉한 타임아웃을 준다. 전역 기본값(90초)은 그대로 둔다
// -- 죽은 요청을 다른 화면까지 오래 붙들게 하고 싶지 않다.
const ANALYSIS_QUERY_TIMEOUT_MS = 180_000;

// 업로드·삭제·즐겨찾기 등 getJson/postJson을 거치지 않는 raw fetch도
// 반드시 이 함수로 부른다 -- AbortController+타임아웃이 없으면 연결이
// 멎었을 때(서버 다운, 네트워크 끊김) "업로드 중…"/"삭제 중…"이 무한정
// 떠 있는다. 에러 메시지 문구는 호출부마다 다르므로 여기서는 fetch에
// 타임아웃을 거는 부분만 공유한다.
async function timedFetch(url: string, init: RequestInit = {}, timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

// API 베이스 URL의 단일 소스 -- useApiStatus의 /health 폴링까지 포함해
// 모든 호출부가 이 함수를 쓴다. 프로덕션에서 미설정일 때 조용히
// 127.0.0.1로 폴백하면 상태 배지가 "연결 끊김"만 보여주고 진짜 원인
// (NEXT_PUBLIC_API_BASE_URL 미설정)을 감추므로, 개발 환경에서만 폴백하고
// 그 외에는 곧장 에러를 던진다.
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

// 학습 쪽 "내장 데이터로 되돌리기"는 분석 쪽(deactivateDataset)과
// 달리 등록 해제 상태가 없다 -- 유일한 되돌리기 방법은 내장 train.CSV로
// 즉시 재학습하는 것뿐이다. 잡 큐(createTrainingJob)를 거치지 않고
// 서버가 학습을 끝낼 때까지 기다리는 동기 호출이다(수 초~수십 초).
export function retrainBundled(): Promise<{ model_id: string }> {
  return postJson("/api/train/bundled", {}, UPLOAD_TIMEOUT_MS);
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
  return getJson(`/api/screening/scatter?${new URLSearchParams({ dataset, target, feature }).toString()}`, ANALYSIS_QUERY_TIMEOUT_MS);
}

export function getScreeningScatterCategorical(dataset: string, target: string, feature: string): Promise<CategoricalScatterResponse> {
  return getJson(`/api/screening/scatter/categorical?${new URLSearchParams({ dataset, target, feature }).toString()}`, ANALYSIS_QUERY_TIMEOUT_MS);
}

// y(=100 − Σ Y1~Y5) 오름차순 전체 목록(신뢰도==0 웨이퍼 제외) -- 수율
// 예측 화면이 쓰는 유일한 판정 엔드포인트다. 상위 10/전체 보기·검색·정렬은
// 프런트가 이 전체 목록 위에서 수행한다: 검색 중에는 상위 10 제한을
// 해제해야 하므로 서버가 미리 잘라 보내면 안 된다.
export function getYieldPrediction(trainDataset: string, evalDataset: string): Promise<YieldPredictionResponse> {
  const params = new URLSearchParams({ train: trainDataset, eval: evalDataset });
  return getJson(`/api/alerts/ranking?${params.toString()}`, ANALYSIS_QUERY_TIMEOUT_MS);
}

export function getModelPerformance(): Promise<ModelPerformanceResponse> {
  return getJson("/api/models/performance");
}

// 승격 여부와 무관한 학습 시도 이력. 모델
// 학습·자동화 팝업이 "게이트 미달로 교체되지 않음"을 보여주는 데 쓴다.
export function getPromotionHistory(limit = 5): Promise<PromotionHistoryResponse> {
  return getJson(`/api/models/promotion-history?${new URLSearchParams({ limit: String(limit) }).toString()}`);
}

// 지표 토글은 없다 -- 히트맵은 항상 Adjusted R²(농도·표시 숫자)와 적합
// 차수/관계 형태/꼭짓점(방향)을 함께 받는 numeric 보기 하나만 조회한다.
export function getScreeningHeatmap(dataset: string): Promise<HeatmapResponse> {
  const params = new URLSearchParams({ dataset });
  return getJson(`/api/screening/heatmap?${params.toString()}`, ANALYSIS_QUERY_TIMEOUT_MS);
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

export function getConfigTreemap(dataset: string, step: number, target = "Y1"): Promise<ConfigTreemapResponse> {
  return getJson(`/api/monitoring/config-treemap?${new URLSearchParams({ dataset, step: String(step), target }).toString()}`);
}

// -- 알림 연동 -----------------------------------------------------------
// state/latest가 마운트 시 1번에 알림 설정을 함께 실어 온다 -- 이 함수들은
// 사용자가 설정 패널에서 실제로 무언가를 바꿀 때만 호출된다.

export function connectSlack(webhookUrl: string, channel: string | null): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/slack", { webhook_url: webhookUrl, channel });
}

// webhookUrl을 생략하면(이미 연결된 채널을 테스트할 때) 서버가
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

// "알림·자동화 설정" 팝업의 자동화 섹션 -- 서버 주소·사용자명·
// refresh time·켜짐 여부를 저장한다. 비밀번호 필드는 없다(환경변수로만
// 받는다).
export function saveAutomationSettings(body: AutomationSaveRequest): Promise<NotificationSettingsSummary> {
  return postJson("/api/notify/automation", body);
}

export function testAutomationConnection(): Promise<AutomationTestResponse> {
  return postJson("/api/notify/automation/test", {});
}

// 알림 기록 화면 -- 발송/건너뜀 이력과 발송 당시 메시지 전문.
export function getNotifyHistory(limit = 100): Promise<NotifyHistoryListResponse> {
  return getJson(`/api/notify/history?${new URLSearchParams({ limit: String(limit) }).toString()}`);
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

// 원인 분석 실행 직후 fire-and-forget으로 호출된다 -- 실패해도 분석 결과
// 표시를 막으면 안 되지만, 발송 실패 자체는 사용자에게 보여야 한다.
// 호출부는 실패 시 오류 배너를 띄우고, 이미 렌더된 분석 결과는 그대로 둔다.
export function dispatchAlarmNotifications(trainDataset: string, evalDataset: string): Promise<DispatchResponse> {
  return postJson("/api/notify/dispatch", { train_dataset: trainDataset, eval_dataset: evalDataset, dashboard_url: null });
}

// 수율 예측 화면 "알림 전송" 버튼 -- 확인 다이얼로그에서 연결된
// 채널을 보여주려면 먼저 현재 설정을 읽어야 한다.
export function getNotificationSettings(): Promise<NotificationSettingsSummary> {
  return getJson("/api/notify/settings");
}

// 버튼을 눌러 지금 바로 발송한다 -- `/dispatch`와는 별도 엔드포인트라
// AUC 게이트·목표 수율 설정과 무관하게 동작하지만,
// 억제 규칙(신규분만·시간당 예산·최소 간격 10분)은 그대로 적용된다.
export function dispatchYieldUpdateNotification(trainDataset: string, evalDataset: string): Promise<DispatchResponse> {
  return postJson("/api/notify/yield-update/dispatch", {
    train_dataset: trainDataset,
    eval_dataset: evalDataset,
    dashboard_url: null,
  });
}

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Called once on app mount by AnalysisStateProvider -- never per
// tab-switch. A short timeout keeps a slow/unreachable API from stalling
// first paint; the provider treats any failure the same as "no saved
// result yet", since a failed restore must never block the app.
export function getLatestState(): Promise<LatestStateResponse> {
  return getJson("/api/state/latest", 15_000);
}

// 자동 갱신 스냅샷. `getSnapshotMeta`는 60초 폴링·포커스 복귀 때 가볍게
// 부르는 용도(created_at만 온다) -- 그 값이 캐시보다 최신일 때만
// `getSnapshot`으로 전체를 다시 받고, 같으면 아무것도 하지 않는다.
export function getSnapshotMeta(): Promise<SnapshotMetaResponse> {
  return getJson("/api/state/snapshot/meta", 10_000);
}

export function getSnapshot(): Promise<SnapshotResponse> {
  return getJson("/api/state/snapshot", 15_000);
}

// "모델 분석" 팝업의 [분석 시작] 버튼 -- 네 화면(모니터링/Config별
// 트리맵/원인 분석/수율 예측)을 한 번에 갱신하는 유일한 실행 경로다
// ("새로고침 역할" 겸함). 백그라운드로 실행되어 응답은 즉시 온다 --
// 완료는 getSnapshotMeta 폴링이 감지한다. 이미 실행 중이면 409
// (ApiResponseError.status === 409).
export function triggerRefresh(): Promise<{ triggered: boolean }> {
  return postJson("/api/state/refresh", {}, 10_000);
}

// "모델 분석" 팝업에서 파일을 선택하거나 데이터베이스에서 불러온 뒤
// 부른다 -- 그 데이터셋을 활성 분석 데이터로 등록할 뿐, 4화면 분석을
// 자동으로 실행하지 않는다(등록과 실행은 분리돼 있다). 실제 계산은 별도로
// triggerRefresh()를 호출해야 시작된다.
export function activateDataset(datasetId: string): Promise<{ activated: boolean; dataset_id: string }> {
  return postJson("/api/state/activate-dataset", { dataset_id: datasetId }, 10_000);
}

// "데이터베이스에서 불러오기" -- "알림·자동화 설정"에 등록된 서버와
// 같은 소스에서 최신 배치를 가져와 데이터셋으로 등록한다(업로드 응답과
// 같은 모양). 등록만 할 뿐 활성화하지 않는다 -- 이어서 activateDataset을
// 호출해야 분석 데이터로 반영된다.
export function fetchFromDb(): Promise<DatasetUploadResponse> {
  return postJson("/api/state/fetch-from-db", {}, 30_000);
}

// 수동 override를 지워 다음 [분석 시작]부터 원래 소스(SQL/폴백)로
// 되돌아가게 한다. 등록만 할 뿐 분석을 자동으로 다시 실행하지 않는다.
export function deactivateDataset(): Promise<{ deactivated: boolean }> {
  return postJson("/api/state/deactivate-dataset", {}, 10_000);
}

// Fire-and-forget from the caller's point of view -- a save failure must
// never surface as an analysis/training failure. These still return a
// Promise so a caller that wants to log a failure can, but every call site
// here is expected to `.catch(() => {})`.
// `schedule_applied`가 false면 상태 저장은 됐지만 자동 수집 주기
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

// -- 즐겨찾기 -- 서버에 저장한다(브라우저 저장소는 쓰지 않는다). 점
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

// "no_llm"/"no_analysis" are terminal states a retry can't fix -- only
// "timeout" and "other" get a 재시도 button.
export type ChatErrorKind = "no_llm" | "no_analysis" | "timeout" | "other";

export type ChatStreamHandlers = {
  onDelta: (text: string) => void;
  onDone: () => void;
  onError: (message: string, kind: ChatErrorKind) => void;
};

// 총소요 기준이 아니라 idle(무수신) 기준이다 -- SUNI 보고서 응답은
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
  // 매 청크(응답 시작 포함)마다 다시 설정한다 -- 이게 idle 타임아웃의
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
        resetIdleTimer(); // 바이트가 도착했다 -- idle이 아니다.
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
