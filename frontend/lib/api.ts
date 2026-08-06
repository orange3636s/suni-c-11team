import type {
  AlarmListResponse,
  AlarmSummaryResponse,
  AnalysisReportResponse,
  CategoricalScatterResponse,
  ControlRangeListResponse,
  DatasetListResponse,
  DatasetSchemaResponse,
  DatasetUploadResponse,
  DeleteModelResponse,
  HeatmapMetric,
  HeatmapResponse,
  LatestAlarmsPayload,
  LatestAnalysisPayload,
  LatestStateResponse,
  LatestTrainingPayload,
  MeasurementExpansionResponse,
  ModelDetail,
  ModelListResponse,
  ModelPerformanceResponse,
  ParetoRankingResponse,
  PreprocessResponse,
  RecommendationListResponse,
  ScreeningScatterResponse,
  TrainResponse,
  TrainingJobCreateResponse,
  TrainingJobStatusResponse,
  ValidationResponse,
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

export type LatestModelMetadata = {
  model_id: string;
  model_name?: string | null;
  target: "Y";
  version?: string | null;
  trained_at?: string | null;
  source_filename?: string | null;
  row_count?: number | null;
  feature_columns?: string[];
  categorical_columns?: string[];
  metrics?: Record<string, { r2?: number | null; rmse?: number | null; mae?: number | null }>;
};

async function championRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<T>;
}

export function getLatestModel() {
  return championRequest<{ latest_model: LatestModelMetadata | null }>("/api/model/latest");
}

export type ApiHealth = {
  status: string;
  service?: string;
  environment?: string;
  version?: string;
  model_directory_ready?: boolean;
};

function getApiBaseUrl(): string {
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

async function postCsv<T>(path: string, file: File): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new ApiResponseError(response.status, await getErrorMessage(response));
  }

  return response.json() as Promise<T>;
}

export async function getApiHealth(): Promise<ApiHealth> {
  const response = await fetch(`${getApiBaseUrl()}/health`, {
    method: "GET",
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API 상태 확인 실패: ${response.status}`);
  }

  return response.json() as Promise<ApiHealth>;
}

export function validateCsv(file: File): Promise<ValidationResponse> {
  return postCsv<ValidationResponse>("/api/validate", file);
}

export function preprocessCsv(file: File): Promise<PreprocessResponse> {
  return postCsv<PreprocessResponse>("/api/preprocess", file);
}

export async function trainModel(
  file: File,
): Promise<TrainResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/train`, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "FastAPI 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  return response.json() as Promise<TrainResponse>;
}

export async function createTrainingJob(
  file: File,
): Promise<TrainingJobCreateResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/train/jobs`, {
      method: "POST",
      body: formData,
    });
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

export async function getModels(): Promise<ModelListResponse> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/models`, {
      method: "GET",
      cache: "no-store",
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "FastAPI 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.json() as Promise<ModelListResponse>;
}

export async function getModelDetail(modelId: string, signal?: AbortSignal): Promise<ModelDetail> {
  let response: Response;
  try {
    response = await fetch(
      `${getApiBaseUrl()}/api/models/${encodeURIComponent(modelId)}`,
      {
        method: "GET",
        cache: "no-store",
        signal,
      },
    );
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "모델 상세 정보를 불러올 수 없습니다. 백엔드 상태를 확인해 주세요.",
    );
  }

  if (!response.ok) {
    throw new ApiResponseError(response.status, await getErrorMessage(response));
  }
  return response.json() as Promise<ModelDetail>;
}

export async function deleteModel(modelId: string): Promise<DeleteModelResponse> {
  let response: Response;
  try {
    response = await fetch(
      `${getApiBaseUrl()}/api/models/${encodeURIComponent(modelId)}`,
      {
        method: "DELETE",
        cache: "no-store",
        headers: { Accept: "application/json" },
      },
    );
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "모델 삭제 서버에 연결할 수 없습니다. 백엔드 실행 상태를 확인해 주세요.",
    );
  }

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const result = await response.json() as DeleteModelResponse;
  if (
    !result.deleted ||
    result.model_id !== modelId ||
    result.prediction_history_kept !== true ||
    result.analysis_history_kept !== true
  ) {
    throw new Error(
      "모델 삭제 응답의 model_id 또는 History 보존 상태가 API 계약과 일치하지 않습니다.",
    );
  }
  return result;
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
  const response = await fetch(`${getApiBaseUrl()}/api/datasets`, { method: "POST", body: formData });
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<DatasetUploadResponse>;
}

export async function deleteDataset(datasetId: string): Promise<{ success: boolean; dataset_id: string }> {
  const response = await fetch(`${getApiBaseUrl()}/api/datasets/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json();
}

export function getDatasetSchema(datasetId: string): Promise<DatasetSchemaResponse> {
  return getJson(`/api/datasets/${encodeURIComponent(datasetId)}/schema`);
}

export async function downloadDatasetFile(datasetId: string, filename: string): Promise<File> {
  const response = await fetch(`${getApiBaseUrl()}/api/datasets/${encodeURIComponent(datasetId)}/download`);
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  const blob = await response.blob();
  return new File([blob], filename, { type: "text/csv" });
}

export function getScreeningScatter(dataset: string, target: string, feature: string): Promise<ScreeningScatterResponse> {
  return getJson(`/api/screening/scatter?${new URLSearchParams({ dataset, target, feature }).toString()}`);
}

export function getScreeningScatterCategorical(dataset: string, target: string, feature: string): Promise<CategoricalScatterResponse> {
  return getJson(`/api/screening/scatter/categorical?${new URLSearchParams({ dataset, target, feature }).toString()}`);
}

export function getControlRanges(dataset: string): Promise<ControlRangeListResponse> {
  return getJson(`/api/control-ranges?${new URLSearchParams({ dataset }).toString()}`);
}

export function getAlarms(trainDataset: string, evalDataset: string, severity?: string): Promise<AlarmListResponse> {
  const params = new URLSearchParams({ train: trainDataset, eval: evalDataset });
  if (severity) params.set("severity", severity);
  return getJson(`/api/alarms?${params.toString()}`);
}

export function getAlarmSummary(trainDataset: string, evalDataset: string): Promise<AlarmSummaryResponse> {
  return getJson(`/api/alarms/summary?${new URLSearchParams({ train: trainDataset, eval: evalDataset }).toString()}`);
}

export function getRecommendations(trainDataset: string, evalDataset: string): Promise<RecommendationListResponse> {
  return getJson(`/api/recommendations?${new URLSearchParams({ train: trainDataset, eval: evalDataset }).toString()}`);
}

export function getModelPerformance(): Promise<ModelPerformanceResponse> {
  return getJson("/api/models/performance");
}

export function getScreeningHeatmap(dataset: string, metric: HeatmapMetric): Promise<HeatmapResponse> {
  return getJson(`/api/screening/heatmap?${new URLSearchParams({ dataset, metric }).toString()}`);
}

export function getScreeningPareto(dataset: string, target: string): Promise<ParetoRankingResponse> {
  return getJson(`/api/screening/pareto?${new URLSearchParams({ dataset, target }).toString()}`);
}

export function getAnalysisReport(dataset: string): Promise<AnalysisReportResponse> {
  return getJson(`/api/analysis/report?${new URLSearchParams({ dataset }).toString()}`);
}

export function getMeasurementExpansion(dataset: string): Promise<MeasurementExpansionResponse> {
  return getJson(`/api/analysis/measurement-expansion?${new URLSearchParams({ dataset }).toString()}`);
}

// -- 학습·분석 결과 상태 유지 (탭 이동·재접속) --------------------------
// Called once on app mount by AnalysisStateProvider (spec §4-2/§6) --
// never per tab-switch. A short timeout keeps a slow/unreachable API from
// stalling first paint; the provider treats any failure the same as "no
// saved result yet" (spec: "복원 실패가 앱을 막으면 안 된다").
export function getLatestState(): Promise<LatestStateResponse> {
  return getJson("/api/state/latest", 15_000);
}

// Fire-and-forget from the caller's point of view (spec §3-2: a save
// failure must never surface as an analysis/training failure) -- these
// still return a Promise so a caller that wants to log a failure can,
// but every call site here is expected to `.catch(() => {})`.
export function saveTrainingState(dataset: string, payload: LatestTrainingPayload): Promise<{ saved: boolean }> {
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

// SUNI 보고서 응답은 20~60초가 걸릴 수 있어 다른 요청보다 넉넉한 상한선을 둔다
// (spec §3-3: 백엔드 타임아웃도 90초).
const CHAT_STREAM_TIMEOUT_MS = 90_000;

/** Streams /api/chat's SSE body, decoding `data: {...}\n\n` frames as they
 * arrive. Returns a handle to cancel the in-flight request (component
 * unmount, user navigates away mid-stream). */
export function streamChat(
  params: { message: string; mode: ChatMode; dataset: string; history?: ChatHistoryTurn[] },
  handlers: ChatStreamHandlers,
): { cancel: () => void } {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), CHAT_STREAM_TIMEOUT_MS);
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
        finishError("응답 시간이 초과되었습니다. 다시 시도해 주세요.");
      } else {
        finishError("답변을 생성하지 못했습니다. 다시 시도해 주세요.");
      }
    }
  })();

  return { cancel: () => controller.abort() };
}
