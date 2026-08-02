import type {
  AlertListResponse,
  AlertStatus,
  AlertSummary,
  AnalysisHistoryDetail,
  AnalysisHistorySummary,
  DeleteModelResponse,
  ExplainOptions,
  ExplainResponse,
  ModelDetail,
  ModelListResponse,
  OverviewDashboardResponse,
  PreprocessResponse,
  PredictionResponse,
  PredictionHistoryDetail,
  PredictionHistorySummary,
  PredictionThresholds,
  RelationshipAnalysisResponse,
  TrainResponse,
  TrainingJobCreateResponse,
  TrainingJobStatusResponse,
  ValidationResponse,
  HistoryList,
  HistoryResetResponse,
  HistoryResetSummary,
} from "@/types/data";
import { normalizeOverviewAnalysis } from "@/lib/overview";
import {
  normalizeAnalysisHistoryDetail,
  normalizeExplainResponse,
  normalizeRelationshipResponse,
} from "@/lib/root-cause";

export const MODEL_UNAVAILABLE_MESSAGE =
  "선택한 모델을 현재 서버에서 사용할 수 없습니다.\n새 모델을 선택하거나 다시 학습해 주세요.";

export class ApiResponseError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiResponseError";
    this.status = status;
  }
}

export type CumulativeDataStatus = {
  dataset_version: string;
  total_rows: number; total_lots: number; labeled_rows: number;
  pending_label_rows: number; conflict_rows: number;
  new_labeled_rows_since_active_model: number; new_lots_since_active_model: number;
  retraining_required: boolean;
};

async function championRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw new ApiResponseError(response.status, await getErrorMessage(response));
  return response.json() as Promise<T>;
}

export function getCumulativeDataStatus() { return championRequest<CumulativeDataStatus>("/api/data/status"); }
export function ingestProcessData(file: File) { const body = new FormData(); body.append("file", file); return championRequest<Record<string, unknown>>("/api/data/ingest", { method: "POST", body }); }
export function createModelUpdate() { return championRequest<{job_id: string; status: string}>("/api/model/update", { method: "POST" }); }
export function getModelUpdate(jobId: string) { return championRequest<Record<string, unknown>>(`/api/model/update/${encodeURIComponent(jobId)}`); }
export function getActiveModel() { return championRequest<{active_model: Record<string, unknown> | null}>("/api/model/active"); }

function historyResetErrorMessage(status: number): string {
  if (status === 400) return "초기화 확인값이 올바르지 않습니다.";
  if (status === 409) return "현재 실행 중인 작업이 있어 초기화할 수 없습니다.";
  if (status === 429) return "초기화 요청이 너무 많습니다. 10분 뒤 다시 시도해 주세요.";
  if (status === 502 || status === 503) return "초기화 서버에 연결할 수 없습니다.";
  return "이력 초기화 중 서버 오류가 발생했습니다.";
}

async function requestHistoryResetApi<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      cache: "no-store",
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new Error("초기화 서버에 연결할 수 없습니다.");
  }
  if (!response.ok) {
    throw new ApiResponseError(
      response.status,
      historyResetErrorMessage(response.status),
    );
  }
  try {
    return await response.json() as T;
  } catch {
    throw new Error("이력 초기화 중 서버 오류가 발생했습니다.");
  }
}

export function getHistoryResetSummary(): Promise<HistoryResetSummary> {
  return requestHistoryResetApi("/api/admin/history/summary");
}

export function resetAllHistory(): Promise<HistoryResetResponse> {
  return requestHistoryResetApi("/api/admin/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation: "RESET_ALL_HISTORY" }),
  });
}

function isModelUnavailableDetail(detail: string): boolean {
  return /모델|model(?:_id)?|호환|dependency|xgboost/i.test(detail);
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

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    cache: "no-store",
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(await getErrorMessage(response));
  return response.json() as Promise<T>;
}

async function requestUnknown(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    cache: "no-store",
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(await getErrorMessage(response));
  return response.json();
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

function predictionFormData(
  file: File,
  modelId: string | null,
  thresholds: PredictionThresholds,
): FormData {
  const formData = new FormData();
  formData.append("file", file);
  if (modelId) formData.append("model_id", modelId);
  formData.append(
    "warning_threshold",
    String(thresholds.warning_threshold),
  );
  formData.append(
    "danger_threshold",
    String(thresholds.danger_threshold),
  );
  return formData;
}

export async function predictCsv(
  file: File,
  modelId: string | null,
  thresholds: PredictionThresholds,
): Promise<PredictionResponse> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/predict`, {
      method: "POST",
      body: predictionFormData(file, modelId, thresholds),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "예측 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.json() as Promise<PredictionResponse>;
}

export async function downloadPredictions(
  file: File,
  modelId: string | null,
  thresholds: PredictionThresholds,
): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/predict/download`, {
      method: "POST",
      body: predictionFormData(file, modelId, thresholds),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "예측 다운로드 서버에 연결할 수 없습니다.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.blob();
}

function explanationFormData(
  file: File,
  modelId: string | null | undefined,
  options: ExplainOptions,
): FormData {
  const formData = new FormData();
  formData.append("file", file);
  if (modelId?.trim()) formData.append("model_id", modelId);
  formData.append("max_rows", String(options.max_rows));
  formData.append("top_n", String(options.top_n));
  formData.append("per_wafer_top_n", String(options.per_wafer_top_n));
  return formData;
}

export async function explainCsv(
  file: File,
  modelId: string,
  options: ExplainOptions,
): Promise<ExplainResponse> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/explain`, {
      method: "POST",
      body: explanationFormData(file, modelId, options),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "원인 분석 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  const payload: unknown = await response.json();
  return normalizeExplainResponse(payload);
}

export async function analyzeRelationships(
  file: File,
  modelId: string | null,
  options: ExplainOptions,
  correlationMethod: "pearson" | "spearman",
  analysisUnit: "wafer_observed_only" | "lot_aggregated" = "wafer_observed_only",
  thresholds: PredictionThresholds = { warning_threshold: 90, danger_threshold: 85 },
  analysisTarget = "Y",
  predictionId?: string | null,
): Promise<RelationshipAnalysisResponse> {
  const formData = explanationFormData(file, modelId, options);
  formData.append("correlation_method", correlationMethod);
  formData.append("analysis_unit", analysisUnit);
  formData.append("warning_threshold", String(thresholds.warning_threshold));
  formData.append("danger_threshold", String(thresholds.danger_threshold));
  formData.append("analysis_target", analysisTarget);
  if (predictionId) formData.append("prediction_id", predictionId);
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/relationships`, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error("연관 분석 서버에 연결할 수 없습니다.");
  }
  if (!response.ok) {
    const detail = await getErrorMessage(response);
    if (
      (response.status === 400 || response.status === 404 || response.status === 422) &&
      isModelUnavailableDetail(detail)
    ) {
      throw new ApiResponseError(response.status, MODEL_UNAVAILABLE_MESSAGE);
    }
    throw new ApiResponseError(response.status, detail);
  }
  const payload: unknown = await response.json();
  return normalizeRelationshipResponse(payload);
}

type HistoryQuery = {
  limit?: number;
  offset?: number;
  model_id?: string;
  prediction_id?: string;
  filename?: string;
  search?: string;
  target?: string;
  status?: string;
  date_from?: string;
  date_to?: string;
  sort?: "newest" | "oldest";
};

function historyPath(path: string, query: HistoryQuery = {}): string {
  const params = new URLSearchParams();
  Object.entries({ limit: 100, ...query }).forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value));
  });
  return `${path}?${params.toString()}`;
}

export function getPredictionHistory(query: HistoryQuery = {}): Promise<HistoryList<PredictionHistorySummary>> {
  return requestJson(historyPath("/api/predictions/history", query));
}

export function getPredictionHistoryDetail(predictionId: string): Promise<PredictionHistoryDetail> {
  return requestJson(`/api/predictions/history/${encodeURIComponent(predictionId)}`);
}

export function deletePredictionHistory(predictionId: string): Promise<{ success: boolean }> {
  return requestJson(`/api/predictions/history/${encodeURIComponent(predictionId)}`, { method: "DELETE" });
}

export function getAnalysisHistory(query: HistoryQuery = {}): Promise<HistoryList<AnalysisHistorySummary>> {
  return requestJson(historyPath("/api/analyses/history", query));
}

export async function getAnalysisHistoryDetail(analysisId: string): Promise<AnalysisHistoryDetail> {
  const payload = await requestUnknown(`/api/analyses/history/${encodeURIComponent(analysisId)}`);
  return normalizeAnalysisHistoryDetail(payload, analysisId);
}

export function deleteAnalysisHistory(analysisId: string): Promise<{ success: boolean }> {
  return requestJson(`/api/analyses/history/${encodeURIComponent(analysisId)}`, { method: "DELETE" });
}

export async function getDashboardOverview(
  analysisId?: string,
  signal?: AbortSignal,
): Promise<OverviewDashboardResponse> {
  const params = new URLSearchParams();
  if (analysisId) params.set("analysis_id", analysisId);
  const query = params.size ? `?${params.toString()}` : "";
  const payload = await requestUnknown(`/api/dashboard/overview${query}`, { signal });
  return normalizeOverviewAnalysis(payload);
}

export async function downloadExplanation(
  file: File,
  modelId: string,
  options: ExplainOptions,
): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/explain/download`, {
      method: "POST",
      body: explanationFormData(file, modelId, options),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error("원인 분석 결과 다운로드 서버에 연결할 수 없습니다.");
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.blob();
}

async function runtimeGet<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, { cache: "no-store" });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error("Dashboard API에 연결할 수 없습니다.");
  }
  if (!response.ok) throw new Error(await getErrorMessage(response));
  return response.json() as Promise<T>;
}

export function getAlerts(query = ""): Promise<AlertListResponse> {
  return runtimeGet(`/api/alerts${query ? `?${query}` : ""}`);
}

export function getAlertSummary(): Promise<AlertSummary> {
  return runtimeGet("/api/alerts/summary");
}

export async function updateAlertStatus(alertId: string, status: AlertStatus): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/alerts/${encodeURIComponent(alertId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!response.ok) throw new Error(await getErrorMessage(response));
}
