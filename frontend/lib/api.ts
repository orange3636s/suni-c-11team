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
  ReportOptions,
  ReportResponse,
  TrainResponse,
  ValidationResponse,
  HistoryList,
} from "@/types/data";
import { normalizeOverviewAnalysis } from "@/lib/overview";
import {
  normalizeAnalysisHistoryDetail,
  normalizeExplainResponse,
  normalizeRelationshipResponse,
} from "@/lib/root-cause";

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
    throw new Error(await getErrorMessage(response));
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
    throw new Error(await getErrorMessage(response));
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
  modelId: string,
  thresholds: PredictionThresholds,
): FormData {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_id", modelId);
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
  modelId: string,
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
  modelId: string,
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
  modelId: string,
  options: ExplainOptions,
): FormData {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_id", modelId);
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
  modelId: string,
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
    throw new Error(await getErrorMessage(response));
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

function reportFormData(
  file: File,
  modelId: string,
  options: ReportOptions,
): FormData {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_id", modelId);
  formData.append(
    "warning_threshold",
    String(options.warning_threshold),
  );
  formData.append(
    "danger_threshold",
    String(options.danger_threshold),
  );
  formData.append("max_rows", String(options.max_rows));
  formData.append("top_n", String(options.top_n));
  return formData;
}

export async function generateReport(
  file: File,
  modelId: string,
  options: ReportOptions,
): Promise<ReportResponse> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/report`, {
      method: "POST",
      body: reportFormData(file, modelId, options),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error(
      "분석 보고서 서버에 연결할 수 없습니다. 백엔드 실행 상태를 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.json() as Promise<ReportResponse>;
}

export async function downloadReport(
  file: File,
  modelId: string,
  options: ReportOptions,
): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/report/download`, {
      method: "POST",
      body: reportFormData(file, modelId, options),
    });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error("HTML 보고서 다운로드 서버에 연결할 수 없습니다.");
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
