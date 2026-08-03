import type {
  AlarmListResponse,
  AlarmSummaryResponse,
  ControlRangeListResponse,
  DatasetListResponse,
  DatasetSchemaResponse,
  DatasetUploadResponse,
  DeleteModelResponse,
  HeatmapMetric,
  HeatmapResponse,
  ModelDetail,
  ModelListResponse,
  ModelPerformanceResponse,
  PreprocessResponse,
  ScreeningResponse,
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

async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}${path}`, { method: "GET", cache: "no-store" });
  } catch (error) {
    rethrowApiConfigurationError(error);
    throw new Error("서버에 연결할 수 없습니다.");
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

export function getScreening(dataset: string): Promise<ScreeningResponse> {
  return getJson(`/api/screening?${new URLSearchParams({ dataset }).toString()}`);
}

export function getScreeningScatter(dataset: string, target: string, feature: string): Promise<ScreeningScatterResponse> {
  return getJson(`/api/screening/scatter?${new URLSearchParams({ dataset, target, feature }).toString()}`);
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

export function getModelPerformance(): Promise<ModelPerformanceResponse> {
  return getJson("/api/models/performance");
}

export function getScreeningHeatmap(dataset: string, metric: HeatmapMetric): Promise<HeatmapResponse> {
  return getJson(`/api/screening/heatmap?${new URLSearchParams({ dataset, metric }).toString()}`);
}
