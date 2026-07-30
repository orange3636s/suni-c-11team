import type {
  ExplainOptions,
  ExplainResponse,
  ModelListResponse,
  PreprocessResponse,
  PredictionResponse,
  PredictionThresholds,
  ReportOptions,
  ReportResponse,
  TrainResponse,
  ValidationResponse,
} from "@/types/data";

export type ApiHealth = {
  status: string;
};

function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";
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
  target: string,
): Promise<TrainResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);

  let response: Response;
  try {
    response = await fetch(`${getApiBaseUrl()}/api/train`, {
      method: "POST",
      body: formData,
    });
  } catch {
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
  } catch {
    throw new Error(
      "FastAPI 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.json() as Promise<ModelListResponse>;
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
  } catch {
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
  } catch {
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
  } catch {
    throw new Error(
      "원인 분석 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해 주세요.",
    );
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.json() as Promise<ExplainResponse>;
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
  } catch {
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
  } catch {
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
  } catch {
    throw new Error("HTML 보고서 다운로드 서버에 연결할 수 없습니다.");
  }
  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }
  return response.blob();
}
