import type {
  PreprocessResponse,
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
