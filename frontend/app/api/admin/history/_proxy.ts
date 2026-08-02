import { NextResponse } from "next/server";


type AdminProxyOptions = {
  method: "GET" | "DELETE";
  body?: string;
};

function jsonError(status: number, detail: string): NextResponse {
  return NextResponse.json(
    { detail },
    { status, headers: { "Cache-Control": "no-store" } },
  );
}

function backendApiBaseUrl(): string | null {
  const configured = (
    process.env.BACKEND_API_BASE_URL
    ?? process.env.NEXT_PUBLIC_API_BASE_URL
    ?? ""
  ).trim().replace(/\/+$/, "");
  if (configured) return configured;
  return process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : null;
}

export async function proxyAdminHistory(
  path: string,
  options: AdminProxyOptions,
): Promise<Response> {
  const secret = process.env.ADMIN_RESET_SECRET?.trim();
  if (!secret) return jsonError(403, "초기화 권한이 없습니다.");

  const apiBaseUrl = backendApiBaseUrl();
  if (!apiBaseUrl) {
    return jsonError(500, "이력 초기화 중 서버 오류가 발생했습니다.");
  }

  try {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      method: options.method,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "X-Admin-Reset-Secret": secret,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
      },
      body: options.body,
    });
    const responseBody = await response.text();
    return new Response(responseBody, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return jsonError(502, "초기화 서버에 연결할 수 없습니다.");
  }
}
