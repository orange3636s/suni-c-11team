import { NextResponse } from "next/server";

import { proxyAdminHistory } from "./_proxy";


export const dynamic = "force-dynamic";

export async function DELETE(request: Request): Promise<Response> {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "초기화 확인값이 올바르지 않습니다." },
      { status: 400 },
    );
  }
  return proxyAdminHistory("/api/admin/history", {
    method: "DELETE",
    body: JSON.stringify(payload),
  });
}
