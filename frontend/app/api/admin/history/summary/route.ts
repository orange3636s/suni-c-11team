import { proxyAdminHistory } from "../_proxy";


export const dynamic = "force-dynamic";

export function GET(): Promise<Response> {
  return proxyAdminHistory("/api/admin/history/summary", { method: "GET" });
}
