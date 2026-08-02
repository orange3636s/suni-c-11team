import { redirect } from "next/navigation";

export default function LegacyReportRedirect() {
  redirect("/root-cause?tab=report");
}
