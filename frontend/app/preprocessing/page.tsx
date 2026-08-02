import { redirect } from "next/navigation";

export default function LegacyPreprocessingRedirect() {
  redirect("/training?source=preprocessing-redirect");
}
