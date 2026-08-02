import { redirect } from "next/navigation";

export default function LegacyDataPreprocessingRedirect() {
  redirect("/training?source=preprocessing-redirect");
}
