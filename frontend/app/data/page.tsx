import { redirect } from "next/navigation";

export default function LegacyDataRedirect() {
  redirect("/training?source=preprocessing-redirect");
}
