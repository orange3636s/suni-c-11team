import { redirect } from "next/navigation";

export default function LegacyUploadRedirect() {
  redirect("/training?source=preprocessing-redirect");
}
