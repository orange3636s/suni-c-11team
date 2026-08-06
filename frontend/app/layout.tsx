import type { Metadata, Viewport } from "next";
import { cookies } from "next/headers";

import AnalysisStateProvider from "@/components/AnalysisStateProvider";
import PanelStateProvider from "@/components/PanelStateProvider";
import ThemeProvider from "@/components/ThemeProvider";

import "./globals.css";

export const metadata: Metadata = {
  title: "제조 공정 불량 예측 & 원인 분석 AI",
  description:
    "공정 데이터 검증, 수율 위험 예측, 원인 분석 및 사전 알림 시스템",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F5F5F7" },
    { media: "(prefers-color-scheme: dark)", color: "#1C1C1E" },
  ],
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Panel open/collapsed defaults are decided here, at server-render time,
  // from cookies (not localStorage) precisely so the first HTML the
  // browser paints already matches the saved state -- no client-side
  // useEffect correction, no collapsed->open flash on load.
  const cookieStore = await cookies();
  const initialSidebarCollapsed = cookieStore.get("sidebar-collapsed")?.value === "true";
  const initialAiPanelOpen = cookieStore.get("ai-panel-open")?.value !== "false";

  return (
    <html lang="ko" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("dashboard-theme")||"system";var d=t==="dark"||(t==="system"&&matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.dataset.theme=d?"dark":"light";document.documentElement.style.colorScheme=d?"dark":"light"}catch(e){}`,
          }}
        />
      </head>
      <body>
        <ThemeProvider>
          <PanelStateProvider
            initialSidebarCollapsed={initialSidebarCollapsed}
            initialAiPanelOpen={initialAiPanelOpen}
          >
            <AnalysisStateProvider>{children}</AnalysisStateProvider>
          </PanelStateProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
