import type { Metadata } from "next";

import ThemeProvider from "@/components/ThemeProvider";

import "./globals.css";

export const metadata: Metadata = {
  title: "제조 공정 불량 예측 및 원인 분석 AI",
  description:
    "공정 데이터 검증, 수율 위험 예측, 불량 원인 후보 분석 및 사전 알림 시스템",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
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
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
