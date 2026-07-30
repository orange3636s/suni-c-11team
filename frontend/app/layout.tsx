import type { Metadata } from "next";

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
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
