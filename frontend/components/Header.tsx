"use client";

import { useEffect, useState } from "react";

import { getApiHealth } from "@/lib/api";

type ApiState = "확인 중" | "연결됨" | "연결 실패";

export default function Header() {
  const [apiState, setApiState] = useState<ApiState>("확인 중");
  const [updatedAt, setUpdatedAt] = useState("확인 중");

  useEffect(() => {
    let isMounted = true;

    async function checkApi() {
      try {
        const result = await getApiHealth();
        if (isMounted) {
          setApiState(result.status === "ok" ? "연결됨" : "연결 실패");
        }
      } catch {
        if (isMounted) {
          setApiState("연결 실패");
        }
      } finally {
        if (isMounted) {
          setUpdatedAt(
            new Intl.DateTimeFormat("ko-KR", {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }).format(new Date()),
          );
        }
      }
    }

    void checkApi();
    return () => {
      isMounted = false;
    };
  }, []);

  const apiStatusClass =
    apiState === "연결됨"
      ? "normal"
      : apiState === "연결 실패"
        ? "danger"
        : "warning";

  return (
    <header className="topHeader">
      <div className="headerStatus">
        <span className="statusDot normal" aria-hidden="true" />
        <div>
          <span>시스템 상태</span>
          <strong>정상</strong>
        </div>
      </div>
      <div className="headerStatus">
        <span className={`statusDot ${apiStatusClass}`} aria-hidden="true" />
        <div>
          <span>API 연결 상태</span>
          <strong>API {apiState}</strong>
        </div>
      </div>
      <div className="headerStatus updated">
        <div>
          <span>최근 업데이트 시간</span>
          <strong>{updatedAt}</strong>
        </div>
      </div>
    </header>
  );
}
