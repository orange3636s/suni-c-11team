"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { ChatMode } from "@/lib/api";

type BoolUpdater = boolean | ((previous: boolean) => boolean);

type PendingChatRequest = { message: string; mode: ChatMode; nonce: number };

type PanelStateValue = {
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (value: BoolUpdater) => void;
  // 모바일 반응형 패치 S-1: ≤767px에서 사이드바가 오프캔버스 드로어로
  // 바뀐다 -- 열림/닫힘은 sidebarCollapsed(≥768px 전용, 아이콘 레일
  // 접힘)와 별개 상태다. 세션 간 지속할 이유가 없어(매번 닫힌 채로
  // 시작) 쿠키에 쓰지 않는다.
  sidebarDrawerOpen: boolean;
  setSidebarDrawerOpen: (value: BoolUpdater) => void;
  aiPanelOpen: boolean;
  setAiPanelOpen: (value: BoolUpdater) => void;
  // 설정 패널 신설 §A-3: 챗봇 패널과 동시에 열리지 않는다 -- 둘 중 하나를
  // 열면 다른 하나는 자동으로 닫힌다 (아래 setter들이 서로를 닫는다).
  settingsPanelOpen: boolean;
  setSettingsPanelOpen: (value: BoolUpdater) => void;
  // 지시서 L-1: 모델 학습 팝업도 설정·챗봇 패널과 상호 배타적이다.
  trainingPanelOpen: boolean;
  setTrainingPanelOpen: (value: BoolUpdater) => void;
  // RA-1: 모델 학습·자동화가 둘로 나뉘면서 생긴 두 번째 팝업(모델
  // 분석·자동화) -- trainingPanelOpen과도 서로 배타적이다(동시에 열리지
  // 않는다).
  analysisPanelOpen: boolean;
  setAnalysisPanelOpen: (value: BoolUpdater) => void;
  // The dataset id that 원인 분석 last completed for, or null if no
  // analysis has finished yet (or it was invalidated by a dataset change).
  // SUNI's AI panel reads this to know whether it has anything to answer
  // from and which dataset to ground its context in -- set by the
  // root-cause page, not persisted (a reload means "run it again").
  analysisDataset: string | null;
  setAnalysisDataset: (value: string | null) => void;
  // A message another page (e.g. the alarm table's "해설" button) wants
  // AiPanel to send on its behalf. AiPanel owns the actual chat state and
  // isn't reachable from outside, so this is the one shared inbox for
  // "open the panel and send this" -- `nonce` makes even a repeated,
  // identical message distinguishable as a new request.
  pendingChatRequest: PendingChatRequest | null;
  requestChat: (message: string, mode?: ChatMode) => void;
  clearPendingChatRequest: () => void;
};

const SIDEBAR_COOKIE = "sidebar-collapsed";
const AI_PANEL_COOKIE = "ai-panel-open";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

const PanelStateContext = createContext<PanelStateValue | null>(null);

function writeCookie(name: string, value: boolean) {
  document.cookie = `${name}=${value}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`;
}

// Initial open/collapsed state is decided server-side from these same
// cookies (see app/layout.tsx) so the first paint already matches --
// no client-side correction after mount, no flash.
export default function PanelStateProvider({
  initialSidebarCollapsed,
  initialAiPanelOpen,
  children,
}: {
  initialSidebarCollapsed: boolean;
  initialAiPanelOpen: boolean;
  children: ReactNode;
}) {
  const [sidebarCollapsed, setSidebarCollapsedState] = useState(initialSidebarCollapsed);
  const [sidebarDrawerOpen, setSidebarDrawerOpen] = useState(false);
  const [aiPanelOpen, setAiPanelOpenState] = useState(initialAiPanelOpen);
  const [settingsPanelOpen, setSettingsPanelOpenState] = useState(false);
  const [trainingPanelOpen, setTrainingPanelOpenState] = useState(false);
  const [analysisPanelOpen, setAnalysisPanelOpenState] = useState(false);
  const [analysisDataset, setAnalysisDataset] = useState<string | null>(null);
  const [pendingChatRequest, setPendingChatRequest] = useState<PendingChatRequest | null>(null);

  const setSidebarCollapsed = useCallback((value: BoolUpdater) => {
    setSidebarCollapsedState((previous) => {
      const next = typeof value === "function" ? value(previous) : value;
      writeCookie(SIDEBAR_COOKIE, next);
      return next;
    });
  }, []);

  const setAiPanelOpen = useCallback((value: BoolUpdater) => {
    setAiPanelOpenState((previous) => {
      const next = typeof value === "function" ? value(previous) : value;
      writeCookie(AI_PANEL_COOKIE, next);
      if (next) {
        setSettingsPanelOpenState(false);
        setTrainingPanelOpenState(false);
        setAnalysisPanelOpenState(false);
      }
      return next;
    });
  }, []);

  const setSettingsPanelOpen = useCallback((value: BoolUpdater) => {
    setSettingsPanelOpenState((previous) => {
      const next = typeof value === "function" ? value(previous) : value;
      if (next) {
        setAiPanelOpenState(false);
        writeCookie(AI_PANEL_COOKIE, false);
        setTrainingPanelOpenState(false);
        setAnalysisPanelOpenState(false);
      }
      return next;
    });
  }, []);

  const setTrainingPanelOpen = useCallback((value: BoolUpdater) => {
    setTrainingPanelOpenState((previous) => {
      const next = typeof value === "function" ? value(previous) : value;
      if (next) {
        setAiPanelOpenState(false);
        writeCookie(AI_PANEL_COOKIE, false);
        setSettingsPanelOpenState(false);
        setAnalysisPanelOpenState(false);
      }
      return next;
    });
  }, []);

  // RA-1: 모델 분석·자동화 팝업 -- 챗봇/설정/모델 학습 팝업과 상호
  // 배타적이다(같은 규칙을 그대로 따른다).
  const setAnalysisPanelOpen = useCallback((value: BoolUpdater) => {
    setAnalysisPanelOpenState((previous) => {
      const next = typeof value === "function" ? value(previous) : value;
      if (next) {
        setAiPanelOpenState(false);
        writeCookie(AI_PANEL_COOKIE, false);
        setSettingsPanelOpenState(false);
        setTrainingPanelOpenState(false);
      }
      return next;
    });
  }, []);

  const requestChat = useCallback((message: string, mode: ChatMode = "chat") => {
    setPendingChatRequest({ message, mode, nonce: Date.now() + Math.random() });
  }, []);

  const clearPendingChatRequest = useCallback(() => setPendingChatRequest(null), []);

  const value = useMemo(
    () => ({
      sidebarCollapsed,
      setSidebarCollapsed,
      sidebarDrawerOpen,
      setSidebarDrawerOpen,
      aiPanelOpen,
      setAiPanelOpen,
      settingsPanelOpen,
      setSettingsPanelOpen,
      trainingPanelOpen,
      setTrainingPanelOpen,
      analysisPanelOpen,
      setAnalysisPanelOpen,
      analysisDataset,
      setAnalysisDataset,
      pendingChatRequest,
      requestChat,
      clearPendingChatRequest,
    }),
    [
      sidebarCollapsed,
      setSidebarCollapsed,
      sidebarDrawerOpen,
      aiPanelOpen,
      setAiPanelOpen,
      settingsPanelOpen,
      setSettingsPanelOpen,
      trainingPanelOpen,
      setTrainingPanelOpen,
      analysisPanelOpen,
      setAnalysisPanelOpen,
      analysisDataset,
      pendingChatRequest,
      requestChat,
      clearPendingChatRequest,
    ],
  );

  return <PanelStateContext.Provider value={value}>{children}</PanelStateContext.Provider>;
}

export function usePanelState() {
  const context = useContext(PanelStateContext);
  if (!context) {
    throw new Error("usePanelState must be used within PanelStateProvider.");
  }
  return context;
}
