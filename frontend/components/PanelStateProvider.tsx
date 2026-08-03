"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

type BoolUpdater = boolean | ((previous: boolean) => boolean);

type PanelStateValue = {
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (value: BoolUpdater) => void;
  aiPanelOpen: boolean;
  setAiPanelOpen: (value: BoolUpdater) => void;
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
  const [aiPanelOpen, setAiPanelOpenState] = useState(initialAiPanelOpen);

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
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ sidebarCollapsed, setSidebarCollapsed, aiPanelOpen, setAiPanelOpen }),
    [sidebarCollapsed, setSidebarCollapsed, aiPanelOpen, setAiPanelOpen],
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
