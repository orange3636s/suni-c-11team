"use client";

import { useEffect, useRef, type RefObject } from "react";

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** SettingsPanel/TrainingPanel/AiPanel(오버레이) 세 모달이 공유하는
 * 포커스 트랩 -- 열릴 때 컨테이너 안 첫 포커스 가능 요소로 이동시키고,
 * Tab 순환을 컨테이너 밖으로 나가지 않게 막고, 닫힐 때(active가
 * false로 바뀔 때) 열기 전 포커스였던 요소로 되돌린다. Esc로 닫는
 * 로직은 각 컴포넌트가 이미 조건(예: 다른 열린 모달과의 충돌 회피)까지
 * 갖춰 두고 있어 여기서 통합하지 않는다 -- 포커스 이동/순환/복귀만
 * 책임진다. */
export function useFocusTrap(containerRef: RefObject<HTMLElement | null>, active: boolean) {
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active) return;
    const container = containerRef.current;
    if (!container) return;

    previouslyFocused.current = document.activeElement as HTMLElement | null;

    const focusables = () => Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
    const first = focusables()[0];
    (first ?? container).focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const firstEl = items[0];
      const lastEl = items[items.length - 1];
      if (event.shiftKey && document.activeElement === firstEl) {
        event.preventDefault();
        lastEl.focus();
      } else if (!event.shiftKey && document.activeElement === lastEl) {
        event.preventDefault();
        firstEl.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused.current?.focus();
    };
  }, [active, containerRef]);
}
