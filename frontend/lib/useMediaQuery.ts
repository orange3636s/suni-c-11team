"use client";

import { useEffect, useState } from "react";

/** Reactive `window.matchMedia` read -- the one JS-driven breakpoint hook
 * in the codebase (everything else is CSS-only `@media`); needed wherever
 * React itself has to branch on viewport width, not just restyle it (e.g.
 * swapping the sidebar for a horizontal tab bar, or a chart's numeric
 * `height` prop). SSR-safe: starts `false` and corrects on mount, same
 * "no client flash worth avoiding a hook over" tradeoff `useResolvedTheme`
 * already makes for `data-theme`.
 */
export function useMediaQuery(query: string): boolean {
  // Always starts `false`, even on the client's very first (hydrating)
  // render -- reading `window.matchMedia` in a lazy initializer would
  // return the real value immediately on that first client pass, which
  // is exactly what a Next.js server render can never know in advance,
  // producing a hydration mismatch on every component that branches its
  // JSX structure on this hook (not just a restyle -- see DashboardShell,
  // which mounts either <Sidebar> or <MobileTabBar>). The correction
  // happens in the effect below instead, one render after hydration has
  // already reconciled cleanly against the server's `false`-based HTML.
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mql = window.matchMedia(query);
    // Correcting an SSR-unknowable value (real viewport width) right after
    // mount is exactly the "sync with an external system" case this rule
    // expects an effect for; the alternative (reading it in a lazy
    // useState initializer) is what caused the hydration mismatch this
    // hook exists to avoid in the first place.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMatches(mql.matches);
    const listener = (event: MediaQueryListEvent) => setMatches(event.matches);
    mql.addEventListener("change", listener);
    return () => mql.removeEventListener("change", listener);
  }, [query]);
  return matches;
}

// Named breakpoints for the mobile layout switch -- kept as constants so
// every callsite uses the exact same cutoff CSS also uses.
export const BREAKPOINT_TABBAR_MAX = "(max-width: 1023px)";
export const BREAKPOINT_MOBILE_MAX = "(max-width: 767px)";

export function useIsTabBarLayout(): boolean {
  return useMediaQuery(BREAKPOINT_TABBAR_MAX);
}

export function useIsMobileLayout(): boolean {
  return useMediaQuery(BREAKPOINT_MOBILE_MAX);
}
