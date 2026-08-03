"use client";

import { useEffect, useState } from "react";

// ThemeProvider stamps the resolved (system-independent) theme onto
// <html data-theme="...">; this reads that attribute reactively for
// components that need it to pick a JS-computed color (e.g. a
// continuous color-scale fill) rather than something expressible in CSS.
export function useResolvedTheme(): "light" | "dark" {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    const root = document.documentElement;
    const read = () => setTheme(root.dataset.theme === "dark" ? "dark" : "light");
    read();
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return theme;
}
