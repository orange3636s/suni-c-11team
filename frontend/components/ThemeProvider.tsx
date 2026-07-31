"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

export type ThemePreference = "system" | "light" | "dark";

type ThemeContextValue = {
  theme: ThemePreference;
  setTheme: (theme: ThemePreference) => void;
};

const STORAGE_KEY = "dashboard-theme";
const ThemeContext = createContext<ThemeContextValue | null>(null);

function applyTheme(theme: ThemePreference) {
  const resolvedTheme =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;

  document.documentElement.dataset.theme = resolvedTheme;
  document.documentElement.style.colorScheme = resolvedTheme;
}

export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [theme, setThemeState] = useState<ThemePreference>("system");

  useEffect(() => {
    const savedTheme = window.localStorage.getItem(STORAGE_KEY);
    if (
      savedTheme === "system" ||
      savedTheme === "light" ||
      savedTheme === "dark"
    ) {
      applyTheme(savedTheme);
      queueMicrotask(() => setThemeState(savedTheme));
    } else {
      applyTheme("system");
    }
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleSystemThemeChange = () => {
      if (theme === "system") applyTheme("system");
    };

    media.addEventListener("change", handleSystemThemeChange);
    return () => media.removeEventListener("change", handleSystemThemeChange);
  }, [theme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme(nextTheme: ThemePreference) {
        setThemeState(nextTheme);
        window.localStorage.setItem(STORAGE_KEY, nextTheme);
        applyTheme(nextTheme);
      },
    }),
    [theme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider.");
  }
  return context;
}
