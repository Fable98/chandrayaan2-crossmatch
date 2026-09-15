"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark";

const STORAGE_KEY = "chandrayaan-theme";

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "light",
  toggle: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

function getInitialTheme(): Theme {
  // The inline <script> in layout.tsx already applied the stored class before
  // hydration, so read the DOM first — this avoids a light-flash / perceived
  // "revert" on full page loads and route remounts.
  try {
    if (typeof document !== "undefined" && document.documentElement.classList.contains("dark")) {
      return "dark";
    }
    if (typeof window !== "undefined") {
      return window.localStorage.getItem(STORAGE_KEY) === "dark" ? "dark" : "light";
    }
    return "light";
  } catch {
    return "light";
  }
}

function applyTheme(next: Theme) {
  document.documentElement.classList.toggle("dark", next === "dark");
  // Keep native chrome (scrollbars, form controls) in sync with the theme.
  document.documentElement.style.colorScheme = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // ignore persistence failures
  }
}

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  // Single place that touches the DOM / storage: runs on mount and on every
  // theme change. The toggle below stays a pure state update, which keeps it
  // safe under StrictMode / concurrent rendering (side-effects must never
  // live inside a state updater).
  useEffect(() => {
    try {
      applyTheme(theme);
    } catch {
      // document unavailable (SSR) — nothing to sync
    }
  }, [theme]);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      // DOM class (set pre-hydration) wins over a stale stored value.
      const initial: Theme = document.documentElement.classList.contains("dark")
        ? "dark"
        : stored === "dark"
          ? "dark"
          : "light";
      setTheme(initial);
    } catch {
      // localStorage unavailable — stay on current theme
    }
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  return <ThemeContext.Provider value={{ theme, toggle }}>{children}</ThemeContext.Provider>;
}
