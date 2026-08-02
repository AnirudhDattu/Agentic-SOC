import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Severity } from "@/lib/soc";

export type SocSettings = {
  liveMode: boolean;
  alertSeverities: Severity[];
  model: string;
  density: "comfortable" | "compact";
};

const DEFAULTS: SocSettings = {
  liveMode: true,
  alertSeverities: ["High"],
  model: "random-forest + groq/llama-3.3-70b",
  density: "comfortable",
};

const STORAGE_KEY = "sentinel.settings";
const THEME_KEY = "sentinel.theme";

type Ctx = {
  settings: SocSettings;
  update: (patch: Partial<SocSettings>) => void;
  theme: "dark" | "light";
  setTheme: (t: "dark" | "light") => void;
  hydrated: boolean;
};

const SettingsContext = createContext<Ctx | null>(null);

export function SocSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<SocSettings>(DEFAULTS);
  const [theme, setThemeState] = useState<"dark" | "light">("dark");
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setSettings({ ...DEFAULTS, ...JSON.parse(raw) });
      const storedTheme = localStorage.getItem(THEME_KEY);
      if (storedTheme === "light" || storedTheme === "dark") setThemeState(storedTheme);
    } catch {
      /* ignore corrupt storage */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    root.style.colorScheme = theme;
  }, [theme]);

  const update = useCallback((patch: Partial<SocSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const setTheme = useCallback((t: "dark" | "light") => {
    setThemeState(t);
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(
    () => ({ settings, update, theme, setTheme, hydrated }),
    [settings, update, theme, setTheme, hydrated],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSocSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSocSettings must be used inside SocSettingsProvider");
  return ctx;
}
