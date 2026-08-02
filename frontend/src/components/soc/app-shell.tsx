import { Link } from "@tanstack/react-router";
import {
  Activity,
  Globe2,
  LayoutDashboard,
  Moon,
  Radio,
  Settings as SettingsIcon,
  ShieldAlert,
  Sun,
  Table2,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useRealtimeIncidents } from "@/context/realtime-incidents";
import { useSocSettings } from "@/context/soc-settings";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/incidents", label: "Incidents", icon: Table2 },
  { to: "/threat-map", label: "Threat Map", icon: Globe2 },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
] as const;

function Clock() {
  const [now, setNow] = useState<string>("--:--:--");
  useEffect(() => {
    const tick = () =>
      setNow(new Date().toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" }));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <span className="hidden font-mono text-xs text-muted-foreground tabular md:inline">
      {now} <span className="text-muted-foreground/60">UTC</span>
    </span>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const { theme, setTheme, settings } = useSocSettings();
  const { connected, demoFeed, setDemoFeed } = useRealtimeIncidents();

  return (
    <div className="flex min-h-screen bg-background">
      <aside className="sticky top-0 flex h-screen w-14 shrink-0 flex-col border-r border-border bg-surface lg:w-52">
        <div className="flex h-12 items-center gap-2 border-b border-border px-3">
          <ShieldAlert className="size-5 shrink-0 text-primary" aria-hidden />
          <span className="hidden font-mono text-[13px] font-bold tracking-[0.16em] lg:inline">
            SENTINEL
          </span>
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 p-1.5" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact: to === "/" }}
              className="flex items-center gap-2.5 rounded-sm px-2.5 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground data-[status=active]:bg-accent data-[status=active]:text-accent-foreground data-[status=active]:font-medium"
            >
              <Icon className="size-4 shrink-0" aria-hidden />
              <span className="hidden lg:inline">{label}</span>
            </Link>
          ))}
        </nav>
        <div className="hidden border-t border-border p-3 lg:block">
          <p className="font-mono text-[10px] leading-relaxed tracking-wide text-muted-foreground uppercase">
            Pipeline
          </p>
          <p className="mt-1 font-mono text-[11px] break-all text-foreground/80">
            {settings.model}
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-12 items-center justify-between gap-3 border-b border-border bg-surface/95 px-3 backdrop-blur">
          <div className="flex min-w-0 items-center gap-2">
            <Activity className="size-4 text-primary" aria-hidden />
            <h1 className="truncate font-mono text-[12px] font-semibold tracking-[0.14em] uppercase">
              Security Operations Center
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Clock />
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-[10px] tracking-wider uppercase",
                settings.liveMode && connected
                  ? "border-sev-low/50 text-sev-low"
                  : "border-border-strong text-muted-foreground",
              )}
            >
              <span
                className={cn(
                  "size-1.5 rounded-full",
                  settings.liveMode && connected
                    ? "animate-pulse bg-sev-low"
                    : "bg-muted-foreground",
                )}
              />
              {settings.liveMode ? (connected ? "Live" : "Connecting") : "Paused"}
            </span>
            <button
              type="button"
              onClick={() => setDemoFeed(!demoFeed)}
              aria-pressed={demoFeed}
              title="Simulate the pipeline writing one incident every 2 seconds"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-colors",
                demoFeed
                  ? "border-primary/60 bg-accent text-accent-foreground"
                  : "border-border-strong text-muted-foreground hover:bg-secondary",
              )}
            >
              <Radio className="size-3" aria-hidden />
              <span className="hidden sm:inline">Demo feed</span>
            </button>
            <button
              type="button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              className="rounded-sm border border-border-strong p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            >
              {theme === "dark" ? (
                <Sun className="size-3.5" aria-hidden />
              ) : (
                <Moon className="size-3.5" aria-hidden />
              )}
            </button>
          </div>
        </header>
        <main className="min-w-0 flex-1 p-3">{children}</main>
      </div>
    </div>
  );
}
