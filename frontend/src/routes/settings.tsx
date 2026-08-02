import { createFileRoute } from "@tanstack/react-router";
import { Panel } from "@/components/soc/panel";
import { useRealtimeIncidents } from "@/context/realtime-incidents";
import { useSocSettings } from "@/context/soc-settings";
import { SEVERITIES, type Severity } from "@/lib/soc";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — Sentinel SOC" },
      {
        name: "description",
        content:
          "Configure live monitoring, alert thresholds, table density, theme, and the demo incident feed.",
      },
      { property: "og:title", content: "Settings — Sentinel SOC" },
      {
        property: "og:description",
        content: "Console preferences for the Sentinel security operations dashboard.",
      },
    ],
  }),
  component: SettingsPage,
});

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full border transition-colors",
        checked ? "border-accent bg-accent/30" : "border-border bg-secondary",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 h-3.5 w-3.5 rounded-full transition-all",
          checked ? "left-[18px] bg-accent" : "left-0.5 bg-muted-foreground",
        )}
      />
    </button>
  );
}

function Row({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border/60 px-3 py-3 last:border-b-0">
      <div className="min-w-0">
        <p className="text-xs font-semibold">{title}</p>
        <p className="mt-0.5 text-[11px] text-muted-foreground">{description}</p>
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function SettingsPage() {
  const { settings, update, theme, setTheme } = useSocSettings();
  const { demoFeed, setDemoFeed, connected } = useRealtimeIncidents();

  const toggleSeverity = (sev: Severity) => {
    const has = settings.alertSeverities.includes(sev);
    update({
      alertSeverities: has
        ? settings.alertSeverities.filter((s) => s !== sev)
        : [...settings.alertSeverities, sev],
    });
  };

  return (
    <div className="grid max-w-4xl gap-3">
      <Panel title="Monitoring" bodyClassName="p-0">
        <Row
          title="Live mode"
          description="Stream new detections into the overview and feed as they arrive."
        >
          <Toggle
            label="Live mode"
            checked={settings.liveMode}
            onChange={(v) => update({ liveMode: v })}
          />
        </Row>
        <Row
          title="Demo feed"
          description="Synthesize realistic incidents locally when no detector is publishing."
        >
          <Toggle label="Demo feed" checked={demoFeed} onChange={setDemoFeed} />
        </Row>
        <Row title="Realtime channel" description="Current connection to the incidents stream.">
          <span
            className={cn(
              "font-mono text-[11px] tracking-wider uppercase",
              connected ? "text-sev-low" : "text-sev-medium",
            )}
          >
            {connected ? "connected" : "offline"}
          </span>
        </Row>
      </Panel>

      <Panel title="Alerting" bodyClassName="p-0">
        <Row
          title="Toast severities"
          description="Raise a desktop toast when an incident of these severities arrives."
        >
          <div className="flex gap-1">
            {SEVERITIES.map((s) => {
              const active = settings.alertSeverities.includes(s);
              return (
                <button
                  key={s}
                  type="button"
                  aria-pressed={active}
                  onClick={() => toggleSeverity(s)}
                  className={cn(
                    "rounded-sm border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-colors",
                    active
                      ? "border-accent bg-accent/20 text-foreground"
                      : "border-border text-muted-foreground hover:bg-secondary",
                  )}
                >
                  {s}
                </button>
              );
            })}
          </div>
        </Row>
        <Row title="Detection model" description="Label shown across the console for provenance.">
          <input
            value={settings.model}
            onChange={(e) => update({ model: e.target.value })}
            aria-label="Detection model"
            className="w-56 rounded-sm border border-border bg-input px-2 py-1 font-mono text-[11px] outline-none focus:border-accent"
          />
        </Row>
      </Panel>

      <Panel title="Appearance" bodyClassName="p-0">
        <Row title="Theme" description="Dark is the default for low-light operations rooms.">
          <div className="flex gap-1">
            {(["dark", "light"] as const).map((t) => (
              <button
                key={t}
                type="button"
                aria-pressed={theme === t}
                onClick={() => setTheme(t)}
                className={cn(
                  "rounded-sm border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-colors",
                  theme === t
                    ? "border-accent bg-accent/20 text-foreground"
                    : "border-border text-muted-foreground hover:bg-secondary",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </Row>
        <Row title="Table density" description="Compact fits more rows per screen in the grid.">
          <div className="flex gap-1">
            {(["comfortable", "compact"] as const).map((d) => (
              <button
                key={d}
                type="button"
                aria-pressed={settings.density === d}
                onClick={() => update({ density: d })}
                className={cn(
                  "rounded-sm border px-2 py-1 font-mono text-[10px] tracking-wider uppercase transition-colors",
                  settings.density === d
                    ? "border-accent bg-accent/20 text-foreground"
                    : "border-border text-muted-foreground hover:bg-secondary",
                )}
              >
                {d}
              </button>
            ))}
          </div>
        </Row>
      </Panel>

      <Panel title="Data pipeline">
        <p className="text-xs text-muted-foreground">
          Incidents are read from a local SQLite file (
          <code className="font-mono text-[11px] text-foreground">Agentic-SOC/infra/soc.db</code>)
          that the Python pipeline writes to directly — no cloud service involved. The UI polls
          for new rows every 2s. Each row has{" "}
          <code className="font-mono text-[11px] text-foreground">status</code>,{" "}
          <code className="font-mono text-[11px] text-foreground">attack_type</code>,{" "}
          <code className="font-mono text-[11px] text-foreground">severity</code>,{" "}
          <code className="font-mono text-[11px] text-foreground">recommendation</code>, the model
          input vector in{" "}
          <code className="font-mono text-[11px] text-foreground">raw_features</code>, and{" "}
          <code className="font-mono text-[11px] text-foreground">reasoning_method</code> (
          <code className="font-mono text-[11px] text-foreground">llm</code> or{" "}
          <code className="font-mono text-[11px] text-foreground">rules</code>).
        </p>
      </Panel>
    </div>
  );
}
