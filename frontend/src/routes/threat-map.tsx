import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { feature } from "topojson-client";
import worldTopo from "world-atlas/countries-110m.json";
import { IncidentDrawer } from "@/components/soc/incident-drawer";
import { SeverityBadge } from "@/components/soc/badges";
import { EmptyState, ErrorState, Panel, SkeletonBlock } from "@/components/soc/panel";
import { recentIncidentsQuery } from "@/lib/incidents";
import { isSeverity, relativeTime, severityMeta, type Incident } from "@/lib/soc";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/threat-map")({
  head: () => ({
    meta: [
      { title: "Threat Map — Sentinel SOC" },
      {
        name: "description",
        content:
          "Geographic view of attack sources plotted by severity, with a live list of the most recent origins.",
      },
      { property: "og:title", content: "Threat Map — Sentinel SOC" },
      {
        property: "og:description",
        content: "Attack origins plotted on a world map and sized by severity.",
      },
    ],
  }),
  component: ThreatMapPage,
});

const W = 1000;
const H = 500;

const project = (lon: number, lat: number): [number, number] => [
  ((lon + 180) / 360) * W,
  ((90 - lat) / 180) * H,
];

type Ring = [number, number][];

function ringsToPath(rings: Ring[]): string {
  return rings
    .map((ring) => {
      const pts = ring.map(([lon, lat]) => project(lon, lat));
      return `M${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join("L")}Z`;
    })
    .join("");
}

function useWorldPaths() {
  return useMemo(() => {
    const topo = worldTopo as unknown as Parameters<typeof feature>[0];
    const collection = feature(
      topo,
      (topo as unknown as { objects: Record<string, never> }).objects["countries"]!,
    ) as unknown as {
      features: { geometry: { type: string; coordinates: unknown } }[];
    };
    return collection.features.map((f) => {
      const rings =
        f.geometry.type === "Polygon"
          ? (f.geometry.coordinates as Ring[])
          : (f.geometry.coordinates as Ring[][]).flat();
      return ringsToPath(rings);
    });
  }, []);
}

function ThreatMapPage() {
  const [selected, setSelected] = useState<Incident | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string | null>(null);
  const paths = useWorldPaths();
  const { data, isLoading, isError, error, refetch } = useQuery(recentIncidentsQuery(7));

  const attacks = (data ?? []).filter(
    (i) =>
      i.status === "Attack" &&
      i.source_lat != null &&
      i.source_lon != null &&
      (!severityFilter || i.severity === severityFilter),
  );
  const markers = attacks.slice(0, 300);

  const byCountry = useMemo(() => {
    const map = new Map<string, { country: string; count: number; high: number }>();
    for (const i of attacks) {
      const key = i.source_country ?? "Unknown";
      const entry = map.get(key) ?? { country: key, count: 0, high: 0 };
      entry.count += 1;
      if (i.severity === "High") entry.high += 1;
      map.set(key, entry);
    }
    return [...map.values()].sort((a, b) => b.count - a.count);
  }, [attacks]);

  if (isError) {
    return (
      <Panel title="Threat map">
        <ErrorState message={(error as Error).message} onRetry={() => void refetch()} />
      </Panel>
    );
  }

  return (
    <div className="grid gap-3 xl:grid-cols-[1fr_300px]">
      <Panel
        title="Global attack origins"
        subtitle="simulated coordinates · last 7 days"
        bodyClassName="p-0"
        actions={
          <div className="flex gap-0.5 rounded-sm border border-border p-0.5">
            {[null, "High", "Medium", "Low"].map((s) => (
              <button
                key={s ?? "all"}
                type="button"
                onClick={() => setSeverityFilter(s)}
                aria-pressed={severityFilter === s}
                className={cn(
                  "rounded-[3px] px-2 py-0.5 font-mono text-[10px] tracking-wider uppercase transition-colors",
                  severityFilter === s
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-secondary",
                )}
              >
                {s ?? "All"}
              </button>
            ))}
          </div>
        }
      >
        {isLoading ? (
          <div className="p-3">
            <SkeletonBlock className="h-[420px] w-full" />
          </div>
        ) : !markers.length ? (
          <EmptyState message="No geolocated attacks in this window" />
        ) : (
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="grid-bg h-auto w-full"
            role="img"
            aria-label={`World map with ${markers.length} plotted attack sources`}
          >
            {paths.map((d, i) => (
              <path
                key={i}
                d={d}
                fill="var(--surface-2)"
                stroke="var(--border-strong)"
                strokeWidth={0.5}
              />
            ))}
            {markers.map((m) => {
              const [x, y] = project(m.source_lon!, m.source_lat!);
              const color = isSeverity(m.severity)
                ? severityMeta[m.severity].chartVar
                : "var(--muted-foreground)";
              const r = m.severity === "High" ? 5 : m.severity === "Medium" ? 3.5 : 2.5;
              return (
                <g key={m.id} onClick={() => setSelected(m)} className="cursor-pointer">
                  <title>{`${m.attack_type} · ${m.severity} · ${m.source_country} · ${m.source_ip}`}</title>
                  {m.severity === "High" && (
                    <circle cx={x} cy={y} r={r} fill={color} opacity={0.5} className="pulse-ring" />
                  )}
                  <circle cx={x} cy={y} r={r} fill={color} fillOpacity={0.75} stroke={color} />
                </g>
              );
            })}
          </svg>
        )}
      </Panel>

      <div className="flex flex-col gap-3">
        <Panel title="Top source countries" bodyClassName="p-0">
          {isLoading ? (
            <div className="p-3">
              <SkeletonBlock className="h-40 w-full" />
            </div>
          ) : !byCountry.length ? (
            <EmptyState message="No sources" />
          ) : (
            <ul className="divide-y divide-border/60">
              {byCountry.slice(0, 10).map((c) => (
                <li key={c.country} className="flex items-center justify-between px-3 py-1.5">
                  <span className="truncate text-xs">{c.country}</span>
                  <span className="flex items-center gap-2 font-mono text-[11px] tabular">
                    <span className="text-sev-high">{c.high}</span>
                    <span className="text-muted-foreground">/ {c.count}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Recent origins" bodyClassName="p-0">
          {isLoading ? (
            <div className="p-3">
              <SkeletonBlock className="h-40 w-full" />
            </div>
          ) : (
            <ul className="max-h-[320px] divide-y divide-border/60 overflow-y-auto">
              {markers.slice(0, 40).map((m) => (
                <li key={m.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(m)}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-secondary/70"
                  >
                    <SeverityBadge severity={m.severity} />
                    <span className="min-w-0 flex-1 truncate font-mono text-[11px]">
                      {m.source_ip}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {relativeTime(m.timestamp)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <IncidentDrawer incident={selected} onOpenChange={(open) => !open && setSelected(null)} />
    </div>
  );
}
