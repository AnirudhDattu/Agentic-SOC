import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AgentPipeline } from "@/components/soc/agent-pipeline";
import { AttackTypeChart, SeverityDonut, VolumeChart } from "@/components/soc/charts";
import { IncidentDrawer } from "@/components/soc/incident-drawer";
import { KpiTile } from "@/components/soc/kpi-tile";
import { LiveFeed } from "@/components/soc/live-feed";
import { EmptyState, ErrorState, Panel, SkeletonBlock } from "@/components/soc/panel";
import {
  attackTypeCounts,
  computeKpis,
  severityCounts,
  volumeSeries,
  withinRange,
  type Range,
} from "@/lib/aggregate";
import { recentIncidentsQuery } from "@/lib/incidents";
import { severityMeta, type Incident } from "@/lib/soc";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Live Overview — Sentinel SOC" },
      {
        name: "description",
        content:
          "Real-time KPIs, attack-type breakdown, severity distribution and a live incident feed for AI-detected network threats.",
      },
      { property: "og:title", content: "Live Overview — Sentinel SOC" },
      {
        property: "og:description",
        content: "Real-time KPIs and a live feed of AI-detected network threats.",
      },
    ],
  }),
  component: Overview,
});

const RANGES: Range[] = ["1h", "24h", "7d"];

function Overview() {
  const [range, setRange] = useState<Range>("24h");
  const [selected, setSelected] = useState<Incident | null>(null);
  const { data, isLoading, isError, error, refetch } = useQuery(recentIncidentsQuery(7));

  const incidents = data ?? [];
  const kpis = computeKpis(incidents);
  const ranged = withinRange(incidents, range);
  const types = attackTypeCounts(ranged);
  const severities = severityCounts(ranged);
  const volume = volumeSeries(incidents, range);
  const feed = incidents.slice(0, 60);

  if (isError) {
    return (
      <Panel title="Live overview">
        <ErrorState message={(error as Error).message} onRetry={() => void refetch()} />
      </Panel>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <KpiTile label="Total incidents" value={kpis.total.value} delta={kpis.total.delta} loading={isLoading} accent="primary" />
        <KpiTile label="Total attacks" value={kpis.attacks.value} delta={kpis.attacks.delta} loading={isLoading} />
        <KpiTile label="High severity" value={kpis.high.value} delta={kpis.high.delta} loading={isLoading} accent="high" />
        <KpiTile label="Medium severity" value={kpis.medium.value} delta={kpis.medium.delta} loading={isLoading} accent="medium" />
        <KpiTile label="Low severity" value={kpis.low.value} delta={kpis.low.delta} loading={isLoading} accent="low" />
      </div>

      <AgentPipeline latest={incidents.slice(0, 25)} />

      <div className="grid gap-3 xl:grid-cols-3">
        <Panel title="Attack-type frequency" subtitle={`window: ${range}`}>
          {isLoading ? (
            <SkeletonBlock className="h-[220px] w-full" />
          ) : types.every((t) => t.count === 0) ? (
            <EmptyState message="No attacks in this window" />
          ) : (
            <AttackTypeChart data={types} />
          )}
        </Panel>

        <Panel title="Severity distribution" subtitle={`window: ${range}`}>
          {isLoading ? (
            <SkeletonBlock className="h-[220px] w-full" />
          ) : !severities.length ? (
            <EmptyState message="No incidents in this window" />
          ) : (
            <>
              <SeverityDonut data={severities} height={180} />
              <ul className="mt-1 flex justify-center gap-4">
                {severities.map((s) => (
                  <li key={s.name} className="flex items-center gap-1.5 font-mono text-[11px]">
                    <span
                      className={cn(
                        "size-2 rounded-full",
                        severityMeta[s.name as keyof typeof severityMeta].dot,
                      )}
                    />
                    {s.name} <span className="text-muted-foreground tabular">{s.value}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Panel>

        <Panel
          title="Incident volume"
          subtitle="attacks vs benign"
          actions={
            <div className="flex gap-0.5 rounded-sm border border-border p-0.5">
              {RANGES.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRange(r)}
                  aria-pressed={range === r}
                  className={cn(
                    "rounded-[3px] px-2 py-0.5 font-mono text-[10px] tracking-wider uppercase transition-colors",
                    range === r
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-secondary",
                  )}
                >
                  {r}
                </button>
              ))}
            </div>
          }
        >
          {isLoading ? <SkeletonBlock className="h-[220px] w-full" /> : <VolumeChart data={volume} />}
        </Panel>
      </div>

      <Panel
        title="Live incident feed"
        subtitle="newest first · click a row for full detail"
        bodyClassName="p-0"
      >
        <div className="max-h-[560px] overflow-y-auto">
          <LiveFeed incidents={feed} loading={isLoading} onSelect={setSelected} />
        </div>
      </Panel>

      <IncidentDrawer incident={selected} onOpenChange={(open) => !open && setSelected(null)} />
    </div>
  );
}
