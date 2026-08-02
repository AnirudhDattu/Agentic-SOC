import { useRealtimeIncidents } from "@/context/realtime-incidents";
import { formatTime, relativeTime, type Incident } from "@/lib/soc";
import { cn } from "@/lib/utils";
import { SeverityBadge, StatusBadge, WorkflowBadge } from "./badges";
import { EmptyState, SkeletonRows } from "./panel";

export function LiveFeed({
  incidents,
  loading,
  onSelect,
}: {
  incidents: Incident[];
  loading?: boolean;
  onSelect: (incident: Incident) => void;
}) {
  const { newIds } = useRealtimeIncidents();

  if (loading) return <SkeletonRows rows={8} height={34} />;
  if (!incidents.length)
    return (
      <EmptyState
        message="No incidents yet"
        hint="Waiting for the detection pipeline to write its first record."
      />
    );

  return (
    <ul className="divide-y divide-border/60">
      {incidents.map((incident) => (
        <li key={incident.id}>
          <button
            type="button"
            onClick={() => onSelect(incident)}
            className={cn(
              "flex w-full items-center gap-2.5 px-2 py-2 text-left transition-colors hover:bg-secondary/70",
              newIds.has(incident.id) && "row-flash",
            )}
          >
            <span className="w-16 shrink-0 font-mono text-[11px] text-muted-foreground tabular">
              {formatTime(incident.timestamp)}
            </span>
            <SeverityBadge severity={incident.severity} className="w-16 shrink-0 justify-center" />
            <span className="w-32 shrink-0 truncate text-xs font-medium">
              {incident.attack_type}
            </span>
            <span className="hidden w-32 shrink-0 truncate font-mono text-[11px] text-muted-foreground md:inline">
              {incident.source_ip}
            </span>
            <span className="hidden min-w-0 flex-1 truncate text-[11px] text-muted-foreground lg:inline">
              {incident.action_taken}
            </span>
            <span className="ml-auto flex shrink-0 items-center gap-2">
              <span className="hidden sm:inline">
                <StatusBadge status={incident.status} />
              </span>
              <WorkflowBadge status={incident.workflow_status} />
              <span className="hidden w-14 text-right font-mono text-[10px] text-muted-foreground xl:inline">
                {relativeTime(incident.timestamp)}
              </span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
