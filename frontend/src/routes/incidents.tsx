import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Download, Search, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { SeverityBadge, StatusBadge, WorkflowBadge } from "@/components/soc/badges";
import { IncidentDrawer } from "@/components/soc/incident-drawer";
import { EmptyState, ErrorState, Panel, SkeletonRows } from "@/components/soc/panel";
import { useRealtimeIncidents } from "@/context/realtime-incidents";
import { fetchAllFiltered, incidentPageQuery, type IncidentFilters } from "@/lib/incidents";
import {
  ATTACK_TYPES,
  SEVERITIES,
  STATUSES,
  WORKFLOW_STATUSES,
  downloadCsv,
  formatDateTime,
  toCsv,
  type Incident,
} from "@/lib/soc";
import { cn } from "@/lib/utils";

type Search = {
  types?: string | undefined;
  sev?: string | undefined;
  status?: string | undefined;
  wf?: string | undefined;
  from?: string | undefined;
  to?: string | undefined;
  q?: string | undefined;
  page?: number | undefined;
  sort?: string | undefined;
  dir?: "asc" | "desc" | undefined;
  size?: number | undefined;
};

const str = (v: unknown) => (typeof v === "string" && v !== "" ? v : undefined);

export const Route = createFileRoute("/incidents")({
  validateSearch: (raw: Record<string, unknown>): Search => ({
    types: str(raw["types"]),
    sev: str(raw["sev"]),
    status: str(raw["status"]),
    wf: str(raw["wf"]),
    from: str(raw["from"]),
    to: str(raw["to"]),
    q: str(raw["q"]),
    page: raw["page"] ? Number(raw["page"]) : undefined,
    sort: str(raw["sort"]),
    dir: raw["dir"] === "asc" ? "asc" : raw["dir"] === "desc" ? "desc" : undefined,
    size: raw["size"] ? Number(raw["size"]) : undefined,
  }),

  head: () => ({
    meta: [
      { title: "Incidents — Sentinel SOC" },
      {
        name: "description",
        content:
          "Filter, sort, paginate and export every AI-classified network incident, with a detail drawer for raw flow features.",
      },
      { property: "og:title", content: "Incidents — Sentinel SOC" },
      {
        property: "og:description",
        content: "Searchable, shareable incident table with CSV export and analyst workflow.",
      },
    ],
  }),
  component: IncidentsPage,
});

const split = (v?: string) => (v ? v.split(",").filter(Boolean) : []);

const COLUMNS: { key: string; label: string; sortable: boolean; className?: string }[] = [
  { key: "timestamp", label: "Timestamp", sortable: true, className: "w-44" },
  { key: "severity", label: "Severity", sortable: true, className: "w-20" },
  { key: "status", label: "Status", sortable: true, className: "w-20" },
  { key: "attack_type", label: "Attack type", sortable: true, className: "w-36" },
  { key: "source_ip", label: "Source", sortable: true, className: "w-40" },
  { key: "action_taken", label: "Action taken", sortable: false },
  { key: "workflow_status", label: "Workflow", sortable: true, className: "w-28" },
];

function MultiSelect({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly string[];
  value: string[];
  onChange: (next: string[]) => void;
}) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1 font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
        {label}
      </legend>
      <div className="flex flex-wrap gap-1">
        {options.map((opt) => {
          const active = value.includes(opt);
          return (
            <button
              key={opt}
              type="button"
              aria-pressed={active}
              onClick={() =>
                onChange(active ? value.filter((v) => v !== opt) : [...value, opt])
              }
              className={cn(
                "rounded-sm border px-2 py-0.5 text-[11px] transition-colors",
                active
                  ? "border-primary bg-accent text-accent-foreground"
                  : "border-border-strong text-muted-foreground hover:bg-secondary",
              )}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

function IncidentsPage() {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: "/incidents" });
  const { newIds } = useRealtimeIncidents();
  const [selected, setSelected] = useState<Incident | null>(null);
  const [term, setTerm] = useState(search.q ?? "");
  const [exporting, setExporting] = useState(false);

  useEffect(() => setTerm(search.q ?? ""), [search.q]);

  const filters: IncidentFilters = {
    attackTypes: split(search.types),
    severities: split(search.sev),
    statuses: split(search.status),
    workflow: split(search.wf),
    from: search.from,
    to: search.to,
    q: search.q,
  };
  const page = search.page && search.page > 0 ? search.page : 1;
  const pageSize = search.size ?? 25;
  const sortBy = search.sort ?? "timestamp";
  const sortDir = search.dir ?? "desc";

  const setSearch = (patch: Partial<Search>) =>
    navigate({ search: (prev: Search) => ({ ...prev, page: 1, ...patch }), replace: true });

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery(
    incidentPageQuery({ filters, page, pageSize, sortBy, sortDir }),
  );

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const activeFilterCount =
    filters.attackTypes.length +
    filters.severities.length +
    filters.statuses.length +
    filters.workflow.length +
    (filters.from ? 1 : 0) +
    (filters.to ? 1 : 0) +
    (filters.q ? 1 : 0);

  const onExport = async () => {
    setExporting(true);
    try {
      const all = await fetchAllFiltered(filters);
      if (!all.length) {
        toast.info("Nothing to export for the current filters");
        return;
      }
      downloadCsv(`incidents-${new Date().toISOString().slice(0, 19)}.csv`, toCsv(all));
      toast.success(`Exported ${all.length} incidents`);
    } catch (e) {
      toast.error("Export failed", { description: (e as Error).message });
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <Panel
        title="Filters"
        subtitle={`${activeFilterCount} active · shareable via URL`}
        actions={
          <>
            {activeFilterCount > 0 && (
              <button
                type="button"
                onClick={() =>
                  navigate({ search: {}, replace: true })
                }
                className="inline-flex items-center gap-1 rounded-sm border border-border-strong px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-secondary"
              >
                <X className="size-3" aria-hidden /> Clear
              </button>
            )}
            <button
              type="button"
              onClick={() => void onExport()}
              disabled={exporting}
              className="inline-flex items-center gap-1 rounded-sm bg-primary px-2 py-1 text-[11px] font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              <Download className="size-3" aria-hidden />
              {exporting ? "Exporting…" : "Export CSV"}
            </button>
          </>
        }
      >
        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-4">
          <MultiSelect
            label="Attack type"
            options={ATTACK_TYPES}
            value={filters.attackTypes}
            onChange={(next) => setSearch({ types: next.join(",") || undefined })}
          />
          <MultiSelect
            label="Severity"
            options={SEVERITIES}
            value={filters.severities}
            onChange={(next) => setSearch({ sev: next.join(",") || undefined })}
          />
          <MultiSelect
            label="Status"
            options={STATUSES}
            value={filters.statuses}
            onChange={(next) => setSearch({ status: next.join(",") || undefined })}
          />
          <MultiSelect
            label="Workflow"
            options={WORKFLOW_STATUSES}
            value={filters.workflow}
            onChange={(next) => setSearch({ wf: next.join(",") || undefined })}
          />
        </div>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
              From
            </span>
            <input
              type="date"
              value={search.from ?? ""}
              onChange={(e) => setSearch({ from: e.target.value || undefined })}
              className="rounded-sm border border-input bg-background px-2 py-1 font-mono text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
              To
            </span>
            <input
              type="date"
              value={search.to ?? ""}
              onChange={(e) => setSearch({ to: e.target.value || undefined })}
              className="rounded-sm border border-input bg-background px-2 py-1 font-mono text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
          </label>
          <form
            className="flex min-w-56 flex-1 flex-col gap-1"
            onSubmit={(e) => {
              e.preventDefault();
              setSearch({ q: term || undefined });
            }}
          >
            <span className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
              Search
            </span>
            <div className="flex items-center gap-1.5 rounded-sm border border-input bg-background px-2">
              <Search className="size-3.5 text-muted-foreground" aria-hidden />
              <input
                value={term}
                onChange={(e) => setTerm(e.target.value)}
                placeholder="IP, country, action, recommendation…"
                aria-label="Free-text search"
                className="w-full bg-transparent py-1 font-mono text-xs outline-none"
              />
            </div>
          </form>
        </div>
      </Panel>

      <Panel
        title="All incidents"
        subtitle={isLoading ? "loading…" : `${total.toLocaleString("en-US")} matching records`}
        bodyClassName="p-0"
        actions={
          isFetching && !isLoading ? (
            <span className="font-mono text-[10px] text-muted-foreground uppercase">syncing…</span>
          ) : null
        }
      >
        {isError ? (
          <ErrorState message={(error as Error).message} onRetry={() => void refetch()} />
        ) : isLoading ? (
          <div className="p-3">
            <SkeletonRows rows={10} height={30} />
          </div>
        ) : !rows.length ? (
          <EmptyState
            message="No incidents match these filters"
            hint="Try widening the date range or clearing a filter."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] border-collapse text-left">
              <thead>
                <tr className="border-b border-border bg-surface-2">
                  {COLUMNS.map((col) => {
                    const active = sortBy === col.key;
                    return (
                      <th
                        key={col.key}
                        scope="col"
                        className={cn(
                          "px-2.5 py-1.5 font-mono text-[10px] font-semibold tracking-[0.12em] text-muted-foreground uppercase",
                          col.className,
                        )}
                      >
                        {col.sortable ? (
                          <button
                            type="button"
                            onClick={() =>
                              setSearch({
                                sort: col.key,
                                dir: active && sortDir === "desc" ? "asc" : "desc",
                              })
                            }
                            className={cn(
                              "inline-flex items-center gap-1 transition-colors hover:text-foreground",
                              active && "text-foreground",
                            )}
                          >
                            {col.label}
                            {active && <span aria-hidden>{sortDir === "desc" ? "↓" : "↑"}</span>}
                          </button>
                        ) : (
                          col.label
                        )}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {rows.map((incident) => (
                  <tr
                    key={incident.id}
                    onClick={() => setSelected(incident)}
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setSelected(incident)}
                    className={cn(
                      "cursor-pointer border-b border-border/50 transition-colors hover:bg-secondary/70",
                      newIds.has(incident.id) && "row-flash",
                    )}
                  >
                    <td className="px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground tabular">
                      {formatDateTime(incident.timestamp)}
                    </td>
                    <td className="px-2.5 py-1.5">
                      <SeverityBadge severity={incident.severity} />
                    </td>
                    <td className="px-2.5 py-1.5">
                      <StatusBadge status={incident.status} />
                    </td>
                    <td className="px-2.5 py-1.5 text-xs font-medium">{incident.attack_type}</td>
                    <td className="px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground">
                      {incident.source_ip}
                      <span className="ml-1 text-muted-foreground/70">
                        {incident.source_country}
                      </span>
                    </td>
                    <td className="truncate px-2.5 py-1.5 text-xs text-muted-foreground">
                      {incident.action_taken}
                    </td>
                    <td className="px-2.5 py-1.5">
                      <WorkflowBadge status={incident.workflow_status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-3 py-2">
          <div className="flex items-center gap-2">
            <label className="font-mono text-[10px] tracking-wider text-muted-foreground uppercase">
              Rows
              <select
                value={pageSize}
                onChange={(e) => setSearch({ size: Number(e.target.value) })}
                className="ml-1.5 rounded-sm border border-input bg-background px-1.5 py-0.5 font-mono text-[11px] text-foreground"
              >
                {[25, 50, 100].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] text-muted-foreground tabular">
              Page {page} / {pages}
            </span>
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => navigate({ search: (p: Search) => ({ ...p, page: page - 1 }), replace: true })}
              className="rounded-sm border border-border-strong px-2 py-1 text-[11px] transition-colors hover:bg-secondary disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              disabled={page >= pages}
              onClick={() => navigate({ search: (p: Search) => ({ ...p, page: page + 1 }), replace: true })}
              className="rounded-sm border border-border-strong px-2 py-1 text-[11px] transition-colors hover:bg-secondary disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </Panel>

      <IncidentDrawer incident={selected} onOpenChange={(open) => !open && setSelected(null)} />
    </div>
  );
}
