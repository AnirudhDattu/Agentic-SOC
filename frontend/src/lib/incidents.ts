import { queryOptions } from "@tanstack/react-query";
import type { Incident } from "./soc";

export type IncidentFilters = {
  attackTypes: string[];
  severities: string[];
  statuses: string[];
  workflow: string[];
  from?: string;
  to?: string;
  q?: string;
};

export const emptyFilters: IncidentFilters = {
  attackTypes: [],
  severities: [],
  statuses: [],
  workflow: [],
};

export async function fetchRecentIncidents(days = 7): Promise<Incident[]> {
  const res = await fetch(`/api/incidents?mode=recent&days=${days}`);
  if (!res.ok) throw new Error(`Failed to load incidents: ${res.status}`);
  return (await res.json()) as Incident[];
}

export const recentIncidentsQuery = (days = 7) =>
  queryOptions({
    queryKey: ["incidents", "recent", days],
    queryFn: () => fetchRecentIncidents(days),
    staleTime: 15_000,
  });

type PageArgs = {
  filters: IncidentFilters;
  page: number;
  pageSize: number;
  sortBy: string;
  sortDir: "asc" | "desc";
};

export type IncidentPage = { rows: Incident[]; total: number };

function buildPageParams({ filters, page, pageSize, sortBy, sortDir }: PageArgs): URLSearchParams {
  const params = new URLSearchParams();
  params.set("page", String(page));
  params.set("pageSize", String(pageSize));
  params.set("sortBy", sortBy);
  params.set("sortDir", sortDir);
  for (const t of filters.attackTypes) params.append("attackType", t);
  for (const s of filters.severities) params.append("severity", s);
  for (const s of filters.statuses) params.append("status", s);
  for (const w of filters.workflow) params.append("workflow", w);
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  if (filters.q) params.set("q", filters.q);
  return params;
}

export async function fetchIncidentPage(args: PageArgs): Promise<IncidentPage> {
  const params = buildPageParams(args);
  const res = await fetch(`/api/incidents?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to load incidents: ${res.status}`);
  return (await res.json()) as IncidentPage;
}

export const incidentPageQuery = (args: PageArgs) =>
  queryOptions({
    queryKey: ["incidents", "page", args],
    queryFn: () => fetchIncidentPage(args),
    staleTime: 5_000,
  });

export async function fetchAllFiltered(filters: IncidentFilters): Promise<Incident[]> {
  const { rows } = await fetchIncidentPage({
    filters,
    page: 1,
    pageSize: 5000,
    sortBy: "timestamp",
    sortDir: "desc",
  });
  return rows;
}

export async function updateIncident(
  id: string,
  patch: { workflow_status?: string; analyst_note?: string | null },
) {
  const res = await fetch("/api/incidents", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, ...patch }),
  });
  if (!res.ok) throw new Error(`Failed to update incident: ${res.status}`);
  return (await res.json()) as Incident;
}
