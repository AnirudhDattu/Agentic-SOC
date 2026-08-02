import { createFileRoute } from "@tanstack/react-router";
import { randomUUID } from "node:crypto";
import { getDb, type IncidentRow } from "@/lib/db.server";

const SORTABLE = new Set(["timestamp", "severity", "attack_type", "status", "workflow_status"]);

function toIncident(row: IncidentRow) {
  let raw_features: Record<string, number | string | null> = {};
  try {
    raw_features = JSON.parse(row.raw_features);
  } catch {
    raw_features = {};
  }
  return { ...row, raw_features };
}

export const Route = createFileRoute("/api/incidents")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const url = new URL(request.url);
        const db = getDb();

        if (url.searchParams.get("mode") === "recent") {
          const days = Number(url.searchParams.get("days") ?? "7");
          const since = new Date(Date.now() - days * 86400_000).toISOString();
          const rows = db
            .prepare(
              "SELECT * FROM incidents WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 3000",
            )
            .all(since) as unknown as IncidentRow[];
          return Response.json(rows.map(toIncident));
        }

        const page = Math.max(1, Number(url.searchParams.get("page") ?? "1"));
        const pageSize = Math.max(1, Number(url.searchParams.get("pageSize") ?? "50"));
        const sortByRaw = url.searchParams.get("sortBy") ?? "timestamp";
        const sortBy = SORTABLE.has(sortByRaw) ? sortByRaw : "timestamp";
        const sortDir = url.searchParams.get("sortDir") === "asc" ? "ASC" : "DESC";

        const attackTypes = url.searchParams.getAll("attackType");
        const severities = url.searchParams.getAll("severity");
        const statuses = url.searchParams.getAll("status");
        const workflow = url.searchParams.getAll("workflow");
        const from = url.searchParams.get("from");
        const to = url.searchParams.get("to");
        const q = url.searchParams.get("q");

        const where: string[] = [];
        const params: (string | number)[] = [];

        const inClause = (col: string, values: string[]) => {
          if (!values.length) return;
          where.push(`${col} IN (${values.map(() => "?").join(",")})`);
          params.push(...values);
        };
        inClause("attack_type", attackTypes);
        inClause("severity", severities);
        inClause("status", statuses);
        inClause("workflow_status", workflow);

        if (from) {
          where.push("timestamp >= ?");
          params.push(new Date(from).toISOString());
        }
        if (to) {
          const t = new Date(to);
          t.setHours(23, 59, 59, 999);
          where.push("timestamp <= ?");
          params.push(t.toISOString());
        }
        if (q) {
          const term = `%${q}%`;
          where.push(
            "(source_ip LIKE ? OR source_country LIKE ? OR attack_type LIKE ? OR action_taken LIKE ? OR recommendation LIKE ? OR analyst_note LIKE ?)",
          );
          params.push(term, term, term, term, term, term);
        }

        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
        const total = db.prepare(`SELECT COUNT(*) as c FROM incidents ${whereSql}`).get(...params) as {
          c: number;
        };
        const rows = db
          .prepare(
            `SELECT * FROM incidents ${whereSql} ORDER BY ${sortBy} ${sortDir} LIMIT ? OFFSET ?`,
          )
          .all(...params, pageSize, (page - 1) * pageSize) as unknown as IncidentRow[];

        return Response.json({ rows: rows.map(toIncident), total: total.c });
      },

      POST: async ({ request }) => {
        const body = (await request.json()) as Partial<IncidentRow> & {
          raw_features?: Record<string, unknown>;
        };
        const db = getDb();
        const now = new Date().toISOString();
        const row: IncidentRow = {
          id: randomUUID(),
          timestamp: body.timestamp ?? now,
          status: body.status ?? "Benign",
          attack_type: body.attack_type ?? "None",
          severity: body.severity ?? "Low",
          recommendation: body.recommendation ?? "",
          action_taken: body.action_taken ?? "Logged incident",
          raw_features: JSON.stringify(body.raw_features ?? {}),
          source_ip: body.source_ip ?? null,
          source_country: body.source_country ?? null,
          source_lat: body.source_lat ?? null,
          source_lon: body.source_lon ?? null,
          workflow_status: body.workflow_status ?? "New",
          analyst_note: body.analyst_note ?? null,
          reasoning_method: "rules",
          created_at: now,
          updated_at: now,
        };

        db.prepare(
          `INSERT INTO incidents (
            id, timestamp, status, attack_type, severity, recommendation,
            action_taken, raw_features, source_ip, source_country,
            source_lat, source_lon, workflow_status, analyst_note,
            reasoning_method, created_at, updated_at
          ) VALUES (
            @id, @timestamp, @status, @attack_type, @severity, @recommendation,
            @action_taken, @raw_features, @source_ip, @source_country,
            @source_lat, @source_lon, @workflow_status, @analyst_note,
            @reasoning_method, @created_at, @updated_at
          )`,
        ).run(row);

        return Response.json(toIncident(row));
      },

      PATCH: async ({ request }) => {
        const body = (await request.json()) as {
          id?: string;
          workflow_status?: string;
          analyst_note?: string | null;
        };
        if (!body.id) {
          return Response.json({ error: "id is required" }, { status: 400 });
        }
        const db = getDb();
        const now = new Date().toISOString();

        const sets: string[] = [];
        const params: Record<string, string | null> = { id: body.id, updated_at: now };
        if (body.workflow_status !== undefined) {
          sets.push("workflow_status = @workflow_status");
          params["workflow_status"] = body.workflow_status;
        }
        if (body.analyst_note !== undefined) {
          sets.push("analyst_note = @analyst_note");
          params["analyst_note"] = body.analyst_note;
        }
        if (!sets.length) {
          return Response.json({ error: "no fields to update" }, { status: 400 });
        }

        db.prepare(`UPDATE incidents SET ${sets.join(", ")}, updated_at = @updated_at WHERE id = @id`).run(
          params,
        );
        const row = db.prepare("SELECT * FROM incidents WHERE id = ?").get(body.id) as
          | IncidentRow
          | undefined;
        if (!row) return Response.json({ error: "not found" }, { status: 404 });
        return Response.json(toIncident(row));
      },
    },
  },
});
