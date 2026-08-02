# Sentinel — SOC Dashboard

React + [TanStack Start](https://tanstack.com/start) frontend for the Agentic SOC pipeline
(see the [repo root README](../README.md) for the full system). Originally scaffolded with
[Lovable](https://lovable.dev), since rewired off Supabase onto a local SQLite file shared
directly with the Python backend.

## Run it

```bash
npm install
npm run dev
```

Open **http://localhost:8080** (auto-bumps to 8081/8082 if that port's taken). Requires the
Python pipeline (`../infra/log_streamer.py` + `../main.py`) to be running to see live data —
without it, the dashboard is just empty (or toggle **Demo feed** in the header to synthesize
incidents client-side for a UI preview).

## How it talks to the backend

No API server, no cloud database. `src/lib/db.server.ts` opens `../infra/soc.db` directly using
Node's built-in `node:sqlite` — the same SQLite file the Python `ResponseAgent` writes rows into.
`src/routes/api/incidents.ts` is a thin TanStack Start server route (GET/POST/PATCH) over that
file for the browser to call. There's no websocket/Realtime service either — `RealtimeIncidents
Provider` polls every 2 seconds (matching the pipeline's own emission rate) and diffs incident
IDs to drive the live-feed animations and toasts.

Override the DB location with `SOC_DB_PATH` in `.env` if you're not running this next to the
backend as `Agentic-SOC/frontend/`.

## Pages

- `/` — Overview: KPIs, charts, live agent-pipeline visualization, live incident feed
- `/incidents` — full filterable/paginated table, CSV export, per-incident drawer
- `/threat-map` — simulated geographic origins
- `/settings` — live-mode, alert thresholds, theme, demo feed

## Stack

React 19, TanStack Router/Start, Tailwind v4, shadcn/ui (Radix), Recharts, `motion` for the
pipeline animation, `node:sqlite` for storage — no ORM, no external services.
