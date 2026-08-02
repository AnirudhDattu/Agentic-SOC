# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Agentic SOC is a multi-agent, LLM-assisted Security Operations Center. It streams network-flow
records from the UNSW-NB15 dataset, classifies each as attack/benign with a scikit-learn Random
Forest, reasons about attack type/severity/recommendation via a real CrewAI `Agent`/`Task`/`Crew`
with tool-use (backed by Groq's free-tier LLM API, with a rule-based fallback grounded in the
dataset's real `attack_cat` ground truth), takes a simulated defensive action, and writes each
incident to a local SQLite file. A React/TanStack Start frontend in `frontend/` reads that same
file directly and renders a live dashboard. No cloud services anywhere in the stack.

This is a monorepo: Python backend at the root, frontend in `frontend/` (a separate npm project,
originally scaffolded by Lovable, later rewired off Supabase onto local SQLite).

## Commands

### Backend (root)

No build step, test runner, or linter configured for the Python side.

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

python infra/log_streamer.py     # samples one row from data/network_logs.csv every 2s → infra/live_log.json
python main.py                   # reads live_log.json, runs the 3-agent pipeline in a loop, writes infra/soc.db
```
Or launch both plus the frontend together: `python start_soc.py`.

Train / evaluate the detection model:
```bash
python models/train_model.py     # trains RandomForestClassifier, writes models/detector.pkl
python models/test_model.py      # evaluates detector.pkl against data/UNSW_NB15_testing-set.csv
```

Optional LLM reasoning: put a free key from `console.groq.com/keys` in `.env` as `GROQ_API_KEY`
(copy `.env.example`). Without it, `ReasoningAgent` runs in rule-based-only mode — this is a
normal, fully-supported operating mode, not a degraded error state.

### Frontend (`frontend/`)

```bash
cd frontend
npm install
npm run dev        # http://localhost:8080 (auto-bumps if taken)
```
No test/build commands are exercised in normal dev workflow here; `npm run build` / `npm run lint`
exist (see `frontend/package.json`) but aren't part of the usual iteration loop for this project.

## Architecture

```
data/network_logs.csv (UNSW-NB15) → infra/log_streamer.py → infra/live_log.json
                                                                    │
main.py loop, every 2s:                                            ▼
  d = DetectionAgent.run(log)        # RandomForest → {"status": "Attack"|"Benign", **log}
  r = ReasoningAgent.run(d)          # CrewAI Crew+Task+tool, Groq LLM or rule fallback
  a = ResponseAgent.run(d, r)        # action lookup + INSERT INTO infra/soc.db

frontend/ polls infra/soc.db every 2s via src/routes/api/incidents.ts (node:sqlite, no ORM)
```

- **`agents/detection_agent.py`** — loads `models/detector.pkl` at construction via
  `object.__setattr__` (bypasses Pydantic field restrictions on `crewai.Agent`). Filters the
  incoming log dict to `model.feature_names_in_` before predicting (tolerates extra/missing
  columns), then returns `{"status": status, **log_data}` — the raw features and the dataset's
  `attack_cat` ground-truth tag ride along in the dict for the next stage to use. If you ever see
  `ReasoningAgent` behaving as if it can't see `dur`/`spkts`/`attack_cat`, check this spread first.
- **`agents/reasoning_agent.py`** — this is the actually-agentic part. `ReasoningAgent` subclasses
  `crewai.Agent` and is invoked through a real `Crew([self], [task], Process.sequential).kickoff()`
  per event, with a `lookup_similar_incidents` tool (`agents/tools.py`, queries `infra/soc.db` for
  historical incidents of a candidate category) that the agent may call at its own discretion.
  LLM output is authoritative — not overridden by rules afterward. Rules only run as a fallback
  when there's no `GROQ_API_KEY`, the rate limiter (`infra/rate_limiter.py`) says no, or the call
  errors — and the fallback uses the dataset's real `attack_cat`, not a guessed numeric-threshold
  heuristic. `os.environ.setdefault("CREWAI_DISABLE_TELEMETRY"/"OTEL_SDK_DISABLED"/"CREWAI_TESTING", "true")`
  at the top of this file (and `main.py`) is load-bearing, not cleanup: without them, `crewai`
  either hangs trying to phone home on a sandboxed network, or blocks on an interactive
  "enable tracing?" / "view execution traces?" prompt for up to 20s per event.
- **`agents/response_agent.py`** — attack category → simulated defensive action lookup, writes
  one row per event to `infra/soc.db` (schema in `infra/db.py`), including which reasoning path
  produced it (`reasoning_method`: `"llm"` or `"rules"` — the frontend's pipeline visualization
  depends on this field being accurate).
- **`infra/log_streamer.py`** — the only producer of `infra/live_log.json`; writes atomically
  (`os.replace` after writing to a `.tmp` file, with a retry loop for Windows file-lock races)
  since `main.py` reads the same path concurrently. Carries `attack_cat` through from the CSV
  alongside the 19 ML features — that's ground truth, not derived by the model.
- **`frontend/src/lib/db.server.ts`** — opens `../infra/soc.db` directly via Node's built-in
  `node:sqlite` (no native module, ships with Node 22+). Both the Python writer and the Node
  reader use WAL mode; this is the entire "backend API" for the frontend, no separate server.
- **`frontend/src/components/soc/agent-pipeline.tsx`** — the live pipeline visualization on the
  Overview page. Driven entirely by real incidents arriving via the existing 2s poll (see
  `realtime-incidents.tsx`), not a scripted/fake animation.

### Feature set contract

The 19 ML feature names are duplicated across `infra/log_streamer.py`, `models/train_model.py`,
`models/test_model.py`, and implicitly required by whatever `models/detector.pkl` was trained on
(`detection_agent.py` reads `model.feature_names_in_` rather than hardcoding the list). If you
change the feature list in one place, retrain the model and update the others.

### Attack taxonomy

Both backend (`reasoning_agent.py`'s `VALID_ATTACK_TYPES`/`CATEGORY_SEVERITY`/
`CATEGORY_RECOMMENDATION`) and frontend (`frontend/src/lib/soc.ts`'s `ATTACK_TYPES`) must agree
on the category set: `Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor,
Shellcode, Worms, Unknown, None`. These are the UNSW-NB15 dataset's real categories — don't
reintroduce an invented taxonomy (an earlier version of this project used `DDoS`/`Port Scan`/
`Brute Force`/`Exploit`/`Data Exfiltration`, which don't match the dataset at all and produced a
chart that was ~90% "Unknown"). If you add a category on one side, add it on the other, and update
`agents/response_agent.py`'s action-mapping switch to match.

### State/data files (not source, mutated at runtime — gitignored)

- `infra/live_log.json` — current in-flight log record, overwritten every 2s.
- `infra/soc.db` (+ `-wal`/`-shm`) — the incident store; source of truth for the dashboard. Safe
  to delete to reset all history — both `infra/db.py` and `frontend/src/lib/db.server.ts` run
  `CREATE TABLE IF NOT EXISTS` on first connection.
- `models/detector.pkl` — trained model artifact; regenerate with `models/train_model.py` if
  missing or if the feature set changes.

## Known constraints worth remembering

- **Groq free tier**: 1,000 req/day for `llama-3.3-70b-versatile`. The rate limiter caps at 400/day
  with margin, since a single agentic `kickoff()` can cost more than one request if the tool gets
  used. Don't raise this cap without checking current Groq limits.
- **No account/secret creation on the user's behalf** — if a task needs a new API key or service
  signup, that's the user's step, not something to script around.
- **`frontend/` assumes it's a sibling of `infra/`** (i.e. run from inside this repo as
  `Agentic-SOC/frontend/`) — override via `SOC_DB_PATH` in `frontend/.env` if relocated.
