# 🧠 Agentic SOC — LLM-Powered Cyber Defense

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.201.1-green.svg)](https://www.crewai.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-orange.svg)](https://groq.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TanStack%20Start-61dafb.svg)](https://tanstack.com/start)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A multi-agent Security Operations Center that classifies real network traffic (UNSW-NB15),
> reasons about it with a live LLM through genuine agent orchestration, and streams the result
> to a live dashboard — entirely local, no cloud services required.**

---

## What this actually is

Most "agentic AI" demos wire an LLM into a single `if/else` and call it a day. This one uses
[CrewAI](https://www.crewai.com/)'s real `Crew` / `Task` / tool-use engine: the reasoning agent
decides for itself whether to consult a tool (a local incident-history lookup) before answering,
and the LLM's decision is what actually gets stored — not silently overridden by hardcoded rules
afterward.

Everything runs on your machine. The backend writes to a local SQLite file; the frontend reads
that same file directly. No Supabase, no hosted database, no telemetry. The only network calls
are to Groq's free-tier API for LLM reasoning (optional — the pipeline runs fine without it,
falling back to rule-based classification).

## Architecture

```
data/network_logs.csv (UNSW-NB15, 82k labeled flows)
        │
        ▼
infra/log_streamer.py ──▶ infra/live_log.json (1 sampled row / 2s)
        │
        ▼
main.py loop:
  DetectionAgent   → RandomForestClassifier → Attack / Benign
  ReasoningAgent   → CrewAI Crew+Task, Groq llama-3.3-70b, tool-use over
                     incident history → attack_type / severity / recommendation
                     (falls back to rule-based classification using the
                     dataset's real attack_cat ground truth if no LLM key,
                     rate-limited, or the call fails)
  ResponseAgent    → maps category → simulated defensive action,
                     writes one row to infra/soc.db (SQLite)
        │
        ▼
frontend/ (React + TanStack Start) ──▶ polls infra/soc.db every 2s ──▶ dashboard
```

### Why the reasoning stage is genuinely agentic

- Built on CrewAI's actual `Agent` / `Task` / `Crew(process=sequential).kickoff()` engine —
  not a hand-rolled script that happens to import the class.
- Gives the agent a **tool** (`lookup_similar_incidents`) that queries the local incident
  history. The agent decides on its own whether calling it would help.
- LLM output is the final answer, not decoration papered over by hardcoded thresholds
  afterward (a bug the original single-agent version had — fixed here).
- Rate-limited (daily cap + minimum interval) so it can run unattended on a free API tier
  without ever exceeding it, degrading gracefully to rule-based reasoning instead of failing.

### Why the data is real, not guessed

The UNSW-NB15 dataset includes a genuine ground-truth `attack_cat` column (Generic, Exploits,
Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms, Normal). That signal is
carried through the whole pipeline and given to the reasoning agent as context — instead of
asking a model to reinvent a classification from 19 abstract numbers with no ground truth at
all, which is what earlier versions of this project did.

## Repo layout

```
Agentic-SOC/
├── agents/                  # DetectionAgent, ReasoningAgent, ResponseAgent, CrewAI tools
├── infra/                   # log_streamer.py, local SQLite (db.py), rate limiter
├── models/                  # train_model.py / test_model.py, detector.pkl
├── data/                    # UNSW-NB15 CSVs
├── main.py                  # the pipeline loop
├── frontend/                # React + TanStack Start dashboard (see frontend/README.md)
└── requirements.txt
```

## Quick start

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Optional — enable real LLM reasoning (free, no card required):
1. Sign up at [console.groq.com/keys](https://console.groq.com/keys) (email only)
2. Copy `.env.example` to `.env` and paste your key into `GROQ_API_KEY`

Without a key, the pipeline runs fine — `ReasoningAgent` falls back to rule-based classification
using the dataset's ground-truth `attack_cat`.

Run the pipeline (two terminals):
```bash
python infra/log_streamer.py     # feeds sampled network flows, one every 2s
python main.py                   # detection → reasoning → response, writes infra/soc.db
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:8080**. It reads `../infra/soc.db` directly (no API server, no cloud —
just a Node server route reading a local SQLite file) and polls every 2 seconds for new rows.

### 3. Train the detection model (optional — a trained model is already included)

```bash
python models/train_model.py     # trains RandomForestClassifier → models/detector.pkl
python models/test_model.py      # evaluates against the held-out test set
```

## The dashboard

- **Overview** — live KPI tiles, attack-type frequency, severity distribution, incident volume
  over time, a live incident feed, and a **live agent pipeline visualization**: an animated
  diagram showing each real incident traveling through Dataset → Streamer → Detection → Reasoning
  → Response as it happens, with a running log tagged `LLM` or `RULES` per decision.
- **Incidents** — full sortable/filterable/paginated table, CSV export, per-incident detail
  drawer with the raw 19-feature vector, workflow status (New/Investigating/Resolved) with
  analyst notes.
- **Threat Map** — simulated geographic origins, sized/colored by severity.
- **Settings** — live-mode toggle, alert severity thresholds, theme, table density, and a demo
  feed toggle that synthesizes incidents client-side when nothing's actively piping in.

## Technology stack

| Layer | Tech |
|---|---|
| Detection | scikit-learn RandomForestClassifier (19 UNSW-NB15 flow features) |
| Reasoning | CrewAI (Agent/Task/Crew, tool-use) + Groq `llama-3.3-70b-versatile` |
| Storage | SQLite (`infra/soc.db`) — shared directly between Python and Node |
| Frontend | React 19, TanStack Start/Router, Tailwind, Recharts, motion |
| Data | UNSW-NB15 (Australian Centre for Cyber Security) — 82k labeled network flows |

## Model training details

```python
RandomForestClassifier(
    n_estimators=120,
    max_depth=20,
    min_samples_split=3,
    random_state=42,
)
```
Trained on 19 flow-level features (`dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sttl`, `dttl`,
`sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `ct_srv_src`,
`ct_srv_dst`, `ct_dst_sport_ltm`), ~95%+ accuracy on the held-out UNSW-NB15 test set.

## Configuration

- **LLM model**: override via `GROQ_MODEL` in `.env` (default `groq/llama-3.3-70b-versatile`).
  Any [Groq-hosted model](https://console.groq.com/docs/models) works.
- **Rate limits**: `infra/rate_limiter.py` — defaults to 400 calls/day, 2s minimum interval,
  safely under Groq's free 1,000/day tier.
- **Stream rate**: `time.sleep(2)` in `infra/log_streamer.py`.
- **Action mapping**: `agents/response_agent.py` — attack category → simulated defensive action.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

**Built by Anirudh Dattu**

⭐ Star this repo if you find it useful!

</div>
