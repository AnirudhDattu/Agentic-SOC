// Server-only local SQLite access. The Python pipeline writes into the same
// file (../infra/soc.db, relative to this frontend/ directory); this module
// only reads/updates it.
import { DatabaseSync } from "node:sqlite";
import path from "node:path";

const DB_PATH =
  process.env["SOC_DB_PATH"] ?? path.resolve(process.cwd(), "..", "infra", "soc.db");

const SCHEMA = `
CREATE TABLE IF NOT EXISTS incidents (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'Benign',
  attack_type TEXT NOT NULL DEFAULT 'None',
  severity TEXT NOT NULL DEFAULT 'Low',
  recommendation TEXT NOT NULL DEFAULT '',
  action_taken TEXT NOT NULL DEFAULT 'Logged incident',
  raw_features TEXT NOT NULL DEFAULT '{}',
  source_ip TEXT,
  source_country TEXT,
  source_lat REAL,
  source_lon REAL,
  workflow_status TEXT NOT NULL DEFAULT 'New',
  analyst_note TEXT,
  reasoning_method TEXT NOT NULL DEFAULT 'rules',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS incidents_timestamp_idx ON incidents (timestamp DESC);
CREATE INDEX IF NOT EXISTS incidents_severity_idx ON incidents (severity);
CREATE INDEX IF NOT EXISTS incidents_attack_type_idx ON incidents (attack_type);
`;

let _db: DatabaseSync | undefined;

export function getDb(): DatabaseSync {
  if (!_db) {
    _db = new DatabaseSync(DB_PATH);
    _db.exec("PRAGMA journal_mode = WAL;");
    _db.exec("PRAGMA busy_timeout = 5000;");
    _db.exec(SCHEMA);
  }
  return _db;
}

export type IncidentRow = {
  id: string;
  timestamp: string;
  status: string;
  attack_type: string;
  severity: string;
  recommendation: string;
  action_taken: string;
  raw_features: string;
  source_ip: string | null;
  source_country: string | null;
  source_lat: number | null;
  source_lon: number | null;
  workflow_status: string;
  analyst_note: string | null;
  reasoning_method: string;
  created_at: string;
  updated_at: string;
};
