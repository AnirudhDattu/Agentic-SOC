# infra/db.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "soc.db")

SCHEMA = """
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
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
