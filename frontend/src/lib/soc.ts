// Real UNSW-NB15 attack categories (data/network_logs.csv's attack_cat column)
// — ground truth from the dataset, not a guessed taxonomy.
export const ATTACK_TYPES = [
  "Generic",
  "Exploits",
  "Fuzzers",
  "DoS",
  "Reconnaissance",
  "Analysis",
  "Backdoor",
  "Shellcode",
  "Worms",
  "Unknown",
  "None",
] as const;

export const SEVERITIES = ["High", "Medium", "Low"] as const;
export const STATUSES = ["Attack", "Benign"] as const;
export const WORKFLOW_STATUSES = ["New", "Investigating", "Resolved"] as const;

export type Severity = (typeof SEVERITIES)[number];
export type IncidentStatus = (typeof STATUSES)[number];
export type WorkflowStatus = (typeof WORKFLOW_STATUSES)[number];

export type RawFeatures = Record<string, number | string | null>;

export type Incident = {
  id: string;
  timestamp: string;
  status: string;
  attack_type: string;
  severity: string;
  recommendation: string;
  action_taken: string;
  raw_features: RawFeatures;
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

export const FEATURE_FIELDS: { key: string; label: string; unit?: string }[] = [
  { key: "dur", label: "Duration", unit: "s" },
  { key: "spkts", label: "Source packets" },
  { key: "dpkts", label: "Dest packets" },
  { key: "sbytes", label: "Source bytes", unit: "B" },
  { key: "dbytes", label: "Dest bytes", unit: "B" },
  { key: "sttl", label: "Source TTL" },
  { key: "dttl", label: "Dest TTL" },
  { key: "sload", label: "Source load", unit: "bps" },
  { key: "dload", label: "Dest load", unit: "bps" },
  { key: "sinpkt", label: "Src inter-packet", unit: "ms" },
  { key: "dinpkt", label: "Dst inter-packet", unit: "ms" },
  { key: "sjit", label: "Source jitter", unit: "ms" },
  { key: "djit", label: "Dest jitter", unit: "ms" },
  { key: "swin", label: "Source TCP window" },
  { key: "stcpb", label: "Source TCP base seq" },
  { key: "dtcpb", label: "Dest TCP base seq" },
  { key: "ct_srv_src", label: "Conns same srv/src" },
  { key: "ct_srv_dst", label: "Conns same srv/dst" },
  { key: "ct_dst_sport_ltm", label: "Conns dst/sport (ltm)" },
];

export const severityMeta: Record<
  Severity,
  { label: string; text: string; bg: string; dot: string; chartVar: string }
> = {
  High: {
    label: "HIGH",
    text: "text-sev-high",
    bg: "bg-sev-high-soft",
    dot: "bg-sev-high",
    chartVar: "var(--sev-high)",
  },
  Medium: {
    label: "MED",
    text: "text-sev-medium",
    bg: "bg-sev-medium-soft",
    dot: "bg-sev-medium",
    chartVar: "var(--sev-medium)",
  },
  Low: {
    label: "LOW",
    text: "text-sev-low",
    bg: "bg-sev-low-soft",
    dot: "bg-sev-low",
    chartVar: "var(--sev-low)",
  },
};

export function isSeverity(value: string): value is Severity {
  return (SEVERITIES as readonly string[]).includes(value);
}

export function formatNumber(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  if (Number.isInteger(n)) return n.toLocaleString("en-US");
  return n.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

export function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return `${d.toISOString().slice(0, 10)} ${d.toLocaleTimeString("en-GB", { hour12: false })}`;
}

export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.max(0, Math.floor(diff / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function toCsv(rows: Incident[]): string {
  const headers = [
    "id",
    "timestamp",
    "status",
    "attack_type",
    "severity",
    "workflow_status",
    "source_ip",
    "source_country",
    "action_taken",
    "recommendation",
    "analyst_note",
    "reasoning_method",
    ...FEATURE_FIELDS.map((f) => f.key),
  ];
  const escape = (v: unknown) => {
    let s = v === null || v === undefined ? "" : String(v);
    // Neutralize CSV formula injection: a leading =, +, -, @, tab, or CR lets
    // a value execute as a formula when the export is opened in Excel/Sheets.
    if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = rows.map((r) =>
    [
      r.id,
      r.timestamp,
      r.status,
      r.attack_type,
      r.severity,
      r.workflow_status,
      r.source_ip,
      r.source_country,
      r.action_taken,
      r.recommendation,
      r.analyst_note,
      r.reasoning_method,
      ...FEATURE_FIELDS.map((f) => r.raw_features?.[f.key] ?? ""),
    ]
      .map(escape)
      .join(","),
  );
  return [headers.join(","), ...lines].join("\n");
}

export function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
