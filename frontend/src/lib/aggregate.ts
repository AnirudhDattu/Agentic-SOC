import { ATTACK_TYPES, SEVERITIES, severityMeta, type Incident } from "./soc";

export type Range = "1h" | "24h" | "7d";

const RANGE_MS: Record<Range, number> = {
  "1h": 3600_000,
  "24h": 86400_000,
  "7d": 7 * 86400_000,
};

function pct(current: number, prior: number): number | null {
  if (prior === 0) return current === 0 ? 0 : null;
  return Math.round(((current - prior) / prior) * 100);
}

export function computeKpis(incidents: Incident[]) {
  const now = Date.now();
  const day = 86400_000;
  const current = incidents.filter((i) => now - new Date(i.timestamp).getTime() <= day);
  const prior = incidents.filter((i) => {
    const age = now - new Date(i.timestamp).getTime();
    return age > day && age <= day * 2;
  });

  const count = (rows: Incident[], fn: (i: Incident) => boolean) => rows.filter(fn).length;

  return {
    total: { value: current.length, delta: pct(current.length, prior.length) },
    attacks: {
      value: count(current, (i) => i.status === "Attack"),
      delta: pct(
        count(current, (i) => i.status === "Attack"),
        count(prior, (i) => i.status === "Attack"),
      ),
    },
    high: {
      value: count(current, (i) => i.severity === "High" && i.status === "Attack"),
      delta: pct(
        count(current, (i) => i.severity === "High" && i.status === "Attack"),
        count(prior, (i) => i.severity === "High" && i.status === "Attack"),
      ),
    },
    medium: {
      value: count(current, (i) => i.severity === "Medium" && i.status === "Attack"),
      delta: pct(
        count(current, (i) => i.severity === "Medium" && i.status === "Attack"),
        count(prior, (i) => i.severity === "Medium" && i.status === "Attack"),
      ),
    },
    low: {
      value: count(current, (i) => i.severity === "Low"),
      delta: pct(
        count(current, (i) => i.severity === "Low"),
        count(prior, (i) => i.severity === "Low"),
      ),
    },
  };
}

export function withinRange(incidents: Incident[], range: Range) {
  const cutoff = Date.now() - RANGE_MS[range];
  return incidents.filter((i) => new Date(i.timestamp).getTime() >= cutoff);
}

export function attackTypeCounts(incidents: Incident[]) {
  const counts = new Map<string, number>();
  for (const i of incidents) {
    if (i.status !== "Attack") continue;
    counts.set(i.attack_type, (counts.get(i.attack_type) ?? 0) + 1);
  }
  return ATTACK_TYPES.filter((t) => t !== "None")
    .map((type) => ({ type, count: counts.get(type) ?? 0 }))
    .sort((a, b) => a.count - b.count);
}

export function severityCounts(incidents: Incident[]) {
  return SEVERITIES.map((name) => ({
    name,
    value: incidents.filter((i) => i.severity === name).length,
    color: severityMeta[name].chartVar,
  })).filter((d) => d.value > 0);
}

export function volumeSeries(incidents: Incident[], range: Range) {
  const now = Date.now();
  const span = RANGE_MS[range];
  const buckets = range === "1h" ? 12 : range === "24h" ? 24 : 28;
  const size = span / buckets;
  const series = Array.from({ length: buckets }, (_, idx) => {
    const start = now - span + idx * size;
    const d = new Date(start + size);
    const label =
      range === "7d"
        ? `${d.getUTCMonth() + 1}/${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2, "0")}h`
        : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
    return { bucket: label, attacks: 0, benign: 0, start, end: start + size };
  });

  for (const i of incidents) {
    const t = new Date(i.timestamp).getTime();
    if (t < now - span) continue;
    const idx = Math.min(buckets - 1, Math.floor((t - (now - span)) / size));
    if (idx < 0) continue;
    if (i.status === "Attack") series[idx]!.attacks += 1;
    else series[idx]!.benign += 1;
  }

  return series.map(({ bucket, attacks, benign }) => ({ bucket, attacks, benign }));
}
