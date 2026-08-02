import { ATTACK_TYPES } from "./soc";

const GEO = [
  { country: "Russia", lat: 55.75, lon: 37.61 },
  { country: "China", lat: 39.9, lon: 116.4 },
  { country: "United States", lat: 38.9, lon: -77.03 },
  { country: "Brazil", lat: -15.79, lon: -47.88 },
  { country: "Germany", lat: 52.52, lon: 13.4 },
  { country: "India", lat: 28.61, lon: 77.2 },
  { country: "Nigeria", lat: 9.07, lon: 7.49 },
  { country: "Netherlands", lat: 52.37, lon: 4.9 },
  { country: "Vietnam", lat: 21.03, lon: 105.85 },
  { country: "Ukraine", lat: 50.45, lon: 30.52 },
];

const RECS = [
  "Rate-limit the source subnet and enable SYN cookies on edge routers.",
  "Block the offending IP range at the perimeter firewall for 24 hours.",
  "Enforce account lockout after 5 failed attempts and require MFA re-enrolment.",
  "Patch the affected service to the latest release and restart the workload.",
  "Quarantine the host and audit outbound TLS sessions for large transfers.",
  "Capture a full packet trace and escalate to tier-2 for manual review.",
];

const ACTIONS = [
  "Throttled network traffic",
  "Blocked IP range",
  "Logged incident",
  "Isolated endpoint",
  "Reset session tokens",
];

const pick = <T,>(arr: readonly T[]): T => arr[Math.floor(Math.random() * arr.length)]!;
const rnd = (max: number, dp = 0) => Number((Math.random() * max).toFixed(dp));

export function synthesizeIncident() {
  const isAttack = Math.random() < 0.65;
  const geo = pick(GEO);
  const attackTypes = ATTACK_TYPES.filter((t) => t !== "None");
  return {
    timestamp: new Date().toISOString(),
    status: isAttack ? "Attack" : "Benign",
    attack_type: isAttack ? pick(attackTypes) : "None",
    severity: isAttack ? pick(["High", "High", "Medium", "Medium", "Low"]) : "Low",
    recommendation: isAttack
      ? pick(RECS)
      : "No action required. Traffic profile matches known-good baseline.",
    action_taken: isAttack ? pick(ACTIONS) : "Logged incident",
    raw_features: {
      dur: rnd(60, 4),
      spkts: rnd(2000),
      dpkts: rnd(2000),
      sbytes: rnd(500000),
      dbytes: rnd(500000),
      sttl: rnd(255),
      dttl: rnd(255),
      sload: rnd(900000, 3),
      dload: rnd(900000, 3),
      sinpkt: rnd(500, 4),
      dinpkt: rnd(500, 4),
      sjit: rnd(300, 4),
      djit: rnd(300, 4),
      swin: rnd(65535),
      stcpb: rnd(4000000000),
      dtcpb: rnd(4000000000),
      ct_srv_src: rnd(60),
      ct_srv_dst: rnd(60),
      ct_dst_sport_ltm: rnd(40),
    },
    source_ip: `${rnd(223) + 1}.${rnd(255)}.${rnd(255)}.${rnd(254) + 1}`,
    source_country: geo.country,
    source_lat: geo.lat + (Math.random() - 0.5) * 6,
    source_lon: geo.lon + (Math.random() - 0.5) * 6,
    workflow_status: "New",
  };
}

export async function pushDemoIncident() {
  const res = await fetch("/api/incidents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(synthesizeIncident()),
  });
  if (!res.ok) throw new Error(`Failed to push demo incident: ${res.status}`);
}
