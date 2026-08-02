# agents/response_agent.py

from crewai import Agent
import datetime, json, uuid, random
from infra.db import get_connection, init_db

FEATURE_KEYS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_sport_ltm'
]

# Same reference points the frontend's demo feed uses, so incidents land
# in plausible spots on the threat map.
GEO = [
    {"country": "Russia", "lat": 55.75, "lon": 37.61},
    {"country": "China", "lat": 39.90, "lon": 116.40},
    {"country": "United States", "lat": 38.90, "lon": -77.03},
    {"country": "Brazil", "lat": -15.79, "lon": -47.88},
    {"country": "Germany", "lat": 52.52, "lon": 13.40},
    {"country": "India", "lat": 28.61, "lon": 77.20},
    {"country": "Nigeria", "lat": 9.07, "lon": 7.49},
    {"country": "Netherlands", "lat": 52.37, "lon": 4.90},
    {"country": "Vietnam", "lat": 21.03, "lon": 105.85},
    {"country": "Ukraine", "lat": 50.45, "lon": 30.52},
]


def _simulated_source():
    geo = random.choice(GEO)
    ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return {
        "source_ip": ip,
        "source_country": geo["country"],
        "source_lat": geo["lat"] + (random.random() - 0.5) * 6,
        "source_lon": geo["lon"] + (random.random() - 0.5) * 6,
    }


class ResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ResponseAgent",
            role="Automated defender",
            goal="Take defensive actions based on reasoning results.",
            backstory="You are an AI defender acting on attack intelligence."
        )
        init_db()
        print("[ResponseAgent] ✅ Local SQLite store ready (infra/soc.db).")

    def run(self, log_data: dict, reasoning_result: dict):
        attack = reasoning_result.get("attack_type", "Unknown")
        severity = reasoning_result.get("severity", "Medium")
        recommendation = reasoning_result.get("recommendation", "")
        method = reasoning_result.get("_method", "rules")
        status = log_data.get("status", "Unknown")
        action = "Logged incident"

        if attack == "DoS":
            action = "Throttled network traffic"
        elif attack == "Reconnaissance":
            action = "Blocked IP range"
        elif attack == "Fuzzers":
            action = "Blocked IP range"
        elif attack == "Backdoor":
            action = "Isolated host"
        elif attack == "Shellcode":
            action = "Isolated host"
        elif attack == "Worms":
            action = "Quarantined host"
        elif attack == "Exploits":
            action = "Isolated host and flagged for patching"
        elif attack == "Generic":
            action = "Escalated for analyst review"

        raw_features = {k: log_data.get(k) for k in FEATURE_KEYS}
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        source = _simulated_source()

        row = {
            "id": str(uuid.uuid4()),
            "timestamp": now,
            "status": status,
            "attack_type": attack,
            "severity": severity,
            "recommendation": recommendation,
            "action_taken": action,
            "raw_features": json.dumps(raw_features),
            "source_ip": source["source_ip"],
            "source_country": source["source_country"],
            "source_lat": source["source_lat"],
            "source_lon": source["source_lon"],
            "workflow_status": "New",
            "analyst_note": None,
            "reasoning_method": method,
            "created_at": now,
            "updated_at": now,
        }

        try:
            conn = get_connection()
            conn.execute(
                """
                INSERT INTO incidents (
                    id, timestamp, status, attack_type, severity, recommendation,
                    action_taken, raw_features, source_ip, source_country,
                    source_lat, source_lon, workflow_status, analyst_note,
                    reasoning_method, created_at, updated_at
                ) VALUES (
                    :id, :timestamp, :status, :attack_type, :severity, :recommendation,
                    :action_taken, :raw_features, :source_ip, :source_country,
                    :source_lat, :source_lon, :workflow_status, :analyst_note,
                    :reasoning_method, :created_at, :updated_at
                )
                """,
                row,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("[ResponseAgent] ⚠️ Local DB insert failed:", e)

        print(f"[ResponseAgent] {attack} ({severity}) → {action}")
        return {"action": action}
