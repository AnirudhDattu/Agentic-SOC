# agents/tools.py
from crewai.tools import tool
from infra.db import get_connection


@tool("lookup_similar_incidents")
def lookup_similar_incidents(attack_type: str) -> str:
    """Look up how many past incidents of a given attack_type were logged, broken
    down by severity, from the local incident history. Use this before finalizing
    a classification to ground your decision in what's actually been seen before.
    Pass one of: DDoS, Port Scan, Brute Force, Exploit, Data Exfiltration, Unknown.
    """
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT severity, COUNT(*) FROM incidents WHERE attack_type = ? "
            "GROUP BY severity ORDER BY COUNT(*) DESC",
            (attack_type,),
        ).fetchall()
        conn.close()
        if not rows:
            return f"No prior incidents of type '{attack_type}' on record yet."
        summary = ", ".join(f"{sev}: {count}" for sev, count in rows)
        return f"Prior '{attack_type}' incidents by severity — {summary}."
    except Exception as e:
        return f"History lookup failed: {e}"
