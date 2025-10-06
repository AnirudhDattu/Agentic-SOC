# agents/response_agent.py

from crewai import Agent
import datetime, json

class ResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ResponseAgent",
            role="Automated defender",
            goal="Take defensive actions based on reasoning results.",
            backstory="You are an AI defender acting on attack intelligence."
        )

    def run(self, reasoning_result: dict):
        attack = reasoning_result.get("attack_type", "Unknown")
        severity = reasoning_result.get("severity", "Medium")
        recommendation = reasoning_result.get("recommendation", "")
        action = "Logged incident"

        if attack == "DDoS":
            action = "Throttled network traffic"
        elif attack == "Port Scan":
            action = "Blocked IP range"
        elif attack == "Brute Force":
            action = "Locked account"
        elif attack == "Exploit":
            action = "Isolated host"
        elif attack == "Data Exfiltration":
            action = "Terminated outbound connections"

        log_entry = {
            "time": str(datetime.datetime.now()),
            "attack_type": attack,
            "severity": severity,
            "recommendation": recommendation,
            "action": action
        }

        with open("infra/action_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"[ResponseAgent] {attack} ({severity}) → {action}")
        return {"action": action}
