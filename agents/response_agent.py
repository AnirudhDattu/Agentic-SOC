# agents/response_agent.py

from crewai import Agent
import datetime, json

class ResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ResponseAgent",
            role="Executes defensive actions and logs outcomes.",
            goal="Perform immediate mitigations based on reasoning output.",
            backstory="You are an automated security operator responsible for neutralizing detected threats."
        )

    def run(self, reasoning_result: dict):
        action = "None"
        if "High-risk" in reasoning_result["summary"]:
            action = "Blocked source IP (simulated)"
        elif "No malicious" in reasoning_result["summary"]:
            action = "No action required"
        elif "Unable" in reasoning_result["summary"]:
            action = "Manual investigation needed"

        log_entry = {
            "time": str(datetime.datetime.now()),
            "action": action,
            "summary": reasoning_result["summary"]
        }
        with open("infra/action_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"[ResponseAgent] {action}")
        return {"action": action}
