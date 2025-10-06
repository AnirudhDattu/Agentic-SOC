# agents/reasoning_agent.py

from crewai import Agent

class ReasoningAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ReasoningAgent",
            role="Summarizes incidents and suggests response actions.",
            goal="Interpret detection results and generate human-readable insights.",
            backstory="You are a cybersecurity strategist that explains alerts and recommends responses."
        )

    def run(self, detection_result: dict):
        status = detection_result.get("status", "Unknown")
        if status == "Attack":
            summary = "High-risk event detected. Possible malicious communication. Recommend blocking source IP and isolating host."
        elif status == "Benign":
            summary = "No malicious behavior found. Monitoring continues."
        elif status == "Unknown":
            summary = "Unable to analyze log due to missing model or data."
        else:
            summary = "Unexpected status; manual review required."

        print(f"[ReasoningAgent] {summary}")
        return {"summary": summary}
