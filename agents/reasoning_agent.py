# agents/reasoning_agent.py

from crewai import Agent

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
        print(f"[ReasoningAgent] Received: {detection_result}")
        return {"summary": "Reasoning placeholder"}
