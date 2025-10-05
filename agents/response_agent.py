# agents/response_agent.py

from crewai import Agent

from crewai import Agent

class ResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ResponseAgent",
            role="Executes defensive actions and logs outcomes.",
            goal="Perform immediate mitigations based on reasoning output.",
            backstory="You are an automated security operator responsible for neutralizing detected threats."
        )

    def run(self, reasoning_result: dict):
        print(f"[ResponseAgent] Received: {reasoning_result}")
        return {"action": "Mock action executed"}
