# agents/reasoning_agent.py

from crewai import Agent
import ollama
import json

class ReasoningAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ReasoningAgent",
            role="LLM-based security analyst",
            goal="Identify attack type, severity, and recommend mitigation using a local LLM.",
            backstory="You are an AI SOC analyst reasoning about detected attacks."
        )

    def run(self, detection_result: dict):
        if detection_result.get("status") != "Attack":
            return {
                "attack_type": "None",
                "severity": "Low",
                "recommendation": "No threat detected. Continue monitoring."
            }

        prompt = f"""
        You are a cybersecurity expert.
        A network log was detected as an attack:
        {detection_result}

        You must respond **only** in this JSON format (no code blocks, no text):
        {{
            "attack_type": "<one of: DDoS, Port Scan, Brute Force, Exploit, Data Exfiltration, Unknown>",
            "severity": "<Low, Medium, High>",
            "recommendation": "<one short sentence on what to do>"
        }}
        """

        try:
            response = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
            text = response["message"]["content"]

            # sanitize: remove code fences if model adds them
            text = text.replace("```json", "").replace("```", "").strip()

            # parse structured JSON output
            parsed = json.loads(text)
        except Exception as e:
            parsed = {
                "attack_type": "Unknown",
                "severity": "Medium",
                "recommendation": f"LLM parsing error: {e}"
            }

        print(f"[ReasoningAgent] {parsed}")
        return parsed
