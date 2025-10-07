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
You are a cybersecurity expert analyzing live network logs.
Here is one suspicious log entry detected by the AI model:

{json.dumps(detection_result, indent=2)}

Based on duration (dur), source packets (spkts), destination packets (dpkts), and traffic patterns,
decide what kind of attack it most likely is.

Respond ONLY in valid JSON:
{{
  "attack_type": "<DDoS, Port Scan, Brute Force, Exploit, Data Exfiltration, Unknown>",
  "severity": "<Low, Medium, High>",
  "recommendation": "<one actionable short sentence>"
}}
"""


        try:
            response = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
            text = response.message.content.strip()

            # Try to extract JSON if LLM adds prose
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                text = text[start:end]

            parsed = json.loads(text)
            dur = float(detection_result.get("dur", 0))
            spkts = float(detection_result.get("spkts", 0))
            dpkts = float(detection_result.get("dpkts", 0))

            if parsed["attack_type"] == "Unknown":
                if spkts > 300 and dpkts > 300:
                    parsed["attack_type"] = "Port Scan"
                elif spkts > 50 and dur < 0.5:
                    parsed["attack_type"] = "Brute Force"
                elif dur > 20 and dpkts < 10:
                    parsed["attack_type"] = "Data Exfiltration"
                elif dur > 5 and spkts > 1000:
                    parsed["attack_type"] = "Exploit"

            print(f"[ReasoningAgent] Parsed response: {parsed}")
            return parsed

        except Exception as e:
            print(f"[ReasoningAgent] ⚠️ Parsing error: {e}")
            return {
                "attack_type": "Unknown",
                "severity": "Medium",
                "recommendation": "Fallback: monitor and escalate if anomalies persist."
            }
