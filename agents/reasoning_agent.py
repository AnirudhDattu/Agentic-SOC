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
            backstory="You are an AI SOC analyst reasoning about detected attacks and classifying threats accurately."
        )

    def run(self, detection_result: dict):
        # Skip if not attack
        if detection_result.get("status") != "Attack":
            return {
                "attack_type": "None",
                "severity": "Low",
                "recommendation": "No threat detected. Continue monitoring."
            }

        # ---------- Improved prompt ----------
        prompt = f"""
You are a cybersecurity analyst. A network log entry was flagged as a potential attack.

Log details:
{json.dumps(detection_result, indent=2)}

Use these simple heuristics to infer the most probable attack type:
- Very short duration (<0.01 s) with very few packets → Port Scan  
- Many packets from one source in short duration (spkts > 1000, dur < 2) → DDoS  
- Moderate packets (spkts > 50) and short duration (dur < 0.5 s) → Brute Force  
- Long duration (dur > 20 s) with few destination packets (dpkts < 10) → Data Exfiltration  
- Long duration (>5 s) and high packets (>1000) → Exploit  

Respond ONLY in **valid JSON**:
{{
  "attack_type": "<one of: DDoS, Port Scan, Brute Force, Exploit, Data Exfiltration, Unknown>",
  "severity": "<Low, Medium, High>",
  "recommendation": "<one short actionable sentence>"
}}
No explanations, no code, just JSON.
"""

        # ---------- Query Ollama ----------
        try:
            response = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
            text = response.get("message", {}).get("content", "").strip()

            # Extract first valid JSON block if needed
            start, end = text.find("{"), text.rfind("}") + 1
            json_text = text[start:end] if (start != -1 and end != -1) else "{}"
            parsed = json.loads(json_text)
        except Exception as e:
            print(f"[ReasoningAgent] ⚠️ LLM parsing failed: {e}")
            parsed = {
                "attack_type": "Unknown",
                "severity": "Medium",
                "recommendation": "Monitor and escalate if anomalies persist."
            }

        # ---------- Apply rule-based refinement always ----------
        try:
            dur = float(detection_result.get("dur", 0))
            spkts = float(detection_result.get("spkts", 0))
            dpkts = float(detection_result.get("dpkts", 0))

            # refine based on quantitative behavior
            if spkts > 1000 and dur < 2:
                parsed.update({
                    "attack_type": "DDoS",
                    "severity": "High",
                    "recommendation": "Enable rate limiting and block suspicious IPs."
                })
            elif spkts > 300 and dpkts > 300 and dur < 1:
                parsed.update({
                    "attack_type": "Port Scan",
                    "severity": "Medium",
                    "recommendation": "Block scanner IP range and enable IDS alerts."
                })
            elif spkts > 50 and dur < 0.5:
                parsed.update({
                    "attack_type": "Brute Force",
                    "severity": "High",
                    "recommendation": "Lock accounts after multiple failed login attempts."
                })
            elif dur > 20 and dpkts < 10:
                parsed.update({
                    "attack_type": "Data Exfiltration",
                    "severity": "High",
                    "recommendation": "Inspect outbound data flows for leaks."
                })
            elif dur > 5 and spkts > 1000:
                parsed.update({
                    "attack_type": "Exploit",
                    "severity": "High",
                    "recommendation": "Isolate affected host and patch vulnerabilities."
                })

        except Exception as e:
            print(f"[ReasoningAgent] ⚠️ Rule-based reasoning error: {e}")

        # ---------- Final cleanup ----------
        parsed.setdefault("attack_type", "Unknown")
        parsed.setdefault("severity", "Medium")
        parsed.setdefault("recommendation", "No valid reasoning provided.")
        print(f"[ReasoningAgent] Parsed response: {parsed}")
        return parsed
