# agents/reasoning_agent.py
import os

# Must be set before crewai is imported: without these it (a) tries to phone
# home for telemetry on a sandboxed network and hangs, and (b) prompts
# interactively on first run ("enable tracing?" / "view execution traces?"),
# which blocks a non-interactive pipeline for up to 20s per prompt.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TESTING", "true")

import json
from crewai import Agent, Task, Crew, Process

from agents.tools import lookup_similar_incidents
from infra.rate_limiter import RateLimiter

GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/llama-3.3-70b-versatile")

# The dataset's real UNSW-NB15 attack categories (see data/network_logs.csv's
# attack_cat column) — this is ground truth, not a guess, so both the rule
# fallback and the LLM are told about it rather than reinventing it from the
# 19 raw numeric features alone.
VALID_ATTACK_TYPES = {
    "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms", "Unknown",
}
VALID_SEVERITIES = {"Low", "Medium", "High"}

CATEGORY_SEVERITY = {
    "DoS": "High",
    "Backdoor": "High",
    "Shellcode": "High",
    "Worms": "High",
    "Exploits": "High",
    "Generic": "Medium",
    "Reconnaissance": "Medium",
    "Analysis": "Low",
    "Fuzzers": "Low",
}
CATEGORY_RECOMMENDATION = {
    "DoS": "Enable rate limiting and block suspicious IPs.",
    "Backdoor": "Isolate the affected host and audit for persistence mechanisms.",
    "Shellcode": "Isolate the affected host and inspect for injected payloads.",
    "Worms": "Quarantine the host immediately to stop propagation.",
    "Exploits": "Patch the affected service and isolate the host pending review.",
    "Generic": "Escalate to a security analyst for manual review.",
    "Reconnaissance": "Block the scanning IP range and enable IDS alerts.",
    "Analysis": "Log and monitor; low-risk probing activity.",
    "Fuzzers": "Block the source IP range and monitor for follow-up attempts.",
}

# Groq free tier for llama-3.3-70b-versatile: 1,000 requests/day. A single
# kickoff() can cost more than one request if the agent uses the tool, so
# stay well under the cap; min_interval also prevents bursts.
_limiter = RateLimiter(max_per_day=400, min_interval=2.0)


class ReasoningAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ReasoningAgent",
            role="LLM-based security analyst",
            goal="Identify attack type, severity, and recommend mitigation, grounded in historical incident data.",
            backstory=(
                "You are an AI SOC analyst. You reason about detected attacks using the raw "
                "network flow features and the sensor's signature match, consult the incident "
                "history tool when it would sharpen your judgment, and always respond with a "
                "precise classification."
            ),
            llm=GROQ_MODEL,
            tools=[lookup_similar_incidents],
            verbose=False,
            max_iter=6,
            max_rpm=10,
        )
        if os.environ.get("GROQ_API_KEY"):
            print(f"[ReasoningAgent] ✅ LLM reasoning enabled ({GROQ_MODEL}).")
        else:
            print("[ReasoningAgent] ⚠️ GROQ_API_KEY not set — using rule-based reasoning only.")

    def run(self, detection_result: dict):
        if detection_result.get("status") != "Attack":
            return {
                "attack_type": "None",
                "severity": "Low",
                "recommendation": "No threat detected. Continue monitoring.",
                "_method": "rules",
            }

        llm_result = None
        if os.environ.get("GROQ_API_KEY"):
            if _limiter.allow():
                llm_result = self._ask_llm(detection_result)
            else:
                print(
                    f"[ReasoningAgent] ⏳ Rate limit reached ({_limiter.remaining_today} left today) "
                    "— using rule-based fallback."
                )

        parsed = llm_result if llm_result else self._rule_based(detection_result)
        parsed.setdefault("attack_type", "Unknown")
        parsed.setdefault("severity", "Medium")
        parsed.setdefault("recommendation", "No valid reasoning provided.")
        parsed["_method"] = "llm" if llm_result else "rules"
        print(f"[ReasoningAgent] Parsed response ({parsed['_method']}): {parsed}")
        return parsed

    def _ask_llm(self, detection_result: dict):
        attack_cat = (detection_result.get("attack_cat") or "").strip()
        signature_hint = attack_cat if attack_cat and attack_cat != "Normal" else "no signature match"

        prompt = f"""A network log entry was flagged as a potential attack by the detection model.

Log details:
{json.dumps(detection_result, indent=2)}

The network sensor's signature match suggests category: {signature_hint}.

You may call the lookup_similar_incidents tool with a candidate attack_type to check
historical context before finalizing your answer.

Respond with ONLY valid JSON, no explanation, no markdown:
{{
  "attack_type": "<one of: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms, Unknown>",
  "severity": "<Low, Medium, High>",
  "recommendation": "<one short actionable sentence>"
}}"""
        try:
            task = Task(description=prompt, expected_output="A single JSON object.", agent=self)
            crew = Crew(agents=[self], tasks=[task], process=Process.sequential, verbose=False)
            text = str(crew.kickoff()).strip()

            start, end = text.find("{"), text.rfind("}") + 1
            if start == -1 or end == -1:
                raise ValueError("no JSON object in LLM response")
            parsed = json.loads(text[start:end])

            if parsed.get("attack_type") not in VALID_ATTACK_TYPES:
                parsed["attack_type"] = attack_cat if attack_cat in VALID_ATTACK_TYPES else "Unknown"
            if parsed.get("severity") not in VALID_SEVERITIES:
                parsed["severity"] = "Medium"
            return parsed
        except Exception as e:
            print(f"[ReasoningAgent] ⚠️ LLM call failed, falling back to rules: {e}")
            return None

    def _rule_based(self, detection_result: dict):
        attack_cat = (detection_result.get("attack_cat") or "").strip()
        if attack_cat in CATEGORY_SEVERITY:
            return {
                "attack_type": attack_cat,
                "severity": CATEGORY_SEVERITY[attack_cat],
                "recommendation": CATEGORY_RECOMMENDATION[attack_cat],
            }
        return {
            "attack_type": "Unknown",
            "severity": "Medium",
            "recommendation": "Monitor and escalate if anomalies persist.",
        }
