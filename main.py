# main.py
import os

# Must happen before crewai is imported (transitively, via the agents below):
# without these it hangs trying to phone home for telemetry, and prompts
# interactively on first run, which blocks this non-interactive loop.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TESTING", "true")

from dotenv import load_dotenv
load_dotenv()

from agents.detection_agent import DetectionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.response_agent import ResponseAgent
import json, time

# ANSI color codes for pretty console output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

detector = DetectionAgent()
reasoner = ReasoningAgent()
responder = ResponseAgent()

print(f"{BLUE}\n=== 🚀 Agentic SOC 2.0 Live Mode Started ==={RESET}\n")

while True:
    if os.path.exists("infra/live_log.json"):
        with open("infra/live_log.json") as f:
            log = json.load(f)

        # Run agents sequentially
        d = detector.run(log)
        r = reasoner.run(d)
        a = responder.run(d, r)

        attack_type = r.get("attack_type", "None")
        severity = r.get("severity", "Low")
        recommendation = r.get("recommendation", "No recommendation")
        action = a.get("action", "No action")

        # Choose color by severity
        color = (
            RED if severity.lower() == "high"
            else YELLOW if severity.lower() == "medium"
            else GREEN
        )

        print(f"{BLUE}───────────────────────────────────────────────{RESET}")
        print(f"{BLUE}[ Event Detected ]{RESET}")
        print(f"→  Status: {color}{d['status']}{RESET}")
        print(f"→  Attack Type: {color}{attack_type}{RESET}")
        print(f"→  Severity: {color}{severity}{RESET}")
        print(f"→  Recommendation: {recommendation}")
        print(f"→  Action Taken: {action}")
        print(f"{BLUE}───────────────────────────────────────────────{RESET}\n")

    time.sleep(2)
