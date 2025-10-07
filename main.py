#main.py
from agents.detection_agent import DetectionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.response_agent import ResponseAgent
import json, time, os
import requests

detector = DetectionAgent()
reasoner = ReasoningAgent()
responder = ResponseAgent()

print("=== Agentic SOC Live Mode ===")

while True:
    if os.path.exists("infra/live_log.json"):
        with open("infra/live_log.json") as f:
            log = json.load(f)

        # run agents sequentially
        d = detector.run(log)
        r = reasoner.run({**log, **d})
        a = responder.run(r)

        # extract clean info
        attack_type = r.get("attack_type", "None")
        severity = r.get("severity", "Low")
        rec = r.get("recommendation", "No details")

        print(f"\n[Pipeline] {d['status']} → {attack_type} ({severity}) → {rec} → {a['action']}\n")

    time.sleep(2)
