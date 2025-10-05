from agents.detection_agent import DetectionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.response_agent import ResponseAgent
import json, time, os

detector = DetectionAgent()
reasoner = ReasoningAgent()
responder = ResponseAgent()

print("=== Agentic SOC Live Mode ===")

while True:
    if os.path.exists("infra/live_log.json"):
        with open("infra/live_log.json") as f:
            log = json.load(f)

        d = detector.run(log)
        r = reasoner.run(d)
        a = responder.run(r)

        print(f"\n[Pipeline] Log→{d['status']} → {r['summary']} → {a['action']}\n")

    time.sleep(2)
