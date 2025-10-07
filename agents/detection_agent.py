# agents/detection_agent.py

from crewai import Agent
import joblib
import pandas as pd
import os

class DetectionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="DetectionAgent",
            role="Analyzes incoming log events and detects suspicious activity.",
            goal="Identify potential cyber threats from logs using AI models.",
            backstory="You are an AI analyst trained to recognize attack patterns and anomalies in system logs."
        )

        model_path = "models/detector.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Assign safely without triggering Pydantic restrictions
            object.__setattr__(self, "_model", model)
            print("[DetectionAgent] ✅ Model loaded successfully.")
        else:
            object.__setattr__(self, "_model", None)
            print("[DetectionAgent] ⚠️ Model not found, using mock mode.")

    def run(self, log_data: dict):
        model = getattr(self, "_model", None)

        if model is None:
            print("[DetectionAgent] ⚠️ No model available.")
            return {"status": "Unknown", "details": "Model not loaded"}

        try:
            # Ensure input uses model's expected features
            model_features = getattr(model, "feature_names_in_", None)
            df = pd.DataFrame([log_data])

            if model_features is not None:
                # Keep only features model expects
                missing = [f for f in model_features if f not in df.columns]
                if missing:
                    print(f"[DetectionAgent] ⚠️ Missing features in log: {missing}")
                df = df[[f for f in model_features if f in df.columns]]

            pred = model.predict(df)[0]

            status = "Attack" if pred == 1 else "Benign"
            print(f"[DetectionAgent] {status} detected for log: {log_data}")
            return {"status": status}

        except Exception as e:
            print("[DetectionAgent] Error:", e)
            return {"status": "Error", "details": str(e)}
