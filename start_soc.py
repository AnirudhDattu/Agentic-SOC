"""
Unified launcher for Agentic SOC 2.0
Runs: log_streamer, main pipeline, and dashboard together in one terminal.
"""
import runpy
import threading
import os
import time

# optional: hide streamlit logs
os.environ["STREAMLIT_SUPPRESS_CONFIG_WARNINGS"] = "1"

def run_streamer():
    print("\n[Launcher] Starting Log Streamer ...")
    runpy.run_path("infra/log_streamer.py", run_name="__main__")

def run_main():
    print("\n[Launcher] Starting Main SOC Pipeline ...")
    runpy.run_path("main.py", run_name="__main__")

def run_dashboard():
    print("\n[Launcher] Starting Dashboard ...")
    # Streamlit must be called as module for UI
    os.system("streamlit run app/dashboard.py --server.headless=true --server.runOnSave=false")

# create threads
threads = [
    threading.Thread(target=run_streamer, daemon=True),
    threading.Thread(target=run_main, daemon=True),
    threading.Thread(target=run_dashboard, daemon=True)
]

# launch all
print("\n=== 🚀 Launching Agentic SOC 2.0 ===")
for t in threads:
    t.start()
    time.sleep(2)   # stagger a bit for clarity

print("\n[Launcher] All components started.")
print("→ Log Streamer, Main SOC, and Dashboard running.")
print("Visit: http://localhost:8501\n")
print("Press Ctrl+C to stop everything.\n")

# keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[Launcher] 🛑 Shutting down...")
