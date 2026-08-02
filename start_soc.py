"""
Unified launcher for Agentic SOC
Runs: log_streamer, main pipeline, and the frontend dev server together in one terminal.
"""
import runpy
import threading
import os
import time

def run_streamer():
    print("\n[Launcher] Starting Log Streamer ...")
    runpy.run_path("infra/log_streamer.py", run_name="__main__")

def run_main():
    print("\n[Launcher] Starting Main SOC Pipeline ...")
    runpy.run_path("main.py", run_name="__main__")

def run_frontend():
    print("\n[Launcher] Starting Frontend ...")
    os.system("npm --prefix frontend run dev")

# create threads
threads = [
    threading.Thread(target=run_streamer, daemon=True),
    threading.Thread(target=run_main, daemon=True),
    threading.Thread(target=run_frontend, daemon=True),
]

# launch all
print("\n=== 🚀 Launching Agentic SOC ===")
for t in threads:
    t.start()
    time.sleep(2)   # stagger a bit for clarity

print("\n[Launcher] All components started.")
print("→ Log Streamer, Main SOC, and Frontend running.")
print("Visit: http://localhost:8080\n")
print("Press Ctrl+C to stop everything.\n")

# keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[Launcher] 🛑 Shutting down...")
