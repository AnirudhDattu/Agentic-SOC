# run_all.py
import subprocess
import time
import webbrowser
import os
import sys
import requests

# ---------- Utility to safely terminate ----------
def safe_kill(proc):
    try:
        proc.terminate()
    except Exception:
        pass

# ---------- Wait for Ollama to be ready ----------
def wait_for_ollama(timeout=60):
    print("[Launcher] Checking Ollama service availability...")
    for i in range(timeout):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1)
            if r.status_code == 200:
                print("[Launcher] ✅ Ollama is ready.")
                return True
        except Exception:
            pass
        print(f"[Launcher] ...waiting ({i+1}/{timeout})")
        time.sleep(1)
    print("[Launcher] ⚠️ Ollama not responding after 60 seconds. Continuing anyway.")
    return False

# ---------- Main launch process ----------
def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # Confirm Ollama service first
    wait_for_ollama()

    commands = {
        "Streamer": ["python", os.path.join(root, "infra", "log_streamer.py")],
        "Main SOC": ["python", os.path.join(root, "main.py")],
        "Dashboard": [
            "streamlit", "run",
            "--server.headless=true",
            "--server.runOnSave=false",
            "--browser.gatherUsageStats=false",
            os.path.join(root, "app", "dashboard.py")
        ],
    }

    processes = {}

    print("\n=== 🚀 Launching Agentic SOC Services ===\n")

    # Sequentially start processes
    for name, cmd in commands.items():
        print(f"[Launcher] Starting {name} ...")
        processes[name] = subprocess.Popen(cmd)
        time.sleep(3)  # small delay between launches

    print("\n[Launcher] ✅ All components started successfully.")
    print("→ Log Streamer, Main SOC, and Streamlit Dashboard are running.\n")
    print("Visit the dashboard at: http://localhost:8501")
    print("Press Ctrl+C to stop all components.\n")

    # Optional: Do not auto-open browser to prevent duplicate tabs
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    # Keep launcher alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Launcher] 🛑 Stopping all processes...")
        for name, proc in processes.items():
            print(f"[Launcher] Terminating {name} ...")
            safe_kill(proc)
        print("[Launcher] ✅ All processes stopped cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
