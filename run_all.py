# run_all.py
import subprocess
import time
import webbrowser
import os
import sys

# Kill old processes if needed (Windows safe check)
def safe_kill(proc):
    try:
        proc.terminate()
    except Exception:
        pass

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    commands = {
        "Streamer": ["python", os.path.join(root, "infra", "log_streamer.py")],
        "Main SOC": ["python", os.path.join(root, "main.py")],
        # disable Streamlit file watching (prevents double launch)
        "Dashboard": [
            "streamlit", "run",
            "--server.runOnSave=false",
            "--logger.level=error",
            os.path.join(root, "app", "dashboard.py")
        ],
    }

    processes = {}

    # Launch each component
    for name, cmd in commands.items():
        print(f"[Launcher] Starting {name} ...")
        processes[name] = subprocess.Popen(cmd)

        # Small stagger to avoid race conditions
        time.sleep(2)

    print("\n[Launcher] All components started successfully.")
    print("→ Log Streamer, Main SOC, and Streamlit dashboard are running.\n")

    # # Try to open Streamlit dashboard automatically
    # try:
    #     webbrowser.open("http://localhost:8501")
    # except Exception:
    #     pass

    print("Press Ctrl+C to stop all components.\n")

    # Keep launcher alive until user stops
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Launcher] Stopping all processes...")
        for proc in processes.values():
            safe_kill(proc)
        print("[Launcher] All stopped cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    main()
