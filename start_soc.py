"""
Unified launcher for Agentic SOC.

Starts the log streamer, the main pipeline, and the frontend dev server as
three real subprocesses. Ctrl+C (or just closing this terminal) tears down
all three, including the frontend's node.exe child — plain
subprocess.terminate() leaves that orphaned on Windows since npm's process
tree isn't killed by killing just the npm.cmd wrapper, so shutdown uses
`taskkill /T` to kill each process's whole tree instead.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")


def kill_tree(proc: subprocess.Popen, label: str):
    if proc.poll() is not None:
        return
    print(f"[Launcher] Stopping {label} (pid {proc.pid}) ...")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    if not os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")):
        print("[Launcher] ⚠️ frontend/node_modules not found — run `npm install` inside frontend/ first.")
        sys.exit(1)

    print("\n=== 🚀 Launching Agentic SOC ===\n")

    procs = []
    try:
        print("[Launcher] Starting Log Streamer ...")
        procs.append(("Log Streamer", subprocess.Popen(
            [sys.executable, "-u", "infra/log_streamer.py"], cwd=ROOT,
            start_new_session=(os.name != "nt"),
        )))
        time.sleep(1)

        print("[Launcher] Starting Main SOC Pipeline ...")
        procs.append(("Main Pipeline", subprocess.Popen(
            [sys.executable, "-u", "main.py"], cwd=ROOT,
            start_new_session=(os.name != "nt"),
        )))
        time.sleep(1)

        print("[Launcher] Starting Frontend ...")
        procs.append(("Frontend", subprocess.Popen(
            "npm run dev", cwd=FRONTEND_DIR, shell=True,
            start_new_session=(os.name != "nt"),
        )))

        print("\n[Launcher] All components started.")
        print("Visit: http://localhost:8080 (auto-bumps if that port's taken)\n")
        print("Press Ctrl+C to stop everything.\n")

        while True:
            time.sleep(1)
            for label, proc in procs:
                if proc.poll() is not None:
                    print(f"\n[Launcher] {label} exited unexpectedly (code {proc.returncode}) — shutting down the rest.")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[Launcher] 🛑 Shutting down...")
    finally:
        for label, proc in procs:
            kill_tree(proc, label)
        print("[Launcher] All stopped.")


if __name__ == "__main__":
    main()
