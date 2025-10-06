# app/dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="Agentic SOC 2.0 Dashboard", layout="wide")
st.title("🧠 Agentic SOC 2.0 — LLM-Powered Cyber Defense")

# ---------- helper to read latest log lines ----------
def read_logs(path="infra/action_log.json", max_lines=200):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    lines = lines[-max_lines:]
    data = []
    for l in lines:
        try:
            data.append(json.loads(l))
        except Exception:
            start = l.find("{"); end = l.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    data.append(json.loads(l[start:end]))
                except Exception:
                    continue
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time", ascending=False)
        except Exception:
            pass
    # ensure all expected columns exist
    for c in ["attack_type", "severity", "recommendation", "action"]:
        if c not in df.columns:
            df[c] = ""
    return df.reset_index(drop=True)

# ---------- sidebar ----------
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 2 s)", value=True)
limit = st.sidebar.slider("Show last N entries", 10, 300, 50)
st.sidebar.info("Colors indicate severity:\n🟥 High 🟨 Medium 🟩 Low")

placeholder = st.empty()

# ---------- function to apply row colors ----------
def highlight_severity(row):
    sev = str(row.get("severity", "")).lower()
    if sev.startswith("high"):
        color = "background-color: #ffcccc;"   # red
    elif sev.startswith("medium"):
        color = "background-color: #fff2cc;"   # yellow
    elif sev.startswith("low"):
        color = "background-color: #d9ead3;"   # green
    else:
        color = ""
    return [color] * len(row)

# ---------- main loop ----------
while True:
    df = read_logs(max_lines=limit)
    if df.empty:
        placeholder.info("Waiting for logs... (run main.py)")
    else:
        view = df[["time", "attack_type", "severity", "recommendation", "action"]].copy()
        styled = view.style.apply(highlight_severity, axis=1)
        placeholder.dataframe(styled, use_container_width=True)
    if not auto_refresh:
        break
    time.sleep(2)
