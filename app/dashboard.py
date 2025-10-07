# app/dashboard.py 
import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px

st.set_page_config(page_title="Agentic SOC Dashboard", layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 Agentic SOC 2.0 — LLM-Powered Cyber Defense</h1>", unsafe_allow_html=True)
st.divider()

def read_logs(path="infra/action_log.json", max_lines=300):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    data = [json.loads(l) for l in lines[-max_lines:]]
    df = pd.DataFrame(data)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time", ascending=False)
    return df

st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", True)
limit = st.sidebar.slider("Entries to show", 20, 300, 100)
attack_filter = st.sidebar.multiselect("Attack Type", [])
severity_filter = st.sidebar.multiselect("Severity", ["Low", "Medium", "High"])

placeholder = st.empty()

def render_dashboard():
    df = read_logs(max_lines=limit)
    if df.empty:
        st.info("Waiting for logs...")
        return
    if attack_filter:
        df = df[df["attack_type"].isin(attack_filter)]
    if severity_filter:
        df = df[df["severity"].isin(severity_filter)]

    # Stats
    total = len(df)
    high = (df["severity"] == "High").sum()
    medium = (df["severity"] == "Medium").sum()
    low = (df["severity"] == "Low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Logs", total)
    c2.metric("High Severity", high)
    c3.metric("Medium", medium)
    c4.metric("Low", low)

    # Chart
    if not df.empty:
        fig = px.bar(df.groupby("attack_type")["severity"].count().reset_index(),
                     x="attack_type", y="severity",
                     color="attack_type", title="Attack Type Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(df[["time", "attack_type", "severity", "recommendation", "action"]],
                 use_container_width=True)

while True:
    render_dashboard()
    if not auto_refresh:
        break
    time.sleep(2)
    st.rerun()
