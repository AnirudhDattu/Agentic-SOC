# app/dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ======== PAGE CONFIG ========
st.set_page_config(page_title="Agentic SOC Dashboard", layout="wide", page_icon="🧠")

# Optional dark theme (you can comment this out if you prefer light)
st.markdown("""
    <style>
    body, .stApp { background-color: #0E1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00BFFF; }
    h1, h2, h3, h4 { color: #00BFFF; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🧠 Agentic SOC 2.0 — LLM-Powered Cyber Defense</h1>",
    unsafe_allow_html=True
)
st.caption("Real-time AI-driven Security Operations Monitoring System")
st.divider()


# ======== READ LOGS ========
def read_logs(path="infra/action_log.json", max_lines=400):
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


# ======== SIDEBAR ========
df_preview = read_logs()  # pre-load for attack types

attack_types = sorted(df_preview["attack_type"].unique()) if not df_preview.empty else []
st.sidebar.header("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", True)
limit = st.sidebar.slider("Entries to show", 20, 400, 100)
attack_filter = st.sidebar.multiselect("Filter by Attack Type", attack_types)
severity_filter = st.sidebar.multiselect("Filter by Severity", ["Low", "Medium", "High"])

placeholder = st.empty()


# ======== DASHBOARD ========
def render_dashboard():
    df = read_logs(max_lines=limit)
    if df.empty:
        st.info("🕐 Waiting for logs... No data yet.")
        return

    if attack_filter:
        df = df[df["attack_type"].isin(attack_filter)]
    if severity_filter:
        df = df[df["severity"].isin(severity_filter)]

    total = len(df)
    attacks = df[df["attack_type"] != "None"]
    benign = df[df["attack_type"] == "None"]
    high = (df["severity"] == "High").sum()
    medium = (df["severity"] == "Medium").sum()
    low = (df["severity"] == "Low").sum()

    # ---- Summary Metrics ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🧾 Total Logs", total)
    c2.metric("⚠️ Detected Attacks", len(attacks))
    c3.metric("🔥 High Severity", high)
    c4.metric("🟡 Medium", medium)
    c5.metric("🟢 Low", low)

    st.divider()

    # ---- Charts Row 1 ----
    col1, col2 = st.columns([2, 1])

    with col1:
        if not attacks.empty:
            attack_counts = attacks["attack_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            fig1 = px.bar(
                attack_counts,
                x="Attack Type",
                y="Count",
                color="Attack Type",
                title="Attack Type Frequency (Live)",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with col2:
        if not df.empty:
            severity_counts = df["severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            fig2 = px.pie(
                severity_counts,
                names="Severity",
                values="Count",
                title="Severity Distribution",
                color="Severity",
                color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ---- Charts Row 2 ----
    col3, col4 = st.columns([1.5, 1])

    with col3:
        if not attacks.empty:
            fig3 = px.scatter(
                attacks,
                x="time",
                y="attack_type",
                color="severity",
                title="Attack Timeline",
                color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
            )
            fig3.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col4:
        high_df = df[df["severity"] == "High"].head(5)
        st.subheader("🚨 Recent High-Severity Incidents")
        if high_df.empty:
            st.info("No high-severity alerts in recent logs.")
        else:
            for _, row in high_df.iterrows():
                st.markdown(
                    f"""
                    <div style='padding:10px;
                                border:1px solid #ff4b4b;
                                border-radius:8px;
                                margin-bottom:6px;
                                background-color:#fff5f5;
                                color:#000;'>
                        <b>{row['attack_type']}</b> — {row['recommendation']}<br>
                        <small><b>Time:</b> {row['time']} | <b>Action:</b> {row['action']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()

    # ---- Data Table ----
    st.subheader("📊 Live Log Table")
    st.dataframe(
        df[["time", "attack_type", "severity", "recommendation", "action"]],
        use_container_width=True,
        height=400
    )


# ======== REFRESH LOOP ========
while True:
    render_dashboard()
    if not auto_refresh:
        break
    time.sleep(2)
    st.rerun()
