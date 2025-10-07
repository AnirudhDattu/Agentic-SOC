# app/dashboard.py
import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import streamlit.components.v1 as components
import warnings
import random
from datetime import datetime

# Optional: import ollama, guard with try/except
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------------
# Page config and style
# -------------------------
st.set_page_config(page_title="Agentic SOC Dashboard", layout="wide", page_icon="🧠")

st.markdown("""
<style>
body, .stApp { background-color: #0E1117; color: white; }
div[data-testid="stMetricValue"] { color: #00BFFF; }
h1, h2, h3, h4 { color: #00BFFF; }
.alert-box {
    padding:10px;
    border:1px solid #ff4b4b;
    border-radius:8px;
    margin-bottom:6px;
    background-color:#fff5f5;
    color:#000;
}
.small-muted { color: #bfbfbf; font-size:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🧠 Agentic SOC 2.0 — LLM-Powered Cyber Defense</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:lime;'>● Live Mode Active</div>", unsafe_allow_html=True)
st.caption("Real-time AI-driven Security Operations Monitoring System")
st.divider()

# -------------------------
# Helpers
# -------------------------
LOG_PATH = "infra/action_log.json"

def read_logs(path=LOG_PATH, max_lines=400):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    data = []
    for l in lines[-max_lines:]:
        try:
            data.append(json.loads(l))
        except Exception:
            # skip broken lines
            continue
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        # create a time column if missing using file read order
        df["time"] = pd.to_datetime(datetime.utcnow())
    df = df.sort_values("time", ascending=False)
    return df

def add_dummy_geo(df):
    """
    Adds dummy lat/lon for each unique source (source_ip or ct_srv_src fallback).
    Returns df with 'lat' and 'lon' columns.
    """
    df = df.copy()
    # determine a key for grouping
    if "source_ip" in df.columns:
        keys = df["source_ip"].fillna("unknown")
    elif "srcip" in df.columns:
        keys = df["srcip"].fillna("unknown")
    else:
        # use ct_srv_src if present or source port fields
        if "ct_srv_src" in df.columns:
            keys = df["ct_srv_src"].astype(str).fillna("unknown")
        else:
            keys = pd.Series([f"row{i}" for i in range(len(df))], index=df.index)

    unique_keys = keys.unique().tolist()
    random.seed(42)
    key_to_coord = {}
    for k in unique_keys:
        # sample sensible lat/lon clusters to look realistic
        lat = random.uniform(-55, 70)
        lon = random.uniform(-170, 170)
        key_to_coord[k] = (lat, lon)

    lats = []
    lons = []
    for k in keys:
        lat, lon = key_to_coord.get(k, (0, 0))
        lats.append(lat)
        lons.append(lon)

    df["lat"] = lats
    df["lon"] = lons
    return df

def compute_performance(df):
    """
    Compute basic performance metrics if timing fields are present.
    Returns dict with keys: avg_detection_time, avg_reasoning_time, avg_response_time, detection_accuracy
    If fields are missing, returns None for that metric.
    """
    out = {"avg_detection_time": None, "avg_reasoning_time": None, "avg_response_time": None, "detection_accuracy": None}

    # detection_time field (seconds) may be present
    if "detection_time" in df.columns:
        try:
            out["avg_detection_time"] = float(df["detection_time"].dropna().astype(float).mean())
        except Exception:
            out["avg_detection_time"] = None

    # reasoning_time field (seconds)
    if "reasoning_time" in df.columns:
        try:
            out["avg_reasoning_time"] = float(df["reasoning_time"].dropna().astype(float).mean())
        except Exception:
            out["avg_reasoning_time"] = None

    # response_time field (seconds)
    if "response_time" in df.columns:
        try:
            out["avg_response_time"] = float(df["response_time"].dropna().astype(float).mean())
        except Exception:
            out["avg_response_time"] = None

    # detection_accuracy: if logs contain ground_truth field
    if "ground_truth" in df.columns and "status" in df.columns:
        try:
            # consider 'status' predicted label where 'Attack' maps to 1 else 0; ground_truth expected 0/1 or 'Attack'/'Benign'
            def to_label(v):
                if pd.isna(v): return None
                if isinstance(v, (int, float)):
                    return int(v)
                s = str(v).lower()
                if s in ("attack", "1", "true", "yes"): return 1
                return 0
            preds = df["status"].apply(to_label)
            truths = df["ground_truth"].apply(to_label)
            mask = preds.notna() & truths.notna()
            if mask.sum() > 0:
                out["detection_accuracy"] = float((preds[mask] == truths[mask]).mean()) * 100.0
        except Exception:
            out["detection_accuracy"] = None

    return out

# -------------------------
# Sidebar controls (dynamic)
# -------------------------
df_preview = read_logs()
attack_types = sorted(df_preview["attack_type"].dropna().unique()) if not df_preview.empty else []
st.sidebar.header("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", True)
limit = st.sidebar.slider("Entries to show", 20, 400, 100)
attack_filter = st.sidebar.multiselect("Filter by Attack Type", attack_types)
severity_filter = st.sidebar.multiselect("Filter by Severity", ["Low", "Medium", "High"])

# -------------------------
# Main render
# -------------------------
def render_dashboard():
    df = read_logs(max_lines=limit)
    if df.empty:
        st.info("🕐 Waiting for logs... No data yet.")
        return

    # Apply filters
    if attack_filter:
        df = df[df["attack_type"].isin(attack_filter)]
    if severity_filter:
        df = df[df["severity"].isin(severity_filter)]

    total = len(df)
    attacks = df[df["attack_type"].fillna("None") != "None"]
    high = (df["severity"] == "High").sum()
    medium = (df["severity"] == "Medium").sum()
    low = (df["severity"] == "Low").sum()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌍 Geo Map", "🧠 LLM Chat", "🚨 Incident Reports"])

    # -------------------------
    # TAB 1: Overview + performance
    # -------------------------
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🧾 Total Logs", total)
        c2.metric("⚠️ Detected Attacks", len(attacks))
        c3.metric("🔥 High Severity", high)
        c4.metric("🟡 Medium", medium)
        c5.metric("🟢 Low", low)

        st.divider()

        # charts
        col1, col2 = st.columns([2, 1])
        with col1:
            if not attacks.empty:
                attack_counts = attacks["attack_type"].value_counts().reset_index()
                attack_counts.columns = ["Attack Type", "Count"]
                fig = px.bar(attack_counts, x="Attack Type", y="Count", color="Attack Type",
                             title="Attack Type Frequency (Live)", color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            severity_counts = df["severity"].fillna("Unknown").value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            fig2 = px.pie(severity_counts, names="Severity", values="Count", title="Severity Distribution",
                          color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"})
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # performance metrics
        perf = compute_performance(df)
        p1, p2, p3 = st.columns(3)
        avg_det = f"{perf['avg_detection_time']:.3f}s" if perf['avg_detection_time'] else "N/A"
        avg_reason = f"{perf['avg_reasoning_time']:.3f}s" if perf['avg_reasoning_time'] else "N/A"
        avg_resp = f"{perf['avg_response_time']:.3f}s" if perf['avg_response_time'] else "N/A"
        acc = f"{perf['detection_accuracy']:.2f}%" if perf['detection_accuracy'] else "N/A"
        p1.metric("⏱️ Avg Detection Time", avg_det)
        p2.metric("🧠 Avg LLM Reasoning Time", avg_reason)
        p3.metric("⚡ Avg Response Time", avg_resp)
        st.caption(f"Detection accuracy (if ground truth present): {acc}")

        st.divider()
        st.subheader("Attack Timeline")
        if not attacks.empty:
            timeline = attacks.copy()
            if "time" not in timeline.columns:
                timeline["time"] = pd.to_datetime(datetime.utcnow())
            fig3 = px.scatter(timeline, x="time", y="attack_type", color="severity",
                              title="Attack Timeline", color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"})
            fig3.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # -------------------------
    # TAB 2: Geo heatmap (dummy)
    # -------------------------
    with tab2:
        st.subheader("🌍 Geo Heatmap of Attack Origins (dummy coordinates for demo)")
        geo_df = add_dummy_geo(df)
        # only plot attacks for clarity
        geo_attacks = geo_df[geo_df["attack_type"].fillna("None") != "None"]
        if geo_attacks.empty:
            st.info("No attack events to plot on map.")
        else:
            # aggregate by lat/lon
            agg = geo_attacks.groupby(["lat", "lon", "attack_type"]).size().reset_index(name="count")
            fig_map = px.scatter_geo(agg, lat="lat", lon="lon", size="count", color="attack_type",
                                     hover_name="attack_type", title="Attack Origins (demo coordinates)")
            fig_map.update_layout(geo=dict(showland=True))
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # -------------------------
    # TAB 3: Ollama LLM chat & summary
    # -------------------------
    with tab3:
        st.subheader("🧠 LLM Summary & Chat (local Ollama)")
        st.caption("You can ask the local LLM about recent logs. Ollama must be running locally for this tab to work.")
        recent_text = df.head(50).to_json(orient="records", date_format="iso")
        # quick autogenerated summary button
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Generate short summary (LLM)"):
                if not OLLAMA_AVAILABLE:
                    st.error("Ollama library not installed or unavailable. Install and run local Ollama server.")
                else:
                    try:
                        prompt = f"Provide a brief SOC summary of these recent events (json list):\n{recent_text}\n\nReturn a 2-3 sentence summary."
                        resp = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
                        content = resp.message.content if hasattr(resp, "message") else resp.get("message", {}).get("content", "")
                        st.success("LLM Summary:")
                        st.write(content)
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        # interactive chat
        with col_b:
            st.markdown("**Interactive Chat**")
        user_msg = st.text_input("Ask the LLM about the logs (e.g. 'What attacks in last 10 entries?')", value="")
        if st.button("Send to LLM") and user_msg.strip():
            if not OLLAMA_AVAILABLE:
                st.error("Ollama not available. Install `ollama` and run local Ollama.")
            else:
                try:
                    prompt = f"Logs: {recent_text}\n\nUser question: {user_msg}\nAnswer concisely."
                    resp = ollama.chat(model="phi3", messages=[{"role": "user", "content": prompt}])
                    content = resp.message.content if hasattr(resp, "message") else resp.get("message", {}).get("content", "")
                    st.markdown("**LLM Answer:**")
                    st.write(content)
                except Exception as e:
                    st.error(f"LLM error: {e}")

    # -------------------------
    # TAB 4: Incident reports + export
    # -------------------------
    with tab4:
        st.subheader("🚨 Incident Reports")
        display_df = df[["time", "attack_type", "severity", "recommendation", "action"]].copy()
        st.dataframe(display_df, use_container_width=True, height=480)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Logs (CSV)", csv, "Agentic_SOC_Logs.csv", "text/csv")

# -------------------------
# Live loop
# -------------------------
while True:
    render_dashboard()
    if not auto_refresh:
        break
    time.sleep(2)
    st.rerun()
