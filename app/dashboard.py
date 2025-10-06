#app/dashboard.py

import streamlit as st
import pandas as pd
import json
import time
import os

st.set_page_config(page_title="Agentic SOC", layout="wide")
st.title("🛰️ Agentic SOC Dashboard")

st.markdown("**Live AI-Driven Security Monitoring**")

placeholder = st.empty()

# Run continuous refresh loop
while True:
    # Read latest actions
    if os.path.exists("infra/action_log.json"):
        with open("infra/action_log.json") as f:
            lines = f.readlines()[-20:]  # last 20 actions
            logs = [json.loads(line) for line in lines]
            df = pd.DataFrame(logs)
            placeholder.dataframe(df)
    else:
        st.info("Waiting for live data...")

    time.sleep(2)
