#infra/log_streamer.py

import pandas as pd
import time
import json
import random

# load dataset
df = pd.read_csv("data/network_logs.csv")

# select only numeric columns needed by model
cols = ['dur', 'spkts', 'dpkts']
df = df[cols]

# randomly stream 1 log every 2 seconds
while True:
    log = df.sample(1).iloc[0].to_dict()
    with open("infra/live_log.json", "w") as f:
        json.dump(log, f)
    print(f"[Streamer] Sent log: {log}")
    time.sleep(2)
