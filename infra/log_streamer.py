# infra/log_streamer.py
import pandas as pd
import time
import json

# Load the UNSW dataset
df = pd.read_csv("data/network_logs.csv")

# Columns that model expects
cols = [
    'dur','spkts','dpkts','sbytes','dbytes','sttl','dttl',
    'sload','dload','sinpkt','dinpkt','sjit','djit',
    'swin','stcpb','dtcpb','ct_srv_src','ct_srv_dst','ct_dst_sport_ltm'
]

# Keep only required columns
df = df[cols]

# Stream one log every 2 seconds
while True:
    log = df.sample(1).iloc[0].to_dict()
    with open("infra/live_log.json", "w") as f:
        json.dump(log, f)
    print(f"[Streamer] Sent log: {log}")
    time.sleep(2)
