# infra/log_streamer.py
import pandas as pd
import time
import json
import os

# Load the UNSW dataset
df = pd.read_csv("data/network_logs.csv")

# Columns the ML model expects, plus attack_cat: the dataset's real ground-truth
# signature tag (Generic/Exploits/Fuzzers/DoS/Reconnaissance/Analysis/Backdoor/
# Shellcode/Worms/Normal). Carried through as sensor metadata for the reasoning
# stage — not used by the ML model itself, which only sees the 19 numeric features.
cols = [
    'dur','spkts','dpkts','sbytes','dbytes','sttl','dttl',
    'sload','dload','sinpkt','dinpkt','sjit','djit',
    'swin','stcpb','dtcpb','ct_srv_src','ct_srv_dst','ct_dst_sport_ltm',
    'attack_cat',
]

# Keep only required columns
df = df[cols]

# Stream one log every 2 seconds
while True:
    log = df.sample(1).iloc[0].to_dict()
    tmp_path = "infra/live_log.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(log, f)

    for attempt in range(5):
        try:
            os.replace(tmp_path, "infra/live_log.json")
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        print("[Streamer] ⚠️ Skipped write, live_log.json was locked.")

    print(f"[Streamer] Sent log: {log}")
    time.sleep(2)
