# infra/clean_logs.py
import os, json
p = "infra/action_log.json"
if not os.path.exists(p):
    print("No log file.")
    raise SystemExit
lines = [l for l in open(p, "r", encoding="utf-8") if l.strip()]
good = []
for l in lines:
    try:
        json.loads(l)
        good.append(l)
    except:
        start = l.find("{"); end = l.rfind("}")+1
        if start!=-1 and end!=-1:
            try:
                j = json.loads(l[start:end])
                good.append(json.dumps(j) + "\n")
            except:
                continue
# keep last 500
open(p, "w", encoding="utf-8").write("".join(good[-500:]))
print("Cleaned log. Lines:", len(good))
