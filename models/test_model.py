# models/test_model.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load model
model = joblib.load("models/detector.pkl")

# Load testing dataset
df_test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# Use same columns as training
feature_cols = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_sport_ltm'
]

X_test = df_test[feature_cols]
y_test = df_test['label']

# Run predictions
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")
print("\nDetailed report:\n")
print(classification_report(y_test, y_pred))
