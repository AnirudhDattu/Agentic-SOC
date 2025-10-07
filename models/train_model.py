# models/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/network_logs.csv")

# Define rich feature set
features = [
    'dur','spkts','dpkts','sbytes','dbytes','sttl','dttl',
    'sload','dload','sinpkt','dinpkt','sjit','djit',
    'swin','stcpb','dtcpb','ct_srv_src','ct_srv_dst','ct_dst_sport_ltm'
]

# Input / output split
X = df[features]
y = df['label']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model with stronger parameters
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=20,
    min_samples_split=3,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/detector.pkl")
print("✅ Model trained and saved with", len(features), "features.")
