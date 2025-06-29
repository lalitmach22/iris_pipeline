# src/evaluate.py

import pandas as pd
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split

# === Load data ===
df = pd.read_csv("data/iris.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# === Load label encoder and model ===
le = joblib.load("label_encoder.joblib")
model = joblib.load("model_rf.joblib")

# === Encode labels and split ===
y_encoded = le.transform(y)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Predict ===
y_pred = model.predict(X_test)

# === Compute Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=le.classes_)

# === Save metrics to JSON ===
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# === Optional: Print for logs ===
print("Evaluation complete:")
print(json.dumps(metrics, indent=2))
print("\nClassification Report:\n")
print(report)
