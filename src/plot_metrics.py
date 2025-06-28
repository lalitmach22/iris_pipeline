# src/plot_metrics.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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

# === Metrics ===
acc = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# === Plot ===
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax[0], cmap="Blues", values_format='d')
ax[0].set_title("Confusion Matrix")

# Classification Report
import numpy as np
import matplotlib.pyplot as plt

report_df = pd.DataFrame(report_dict).transpose()
ax[1].axis("off")
table = ax[1].table(cellText=report_df.round(2).values,
                    rowLabels=report_df.index,
                    colLabels=report_df.columns,
                    cellLoc='center',
                    loc='center')
table.scale(1, 1.5)
ax[1].set_title("Classification Report", pad=20)

# Save plot
plt.tight_layout()
plt.savefig("metrics.png")
print("Saved metrics to metrics.png")
