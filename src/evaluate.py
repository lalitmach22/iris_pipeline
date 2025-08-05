import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

def evaluate_model():
    """
    Loads artifacts, evaluates the model on the correct test set,
    and saves the performance metrics.
    """
    print("--- Starting Model Evaluation ---")

    # === Load data ===
    # This must be the same dataset used in training
    df = pd.read_csv("data/iris.csv")

    # === Load label encoder and model from the correct path ===
    try:
        le = joblib.load("artifacts/label_encoder.joblib")
        model = joblib.load("artifacts/model.joblib")
        print("Artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.py has run and created artifacts.")
        return

    # === Re-create the exact same train/test split as in training ===
    # This is critical for correct evaluation.
    _, X_test, _, y_test_labels = train_test_split(
        df.drop(columns=['species']),
        df['species'],
        test_size=0.4,
        random_state=42,
        stratify=df['species']
    )
    print(f"Test set created with {len(X_test)} samples.")

    # === Predict ===
    y_pred_labels = model.predict(X_test)

    # === Compute Metrics ===
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average="macro")
    recall = recall_score(y_test_labels, y_pred_labels, average="macro")
    f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
    report = classification_report(y_test_labels, y_pred_labels, target_names=le.classes_)

    # === Save metrics to JSON in the artifacts directory ===
    metrics_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    print("Metrics saved to artifacts/metrics.json")

    # === Print reports for logs ===
    print("\n--- Evaluation Complete ---")
    print(json.dumps(metrics_data, indent=2))
    print("\n--- Classification Report ---\n")
    print(report)
    print("---------------------------\n")


if __name__ == "__main__":
    evaluate_model()
