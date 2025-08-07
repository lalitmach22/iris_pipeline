import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split

def evaluate_model():
    """
    Loads artifacts, evaluates the model on the correct test set,
    and saves the performance metrics.
    """
    print("--- Starting Model Evaluation ---")

    # === Load data ===
    df = pd.read_csv("data/iris.csv")

    # === Load artifacts ===
    try:
        model = joblib.load("artifacts/model.joblib")
        print("Model artifact loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.py has run and created artifacts.")
        return

    # === Prepare data dynamically based on model's expected features ===
    # This is the robust way to avoid hardcoding column names.
    try:
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['species']
    except AttributeError:
        # Fallback for older scikit-learn versions or models without this attribute
        print("Warning: 'feature_names_in_' not found. Falling back to dropping columns.")
        X = df.drop(columns=['species', 'location'], errors='ignore')
        y = df['species']
    except KeyError as e:
        print(f"Error: Model was trained on features not present in the new data: {e}")
        return


    # === Re-create the exact same train/test split as in training ===
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    print(f"Test set created with {len(X_test)} samples.")

    # === Predict ===
    # X_test now has the correct features, matching what the model was trained on.
    y_pred = model.predict(X_test)

    # === Compute Metrics ===
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    # === Save metrics to JSON ===
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
    print("--- Evaluation Complete ---\n")


if __name__ == "__main__":
    evaluate_model()
