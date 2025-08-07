import pandas as pd
import joblib
import json
import os
from fairlearn.metrics import MetricFrame, demographic_parity_difference

def check_model_fairness():
    """
    Loads the trained model and assesses its fairness based on the 'location'
    sensitive feature.
    """
    print("--- Checking Model Fairness ---")

    # Load model and data
    try:
        model = joblib.load("artifacts/model.joblib")
        df = pd.read_csv("data/iris.csv")
        
        if 'location' not in df.columns:
            print("Error: 'location' column not found in data. Please run prepare_data.py first.")
            return

        X = df.drop(columns=['species', 'location']) # Exclude location from features
        y_true = df['species']
        sensitive_feature = df['location'] # Use the new column directly
        
        print("Model and data with 'location' feature loaded.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure artifacts and data exist.")
        return

    # Get model predictions
    y_pred = model.predict(X)

    # Use Fairlearn's MetricFrame to assess fairness
    # We will check the selection rate for the 'versicolor' class as an example
    metrics = {
        'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
        'selection_rate': lambda y_true, y_pred: (y_pred == 'versicolor').mean()
    }
    
    grouped_on_feature = MetricFrame(metrics=metrics,
                                     y_true=y_true,
                                     y_pred=y_pred,
                                     sensitive_features=sensitive_feature)

    print("\nFairness metrics by group (0 vs 1):")
    print(grouped_on_feature.by_group)

    # Calculate overall fairness metrics
    fairness_report = {
        "demographic_parity_difference": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    }

    print("\nOverall Fairness Report:")
    print(json.dumps(fairness_report, indent=2))

    # Save report to artifacts
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/fairness_report.json", "w") as f:
        json.dump(fairness_report, f, indent=4)

    print("\nFairness report saved to artifacts/fairness_report.json")
    print("-----------------------------\n")

if __name__ == "__main__":
    check_model_fairness()
