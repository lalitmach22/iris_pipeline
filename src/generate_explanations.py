import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def generate_global_explanations():
    """
    Loads the trained model and data to generate and save a global SHAP summary plot,
    showing overall feature importance.
    """
    print("--- Generating Global SHAP Explanations (Feature Importance) ---")

    # Load the trained model and the original dataset
    try:
        model = joblib.load("artifacts/model.joblib")
        df = pd.read_csv("data/iris.csv")
        
        # Prepare data dynamically based on model's expected features
        expected_features = model.feature_names_in_
        X = df[expected_features]
        
        print("Model and data loaded successfully.")
    except (FileNotFoundError, AttributeError, KeyError) as e:
        print(f"Error loading model or preparing data: {e}")
        return

    # Create a SHAP explainer for the tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Generate and save the summary bar plot
    print("Generating and saving SHAP summary bar plot...")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=model.classes_, show=False)
    plt.title("Global Feature Importance (SHAP Values)")
    plt.tight_layout()
    
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/shap_summary_global.png")
    
    print("Global SHAP summary plot saved to artifacts/shap_summary_global.png")
    print("-----------------------------------------------------------------\n")

def generate_individual_explanations():
    """
    Generates instance-level explanations using TreeExplainer and saves
    interactive force plots as HTML files.
    """
    print("--- Generating Individual SHAP Explanations (Force Plots) ---")

    # Load the trained model and the original dataset
    try:
        model = joblib.load("artifacts/model.joblib")
        df = pd.read_csv("data/iris.csv")
        
        # Prepare data dynamically based on model's expected features
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['species']

        print("Model and data loaded successfully.")
    except (FileNotFoundError, AttributeError, KeyError) as e:
        print(f"Error loading model or preparing data: {e}")
        return

    # Create the same train/test split as in the training script
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    # --- FIXED: Use TreeExplainer for consistency and efficiency ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Save a force plot for a single prediction (e.g., the first test instance)
    print("Generating and saving force plot for a single prediction...")
    # We explain the prediction for the first class ('setosa')
    p = shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], matplotlib=False)
    shap.save_html("artifacts/shap_force_plot_single.html", p)
    print("Single instance force plot saved to artifacts/shap_force_plot_single.html")

    # 2. Save a force plot for all test predictions (stacked)
    print("Generating and saving force plot for all test predictions...")
    # We explain the predictions for the second class ('versicolor')
    p_all = shap.force_plot(explainer.expected_value[1], shap_values[1], X_test, matplotlib=False)
    shap.save_html("artifacts/shap_force_plot_all.html", p_all)
    print("Stacked force plot for all instances saved to artifacts/shap_force_plot_all.html")
    print("---------------------------------------------------------------------\n")


if __name__ == "__main__":
    generate_global_explanations()
    generate_individual_explanations()
