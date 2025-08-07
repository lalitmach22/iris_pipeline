import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_artifacts_exist():
    """
    Tests if the essential model artifacts from training exist.
    """
    print("\n--- Testing for artifact existence ---")
    assert os.path.exists("artifacts/model.joblib"), "Model artifact not found."
    assert os.path.exists("artifacts/label_encoder.joblib"), "Label encoder artifact not found."
    print("Artifacts exist. Test passed. ✓")

def test_model_accuracy():
    """
    Tests if the trained model's accuracy on the test set is above a
    reasonable threshold.
    """
    print("\n--- Testing model accuracy ---")
    # === Load data ===
    df = pd.read_csv("data/iris.csv")

    # === Load model from the correct path ===
    try:
        model = joblib.load("artifacts/model.joblib")
    except FileNotFoundError:
        assert False, "model.joblib not found. Run train.py first."

    # === Prepare data dynamically based on model's expected features ===
    try:
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['species']
        print("correct features")
    except AttributeError:
        # Fallback for older models or different library versions
        print("Warning: 'feature_names_in_' not found. Falling back to dropping columns.")
        X = df.drop(columns=['species', 'location'], errors='ignore')
        y = df['species']


    # === Re-create the exact same train/test split as in training ===
    # This is critical for a valid test.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # === Predict ===
    predictions = model.predict(X_test)

    # === Assert accuracy is above a threshold ===
    acc = accuracy_score(y_test, predictions)
    print(f"Model accuracy on test set: {acc:.3f}")
    assert acc > 0.9, f"Accuracy ({acc}) is below the 0.9 threshold."
    print("Accuracy test passed. ✓")

if __name__ == "__main__":
    # This allows running tests directly for debugging
    test_artifacts_exist()
    test_model_accuracy()

