import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_models():
    # Load and prepare data
    df = pd.read_csv("data/iris.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Encoded")  
  # Save label encoder for future use
    joblib.dump(le, "label_encoder.joblib")

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "model_rf.joblib")

 

if __name__ == "__main__":
    train_models()

