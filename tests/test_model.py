import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
import numpy as np

def test_random_forest_accuracy():
    df = pd.read_csv("data/iris.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = joblib.load("label_encoder.joblib")
    y_encoded = le.transform(y)

    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = joblib.load("model_rf.joblib")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"RandomForest accuracy: {acc}")
    assert acc > 0.8

def test_keras_model_accuracy():
    df = pd.read_csv("data/iris.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = joblib.load("label_encoder.joblib")
    y_encoded = le.transform(y)

    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = tf.keras.models.load_model("model.keras")
    preds = model.predict(X_test)
    pred_class = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_class)
    print(f"Keras model accuracy: {acc}")
    assert acc > 0.7
