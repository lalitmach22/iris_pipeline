import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def train_models():
    # Load and prepare data
    df = pd.read_csv("data/iris.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder for future use
    joblib.dump(le, "label_encoder.joblib")

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "model_rf.joblib")

    


    # Train Keras model
    keras_model = Sequential([
        tf.keras.Input(shape=(X.shape[1],)),  # Better than using input_shape in Dense
        Dense(16, activation='relu'),
        Dense(12, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes
    ])
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    keras_model.fit(X_train, y_train, epochs=50, verbose=0)
    keras_model.save("model.keras")

if __name__ == "__main__":
    train_models()

