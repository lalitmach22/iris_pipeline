import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import mlflow
from mlflow.models import infer_signature
from google.cloud import aiplatform, storage

print("Demo for week8")
# --- Configuration ---
# In a real pipeline, these would come from environment variables or a config file
PROJECT_ID = "mlopsweek1"  
LOCATION = "us-central1"
BUCKET_URI = "gs://week4_mlops_bucket" # Replace with your bucket URI

MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-8"
REPOSITORY = "iris-classifier-repo"
IMAGE = "iris-classifier-img"
MODEL_DISPLAY_NAME = "iris-classifier"

# --- Initialization ---
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
mlflow.set_tracking_uri("http://35.202.173.100:8100")
mlflow.set_experiment("Iris_Classification_Experiment")

# --- Helper Function for GCS Upload ---
def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")

# --- Main Logic ---
# 1. Load Data
data = pd.read_csv('data/iris.csv')

# 2. Split Data
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# 3. Train Model
params = {
    "max_depth": 4,
    "random_state": 1
}
mod_dt = DecisionTreeClassifier(**params)
mod_dt.fit(X_train, y_train)

# 4. Evaluate Model
prediction = mod_dt.predict(X_test)
accuracy_score = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy_score))

# 5. Save and Upload Artifact
os.makedirs("artifacts", exist_ok=True)
joblib.dump(mod_dt, "artifacts/model.joblib")

# Use the Python function for GCS upload
bucket_name_str = BUCKET_URI.replace("gs://", "")
model_gcs_path = f"{MODEL_ARTIFACT_DIR}/model.joblib"
upload_to_gcs(bucket_name_str, "artifacts/model.joblib", model_gcs_path)

# 6. Log Experiment with MLflow
with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy_score)
    mlflow.set_tag("Training Info", "Decision tree model for IRIS data")

    signature = infer_signature(X_train, mod_dt.predict(X_train))
    
    model_info = mlflow.sklearn.log_model(
        sk_model=mod_dt,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train.head(1),
        registered_model_name="IRIS-classifier-decisiontrees",
    )
    print(f"MLflow Run completed. Run ID: {run.info.run_id}")
