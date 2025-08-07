import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import mlflow
from mlflow.models import infer_signature
from google.cloud import aiplatform, storage

print("DEMO")
# --- Configuration ---
# In a real pipeline, these would come from environment variables or a config file
PROJECT_ID = "mlopsweek1"  # Replace with your Project ID
LOCATION = "us-central1"
BUCKET_URI = "gs://mlops-course-mlopsweek1-unique" # Replace with your bucket URI

MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-8"
REPOSITORY = "iris-classifier-repo"
IMAGE = "iris-classifier-img"
MODEL_DISPLAY_NAME = "iris-classifier"

# --- MLflow Tracking URI based on environment ---
if os.getenv('CI'):
    # In CI, save MLflow data locally
    mlflow_tracking_uri = "file:./mlruns"
    print(f"CI environment detected. Using local MLflow tracking URI: {mlflow_tracking_uri}")
    REGISTERED_MODEL_NAME = ""
else:
    # Replace with the actual external IP of your GCP instance
    EXTERNAL_IP = "http://34.59.44.241:8100"  # Replace this dynamically if needed
    mlflow_tracking_uri = EXTERNAL_IP
    print(f"Local environment detected. Using remote MLflow tracking URI: {mlflow_tracking_uri}")
    REGISTERED_MODEL_NAME = "IRIS-classifier-decisiontrees"

# --- Initialize clients and MLflow ---
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
mlflow.set_tracking_uri(mlflow_tracking_uri)
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

### FIXED ###: Create, fit, and save the LabelEncoder
# 3. Fit Label Encoder
print("Fitting LabelEncoder...")
le = LabelEncoder()
le.fit(y_train)

# 4. Train Model
params = {
    "max_depth": 4,
    "random_state": 1
}
mod_dt = DecisionTreeClassifier(**params)
mod_dt.fit(X_train, y_train)

# 5. Evaluate Model
prediction = mod_dt.predict(X_test)
accuracy_score = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy_score))

# 6. Save and Upload Artifacts (Model AND Encoder)
os.makedirs("artifacts", exist_ok=True)
print("Saving model and label encoder artifacts...")
joblib.dump(mod_dt, "artifacts/model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib") ### FIXED ###: Save the encoder

# Use the Python function for GCS upload
bucket_name_str = BUCKET_URI.replace("gs://", "")
model_gcs_path = f"{MODEL_ARTIFACT_DIR}/model.joblib"
encoder_gcs_path = f"{MODEL_ARTIFACT_DIR}/label_encoder.joblib" ### FIXED ###: Define GCS path for encoder

upload_to_gcs(bucket_name_str, "artifacts/model.joblib", model_gcs_path)
upload_to_gcs(bucket_name_str, "artifacts/label_encoder.joblib", encoder_gcs_path) ### FIXED ###: Upload the encoder

# 7. Log Experiment with MLflow
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
        # Use the variable to conditionally register the model
        registered_model_name=REGISTERED_MODEL_NAME if REGISTERED_MODEL_NAME else None, ### FIXED ###
    )
    print(f"MLflow Run completed. Run ID: {run.info.run_id}")
