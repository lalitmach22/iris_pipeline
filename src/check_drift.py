from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

import pandas as pd
import os


def generate_drift_report(reference_path, output_dir="artifacts"):
    """
    Generates a data drift and summary report using Evidently AI.

    It compares the reference dataset against a new, artificially drifted
    version and saves the report as an HTML file.

    Args:
        reference_path (str): Path to the reference (original) dataset.
        output_dir (str): Directory to save the output report.
    """
    print("--- Generating Data Drift Report with Evidently AI ---")

    # 1. Load the reference data
    try:
        reference_df = pd.read_csv(reference_path)
        print(f"Reference data loaded from {reference_path}")
    except FileNotFoundError:
        print(f"Error: Reference data not found at {reference_path}")
        return

    # 2. Create the "current" (drifted) data based on your logic
    print("Creating an artificially drifted 'current' dataset...")
    current_df = reference_df.copy()
    # Intentionally create unseen, noisy data
    noise_data = current_df[current_df['sepal_length'] > 7.5].copy().reset_index(drop=True)
    if not noise_data.empty:
        noise_data['sepal_length'] = 15.0
        noise_data['species'] = 'fakeiris'
        noise_data['petal_length'] = 100.0
        current_df = pd.concat([current_df, noise_data], ignore_index=True)
        print("Noise and new category 'fakeiris' added to current data.")
    else:
        print("No data met the criteria for noise generation (sepal_length > 7.5).")


    # 3. Define the data schema for Evidently
    # This helps Evidently understand column types. We'll include the 'location'
    # column if it exists, as it's part of the dataset now.
    numerical_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    categorical_cols = ["species"]
    if 'location' in reference_df.columns:
        categorical_cols.append('location')

    # 4. Run the drift and summary report
    print("Running Evidently report...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])
    drift_report.run(reference_data=reference_df, current_data=current_df)

    # 5. Save the report to the artifacts directory
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_drift_and_summary_report.html")
    drift_report.save_html(report_path)

    print(f"\nEvidently report saved successfully to {report_path}")
    print("------------------------------------------------------\n")


if __name__ == "__main__":
    # In the pipeline, we'll always use the main data file as the reference
    generate_drift_report(reference_path="data/iris.csv")
