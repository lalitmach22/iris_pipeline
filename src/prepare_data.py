import pandas as pd
import numpy as np
import os

def add_sensitive_feature(data_path):
    """
    Loads the Iris dataset, adds a new artificial 'location' column
    with random binary values (0 or 1), and saves it back in-place.

    Args:
        data_path (str): The path to the iris.csv file.
    """
    print(f"--- Preparing data at: {data_path} ---")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # Check if the column already exists to avoid adding it multiple times
    if 'location' in df.columns:
        print("'location' column already exists. Skipping.")
        return

    # Add the new 'location' column with random 0s and 1s
    np.random.seed(42) # for reproducibility
    df['location'] = np.random.randint(0, 2, df.shape[0])
    
    # Save the modified DataFrame back to the same file
    df.to_csv(data_path, index=False)
    
    print("Successfully added 'location' column to the dataset.")
    print("--------------------------------------\n")

if __name__ == "__main__":
    add_sensitive_feature("data/iris.csv")
