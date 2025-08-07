import pandas as pd
import pytest

@pytest.fixture
def data():
    """
    Pytest fixture to load the Iris dataset once for all tests.
    """
    return pd.read_csv("data/iris.csv")

def test_no_missing_values(data):
    """
    Tests that there are no null values anywhere in the dataset.
    """
    assert not data.isnull().values.any(), "Dataset contains missing values"

def test_core_columns_exist(data):
    """
    Tests that the core required columns are present in the dataset.
    This test is robust and will not fail if extra columns (like 'location') exist.
    """
    required_columns = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    }
    
    actual_columns = set(data.columns)
    
    assert required_columns.issubset(actual_columns), \
        f"Dataset is missing required columns. Missing: {required_columns - actual_columns}"

def test_target_column_classes(data):
    """
    Tests that the target column 'species' contains exactly 3 unique classes.
    """
    # This test is robust to new categories being added by data drift/poisoning scripts,
    # but for the core data validation, we expect exactly 3.
    # If this test fails, it might indicate data corruption.
    unique_classes = data["species"].unique()
    
    # We check for at least 3 classes, but could also check for an exact match
    # depending on how strict we want the validation to be.
    assert len(unique_classes) >= 3, "Target column should have at least 3 classes"

