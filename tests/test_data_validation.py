import pandas as pd
import pytest

@pytest.fixture
def data():
    return pd.read_csv("data/iris.csv")

def test_no_missing_values(data):
    assert not data.isnull().values.any(), "Dataset contains missing values"

def test_expected_columns(data):
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(data.columns) == expected_columns, f"Unexpected columns: {data.columns}"

def test_target_column_classes(data):
    unique_classes = data["species"].unique()
    assert len(unique_classes) == 3, "Target column should have 3 classes"
