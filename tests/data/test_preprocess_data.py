import pytest
import pandas as pd
from src.data.preprocess_data import DataPreprocessor

# Fixtures 
@pytest.fixture
def raw_dataframe():
    """Returns a sample raw DataFrame with missing values."""
    return pd.DataFrame({
        "symbol": ["AAPL", "GOOG", None],
        "price": [150.0, None, 300.0],
        "volume": [1000, 2000, 3000]
    })

@pytest.fixture
def clean_dataframe():
    """Returns a DataFrame with no missing values."""
    return pd.DataFrame({
        "symbol": ["AAPL", "GOOG"],
        "price": [150.0, 120.0],
        "volume": [1000, 2000]
    })

# === Test Case: TC20250615_PreProcess_001 ===
# Description : Test that during preprocess all the rows with a 'None'/missing value gets dropped.
# Component   : src/data/preprocess_data.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-15 21:21
def test_drop_missing_removes_nan_rows(raw_dataframe):
    processor = DataPreprocessor()
    result = processor.drop_missing(raw_dataframe)

    # Only one row should remain (no NaNs)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.isna().sum().sum() == 0

# === Test Case: TC20250615_PreProcess_002===
# Description : Test that preprocess class selects the passed columns from the dataframe.
# Component   : src/data/preprocess_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-15 21:37
def test_select_columns_with_existing_columns(clean_dataframe):
    processor = DataPreprocessor(required_columns=["symbol", "price"])
    result = processor.select_columns(clean_dataframe)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["symbol", "price"]

# === Test Case: TC20250615_PreProcess_003 ===
# Description : Test that nonexistent passed columns are ignored from the dataframe.
# Component   : src/data/preprocess_data.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-15 21:40
def test_select_columns_with_missing_column(clean_dataframe):
    processor = DataPreprocessor(required_columns=["symbol", "price", "nonexistent"])
    result = processor.select_columns(clean_dataframe)

    # It should only keep existing columns, ignore 'nonexistent'
    assert "nonexistent" not in result.columns
    assert "symbol" in result.columns
    assert "price" in result.columns

# === Test Case: TC20250615_PreProcess_004 ===
# Description : Test that Preprocess selects the specified columns and also removes the rows with 'None'/missing values.
# Component   : src/data/preprocess_data.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-15 21:43
def test_preprocess_pipeline(raw_dataframe):
    processor = DataPreprocessor(required_columns=["symbol", "price"])
    result = processor.preprocess(raw_dataframe)

    # Should return a DataFrame with dropped NaNs and only selected columns
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)  # One clean row, two selected columns
    assert list(result.columns) == ["symbol", "price"]

# === Test Case: TC20250615_PreProcess_005 ===
# Description : Test that when no columns are specified, the returned dataframe should consists all the original columns.
# Component   : src/data/preprocess_data.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-15 21:48
def test_no_column_filtering(clean_dataframe):
    processor = DataPreprocessor()  # No filtering
    result = processor.select_columns(clean_dataframe)

    # Should return original columns
    assert list(result.columns) == ["symbol", "price", "volume"]
