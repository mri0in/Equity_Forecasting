import os
import pytest
import pandas as pd
from unittest.mock import patch
from src.data.load_csv import CSVLoader
from tests.mocks.load_csv_mocks import get_fake_dataframe


@pytest.fixture
def sample_csv(tmp_path):
    test_data = "date,open,close\n2023-01-01,100,110\n2023-01-02,110,115"
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(test_data)
    return str(file_path)
  
# === Test Case: TC20250611_CSV_001 ===
# Description : Test Valid CSV file loads correctly.
# Component   : src/data/load_csv.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-11 14:44
def test_load_valid_csv(sample_csv):
    loader = CSVLoader(sample_csv)
    df = loader.load_csv()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["date", "open", "close"]

# === Test Case: TC20250611_CSV_002 ===
# Description : Test Non-existent file path and raise FileNotFoundError.
# Component   : src/data/load_csv.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-11 14:52
def test_load_csv_file_not_found():
    loader = CSVLoader("non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_csv()

# === Test Case: TC20250611_CSV_003 ===
# Description : Test: Input type validation by passing int type instead of str
# Component   : src/data/load_csv.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-11 14:55
def test_load_csv_invalid_input_type():
    with pytest.raises(TypeError):
        CSVLoader(12345) # Invalid type: int instead of str

# === Test Case: TC20250611_CSV_004 ===
# Description : Test to load a CSV file with mocked dataframe.
# Component   : src/data/load_csv.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-11 14:57
@patch("src.data.load_csv.os.path.exists", return_value=True)
@patch("src.data.load_csv.pd.read_csv")
def test_mocked_read_csv(mock_read_csv, mock_exists):
    """
    Test with mocked pandas.read_csv using a synthetic DataFrame.
    """
    fake_df = get_fake_dataframe()
    mock_read_csv.return_value = fake_df

    loader = CSVLoader("fake/path.csv")
    result = loader.load_csv()

    assert result.equals(fake_df)
    mock_read_csv.assert_called_once_with("fake/path.csv")

@pytest.fixture
def malformed_csv(tmp_path):
    content = "date,open\n2023-01-01,100\n2023-01-02"  # Missing one column
    path = tmp_path / "malformed.csv"
    path.write_text(content)
    return str(path)

# === Test Case: TC20250612_CSV_005 ===
# Description : Test that malformed CSVs with missing fields are handled without crashing.
# Component   : src/data/load_csv.py
# Category    : Unit 
# Author      : Mri
# Created On  : 2025-06-12 18:36
def test_load_csv_malformed_csv(malformed_csv):
    loader = CSVLoader(malformed_csv)

    df = loader.load_csv()

    # Assert DataFrame is not empty
    assert not df.empty

    # Assert that there's at least one NaN (missing field)
    assert df.isna().sum().sum() > 0

@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")  
    return str(path)

# === Test Case: TC20250612_CSV_006 ===
# Description : Test to check a exception is raised when loading an empty csv file.
# Component   : src/data/load_csv.py
# Category    : Unit
# Author      : Mri
# Created On  : 2025-06-12 18:44
def test_load_csv_empty_file(empty_csv):
    loader = CSVLoader(empty_csv)
    with pytest.raises(pd.errors.EmptyDataError):
        loader.load_csv()

@pytest.fixture
def extra_columns_csv(tmp_path):
    content = "date,open,close\n2023-01-01,100\n2023-01-02,110,115,EXTRA"
    path = tmp_path / "malformed_path.csv"
    path.write_text(content)
    return str(path)

# === Test Case: TC20250612_CSV_007 ===
# Description : Test to check a malformed csv file is not loaded when there is an extra column of data.
# Component   : src/data/load_csv.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-12 18:50
def test_load_csv_extra_columns(extra_columns_csv):
    loader = CSVLoader(extra_columns_csv)
    
    with pytest.raises(pd.errors.ParserError):
        loader.load_csv()

@pytest.fixture
def header_only_csv(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text("date,open,close")
    return str(path)

# === Test Case: TC20250612_CSV_008 ===
# Description : Test to check csv file is loaded even when only header is present but file is empty.
# Component   : src/data/load_csv.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-12 18:52
def test_load_csv_header_only(header_only_csv):
    loader = CSVLoader(header_only_csv)
    df = loader.load_csv()
    assert df.empty

@pytest.fixture
def special_char_csv(tmp_path):
    content = "date,open,close\n2023-01-01,100,&110"
    path = tmp_path / "special_char.csv"
    path.write_text(content, encoding="utf-8")
    return str(path)

# === Test Case: TC20250612_CSV_009 ===
# Description : Test to check an entry with an special character is also accepted.
# Component   : src/data/load_csv.py
# Category    : Unit / Integration / Functional
# Author      : Mri
# Created On  : 2025-06-12 18:53
def test_load_csv_special_characters(special_char_csv):
    loader = CSVLoader(special_char_csv)
    df = loader.load_csv()

    assert df.iloc[0]["close"] == "&110"

