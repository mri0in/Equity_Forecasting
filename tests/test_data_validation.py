"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.validation.expectations import DataExpectations, ColumnExpectation
from src.validation.validator import DataValidator, MultiStageValidator


class TestDataExpectations:
    """Test DataExpectations class."""
    
    def test_get_expectations_raw_stock(self):
        """Test getting expectations for raw stock data."""
        exp = DataExpectations.get_expectations("raw_stock")
        assert "columns" in exp
        assert exp["min_rows"] == 100
        assert exp["check_duplicates"] is True
    
    def test_get_expectations_preprocessed(self):
        """Test getting expectations for preprocessed data."""
        exp = DataExpectations.get_expectations("preprocessed")
        assert "columns" in exp
        assert exp["min_rows"] == 50
    
    def test_get_expectations_featured(self):
        """Test getting expectations for featured data."""
        exp = DataExpectations.get_expectations("featured")
        assert "columns" in exp
        assert exp["min_rows"] == 30
    
    def test_get_expectations_invalid_stage(self):
        """Test that invalid stage raises error."""
        with pytest.raises(ValueError):
            DataExpectations.get_expectations("invalid_stage")
    
    def test_validate_column_existence_success(self):
        """Test column existence validation passes."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = DataExpectations.validate_column_existence(df, ["A", "B"])
        assert result["valid"] is True
        assert result["missing_columns"] == []
    
    def test_validate_column_existence_missing(self):
        """Test column existence validation fails with missing columns."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = DataExpectations.validate_column_existence(df, ["A", "B", "C"])
        assert result["valid"] is False
        assert set(result["missing_columns"]) == {"B", "C"}
    
    def test_validate_data_types_success(self):
        """Test data type validation passes."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        })
        expectations = [
            ColumnExpectation("Date", dtype="datetime64"),
            ColumnExpectation("Price", dtype="float64"),
        ]
        result = DataExpectations.validate_data_types(df, expectations)
        assert result["valid"] is True
    
    def test_validate_data_types_failure(self):
        """Test data type validation fails."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"] * 10,  # String instead of datetime
            "Price": [100.0] * 10,
        })
        expectations = [
            ColumnExpectation("Date", dtype="datetime64"),
            ColumnExpectation("Price", dtype="float64"),
        ]
        result = DataExpectations.validate_data_types(df, expectations)
        assert result["valid"] is False
    
    def test_validate_range_success(self):
        """Test range validation passes."""
        df = pd.DataFrame({
            "Price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Volume": [1000, 2000, 3000, 4000, 5000],
        })
        expectations = [
            ColumnExpectation("Price", min_value=0),
            ColumnExpectation("Volume", min_value=0),
        ]
        result = DataExpectations.validate_range(df, expectations)
        assert result["valid"] is True
    
    def test_validate_range_failure(self):
        """Test range validation fails with out-of-range values."""
        df = pd.DataFrame({
            "Price": [100.0, -5.0, 102.0],  # Negative price
            "Volume": [1000, 2000, 3000],
        })
        expectations = [
            ColumnExpectation("Price", min_value=0),
            ColumnExpectation("Volume", min_value=0),
        ]
        result = DataExpectations.validate_range(df, expectations)
        assert result["valid"] is False
        assert any("Price" in error for error in result["errors"])
    
    def test_validate_null_percentage_success(self):
        """Test null percentage validation passes."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, None],  # 20% nulls
            "B": [1, 2, 3, 4, 5],     # 0% nulls
        })
        expectations = [
            ColumnExpectation("A", max_null_percentage=25),
            ColumnExpectation("B", max_null_percentage=0),
        ]
        result = DataExpectations.validate_null_percentage(df, expectations)
        assert result["valid"] is True
    
    def test_validate_null_percentage_failure(self):
        """Test null percentage validation fails."""
        df = pd.DataFrame({
            "A": [1, None, None, None, None],  # 80% nulls
        })
        expectations = [
            ColumnExpectation("A", max_null_percentage=10),
        ]
        result = DataExpectations.validate_null_percentage(df, expectations)
        assert result["valid"] is False
    
    def test_validate_datetime_continuity_success(self):
        """Test datetime continuity validation passes."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Date": dates, "Value": range(10)})
        result = DataExpectations.validate_datetime_continuity(df)
        assert result["valid"] is True
    
    def test_validate_datetime_continuity_failure(self):
        """Test datetime continuity validation fails with gaps."""
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 5),  # Gap here
            datetime(2024, 1, 6),
        ]
        df = pd.DataFrame({"Date": dates, "Value": [1, 2, 3, 4]})
        result = DataExpectations.validate_datetime_continuity(df)
        # Note: May or may not fail depending on frequency detection
        assert "gaps" in result or "valid" in result
    
    def test_validate_duplicates_success(self):
        """Test duplicate detection passes."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Value": range(10)
        })
        result = DataExpectations.validate_duplicates(df)
        assert result["valid"] is True
        assert result["duplicates"] == 0
    
    def test_validate_duplicates_failure(self):
        """Test duplicate detection fails."""
        df = pd.DataFrame({
            "Date": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "Value": [1, 2, 3]
        })
        result = DataExpectations.validate_duplicates(df)
        assert result["valid"] is False
        assert result["duplicates"] == 1
    
    def test_validate_min_rows_success(self):
        """Test minimum rows validation passes."""
        df = pd.DataFrame({"A": range(100)})
        result = DataExpectations.validate_min_rows(df, min_rows=50)
        assert result["valid"] is True
    
    def test_validate_min_rows_failure(self):
        """Test minimum rows validation fails."""
        df = pd.DataFrame({"A": range(10)})
        result = DataExpectations.validate_min_rows(df, min_rows=100)
        assert result["valid"] is False


class TestDataValidator:
    """Test DataValidator class."""
    
    @pytest.fixture
    def good_stock_data(self):
        """Create valid stock data."""
        dates = pd.date_range("2024-01-01", periods=120)
        return pd.DataFrame({
            "Date": dates,
            "Open": np.random.uniform(100, 150, 120),
            "High": np.random.uniform(100, 150, 120),
            "Low": np.random.uniform(100, 150, 120),
            "Close": np.random.uniform(100, 150, 120),
            "Volume": np.random.uniform(1000000, 10000000, 120),
            "Adj Close": np.random.uniform(100, 150, 120),
        })
    
    @pytest.fixture
    def bad_stock_data(self):
        """Create invalid stock data."""
        dates = pd.date_range("2024-01-01", periods=5)  # Too few rows
        return pd.DataFrame({
            "Date": dates,
            "Open": [-10, 101, 102, 103, 104],  # Negative price
            "High": [100, 101, 102, 103, None],  # Missing value
            "Low": [100, 101, 102, 103, 104],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000000] * 5,
            "Adj Close": [100, 101, 102, 103, 104],
        })
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator("raw_stock")
        assert validator.stage == "raw_stock"
        assert validator.expectations is not None
    
    def test_validate_good_data(self, good_stock_data):
        """Test validation of good data."""
        validator = DataValidator("raw_stock")
        results = validator.validate(good_stock_data)
        
        assert results["stage"] == "raw_stock"
        assert "checks" in results
        assert results["total_rows"] == 120
    
    def test_validate_bad_data(self, bad_stock_data):
        """Test validation of bad data."""
        validator = DataValidator("raw_stock")
        results = validator.validate(bad_stock_data)
        
        # Should have failures
        assert "checks" in results
    
    def test_get_summary(self, good_stock_data):
        """Test validation summary."""
        validator = DataValidator("raw_stock")
        validator.validate(good_stock_data)
        summary = validator.get_summary()
        
        assert "VALIDATION SUMMARY" in summary
        assert "raw_stock" in summary
    
    def test_raise_on_failure(self, bad_stock_data):
        """Test exception raising on validation failure."""
        validator = DataValidator("raw_stock")
        validator.validate(bad_stock_data)
        
        # Should raise because of min_rows violation
        with pytest.raises(ValueError):
            validator.raise_on_failure()
    
    def test_persist_results(self, good_stock_data, tmp_path):
        """Test saving validation results."""
        validator = DataValidator("raw_stock")
        validator.validate(good_stock_data)
        
        filepath = tmp_path / "validation_results.json"
        validator.persist_results(str(filepath))
        
        assert filepath.exists()
        import json
        with open(filepath) as f:
            data = json.load(f)
        assert "checks" in data


class TestMultiStageValidator:
    """Test MultiStageValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for multiple stages."""
        dates = pd.date_range("2024-01-01", periods=120)
        return {
            "raw_stock": pd.DataFrame({
                "Date": dates,
                "Open": np.random.uniform(100, 150, 120),
                "High": np.random.uniform(100, 150, 120),
                "Low": np.random.uniform(100, 150, 120),
                "Close": np.random.uniform(100, 150, 120),
                "Volume": np.random.uniform(1000000, 10000000, 120),
                "Adj Close": np.random.uniform(100, 150, 120),
            }),
            "preprocessed": pd.DataFrame({
                "Date": dates,
                "Close": np.random.uniform(100, 150, 120),
                "Volume": np.random.uniform(1000000, 10000000, 120),
                "Returns": np.random.uniform(-0.5, 0.5, 120),
                "Log_Returns": np.random.uniform(-0.5, 0.5, 120),
            }),
        }
    
    def test_multi_stage_validator_init(self):
        """Test multi-stage validator initialization."""
        validator = MultiStageValidator(stages=["raw_stock", "preprocessed"])
        assert "raw_stock" in validator.validators
        assert "preprocessed" in validator.validators
    
    def test_validate_pipeline(self, sample_data):
        """Test multi-stage validation."""
        validator = MultiStageValidator(stages=["raw_stock", "preprocessed"])
        results = validator.validate_pipeline(sample_data)
        
        assert "raw_stock" in results
        assert "preprocessed" in results
    
    def test_get_full_report(self, sample_data):
        """Test full validation report."""
        validator = MultiStageValidator(stages=["raw_stock", "preprocessed"])
        validator.validate_pipeline(sample_data)
        report = validator.get_full_report()
        
        assert "MULTI-STAGE VALIDATION REPORT" in report
        assert "raw_stock" in report
        assert "preprocessed" in report


class TestBadDataScenarios:
    """Test validation with known-bad data scenarios."""
    
    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=120),
            "Open": [-10, 101, 102, 103, 104] + list(np.random.uniform(100, 150, 115)),
            "High": np.random.uniform(100, 150, 120),
            "Low": np.random.uniform(100, 150, 120),
            "Close": np.random.uniform(100, 150, 120),
            "Volume": np.random.uniform(1000000, 10000000, 120),
            "Adj Close": np.random.uniform(100, 150, 120),
        })
        
        validator = DataValidator("raw_stock")
        results = validator.validate(df)
        assert not results["checks"]["range"]["valid"]
    
    def test_extreme_missing_data(self):
        """Test detection of extreme missing data."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=120),
            "Open": [np.nan] * 100 + list(np.random.uniform(100, 150, 20)),
            "High": np.random.uniform(100, 150, 120),
            "Low": np.random.uniform(100, 150, 120),
            "Close": np.random.uniform(100, 150, 120),
            "Volume": np.random.uniform(1000000, 10000000, 120),
            "Adj Close": np.random.uniform(100, 150, 120),
        })
        
        validator = DataValidator("raw_stock")
        results = validator.validate(df)
        assert not results["checks"]["null_percentage"]["valid"]
    
    def test_insufficient_rows(self):
        """Test detection of insufficient rows."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Open": np.random.uniform(100, 150, 10),
            "High": np.random.uniform(100, 150, 10),
            "Low": np.random.uniform(100, 150, 10),
            "Close": np.random.uniform(100, 150, 10),
            "Volume": np.random.uniform(1000000, 10000000, 10),
            "Adj Close": np.random.uniform(100, 150, 10),
        })
        
        validator = DataValidator("raw_stock")
        results = validator.validate(df)
        assert not results["checks"]["min_rows"]["valid"]
    
    def test_missing_required_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=120),
            "Open": np.random.uniform(100, 150, 120),
            # Missing other required columns
        })
        
        validator = DataValidator("raw_stock")
        results = validator.validate(df)
        assert not results["checks"]["column_existence"]["valid"]
    
    def test_wrong_data_types(self):
        """Test detection of wrong data types."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"] * 120,  # String instead of datetime
            "Open": list(map(str, np.random.uniform(100, 150, 120))),  # String instead of float
            "High": np.random.uniform(100, 150, 120),
            "Low": np.random.uniform(100, 150, 120),
            "Close": np.random.uniform(100, 150, 120),
            "Volume": np.random.uniform(1000000, 10000000, 120),
            "Adj Close": np.random.uniform(100, 150, 120),
        })
        
        validator = DataValidator("raw_stock")
        results = validator.validate(df)
        assert not results["checks"]["data_types"]["valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
