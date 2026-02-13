"""Data validation expectations and rules for equity forecasting."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ColumnExpectation:
    """Column validation expectations."""
    name: str
    dtype: Optional[str] = None
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_null_percentage: float = 10.0  # Allow max 10% nulls by default


class DataExpectations:
    """Define validation expectations for different data stages."""
    
    STOCK_DATA_EXPECTATIONS = {
        "columns": [
            ColumnExpectation("Date", dtype="datetime64", required=True, max_null_percentage=0),
            ColumnExpectation("Open", dtype="float64", required=True, min_value=0, max_null_percentage=5),
            ColumnExpectation("High", dtype="float64", required=True, min_value=0, max_null_percentage=5),
            ColumnExpectation("Low", dtype="float64", required=True, min_value=0, max_null_percentage=5),
            ColumnExpectation("Close", dtype="float64", required=True, min_value=0, max_null_percentage=5),
            ColumnExpectation("Volume", dtype="float64", required=True, min_value=0, max_null_percentage=5),
            ColumnExpectation("Adj Close", dtype="float64", required=True, min_value=0, max_null_percentage=5),
        ],
        "min_rows": 100,
        "check_duplicates": True,
        "check_datetime_continuity": True,
    }
    
    PREPROCESSED_DATA_EXPECTATIONS = {
        "columns": [
            ColumnExpectation("Date", dtype="datetime64", required=True, max_null_percentage=0),
            ColumnExpectation("Close", dtype="float64", required=True, min_value=0, max_null_percentage=0),
            ColumnExpectation("Volume", dtype="float64", required=True, min_value=0, max_null_percentage=0),
            ColumnExpectation("Returns", dtype="float64", required=True, min_value=-1, max_value=1, max_null_percentage=5),
            ColumnExpectation("Log_Returns", dtype="float64", required=True, min_value=-1, max_value=1, max_null_percentage=5),
        ],
        "min_rows": 50,
        "check_duplicates": True,
        "check_datetime_continuity": False,
    }
    
    FEATURED_DATA_EXPECTATIONS = {
        "columns": [
            ColumnExpectation("Date", dtype="datetime64", required=True, max_null_percentage=0),
        ],
        "min_rows": 30,
        "check_duplicates": True,
        "check_datetime_continuity": False,
        "allow_additional_columns": True,  # Features can add new columns
    }
    
    @classmethod
    def get_expectations(cls, stage: str) -> Dict[str, Any]:
        """Get expectations for a specific pipeline stage.
        
        Args:
            stage: One of 'raw_stock', 'preprocessed', 'featured'
            
        Returns:
            Dictionary of expectations
        """
        if stage == "raw_stock":
            return cls.STOCK_DATA_EXPECTATIONS
        elif stage == "preprocessed":
            return cls.PREPROCESSED_DATA_EXPECTATIONS
        elif stage == "featured":
            return cls.FEATURED_DATA_EXPECTATIONS
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    @staticmethod
    def validate_column_existence(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """Check if all expected columns exist.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            
        Returns:
            Dictionary with validation result
        """
        missing_columns = set(expected_columns) - set(df.columns)
        return {
            "valid": len(missing_columns) == 0,
            "missing_columns": list(missing_columns),
            "message": f"Missing columns: {missing_columns}" if missing_columns else "All columns present"
        }
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame, column_expectations: List[ColumnExpectation]) -> Dict[str, Any]:
        """Validate data types of columns.
        
        Args:
            df: DataFrame to validate
            column_expectations: List of ColumnExpectation objects
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        for expectation in column_expectations:
            if expectation.name not in df.columns:
                continue
            
            col_dtype = str(df[expectation.name].dtype)
            expected_dtype = expectation.dtype
            
            # Map pandas dtypes to expected strings
            dtype_map = {
                "int64": ["int64", "int32", "int"],
                "float64": ["float64", "float32", "float"],
                "object": ["object", "string"],
                "datetime64": ["datetime64[ns]", "datetime64", "datetime"],
                "bool": ["bool", "boolean"],
            }
            
            if expected_dtype:
                expected_base = expected_dtype.split("[")[0]
                if expected_base not in dtype_map:
                    dtype_map[expected_base] = [expected_dtype]
                
                # Check if column dtype matches
                col_matches = False
                for valid_dtype in dtype_map.get(expected_base, [expected_dtype]):
                    if valid_dtype in col_dtype or col_dtype in valid_dtype:
                        col_matches = True
                        break
                
                if not col_matches:
                    errors.append(f"Column '{expectation.name}': expected {expected_dtype}, got {col_dtype}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "Data types valid" if not errors else f"Type errors: {errors}"
        }
    
    @staticmethod
    def validate_range(df: pd.DataFrame, column_expectations: List[ColumnExpectation]) -> Dict[str, Any]:
        """Validate numeric ranges.
        
        Args:
            df: DataFrame to validate
            column_expectations: List of ColumnExpectation objects
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        for expectation in column_expectations:
            if expectation.name not in df.columns:
                continue
            
            col = df[expectation.name].dropna()
            
            # Skip range check for non-numeric columns
            if not pd.api.types.is_numeric_dtype(col):
                continue
            
            if expectation.min_value is not None:
                violations = (col < expectation.min_value).sum()
                if violations > 0:
                    errors.append(f"Column '{expectation.name}': {violations} values < {expectation.min_value}")
            
            if expectation.max_value is not None:
                violations = (col > expectation.max_value).sum()
                if violations > 0:
                    errors.append(f"Column '{expectation.name}': {violations} values > {expectation.max_value}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "All values in valid range" if not errors else f"Range errors: {errors}"
        }
    
    @staticmethod
    def validate_null_percentage(df: pd.DataFrame, column_expectations: List[ColumnExpectation]) -> Dict[str, Any]:
        """Validate null/NaN percentage in columns.
        
        Args:
            df: DataFrame to validate
            column_expectations: List of ColumnExpectation objects
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        for expectation in column_expectations:
            if expectation.name not in df.columns:
                if expectation.required:
                    errors.append(f"Required column '{expectation.name}' missing")
                continue
            
            null_pct = (df[expectation.name].isna().sum() / len(df)) * 100
            
            if null_pct > expectation.max_null_percentage:
                errors.append(
                    f"Column '{expectation.name}': {null_pct:.2f}% nulls "
                    f"(max allowed: {expectation.max_null_percentage}%)"
                )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "Null percentage within limits" if not errors else f"Null errors: {errors}"
        }
    
    @staticmethod
    def validate_datetime_continuity(df: pd.DataFrame, date_column: str = "Date") -> Dict[str, Any]:
        """Check for datetime continuity (no missing days).
        
        Args:
            df: DataFrame to validate
            date_column: Name of date column
            
        Returns:
            Dictionary with validation results
        """
        if date_column not in df.columns:
            return {
                "valid": False,
                "errors": [f"Date column '{date_column}' not found"],
                "message": f"Date column '{date_column}' not found"
            }
        
        try:
            dates = pd.to_datetime(df[date_column]).dropna()
            if len(dates) < 2:
                return {
                    "valid": True,
                    "message": "Not enough dates to check continuity",
                    "gaps": []
                }
            
            # Check for gaps
            date_diff = dates.sort_values().diff()
            expected_freq = date_diff[1:].mode()[0]  # Most common frequency
            gaps = date_diff[date_diff > expected_freq]
            
            return {
                "valid": len(gaps) == 0,
                "message": "No gaps in datetime sequence" if len(gaps) == 0 else f"Found {len(gaps)} gaps",
                "gaps": len(gaps)
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "message": f"Error checking datetime continuity: {str(e)}"
            }
    
    @staticmethod
    def validate_duplicates(df: pd.DataFrame, key_column: str = "Date") -> Dict[str, Any]:
        """Check for duplicate rows (based on key column).
        
        Args:
            df: DataFrame to validate
            key_column: Column to check for duplicates
            
        Returns:
            Dictionary with validation results
        """
        if key_column not in df.columns:
            return {
                "valid": False,
                "errors": [f"Key column '{key_column}' not found"],
                "message": f"Key column '{key_column}' not found",
                "duplicates": 0
            }
        
        duplicates = int(df[key_column].duplicated().sum())
        
        return {
            "valid": bool(duplicates == 0),
            "message": "No duplicates found" if duplicates == 0 else f"Found {duplicates} duplicate rows",
            "duplicates": duplicates
        }
    
    @staticmethod
    def validate_min_rows(df: pd.DataFrame, min_rows: int) -> Dict[str, Any]:
        """Check minimum number of rows.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows
            
        Returns:
            Dictionary with validation results
        """
        actual_rows = len(df)
        
        return {
            "valid": actual_rows >= min_rows,
            "message": f"Row count valid ({actual_rows} >= {min_rows})" if actual_rows >= min_rows 
                      else f"Insufficient rows ({actual_rows} < {min_rows})",
            "actual_rows": actual_rows,
            "min_rows": min_rows
        }
