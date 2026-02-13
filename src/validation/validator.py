"""Data validator class for equity forecasting pipeline."""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging
from .expectations import DataExpectations, ColumnExpectation

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validator for pipeline stages."""
    
    def __init__(self, stage: str = "raw_stock"):
        """Initialize validator with expectations for a specific stage.
        
        Args:
            stage: Pipeline stage ('raw_stock', 'preprocessed', 'featured')
        """
        self.stage = stage
        self.expectations = DataExpectations.get_expectations(stage)
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations on the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with complete validation results
        """
        self.validation_results = {
            "stage": self.stage,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "checks": {}
        }
        
        # Get column expectations
        column_expectations = self.expectations.get("columns", [])
        expected_cols = [e.name for e in column_expectations]
        
        # 1. Column existence check
        logger.info(f"[{self.stage}] Checking column existence...")
        col_check = DataExpectations.validate_column_existence(df, expected_cols)
        self.validation_results["checks"]["column_existence"] = col_check
        if not col_check["valid"]:
            logger.warning(f"Column check failed: {col_check['message']}")
        
        # 2. Data type validation
        logger.info(f"[{self.stage}] Validating data types...")
        dtype_check = DataExpectations.validate_data_types(df, column_expectations)
        self.validation_results["checks"]["data_types"] = dtype_check
        if not dtype_check["valid"]:
            logger.warning(f"Data type check failed: {dtype_check['message']}")
        
        # 3. Range validation
        logger.info(f"[{self.stage}] Validating value ranges...")
        range_check = DataExpectations.validate_range(df, column_expectations)
        self.validation_results["checks"]["range"] = range_check
        if not range_check["valid"]:
            logger.warning(f"Range check failed: {range_check['message']}")
        
        # 4. Null percentage validation
        logger.info(f"[{self.stage}] Validating null percentages...")
        null_check = DataExpectations.validate_null_percentage(df, column_expectations)
        self.validation_results["checks"]["null_percentage"] = null_check
        if not null_check["valid"]:
            logger.warning(f"Null percentage check failed: {null_check['message']}")
        
        # 5. Minimum rows
        min_rows = self.expectations.get("min_rows", 10)
        logger.info(f"[{self.stage}] Checking minimum rows ({min_rows})...")
        min_rows_check = DataExpectations.validate_min_rows(df, min_rows)
        self.validation_results["checks"]["min_rows"] = min_rows_check
        if not min_rows_check["valid"]:
            logger.warning(f"Min rows check failed: {min_rows_check['message']}")
        
        # 6. Datetime continuity (if enabled)
        if self.expectations.get("check_datetime_continuity", False):
            logger.info(f"[{self.stage}] Checking datetime continuity...")
            datetime_check = DataExpectations.validate_datetime_continuity(df)
            self.validation_results["checks"]["datetime_continuity"] = datetime_check
            if not datetime_check["valid"]:
                logger.warning(f"Datetime continuity check failed: {datetime_check['message']}")
        
        # 7. Duplicate detection (if enabled)
        if self.expectations.get("check_duplicates", False):
            logger.info(f"[{self.stage}] Checking for duplicates...")
            dup_check = DataExpectations.validate_duplicates(df, key_column="Date")
            self.validation_results["checks"]["duplicates"] = dup_check
            if not dup_check["valid"]:
                logger.warning(f"Duplicate check failed: {dup_check['message']}")
        
        # Aggregate results
        all_valid = all(check.get("valid", False) for check in self.validation_results["checks"].values())
        self.validation_results["all_valid"] = all_valid
        self.validation_results["status"] = "PASSED" if all_valid else "FAILED"
        
        return self.validation_results
    
    def get_summary(self) -> str:
        """Get human-readable summary of validation results.
        
        Returns:
            String summary of validation
        """
        if not self.validation_results:
            return "No validation run yet"
        
        summary = f"\n{'='*60}\n"
        summary += f"VALIDATION SUMMARY [{self.stage}] - {self.validation_results['status']}\n"
        summary += f"{'='*60}\n\n"
        
        for check_name, result in self.validation_results["checks"].items():
            status = "✓ PASS" if result.get("valid", False) else "✗ FAIL"
            message = result.get("message", "No message")
            summary += f"  {status:8} | {check_name:20} | {message}\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def persist_results(self, filepath: str) -> None:
        """Save validation results to a file.
        
        Args:
            filepath: Path to save results JSON
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {filepath}")
    
    def raise_on_failure(self) -> None:
        """Raise exception if validation failed.
        
        Raises:
            ValueError: If validation did not pass
        """
        if not self.validation_results.get("all_valid", False):
            failures = [
                f"{k}: {v.get('message', 'Unknown error')}"
                for k, v in self.validation_results["checks"].items()
                if not v.get("valid", False)
            ]
            raise ValueError(f"Data validation failed for {self.stage}:\n" + "\n".join(failures))
        logger.info(f"All validation checks passed for {self.stage}")


class MultiStageValidator:
    """Validator for multiple pipeline stages."""
    
    def __init__(self, stages: Optional[List[str]] = None):
        """Initialize multi-stage validator.
        
        Args:
            stages: List of stages to validate ('raw_stock', 'preprocessed', 'featured')
        """
        self.stages = stages or ["raw_stock"]
        self.validators = {stage: DataValidator(stage) for stage in self.stages}
        self.all_results = {}
    
    def validate_pipeline(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate data at multiple stages.
        
        Args:
            data_dict: Dictionary mapping stage names to DataFrames
            
        Returns:
            Dictionary with results for all stages
        """
        for stage, data in data_dict.items():
            if stage in self.validators:
                logger.info(f"Validating {stage}...")
                results = self.validators[stage].validate(data)
                self.all_results[stage] = results
        
        return self.all_results
    
    def get_full_report(self) -> str:
        """Get comprehensive validation report.
        
        Returns:
            String report for all stages
        """
        report = "\n" + "="*60 + "\n"
        report += "MULTI-STAGE VALIDATION REPORT\n"
        report += "="*60 + "\n"
        
        for stage, validator in self.validators.items():
            report += validator.get_summary()
        
        overall_status = "ALL PASSED" if all(
            r.get("all_valid", False) for r in self.all_results.values()
        ) else "SOME FAILED"
        
        report += f"\nOVERALL STATUS: {overall_status}\n"
        report += "="*60 + "\n"
        
        return report
