"""Input sanitization module for cleaning and validating raw inputs."""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize and clean user inputs, DataFrame columns, and API parameters."""
    
    @staticmethod
    def sanitize_ticker(ticker: str, allowed_length: int = 10) -> Optional[str]:
        """Sanitize a ticker symbol.
        
        Args:
            ticker: Ticker symbol (e.g. 'AAPL', 'BRK.B')
            allowed_length: Maximum allowed ticker length
            
        Returns:
            Sanitized ticker or None if invalid
        """
        if not ticker or not isinstance(ticker, str):
            logger.warning(f"Invalid ticker type: {type(ticker)}")
            return None
        
        # Remove whitespace
        ticker = ticker.strip().upper()
        
        # Allow alphanumeric, dots, and dashes
        if not all(c.isalnum() or c in ['.', '-', '_'] for c in ticker):
            logger.warning(f"Ticker contains invalid characters: {ticker}")
            return None
        
        # Check length
        if len(ticker) > allowed_length:
            logger.warning(f"Ticker too long: {ticker} (max {allowed_length})")
            return None
        
        if len(ticker) < 1:
            logger.warning("Ticker is empty")
            return None
        
        return ticker
    
    @staticmethod
    def sanitize_horizon(horizon: Union[int, float], min_days: int = 1, max_days: int = 365) -> Optional[int]:
        """Sanitize forecast horizon parameter.
        
        Args:
            horizon: Number of days to forecast
            min_days: Minimum allowed days
            max_days: Maximum allowed days
            
        Returns:
            Sanitized horizon or None if invalid
        """
        try:
            h = int(horizon)
            
            if h < min_days:
                logger.warning(f"Horizon {h} below minimum {min_days}")
                return None
            
            if h > max_days:
                logger.warning(f"Horizon {h} exceeds maximum {max_days}")
                return None
            
            return h
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert horizon to int: {e}")
            return None
    
    @staticmethod
    def sanitize_dataframe(
        df: pd.DataFrame,
        remove_inf: bool = True,
        remove_nan: bool = False,
        convert_to_numeric: bool = False
    ) -> Optional[pd.DataFrame]:
        """Sanitize a DataFrame by handling invalid values.
        
        Args:
            df: Input DataFrame
            remove_inf: Replace inf/-inf with NaN
            remove_nan: Drop rows with NaN values
            convert_to_numeric: Attempt to convert columns to numeric
            
        Returns:
            Sanitized DataFrame or None if invalid
        """
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Input is not a DataFrame: {type(df)}")
            return None
        
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return None
        
        df = df.copy()
        
        # Replace infinite values
        if remove_inf:
            before_inf = df.isin([np.inf, -np.inf]).sum().sum()
            df = df.replace([np.inf, -np.inf], np.nan)
            if before_inf > 0:
                logger.info(f"Replaced {before_inf} infinite values with NaN")
        
        # Remove rows with NaN
        if remove_nan:
            before_rows = len(df)
            df = df.dropna()
            removed = before_rows - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with NaN values")
            
            if df.empty:
                logger.warning("DataFrame is empty after NaN removal")
                return None
        
        # Convert numeric columns
        if convert_to_numeric:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.debug(f"Could not convert {col} to numeric: {e}")
        
        return df
    
    @staticmethod
    def sanitize_list(
        items: List[Any],
        allowed_type: Optional[type] = None,
        allow_empty: bool = False
    ) -> Optional[List[Any]]:
        """Sanitize a list input.
        
        Args:
            items: List to sanitize
            allowed_type: If specified, filter to items of this type
            allow_empty: Whether empty list is valid
            
        Returns:
            Sanitized list or None if invalid
        """
        if not isinstance(items, list):
            logger.warning(f"Input is not a list: {type(items)}")
            return None
        
        if not items and not allow_empty:
            logger.warning("List is empty but empty not allowed")
            return None
        
        # Filter by type if specified
        if allowed_type is not None:
            filtered = [x for x in items if isinstance(x, allowed_type)]
            if len(filtered) < len(items):
                logger.warning(
                    f"Filtered list from {len(items)} to {len(filtered)} items "
                    f"(removed {len(items) - len(filtered)} non-{allowed_type.__name__} items)"
                )
            items = filtered
        
        if not items and not allow_empty:
            logger.warning("List is empty after filtering")
            return None
        
        return items
    
    @staticmethod
    def sanitize_numeric_range(
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        strict: bool = False
    ) -> Optional[Union[int, float]]:
        """Sanitize a numeric value to be within a range.
        
        Args:
            value: Numeric value
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            strict: If True, reject out-of-range values; if False, clamp them
            
        Returns:
            Sanitized value or None if invalid
        """
        try:
            value = float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert to numeric: {e}")
            return None
        
        # Check for NaN or inf
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Value is NaN or infinite: {value}")
            return None
        
        # Check bounds
        if min_val is not None and value < min_val:
            if strict:
                logger.warning(f"Value {value} below minimum {min_val} (strict mode)")
                return None
            else:
                logger.debug(f"Clamping {value} to minimum {min_val}")
                value = float(min_val)
        
        if max_val is not None and value > max_val:
            if strict:
                logger.warning(f"Value {value} exceeds maximum {max_val} (strict mode)")
                return None
            else:
                logger.debug(f"Clamping {value} to maximum {max_val}")
                value = float(max_val)
        
        return value
    
    @staticmethod
    def sanitize_dict(
        data: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Sanitize a dictionary input.
        
        Args:
            data: Input dictionary
            required_keys: Keys that must be present
            allowed_keys: Only these keys are allowed (if specified)
            
        Returns:
            Sanitized dictionary or None if invalid
        """
        if not isinstance(data, dict):
            logger.warning(f"Input is not a dict: {type(data)}")
            return None
        
        # Check required keys
        if required_keys:
            missing = set(required_keys) - set(data.keys())
            if missing:
                logger.warning(f"Missing required keys: {missing}")
                return None
        
        # Filter to allowed keys
        if allowed_keys:
            data = {k: v for k, v in data.items() if k in allowed_keys}
            logger.debug(f"Filtered dict to allowed keys: {allowed_keys}")
        
        return data
    
    @staticmethod
    def sanitize_string(
        text: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        strip: bool = True
    ) -> Optional[str]:
        """Sanitize a string input.
        
        Args:
            text: Input string
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            strip: Whether to strip whitespace
            
        Returns:
            Sanitized string or None if invalid
        """
        if not isinstance(text, str):
            logger.warning(f"Input is not a string: {type(text)}")
            return None
        
        if strip:
            text = text.strip()
        
        if not text:
            logger.warning("String is empty after stripping")
            return None
        
        if max_length and len(text) > max_length:
            logger.warning(f"String length {len(text)} exceeds maximum {max_length}")
            return None
        
        if allowed_chars:
            import re
            if not re.match(allowed_chars, text):
                logger.warning(f"String contains invalid characters: {text}")
                return None
        
        return text


class BatchSanitizer:
    """Batch sanitization for multiple items."""
    
    def __init__(self, sanitizer: Optional[InputSanitizer] = None):
        """Initialize batch sanitizer.
        
        Args:
            sanitizer: InputSanitizer instance (creates default if None)
        """
        self.sanitizer = sanitizer or InputSanitizer()
        self.results = {}
    
    def sanitize_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """Sanitize multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to sanitization success
        """
        results = {}
        for ticker in tickers:
            sanitized = self.sanitizer.sanitize_ticker(ticker)
            results[ticker] = sanitized is not None
        
        self.results['tickers'] = results
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics.
        
        Returns:
            Dictionary with sanitization stats
        """
        stats = {}
        for key, result_dict in self.results.items():
            if isinstance(result_dict, dict):
                total = len(result_dict)
                passed = sum(1 for v in result_dict.values() if v)
                stats[key] = {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": (passed / total * 100) if total > 0 else 0
                }
        
        return stats
