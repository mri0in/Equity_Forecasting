# src/dashboard/combined_table.py

import streamlit as st
import pandas as pd
from typing import Dict, Optional
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.get_ui_logger(__name__)

# -------------------------------
# Combined Table Panel Class
# -------------------------------
class CombinedTable:
    """
    Displays a combined table of feed-wise sentiment,
    overall sentiment, and forecasted price for a selected equity.
    """

    def __init__(self, equity: str, feed_scores: Optional[Dict[str, float]] = None,
                 overall_sentiment: Optional[float] = None, forecast_prices: Optional[list] = None):
        self.equity = equity.upper()
        self.feed_scores = feed_scores if feed_scores else {}
        self.overall_sentiment = overall_sentiment if overall_sentiment is not None else 0.0
        self.forecast_prices = forecast_prices if forecast_prices else []

    def render_combined_table(self):
        """
        Render the combined table in Streamlit
        """
        if not self.equity:
            st.warning("No equity selected for combined table.")
            return

        data = {
            "Equity": [self.equity],
            "News": [self.feed_scores.get("News", 0.0)],
            "Press": [self.feed_scores.get("Press", 0.0)],
            "Social": [self.feed_scores.get("Social", 0.0)],
            "Web": [self.feed_scores.get("Web", 0.0)],
            "Overall Sentiment": [self.overall_sentiment],
            "Forecast Next Price": [self.forecast_prices[-1] if self.forecast_prices else None]
        }

        df = pd.DataFrame(data)
        st.dataframe(df)

        logger.info(f"Rendered combined table for {self.equity}")
