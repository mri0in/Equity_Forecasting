# src/dashboard/forecast_panel.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Forecast Panel Class
# -------------------------------
class ForecastPanel:
    """
    Displays historical prices and forecast for a selected equity.
    Can overlay a sentiment gauge on the chart.
    """

    def __init__(self, equity: str, horizon: int = 7):
        self.equity = equity.upper()
        self.horizon = horizon
        self.dates: Optional[pd.DatetimeIndex] = None
        self.hist_prices: Optional[np.ndarray] = None
        self.forecast_prices: Optional[np.ndarray] = None

    def simulate_forecast(self):
        """Simulate historical and forecasted prices (replace with real model later)."""
        np.random.seed(hash(self.equity) % 2**32)
        self.dates = pd.date_range('2025-01-01', periods=30)
        self.hist_prices = np.linspace(150, 180, 30) + np.random.normal(0, 2, 30)
        self.forecast_prices = self.hist_prices[-1] + np.cumsum(np.random.normal(0, 0.5, self.horizon))
        logger.info(f"Simulated forecast for {self.equity}, horizon {self.horizon}")

    def render_chart(self, sentiment_data: Optional[dict] = None):
        """Render historical + forecast line chart with optional sentiment overlay."""
        if self.dates is None or self.hist_prices is None or self.forecast_prices is None:
            st.warning("Forecast data not available.")
            return

        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=self.dates,
            y=self.hist_prices,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))

        # Forecast prices
        forecast_dates = pd.date_range(self.dates[-1] + pd.Timedelta(days=1), periods=self.horizon)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=self.forecast_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red')
        ))

        # Optional Sentiment Gauge Overlay
        if sentiment_data and "overall" in sentiment_data:
            overall_sentiment = sentiment_data["overall"]
            # Map sentiment [-1,1] to degrees for gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=overall_sentiment,
                title={"text": "Sentiment"},
                gauge={
                    "axis": {"range": [-1, 1], "tickvals": [-1, 0, 1], "ticktext": ["Negative","Neutral","Positive"]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [-1, 0], "color": "red"},
                        {"range": [0, 1], "color": "green"}
                    ]
                },
                domain={"x": [0, 0.25], "y": [0.75, 1]}  # top-left corner overlay
            ))

        fig.update_layout(
            title=f"{self.equity} - Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render(self, sentiment_data: Optional[dict] = None):
        """Main method to render forecast panel with optional sentiment overlay."""
        if not self.equity:
            st.warning("No equity selected for forecast panel.")
            return
        self.simulate_forecast()
        st.markdown("### Forecast Panel")
        self.render_chart(sentiment_data=sentiment_data)
