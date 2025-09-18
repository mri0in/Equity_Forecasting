# src/dashboard/forecast_panel.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.get_ui_logger(__name__)

# -------------------------------
# Forecast Panel Class
# -------------------------------
class ForecastPanel:
    """
    Displays historical prices and forecast for a selected equity.
    """

    def __init__(self, equity: str, horizon: int = 7):
        self.equity: str = equity.upper()
        self.horizon: int = horizon
        self.dates: Optional[pd.DatetimeIndex] = None
        self.hist_prices: Optional[np.ndarray] = None
        self.forecast_prices: List[float] = []

    def simulate_forecast(self) -> None:
        """
        Simulate historical and forecasted prices.
        Replace this logic later with real model predictions.
        """
        np.random.seed(hash(self.equity) % 2**32)
        self.dates = pd.date_range("2025-01-01", periods=30)
        self.hist_prices = np.linspace(150, 180, 30) + np.random.normal(0, 2, 30)

        forecast = self.hist_prices[-1] + np.cumsum(np.random.normal(0, 0.5, self.horizon))
        self.forecast_prices = forecast.tolist()

        logger.info(f"Simulated forecast for {self.equity}, horizon {self.horizon}")

    def render_chart(self) -> None:
        """
        Render historical + forecast line chart.
        """
        if self.dates is None or self.hist_prices is None or not self.forecast_prices:
            st.warning("Forecast data not available.")
            return

        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=self.dates,
            y=self.hist_prices,
            mode="lines+markers",
            name="Historical",
            line=dict(color="blue")
        ))

        # Forecast prices
        forecast_dates = pd.date_range(self.dates[-1] + pd.Timedelta(days=1), periods=self.horizon)
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=self.forecast_prices,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red")
        ))

        fig.update_layout(
            title=f"{self.equity} - Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    def render(self) -> None:
        """
        Main method to render forecast panel.
        """
        if not self.equity:
            st.warning("No equity selected for forecast panel.")
            return

        self.simulate_forecast()
        self.render_chart()

    def get_forecast(self) -> List[float]:
        """
        Return forecast prices safely.
        Always returns a list (may be empty if no forecast was generated).
        """
        return self.forecast_prices if self.forecast_prices else []
