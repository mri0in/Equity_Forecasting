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
    
    Features:
    - Historical price line chart
    - Forecasted price line chart
    - Uses forecast horizon from sidebar
    """

    def __init__(self, equity: str, horizon: int = 7):
        self.equity = equity.upper()
        self.horizon = horizon
        self.dates: Optional[pd.DatetimeIndex] = None
        self.hist_prices: Optional[np.ndarray] = None
        self.forecast_prices: Optional[np.ndarray] = None

    def simulate_forecast(self):
        """
        Simulate historical and forecasted prices.
        Replace with real model prediction later.
        """
        np.random.seed(hash(self.equity) % 2**32)
        self.dates = pd.date_range('2025-01-01', periods=30)
        self.hist_prices = np.linspace(150, 180, 30) + np.random.normal(0, 2, 30)
        # Forecast horizon can be 1-30 days
        self.forecast_prices = self.hist_prices[-1] + np.cumsum(np.random.normal(0, 0.5, self.horizon))
        logger.info(f"Simulated forecast for {self.equity}, horizon {self.horizon}")

    def render_chart(self):
        """
        Render historical + forecast line chart
        """
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
        fig.update_layout(
            title=f"{self.equity} - Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        """
        Main method to render forecast panel
        """
        if not self.equity:
            st.warning("No equity selected for forecast panel.")
            return
        self.simulate_forecast()
        st.markdown("### Forecast Panel")
        self.render_chart()
