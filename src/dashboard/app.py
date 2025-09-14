# src/dashboard/app.py

import streamlit as st
from src.dashboard.ui_components import render_sidebar
from src.dashboard.sentiment_panel import SentimentPanel
from src.dashboard.forecast_panel import ForecastPanel
from src.dashboard.combined_tabel import CombinedTable
from src.dashboard.history_manager import EquityHistory
from src.dashboard.utils import get_ui_logger

# -------------------------------
# Logger
# -------------------------------
logger = get_ui_logger("app")

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="Equity Forecasting Dashboard", layout="wide")
    st.title("Equity Forecasting Dashboard")

    # ---- Sidebar ----
    current_equity, forecast_horizon, panel_option = render_sidebar()
    logger.info(f"Sidebar selections - Equity: {current_equity}, Horizon: {forecast_horizon}, Panel: {panel_option}")

    # ---- Equity History Persistence ----
    history_manager = EquityHistory()
    history_manager.add_equity(current_equity)

    # ---- Prepare Shared Data ----
    feed_scores = {}
    overall_sentiment = 0.0
    forecast_prices = []

    # ---- Horizontal Layout: 70% | 20% | 10% ----
    col1, col2, col3 = st.columns([7, 2, 1])

    # Forecast Panel (70%)
    with col1:
        if panel_option in ["Show Forecast", "Show Both"]:
            forecast_panel = ForecastPanel(current_equity, forecast_horizon)
            forecast_panel.render()
            forecast_prices = forecast_panel.forecast_prices

    # Sentiment Panel (20%)
    with col2:
        if panel_option in ["Show Sentiment", "Show Both"]:
            sentiment_panel = SentimentPanel(current_equity)
            sentiment_panel.render()
            feed_scores = sentiment_panel.feed_scores
            overall_sentiment = sentiment_panel.overall_sentiment

    # Combined Table (10%)
    with col3:
        combined_table = CombinedTable(
            equity=current_equity,
            feed_scores=feed_scores,
            overall_sentiment=overall_sentiment,
            forecast_prices=forecast_prices
        )
        combined_table.render()

    # ---- Footer / Info ----
    st.sidebar.markdown("---")
    st.sidebar.info("Equity history saved. Use 'Clear History' in the sidebar to reset.")
    logger.info("Dashboard render complete.")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()
