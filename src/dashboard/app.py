# src/dashboard/app.py

import streamlit as st
from src.dashboard.ui_components import render_sidebar
from src.dashboard.sentiment_panel import SentimentPanel
from src.dashboard.forecast_panel import ForecastPanel
from src.dashboard.combined_tabel import CombinedTable
from src.dashboard.history_manager import EquityHistory
from src.dashboard.utils import get_ui_logger

logger = get_ui_logger("app")


def main():
    st.set_page_config(page_title="Equity Forecasting Dashboard", layout="wide")
    st.markdown("<h2 style='text-align:center'>Equity Forecasting Dashboard</h2>", unsafe_allow_html=True)

    # ---- Sidebar ----
    current_equity, forecast_horizon, panel_option = render_sidebar()
    logger.info(
        f"Sidebar selections - Equity: {current_equity}, Horizon: {forecast_horizon}, Panel: {panel_option}"
    )

    # ---- Equity History ----
    history_manager = EquityHistory()
    history_manager.add_equity(current_equity)

    # ---- Panels ----
    feed_scores = {}
    overall_sentiment = 0.0
    forecast_prices = []

    # --- Row 1: Forecast Panel (70%) ---
    with st.container():
        st.markdown("<h4>Forecast Panel</h4>", unsafe_allow_html=True)
        if panel_option in ["Show Forecast", "Show Both"]:
            forecast_panel = ForecastPanel(current_equity, forecast_horizon)
            forecast_panel.render()
            forecast_prices = forecast_panel.forecast_prices

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)  # spacing

    # --- Row 2: Market Sentiment Panel (20%) ---
    with st.container():
        st.markdown("<h4>Market Sentiment Panel</h4>", unsafe_allow_html=True)
        if panel_option in ["Show Sentiment", "Show Both"]:
            sentiment_panel = SentimentPanel(current_equity)
            sentiment_panel.render()  # internally handles 70:30 vertical split
            feed_scores = sentiment_panel.feed_scores
            overall_sentiment = sentiment_panel.overall_sentiment

    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)  # spacing

    # --- Row 3: Combined Output Table (10%) ---
    with st.container():
        st.markdown("<h4>Combined Output Table</h4>", unsafe_allow_html=True)
        combined_table = CombinedTable(
            equity=current_equity,
            feed_scores=feed_scores,
            overall_sentiment=overall_sentiment,
            forecast_prices=forecast_prices,
        )
        combined_table.render()

    # ---- Footer ----
    st.sidebar.markdown("---")
    st.sidebar.info("Equity history saved. Use 'Clear History' in the sidebar to reset.")
    logger.info("Dashboard render complete.")


if __name__ == "__main__":
    main()
