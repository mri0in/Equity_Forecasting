# src/dashboard/app.py

import streamlit as st
from src.dashboard.ui_components import render_sidebar
from src.dashboard.sentiment_panel import SentimentPanel
from src.dashboard.forecast_panel import ForecastPanel
from src.dashboard.combined_tabel import CombinedTable
from src.config.active_equity import set_active_equity, get_active_equity
from src.dashboard.utils import get_ui_logger

logger = get_ui_logger("app")


def main():
    st.set_page_config(page_title="Equity Forecasting Dashboard", layout="wide")

    # -------------------------------
    # Title
    # -------------------------------
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:20px;'>Equity Forecasting Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ==========================================================
    # Step 1 - Collect raw sidebar inputs
    # This returns possibly invalid equity from ui_components.py
    # ==========================================================
    current_equity, forecast_horizon, panel_option = render_sidebar()
    logger.info(
        f"Sidebar selections - Equity: {current_equity}, Horizon: {forecast_horizon}, Panel: {panel_option}"
    )

    # ==========================================================
    # Step 2 - Validate & set active equity centrally
    # The actual check happens inside set_active_equity()
    # ==========================================================
    # Only attempt to set active equity if user provided or selected something
    if current_equity:
        success = set_active_equity(current_equity)

        if not success:
            st.warning(f"⚠️ '{current_equity}' is not a valid ticker/symbol name.")
            current_equity = None
    else:
        success = False

    
    # ==========================================================
    # Step 3 - Panels
    # ==========================================================
    feed_scores = {}
    overall_sentiment = 0.0
    forecast_prices = []

    # --- Forecast Panel ---
    if current_equity and panel_option in ["Show Forecast", "Show Both"]:
        with st.container():
            st.markdown(
                "<h4 style='border-bottom:1px solid #ccc; padding-bottom:5px;'>Forecast Panel</h4>",
                unsafe_allow_html=True,
            )
            forecast_panel = ForecastPanel(current_equity, forecast_horizon)
            forecast_panel.render_forecast()
            forecast_prices = forecast_panel.get_forecast()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Sentiment Panel ---
    if current_equity and panel_option in ["Show Sentiment", "Show Both"]:
        with st.container():
            st.markdown(
                "<h4 style='border-bottom:1px solid #ccc; padding-bottom:5px;'>Market Sentiment Panel</h4>",
                unsafe_allow_html=True,
            )

        # Custom CSS for vertical separation
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
                border-right: 2px solid #ccc;
                padding-right: 15px;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
                padding-left: 15px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        sentiment_panel = SentimentPanel(current_equity)
        sim_data_used = sentiment_panel.render_sentiment()
        feed_scores = sentiment_panel.feed_scores
        overall_sentiment = sentiment_panel.overall_sentiment

        if sim_data_used:
            st.markdown(
                "<small>'*' is result based on simulated data not real market data.</small>",
                unsafe_allow_html=True
            )
            logger.warning("Displayed sentiment is based on simulated data.")
            
        else:
            st.markdown(
                "<small>'✅' indicates live market feed.</small>",
                unsafe_allow_html=True
            )
            logger.info("Displayed sentiment is based on live market data.")    

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    # --- Combined Output Table ---
    if current_equity:
        with st.container():
            st.markdown(
                "<h4 style='border-bottom:1px solid #ccc; padding-bottom:5px;'>Combined Output Table</h4>",
                unsafe_allow_html=True,
            )
            combined_table = CombinedTable(
                equity=current_equity,
                feed_scores=feed_scores,
                overall_sentiment=overall_sentiment,
                forecast_prices=forecast_prices,
            )
            combined_table.render_combined_table()

    
    logger.info("Dashboard render complete.")


if __name__ == "__main__":
    main()
