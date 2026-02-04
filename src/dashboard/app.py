# src/dashboard/app.py

import streamlit as st

from src.dashboard.ui_components import render_sidebar
from src.dashboard.forecast_panel import ForecastPanel
from src.dashboard.combined_tabel import CombinedTable
from src.config.active_equity import set_active_equity
from src.dashboard.utils import get_ui_logger

logger = get_ui_logger("app")


def main() -> None:
    st.set_page_config(
        page_title="Equity Forecasting Dashboard",
        layout="wide",
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:20px;'>Equity Forecasting Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # STEP 1 — Sidebar inputs (raw, possibly invalid)
    # ------------------------------------------------------------------
    current_equity, forecast_horizon, panel_option = render_sidebar()

    logger.info(
        "Sidebar selections | equity=%s horizon=%s panel=%s",
        current_equity,
        forecast_horizon,
        panel_option,
    )

    # ------------------------------------------------------------------
    # STEP 2 — Validate & set active equity
    # ------------------------------------------------------------------
    if current_equity:
        if not set_active_equity(current_equity):
            st.warning(f"⚠️ '{current_equity}' is not a valid ticker/symbol.")
            logger.warning("Invalid equity selected: %s", current_equity)
            current_equity = None

    # ------------------------------------------------------------------
    # Initialize shared state (ALWAYS defined)
    # ------------------------------------------------------------------
    feed_scores: dict = {}
    overall_sentiment: float | None = None
    forecast_prices: list = []
    sentiment_simulated: bool = False

    # ==================================================================
    # STEP 3 — Forecast Panel 
    # ==================================================================
    if current_equity and panel_option in {"Show Forecast", "Show Both"}:
        with st.container():
            st.markdown(
                "<h4 style='border-bottom:1px solid #ccc;'>Forecast Panel</h4>",
                unsafe_allow_html=True,
            )

            forecast_panel = ForecastPanel(
                equity=current_equity,
                horizon=forecast_horizon,
                sentiment=None,  # Explicit: sentiment is optional
            )

            forecast_simulated = forecast_panel.render_forecast()
            forecast_prices = forecast_panel.get_forecast()
            if forecast_simulated:
                    st.markdown(
                        "<small>⚠️ Forecast based on simulated data.</small>",
                        unsafe_allow_html=True,
                    )
                    logger.warning("Forecast shown using simulated data.")
            else:
                st.markdown(
                    "<small>✅ Live market forecast.</small>",
                    unsafe_allow_html=True,
                )




    # Spacer
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ==================================================================
    # STEP 4 — Sentiment Panel 
    # ==================================================================
    if current_equity and panel_option in {"Show Sentiment", "Show Both"}:
        try:
            from src.dashboard.sentiment_panel import SentimentPanel

            with st.container():
                st.markdown(
                    "<h4 style='border-bottom:1px solid #ccc;'>Market Sentiment Panel</h4>",
                    unsafe_allow_html=True,
                )

                sentiment_panel = SentimentPanel(current_equity)
                sentiment_simulated = sentiment_panel.render_sentiment()

                feed_scores = sentiment_panel.feed_scores
                overall_sentiment = sentiment_panel.overall_sentiment

                if sentiment_simulated:
                    st.markdown(
                        "<small>⚠️ Sentiment based on simulated data.</small>",
                        unsafe_allow_html=True,
                    )
                    logger.warning("Sentiment shown using simulated data.")
                else:
                    st.markdown(
                        "<small>✅ Live market sentiment.</small>",
                        unsafe_allow_html=True,
                    )

        except Exception:
            logger.exception("Sentiment subsystem failed")
            st.error("Sentiment system unavailable.")

    # Spacer
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    # ==================================================================
    # STEP 5 — Combined Output Table (pure presentation)
    # ==================================================================
    if current_equity:
        with st.container():
            st.markdown(
                "<h4 style='border-bottom:1px solid #ccc;'>Combined Output</h4>",
                unsafe_allow_html=True,
            )

            CombinedTable(
                equity=current_equity,
                feed_scores=feed_scores,
                overall_sentiment=overall_sentiment,
                forecast_prices=forecast_prices,
            ).render_combined_table()

    logger.info("Dashboard render complete.")


if __name__ == "__main__":
    main()
