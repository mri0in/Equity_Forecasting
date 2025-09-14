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
    st.title("Equity Forecasting Dashboard")

    # Sidebar
    current_equity, forecast_horizon, panel_option = render_sidebar()
    logger.info(f"Sidebar selections - Equity: {current_equity}, Horizon: {forecast_horizon}, Panel: {panel_option}")

    # Persist history
    history_manager = EquityHistory()
    if current_equity:
        history_manager.add_equity(current_equity)

    # Prepare objects
    sentiment_panel = SentimentPanel(current_equity)
    forecast_panel = ForecastPanel(current_equity, forecast_horizon)

    # Simulate sentiment early if we will need it (overlay or middle column)
    need_sentiment = panel_option in ["Show Sentiment", "Show Both", "Show Forecast"]
    if need_sentiment:
        sentiment_panel.simulate_sentiment()
        sentiment_data = {"overall": sentiment_panel.overall_sentiment}
    else:
        sentiment_data = None

    # Horizontal layout: 70 | 20 | 10
    col1, col2, col3 = st.columns([7, 2, 1])

    # ----- LEFT: Forecast (70%) -----
    with col1:
        if panel_option in ["Show Forecast", "Show Both"]:
            # pass sentiment_data to overlay the overall gauge on the forecast chart
            forecast_panel.render(sentiment_data=sentiment_data)
        else:
            st.info("Forecast hidden (choose 'Show Forecast' or 'Show Both')")

    # ----- MIDDLE: Sentiment (20%) with vertical 70:30 via heights -----
    with col2:
        st.markdown("### Market Sentiment")
        if panel_option in ["Show Sentiment", "Show Both", "Show Forecast"]:
            # enforce pixel heights that approximate 70:30 within this column
            total_pixels = 600  # adjust if you want the vertical size larger/smaller
            top_height = int(total_pixels * 0.70)   # feed-scores area
            bottom_height = int(total_pixels * 0.30)  # overall gauge

            # Feed-wise (top ~70%)
            sentiment_panel.render_feed_scores(height=top_height)

            # Overall (bottom ~30%)
            sentiment_panel.render_overall_gauge(height=bottom_height)
        else:
            st.info("Sentiment hidden (choose 'Show Sentiment' or 'Show Both')")

    # ----- RIGHT: Combined Table (10%) -----
    with col3:
        st.markdown("### Combined Output")
        # feed_scores and overall_sentiment come from simulated sentiment if available
        feed_scores = sentiment_panel.feed_scores if need_sentiment else {}
        overall_sentiment = sentiment_panel.overall_sentiment if need_sentiment else 0.0

        combined_table = CombinedTable(
            equity=current_equity,
            feed_scores=feed_scores,
            overall_sentiment=overall_sentiment,
            forecast_prices=forecast_panel.forecast_prices or []
        )
        combined_table.render()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("Equity history saved. Use 'Clear History' in the sidebar to reset.")
    logger.info("Dashboard render complete.")

if __name__ == "__main__":
    main()
