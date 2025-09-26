# src/dashboard/ui_components.py

import logging
from typing import Optional
import streamlit as st
from src.dashboard.history_manager import EquityHistory
from src.dashboard.utils import get_ui_logger

# ==========================================================
# Logging configuration
# ==========================================================
logger = get_ui_logger(__name__)

# ==========================================================
# Sidebar UI Components Class
# ==========================================================
class SidebarUI:
    """
    Handles the Streamlit sidebar UI components for the Equity Dashboard.

    Features:
    - Single custom equity input
    - Historical equity dropdown feeding into custom input
    - Clear history button
    - Forecast horizon slider
    - Panel selection buttons
    """

    def __init__(self, history_file: str = "datalake/data/raw/equity_history.json"):
        self.history = EquityHistory(history_file)
        self.current_equity: Optional[str] = None
        self.forecast_horizon: int = 1
        self.panel_option: str = "Show Both"

    # ==========================================================
    # Render sidebar components
    # ==========================================================
    def render(self) -> tuple[Optional[str], int, str]:
        """
        Render the sidebar and return current selections.

        Returns:
            tuple: (current_equity, forecast_horizon, panel_option)
        """
        st.sidebar.title("Equity Dashboard Controls")

        # ==========================================================
        # Single Custom Equity Input
        # ==========================================================
        custom_equity = st.sidebar.text_input(
            "Select an Equity",
            placeholder="Select an Equity",
            key="custom_equity_input"
        )

        # ==========================================================
        # Historical Equities Dropdown (feeds into custom input)
        # ==========================================================
        equity_options = self.history.get_history()
        
        if equity_options:
            selected_from_history = st.sidebar.selectbox(
                "Or choose from history",
                options=["-- Select an equity --"] + equity_options,
                key="history_dropdown"
            )
            if selected_from_history != "-- Select an equity --":
                custom_equity = selected_from_history
                st.session_state.custom_equity_input = selected_from_history
                logger.info(f"Equity selected from history: {custom_equity}")

        # ==========================================================
        # Set current equity
        # ==========================================================
        self.current_equity = custom_equity.strip() or None
        logger.info(f"Equity input collected: {self.current_equity}")

        # ==========================================================
        # Clear History Button
        # ==========================================================
        if st.sidebar.button("Clear History"):
            self.history.clear_history()
            logger.info("Cleared equity history")
            # Modern Streamlit rerun approach
            st.experimental_set_query_params(clear="1")
            st.experimental_rerun()

        # ==========================================================
        # Forecast Horizon
        # ==========================================================
        forecast_options = [1, 7, 14, 21, 30]
        self.forecast_horizon = st.sidebar.select_slider(
            "Forecast Horizon (days)",
            options=forecast_options,
            value=self.forecast_horizon,
            format_func=lambda x: f"{x} "
        )
        logger.info(f"Forecast horizon selected: {self.forecast_horizon}")

        # ==========================================================
        # Panel Control
        # ==========================================================
        self.panel_option = st.sidebar.radio(
            "Choose Panel",
            options=["Show Sentiment", "Show Forecast", "Show Both"]
        )
        logger.info(f"Panel option selected: {self.panel_option}")

        # ==========================================================
        # Display current settings
        # ==========================================================
        st.sidebar.markdown(f"**Current Equity (raw):** {self.current_equity}")
        st.sidebar.markdown(f"**Forecast Horizon:** {self.forecast_horizon} days")
        st.sidebar.markdown(f"**Panel Option:** {self.panel_option}")

        return self.current_equity, self.forecast_horizon, self.panel_option


# ==========================================================
# Function to initialize and render sidebar
# ==========================================================
def render_sidebar() -> tuple[Optional[str], int, str]:
    """
    Helper function to render the sidebar and return current selections.

    Returns:
        tuple: (current_equity, forecast_horizon, panel_option)
    """
    sidebar = SidebarUI()
    return sidebar.render()
