# src/dashboard/ui_components.py

import logging
from typing import Optional
import streamlit as st
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
    - Forecast horizon slider
    - Panel selection buttons
    """

    def __init__(self):
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
        self.current_equity = st.sidebar.text_input(
            "Enter Equity Symbol",
            placeholder="e.g., AAPL, RELIANCE"
        ).strip() or None

        logger.info(f"Equity input collected: {self.current_equity}")

        # ==========================================================
        # Forecast Horizon
        # ==========================================================
        forecast_options = [1, 7, 14, 21, 30]
        self.forecast_horizon = st.sidebar.select_slider(
            "Forecast Horizon (days)",
            options=forecast_options,
            value=1,
            format_func=lambda x: f"{x} "
        )

        # ==========================================================
        # Panel Control
        # ==========================================================
        self.panel_option = st.sidebar.radio(
            "Choose Panel",
            options=["Show Sentiment", "Show Forecast", "Show Both"]
        )

        # ==========================================================
        # Display current settings
        # ==========================================================
        st.sidebar.markdown(f"**Current Equity:** {self.current_equity}")
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
