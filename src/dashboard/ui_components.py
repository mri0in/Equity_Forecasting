# src/dashboard/ui_components.py

import streamlit as st
from typing import Optional
from .history_manager import EquityHistory
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Sidebar UI Components Class
# -------------------------------
class SidebarUI:
    """
    Handles the Streamlit sidebar UI components for the Equity Dashboard.
    
    Features:
    - Custom equity input
    - Historical equity dropdown
    - Clear history button
    - Forecast horizon slider
    - Panel selection buttons
    """

    def __init__(self, history_file: str = "src/dashboard/data/equity_history.json"):
        self.history = EquityHistory(history_file)
        self.current_equity: Optional[str] = None
        self.forecast_horizon: int = 7
        self.panel_option: str = "Show Both"

    def render(self):
        """
        Render the sidebar components and update current equity, horizon, and panel selection.
        """
        st.sidebar.title("Equity Dashboard Controls")

        # ---- Custom Equity Input ----
        custom_equity = st.sidebar.text_input(
            "Select an Equity",
            placeholder="Select an Equity"
        )

        if custom_equity:
            self.history.add_equity(custom_equity.upper())
            logger.info(f"Added custom equity: {custom_equity.upper()}")

        # ---- Historical Equities Dropdown ----
        equity_options = self.history.get_history()
        selected_equity = None
        if equity_options:
            selected_equity = st.sidebar.selectbox(
                "Or choose from history",
                options=equity_options,
                index=0
            )

        # Determine current equity
        self.current_equity = selected_equity if selected_equity else custom_equity

        # ---- Clear History Button ----
        if st.sidebar.button("Clear History"):
            self.history.clear_history()
            logger.info("Cleared equity history")
            st.experimental_rerun()  # Refresh sidebar to update dropdown

        # ---- Forecast Horizon Slider ----
        self.forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=1
        )

        # ---- Panel Control Buttons ----
        self.panel_option = st.sidebar.radio(
            "Choose Panel",
            options=["Show Sentiment", "Show Forecast", "Show Both"]
        )

        # ---- Display current settings ----
        st.sidebar.markdown(f"**Current Equity:** {self.current_equity}")
        st.sidebar.markdown(f"**Forecast Horizon:** {self.forecast_horizon} days")
        st.sidebar.markdown(f"**Panel Option:** {self.panel_option}")

        return self.current_equity, self.forecast_horizon, self.panel_option


# -------------------------------
# Function to initialize and render sidebar
# -------------------------------
def render_sidebar():
    """
    Helper function to render the sidebar and return current selections.
    """
    sidebar = SidebarUI()
    return sidebar.render()
