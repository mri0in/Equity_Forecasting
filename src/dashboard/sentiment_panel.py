# src/dashboard/sentiment_panel.py

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional, Any
from streamlit.delta_generator import DeltaGenerator
import numpy as np
import logging

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Sentiment Panel Class
# -------------------------------
class SentimentPanel:
    """
    Displays feed-wise sentiment and overall sentiment gauge for a selected equity.

    Features:
    - Bullet chart for individual feeds (News, Press, Social, Web)
    - Circular gauge for overall sentiment
    """

    def __init__(self, equity: str):
        self.equity = equity.upper()
        self.feeds = ["News", "Press", "Social", "Web"]
        self.feed_scores: Dict[str, float] = {}
        self.overall_sentiment: float = 0.0

    # -------------------------------
    # Data Simulation (placeholder)
    # -------------------------------
    def simulate_sentiment(self) -> None:
        """
        Simulate sentiment scores for demonstration purposes.
        Real implementation should fetch from aggregator.
        """
        np.random.seed(hash(self.equity) % 2**32)  # deterministic per equity
        self.feed_scores = {
            feed: np.round(np.random.uniform(-1, 1), 2) for feed in self.feeds
        }
        self.overall_sentiment = np.round(
            np.mean(list(self.feed_scores.values())), 2
        )
        logger.info(
            f"Simulated sentiment for {self.equity}: "
            f"{self.feed_scores}, overall {self.overall_sentiment}"
        )

    # -------------------------------
    # Charts
    # -------------------------------
    def render_bullet_chart(self, container: Optional[DeltaGenerator] = None) -> None:
        """
        Render feed-wise sentiment as a horizontal bar (bullet chart).

        Args:
            container: Optional Streamlit container (DeltaGenerator) to render into.
                       If None, rendering goes to the main page (st).
        """
        if not self.feed_scores:
            st.warning("No feed sentiment scores available.")
            return

        colors = [
            "green" if s > 0 else "red" if s < 0 else "gray"
            for s in self.feed_scores.values()
        ]
        fig = go.Figure(
            go.Bar(
                x=list(self.feed_scores.values()),
                y=list(self.feed_scores.keys()),
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.2f}" for v in self.feed_scores.values()],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{self.equity} - Feed-wise Sentiment",
            xaxis=dict(title="Sentiment Score [-1,1]", range=[-1, 1]),
            yaxis=dict(title="Feeds"),
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # Render into provided container or top-level Streamlit
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    def render_overall_gauge(self, container: Optional[DeltaGenerator] = None) -> None:
        """
        Render overall sentiment as a circular gauge with needle.

        Args:
            container: Optional Streamlit container (DeltaGenerator) to render into.
        """
        if self.overall_sentiment == 0 and not self.feed_scores:
            st.warning("No overall sentiment available.")
            return

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=self.overall_sentiment,
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [-1, -0.3], "color": "red"},
                        {"range": [-0.3, 0.3], "color": "gray"},
                        {"range": [0.3, 1], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": self.overall_sentiment,
                    },
                },
            )
        )
        fig.update_layout(
            title=f"{self.equity} - Overall Sentiment",
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Layout Renderer
    # -------------------------------
    def render(self) -> None:
        """
        Main method to render the sentiment panel with a left/right split:
        - Left side (70%): feed-wise bullet chart
        - Right side (30%): overall gauge
        """
        if not self.equity:
            st.warning("No equity selected for sentiment panel.")
            return

        # Simulate or fetch sentiment
        self.simulate_sentiment()

        st.markdown("### Market Sentiment Panel")

        # Split into 2 columns (70:30)
        col_a, col_b = st.columns([7, 3])

        # Call rendering methods into specific sub-columns
        with col_a:
            self.render_bullet_chart(container=st)  # main area of the sub-column
        with col_b:
            self.render_overall_gauge(container=st)
