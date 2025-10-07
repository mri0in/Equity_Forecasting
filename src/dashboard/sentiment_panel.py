# src/dashboard/sentiment_panel.py

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional
from streamlit.delta_generator import DeltaGenerator
from src.dashboard.utils import get_ui_logger

from src.features.market_sentiment.sentiment.sentiment_aggregator import SentimentAggregator

# -------------------------------
# Logging configuration
# -------------------------------
logger = get_ui_logger(__name__)

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
        self.feeds = ["Market_News", "Press", "Social", "Web"]
        self.feed_scores: Dict[str, float] = {}
        self.overall_sentiment: float = 0.0
        self.is_sim_data: bool = True  # Flag to indicate if data is simulated

    # -------------------------------
    # Fetch aggregated sentiment
    # -------------------------------
    def fetch_real_sentiment(self) -> bool:
        """
        Fetch feed-wise sentiment and overall sentiment from SentimentAggregator.
        """
        if not self.equity:
            logger.warning("No equity specified for fetching sentiment")
            return

        is_sim_data = True

        try:
            aggregator = SentimentAggregator(equity=self.equity)
            result = aggregator.SentimentRunner()
            self.feed_scores = result.get("feed_scores", {})
            self.overall_sentiment = result.get("overall_sentiment", 0.0)
            is_sim_data = False
            self.is_sim_data = False
            logger.info(
                f"Fetched aggregated sentiment for {self.equity}: "
                f"{self.feed_scores}, overall {self.overall_sentiment}"
            )
        except SentimentAggregator.AllNonSocialFeedsFailed:
            logger.warning(f"Using simulated sentiment for {self.equity} (all non-social feeds failed).")
            self.simulate_sentiment()
            self.is_sim_data = True
        except Exception as e:
            logger.error(f"Fetching aggregated sentiment failed for {self.equity}: {e}")
            raise
        return is_sim_data

    # -------------------------------
    # Data Simulation (fallback)
    # -------------------------------
    def simulate_sentiment(self) -> None:
        """
        Simulate sentiment scores for demonstration purposes.
        """
        import numpy as np

        np.random.seed(hash(self.equity) % 2**32)
        self.feed_scores = {
            feed: np.round(np.random.uniform(-1, 1), 2) for feed in self.feeds
        }

        self.overall_sentiment = round(
            sum(self.feed_scores.values()) / len(self.feed_scores), 2
        )
        
        logger.info(
            f"Simulated sentiment for {self.equity}: "
            f"{self.feed_scores}, overall {self.overall_sentiment}"
        )

    # -------------------------------
    # Charts
    # -------------------------------
    def render_bullet_chart(self, container: Optional[DeltaGenerator] = None) -> None:
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
                y = [f"{k}*" if self.is_sim_data else k for k in self.feed_scores.keys()],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:+.2f}" for v in self.feed_scores.values()],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{self.equity} - Feed-wise Sentiment *" if self.is_sim_data else f"{self.equity} - Feed-wise Sentiment ✅", # (*) for simulated data, ✅ for real data
            xaxis=dict(title="Sentiment Score [-1,1]", range=[-1, 1]),
            yaxis=dict(title="Feeds"),
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    def render_overall_gauge(self, container: Optional[DeltaGenerator] = None) -> None:
        if not self.feed_scores and self.overall_sentiment == 0:
            st.warning("No overall sentiment available.")
            return

        gauge_value = (self.overall_sentiment + 1) / 2

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_value,
            number={'suffix': '', 'valueformat': '.2f'},
            delta={'reference': 0.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            title={'text': f" {self.equity} - Overall Sentiment *" if self.is_sim_data                else f"{self.equity} - Overall Sentiment ✅"}, # (*) for simulated data, ✅ for real data
            
            gauge={
                'axis': {'range': [0, 1], 'tickvals': [0, 0.25, 0.5, 0.75, 1],
                         'ticktext': ['-1', '-0.5', '0', '0.5', '1']},
                'bar': {'color': "black", 'thickness': 0.05},
                'steps': [
                    {'range': [0, 0.35], 'color': "red"},
                    {'range': [0.35, 0.65], 'color': "gray"},
                    {'range': [0.65, 1], 'color': "green"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': gauge_value
                }
            }
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Layout Renderer
    # -------------------------------
    def render_sentiment(self) ->  bool:
        """
        Render the sentiment panel.
        If use_real_data is True, fetch actual sentiment from aggregator.
        Otherwise, fallback to simulation for demo.
        """
        if not self.equity:
            st.warning("No equity selected for sentiment panel.")
            return

        is_sim_data = self.fetch_real_sentiment()
        
        col_a, col_b = st.columns([6, 4])
        with col_a:
            self.render_bullet_chart(container=st)
        with col_b:
            self.render_overall_gauge(container=st)
        
        return is_sim_data