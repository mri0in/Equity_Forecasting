# src/dashboard/sentiment_panel.py

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger("dashboard.sentiment_panel")
logger.setLevel(logging.INFO)

class SentimentPanel:
    """
    Split rendering for feed-wise scores and overall gauge.
    """

    def __init__(self, equity: str):
        self.equity = (equity or "").upper()
        self.feeds = ['News', 'Press', 'Social', 'Web']
        self.feed_scores: Dict[str, float] = {}
        self.overall_sentiment: float = 0.0

    def simulate_sentiment(self) -> None:
        """Deterministic demo sentiment per equity."""
        np.random.seed(hash(self.equity) % 2**32)
        self.feed_scores = {feed: float(np.round(np.random.uniform(-1, 1), 2)) for feed in self.feeds}
        self.overall_sentiment = float(np.round(np.mean(list(self.feed_scores.values())), 2))
        logger.info(f"Simulated sentiment for {self.equity}: {self.feed_scores}, overall {self.overall_sentiment}")

    def render_feed_scores(self, height: int = 420) -> None:
        """
        Render feed-wise sentiment as a horizontal bar chart.
        `height` in pixels controls vertical space (use larger value for the 70% portion).
        """
        if not self.feed_scores:
            st.warning("No feed scores available. Call simulate_sentiment() first.")
            return

        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in self.feed_scores.values()]
        fig = go.Figure(go.Bar(
            x=list(self.feed_scores.values()),
            y=list(self.feed_scores.keys()),
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:+.2f}" for v in self.feed_scores.values()],
            textposition='outside'
        ))
        fig.update_layout(
            title=f"{self.equity} — Feed-wise Sentiment",
            xaxis=dict(title="Sentiment Score [-1,1]", range=[-1, 1]),
            yaxis=dict(title="", automargin=True),
            template="plotly_white",
            height=height
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_overall_gauge(self, height: int = 180) -> None:
        """
        Render an overall sentiment gauge (ring/indicator).
        `height` in pixels controls the bottom 30% portion.
        """
        if not self.feed_scores:
            st.warning("No overall sentiment available. Call simulate_sentiment() first.")
            return

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=self.overall_sentiment,
            number={'suffix': ""},
            gauge={
                'axis': {'range': [-1, 1], 'tickvals': [-1, 0, 1], 'ticktext': ['-1', '0', '+1']},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "gray"},
                    {'range': [0.3, 1], 'color': "green"}
                ],
            }
        ))
        fig.update_layout(title=f"{self.equity} — Overall Sentiment ({self.overall_sentiment:+.2f})", height=height)
        st.plotly_chart(fig, use_container_width=True)
