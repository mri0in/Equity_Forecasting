# src/features/market_sentiment/sentiment/sentiment_aggregator.py

from typing import Dict, Any, List
from src.utils import setup_logger

from src.features.market_sentiment.feeds.news_feed import NewsFeed
from src.features.market_sentiment.feeds.press_feed import PressFeed
from src.features.market_sentiment.feeds.social_feed import SocialFeed
from src.features.market_sentiment.feeds.web_feed import WebFeed

from src.features.market_sentiment.processing.pre_processor import PreProcessor
from src.features.market_sentiment.processing.extractor import Extractor
from src.features.market_sentiment.sentiment.sentiment_model import SentimentModel

"""
Sentiment Aggregator Module

This module is responsible for orchestrating the sentiment analysis pipeline
across multiple market-related feeds (News, Press, Social, Web). It fetches
raw content for a given equity, preprocesses and extracts relevant text,
applies the sentiment model, and aggregates feed-level scores into a single
equity sentiment score.

Core Responsibilities:
    - Initialize feed handlers for a given active equity
    - Fetch and validate raw data from multiple sources
    - Preprocess and extract relevant financial text
    - Perform sentiment analysis using SentimentModel
    - Aggregate sentiment across feeds into an overall equity sentiment metric
    - Log detailed process flow for monitoring and debugging

Output:
    A dictionary containing:
        - equity (str): Equity ticker or name
        - feed_scores (Dict[str, float]): Sentiment scores per feed
        - overall_sentiment (float): Aggregated sentiment score for the equity

"""


logger = setup_logger("sentiment_aggregator")


class SentimentAggregator:
    """
    Aggregates sentiment from multiple feeds (news, press, social, web).
    Orchestrates fetching, preprocessing, extraction, and sentiment scoring.
    Produces an equity-level sentiment summary.
    """

    def __init__(self, equity: str):
        """
        Args:
            equity (str): Active equity ticker or name
        """
        self.equity = equity
        self.feeds = [
            NewsFeed(equity),
            PressFeed(equity),
            SocialFeed(equity),
            WebFeed(equity),
        ]
        self.preprocessor = PreProcessor()
        self.extractor = Extractor()
        self.sentiment_model = SentimentModel()

    def run(self) -> Dict[str, Any]:
        """
        Execute the sentiment aggregation pipeline.

        Returns:
            Dict[str, Any]: Combined sentiment results containing:
                - feed_scores: individual feed-level scores
                - overall_sentiment: aggregated equity sentiment score
        """
        feed_scores: Dict[str, float] = {}

        for feed in self.feeds:
            raw_items = feed.fetch_data()

            if not feed.validate_data(raw_items):
                logger.warning(f"{feed.source_name}: Invalid or empty data, skipping.")
                continue

            processed_texts: List[str] = [
                self.preprocessor.clean(item.text) for item in raw_items
            ]

            extracted_texts: List[str] = [
                self.extractor.extract_relevant_text(text) for text in processed_texts
            ]

            scores: List[float] = [
                self.sentiment_model.analyze(text) for text in extracted_texts
            ]

            avg_score = sum(scores) / len(scores) if scores else 0.0
            feed_scores[feed.source_name] = avg_score
            logger.info(f"{feed.source_name}: Sentiment score {avg_score:.3f}")

        overall_sentiment = (
            sum(feed_scores.values()) / len(feed_scores) if feed_scores else 0.0
        )

        logger.info(f"{self.equity}: Aggregated sentiment {overall_sentiment:.3f}")

        return {
            "equity": self.equity,
            "feed_scores": feed_scores,
            "overall_sentiment": overall_sentiment,
        }
