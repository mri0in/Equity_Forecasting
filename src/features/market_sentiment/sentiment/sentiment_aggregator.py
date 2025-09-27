# src/features/market_sentiment/sentiment/sentiment_aggregator.py

from typing import Dict, Any, List
from src.utils.logger import get_logger

from src.features.market_sentiment.feeds.market_news_feed import MarketNewsFeed
from src.features.market_sentiment.feeds.press_feed import PressFeed
from src.features.market_sentiment.feeds.social_feed import SocialFeed
from src.features.market_sentiment.feeds.web_feed import WebFeed

from src.features.market_sentiment.processing.pre_processor import TextPreProcessor
from src.features.market_sentiment.processing.extractor import Extractor
from src.features.market_sentiment.sentiment.sentiment_model import SentimentModel

"""
Sentiment Aggregator Module

This module orchestrates the end-to-end sentiment pipeline across multiple
market-related feeds (News, Press, Social, Web) for a given equity.

Core Responsibilities:
    - Initialize feed handlers for a given active equity
    - Fetch and validate raw data from multiple sources
    - Preprocess and extract relevant financial text
    - Perform sentiment analysis using SentimentModel
    - Aggregate sentiment across feeds into an overall equity sentiment metric
    - Log detailed process flow for monitoring and debugging

Implementation Notes:
    - SentimentModel backends implemented NOW: TextBlob (baseline) and FinBERT
      (finance-domain transformer). 
"""

logger = get_logger("sentiment_aggregator")


class SentimentAggregator:
    """
    Aggregates sentiment from multiple feeds (news, press, social, web).
    Orchestrates fetching, preprocessing, extraction, and sentiment scoring.
    Produces an equity-level sentiment summary.
    """

    def __init__(self, equity: str, model_backend: str = "finbert"):
        """
        Args:
            equity (str): Active equity ticker or name.
            model_backend (str): Sentiment backend to use: "textblob" or "finbert".
            (Custom ML can be added later behind the same interface.)
        """
        self.equity = equity

        # fallback model
        self.fallback_model = "textblob"

        # Validate requested backend
        allowed_models = ["finbert",  "roberta-financial-news", "distilbert-sst2","textblob"]
        if model_backend not in allowed_models:
            logger.warning(f"Unknown backend '{model_backend}', falling back to {self.fallback_model}")
            model_backend = self.fallback_model
        self.model_backend = model_backend

        # Initialize feed handlers for this equity
        self.feeds = [
            MarketNewsFeed(),
            PressFeed(equity),
            SocialFeed(equity),
            WebFeed(equity),
        ]

        # Processing utilities
        self.preprocessor = TextPreProcessor()
        self.extractor = Extractor()

        # Sentiment model with selected backend (TextBlob/FinBERT implemented now)
        self.sentiment_model = SentimentModel(backend=model_backend)

        logger.info(
            f"SentimentAggregator initialized | equity={equity} | backend={model_backend}"
        )

    def _clean_text(self, text: str) -> str:
        """
        Internal helper to call the appropriate preprocessor method.
        Supports either `clean_text` or `clean` depending on your implementation.
        """
        if hasattr(self.preprocessor, "clean_text"):
            return self.preprocessor.clean_text(text)
        # Fallback for earlier naming
        return self.preprocessor.clean(text)  # type: ignore[attr-defined]

    def SentimentRunner(self) -> Dict[str, Any]:
        """
        Execute the sentiment aggregation pipeline.

        Returns:
            Dict[str, Any]: Combined sentiment results containing:
                - equity: the equity identifier processed
                - feed_scores: average sentiment score per feed
                - overall_sentiment: aggregated equity sentiment score
        """
        feed_scores: Dict[str, float] = {}

        for feed in self.feeds:
            # 1) Fetch
            raw_items = feed.fetch_data()
            logger.info(f"{feed.source_name}: fetched {len(raw_items)} raw items")

            # 2) Validate
            if not feed.validate_data(raw_items):
                logger.warning(f"{feed.source_name}: invalid or empty data, skipping.")
                continue

            # 3) Preprocess -> 4) Extract relevant text
            processed_texts: List[str] = [self._clean_text(item.text) for item in raw_items]
            extracted_texts: List[str] = [
                self.extractor.extract_relevant_text(text) for text in processed_texts
            ]
            logger.info(
                f"{feed.source_name}: processed {len(extracted_texts)} texts for sentiment"
            )

            # 5) Sentiment scoring (TextBlob/FinBERT implemented now)
            scores: List[float] = [self.sentiment_model.analyze(text) for text in extracted_texts]

            # 6) Aggregate per-feed
            avg_score = (sum(scores) / len(scores)) if scores else 0.0
            feed_scores[feed.source_name] = avg_score
            logger.info(
                f"{feed.source_name}: avg sentiment score {avg_score:.3f} "
                f"(backend={self.model_backend})"
            )

        # 7) Aggregate overall
        overall_sentiment = (sum(feed_scores.values()) / len(feed_scores)) if feed_scores else 0.0
        logger.info(
            f"{self.equity}: aggregated sentiment {overall_sentiment:.3f} "
            f"from {len(feed_scores)} feeds (backend={self.model_backend})"
        )

        return {
            "equity": self.equity,
            "feed_scores": feed_scores,
            "overall_sentiment": overall_sentiment,
        }
