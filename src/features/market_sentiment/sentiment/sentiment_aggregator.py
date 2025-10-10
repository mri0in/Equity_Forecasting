# src/features/market_sentiment/sentiment/sentiment_aggregator.py
from statistics import median
from src.utils.logger import get_logger
from src.features.market_sentiment.feeds.news_feed import NewsFeed
from src.features.market_sentiment.feeds.press_feed import PressFeed
from src.features.market_sentiment.feeds.social_feed import SocialFeed
from src.features.market_sentiment.feeds.web_feed import WebFeed
from src.features.market_sentiment.processing.pre_processor import TextPreProcessor
from src.features.market_sentiment.processing.extractor import Extractor
from src.features.market_sentiment.sentiment.sentiment_model import SentimentModel

from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    class AllNonSocialFeedsFailed(Exception):
        """Raised when all feeds except SocialFeed fail to fetch/process sentiment."""
        pass

    def __init__(self, equity: str, model_backend: str = "finbert"):
        """
        Initialize SentimentAggregator.

        Args:
            equity (str): Active equity ticker or name.
            model_backend (str): Sentiment backend to use: "textblob" or "finbert(default)".
        """
        self.equity = equity

        # Fallback model in case of unknown backend
        self.fallback_model = "textblob"    

        # Validate requested backend
        allowed_models = ["finbert", "roberta-financial-news", "distilbert-sst2", "textblob"]
        if model_backend not in allowed_models:
            logger.warning(f"Unknown backend '{model_backend}', falling back to {self.fallback_model}")
            model_backend = self.fallback_model
        self.model_backend = model_backend

        # Initialize feed handlers for this equity
        self.feeds = [
            NewsFeed(),
            PressFeed(),
            SocialFeed(),
            WebFeed(),
        ]

        # Processing utilities
        self.preprocessor = TextPreProcessor()
        self.extractor = Extractor()

        # Sentiment model with selected backend (TextBlob/FinBERT implemented now)
        self.sentiment_model = SentimentModel(model_name=model_backend)

        logger.info(
            f"SentimentAggregator initialized | equity={equity} | backend={model_backend}"
        )

    def _process_feed(self, feed) -> Tuple[str, float]:
        """
        Process a single feed: fetch, validate, preprocess, and score sentiment.
        Returns the feed name and its average sentiment score.

        Steps:
            1) Fetch raw data from the feed
            2) Validate the data
            3) Preprocess text
            4) Extract relevant financial text
            5) Perform sentiment scoring per item
            6) Aggregate per-feed average median sentiment
        """
        try:
            # 1) Fetch raw items from feed
            raw_items = feed.fetch_data()
            logger.info(f"{feed.source_name}: fetched {len(raw_items)} raw items")
        except Exception as e:
            # Log and return 0 if fetching fails
            logger.error(f"{feed.source_name}: failed to fetch data: {e}")
            return feed.source_name, 0.0

        # 2) Validate feed data
        if not feed.validate_data(raw_items):
            logger.warning(f"{feed.source_name}: invalid or empty data, skipping.")
            return feed.source_name, 0.0

        # 3) Preprocess -> 4) Extract relevant text
        processed_texts: List[str] = [self._clean_text(item.text) for item in raw_items]
        extracted_texts: List[str] = [
            self.extractor.extract_relevant_text(text) for text in processed_texts
        ]
        extracted_texts = [text for text in extracted_texts if text]  # filter empty
        
        logger.info(f"{feed.source_name}: processed {len(extracted_texts)} texts for sentiment")
        logger.info(f"{feed.source_name}: extracted_texts = {extracted_texts}")

        # 5) Sentiment scoring (TextBlob/FinBERT implemented now)
        scores: List[float] = []
        for idx, text in enumerate(extracted_texts):
            try:
                score = self.sentiment_model.analyze_text(text)["score"]
                scores.append(score)
            except Exception as e:
                # Log per-item sentiment failures but continue
                logger.error(f"{feed.source_name}: failed to analyze text {idx}: {e}")

        # 6) Aggregate per-feed sentiment
        non_zero_scores = [s for s in scores if s != 0.0]
        median_score = median(non_zero_scores) if non_zero_scores else 0.0
        logger.info(
            f"{feed.source_name}: sentiment score {median_score:.3f} (backend={self.model_backend})"
        )
        return feed.source_name, median_score

    def _clean_text(self, text: str) -> str:
        """
        Internal helper to clean raw text using preprocessor.
        """
        if hasattr(self.preprocessor, "clean_text"):
            return self.preprocessor.clean_text(text)
        # Fallback for older naming
        return self.preprocessor.clean_text(text)  # type: ignore[attr-defined]
    

    def SentimentRunner(self, feed_timeout: float = 15.0) -> Dict[str, Any]:
        """
        Execute the sentiment aggregation pipeline in parallel for all feeds.

        Each feed is processed in its own thread. Failures at any stage
        (fetch, validation, per-item sentiment scoring) are logged but do not
        stop the overall pipeline.

        Args:
        feed_timeout (float): Maximum time in seconds to wait per feed thread

        Returns:
            Dict[str, Any]: Combined sentiment results containing:
                - equity: the equity identifier processed
                - feed_scores: median sentiment score per feed
                - overall_sentiment: aggregated equity sentiment score
        """
        feed_scores: Dict[str, float] = {}

        # Run each feed in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.feeds)) as executor:
            # Submit each feed to the executor
            future_to_feed = {executor.submit(self._process_feed, feed): feed for feed in self.feeds}
            
            # As each thread completes, collect its feed score
            for future in as_completed(future_to_feed):
                feed = future_to_feed[future]
                feed_name = type(feed).__name__
                try:
                    # Wait for thread result with timeout
                    _, median_score = future.result(timeout=feed_timeout)
                    feed_scores[feed_name] = float(f"{median_score:.3f}")
                    logger.info(f"{feed_name}: completed with median_score={median_score:.3f}")
                except TimeoutError:
                    logger.warning(f"{feed_name}: timed out after {feed_timeout} seconds.")
                    feed_scores[feed_name] = 0.0
                except Exception as e:
                    logger.exception(f"{feed_name}: failed during sentiment processing: {e}")
                    feed_scores[feed_name] = 0.0

        # Check if all non-social feeds failed and raise exception if so
        non_social_scores = [
            score for feed_name, score in feed_scores.items()
            if feed_name.lower() != "socialfeed"
        ]
        if all(score == 0.0 for score in non_social_scores):
            logger.error("All non-social feeds failed. Raising exception to trigger simulated data.")
            raise self.AllNonSocialFeedsFailed()

        # 7) Aggregate overall sentiment from all feeds
        non_zero_scores = [s for s in feed_scores.values() if s != 0.0]
        overall_sentiment = median(non_zero_scores) if non_zero_scores else 0.0

        logger.info(
            f"{self.equity}: aggregated sentiment {overall_sentiment:.3f} "
            f"from {len(feed_scores)} feeds (backend={self.model_backend})"
        )

        return {
            "equity": self.equity,
            "feed_scores": feed_scores,
            "overall_sentiment": overall_sentiment,
        }
