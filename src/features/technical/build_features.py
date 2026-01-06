# src/features/technical/build_features.py

"""
Feature Builder
---------------
Pure feature engineering module.

This module is intentionally:
- Stateless
- Cache-agnostic
- Deterministic

All decisions regarding reuse, skipping, persistence, or formats
(CSV vs Parquet) must be handled by the calling pipeline (Pipeline C).
"""

from typing import Final
import pandas as pd

from src.features.technical.candle_features import CandleFeatures
from src.features.technical.trend_features import TrendFeatures
from src.features.technical.momentum_features import MomentumFeatures
from src.features.technical.volume_features import VolumeFeatures
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FeatureBuilder:
    """
    Orchestrates technical feature generation for a single equity.

    This class:
    - Accepts a clean OHLCV DataFrame
    - Applies multiple feature groups
    - Returns a merged feature DataFrame
    """

    def __init__(self, equity: str):
        """
        Parameters
        ----------
        equity : str
            Equity ticker symbol (used only for logging / traceability)
        """
        self.equity: Final[str] = equity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all technical features.

        Parameters
        ----------
        df : pd.DataFrame
            Clean OHLCV data indexed by datetime

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame
        """
        self._validate_input(df)

        logger.info("Starting feature generation for equity=%s", self.equity)

        try:
            df_candle = self._build_candle_features(df)
            df_trend = self._build_trend_features(df)
            df_momentum = self._build_momentum_features(df)
            df_volume = self._build_volume_features(df)

            df_features = pd.concat(
                [df,df_candle, df_trend, df_momentum, df_volume],
                axis=1,
            )

            logger.info(
                "Feature generation completed for equity=%s | shape=%s",
                self.equity,
                df_features.shape,
            )

            return df_features

        except Exception as exc:
            logger.error(
                "Feature generation failed for equity=%s: %s",
                self.equity,
                exc,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """
        Validate input DataFrame structure.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")

    def _build_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building candle features for %s", self.equity)
        return CandleFeatures(df=df, equity=self.equity).compute_all()

    def _build_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building trend features for %s", self.equity)
        return TrendFeatures(df=df, equity=self.equity).compute_all()

    def _build_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building momentum features for %s", self.equity)
        return MomentumFeatures(df=df, equity=self.equity).compute_all()

    def _build_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building volume features for %s", self.equity)
        return VolumeFeatures(df=df, equity=self.equity).compute_all()
