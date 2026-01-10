# src/pipeline/C_feature_gen_pipeline.py
"""
C. Feature Generation Pipeline
------------------------------

This pipeline enriches cleaned time-series datasets with derived features
and a supervised learning 'target' column.

The output artifacts serve as standardized inputs for downstream
optimization, training, and inference pipelines.

Contract
--------
- Accepts a single runtime input: `run_id`
- Resolves ticker universe from ingestion artifacts
- Reads cleaned per-ticker datasets
- Writes per-ticker Parquet feature datasets

Responsibilities
---------------
- Resolve the ticker universe associated with a run
- Load cleaned time-series inputs
- Generate derived features
- Generate a supervised learning target column
- Persist enriched datasets in a columnar format

"""

from __future__ import annotations

from pathlib import Path
from typing import List

import json as jsonlib
import numpy as np
import pandas as pd

from src.features.technical.build_features import FeatureBuilder
from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger


class FeaturePipeline:
    """
    Feature and target generation pipeline driven by ingestion artifacts.
    """

    def __init__(
        self,
        run_id: str,
        clean_root: str = "datalake/data/cache/clean",
        feature_root: str = "datalake/data/cache/features",
        target_column: str = "Close",
        target_horizon: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        run_id : str
            Identifier of the ingestion run.
        clean_root : str
            Directory containing cleaned input datasets.
        feature_root : str
            Directory where feature datasets will be written.
        target_column : str
            Base column used to construct the supervised target.
        target_horizon : int
            Forward shift applied to construct the target.
        """
        if not run_id:
            raise ValueError("run_id must be provided")

        if target_horizon < 1:
            raise ValueError("target_horizon must be >= 1")

        self.run_id = run_id
        self.clean_root = Path(clean_root)
        self.feature_root = Path(feature_root)
        self.target_column = target_column
        self.target_horizon = target_horizon

        self.feature_root.mkdir(parents=True, exist_ok=True)

        self.used_tickers_path = (
            Path("datalake")
            / "runs"
            / run_id
            / "ingestion"
            / "used_tickers.json"
        )

        self.logger = get_logger(self.__class__.__name__)
        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=Path(f"datalake/runs/{run_id}/features"),
            artifact_policy="none",
        )

        self.logger.info(
            "FeaturePipeline initialized | run_id=%s | target=%s | horizon=%d",
            run_id,
            target_column,
            target_horizon,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_used_tickers(self) -> List[str]:
        """
        Load tickers associated with the ingestion run.
        """
        if not self.used_tickers_path.exists():
            raise FileNotFoundError(
                f"used_tickers.json not found for run_id={self.run_id}"
            )

        with self.used_tickers_path.open("r", encoding="utf-8") as fh:
            payload = jsonlib.load(fh)

        tickers = payload.get("tickers")

        if not isinstance(tickers, list) or not all(
            isinstance(t, str) for t in tickers
        ):
            raise ValueError(
                "Invalid used_tickers.json schema: 'tickers' must be List[str]"
            )

        self.logger.info(
            "[FEAT] Loaded %d tickers from ingestion run",
            len(tickers),
        )
        return tickers

    def _is_up_to_date(self, clean_path: Path, feature_path: Path) -> bool:
        """
        Check whether a feature file is newer than its cleaned input.
        """
        if not feature_path.exists():
            return False

        return feature_path.stat().st_mtime >= clean_path.stat().st_mtime

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add supervised learning target column to the dataset.

        Target is constructed as a forward-shifted value of the base
        column (e.g. Close price at t + horizon).
        """
        if self.target_horizon < 1:
            raise ValueError("Target horizon must be >= 1")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target base column '{self.target_column}' not found in dataset"
            )

        df = df.copy()

        df["target"] = df[self.target_column].shift(-self.target_horizon)

        # Drop rows where target cannot be computed
        df = df.dropna(subset=["target"])

        self.logger.info("Added target column with horizon=%d", self.target_horizon)

        return df

    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing NaN or infinite values in feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target column.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with invalid rows removed.
        """
        original_len = len(df)

        # Replace inf values explicitly with NaN which can be dropped later
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows with any NaN
        df_clean = df.dropna(axis=0, how="any")

        removed = original_len - len(df_clean)

        if removed > 0:
            self.logger.info(
                "[Pipeline C] Sanitized features | removed_rows=%d | remaining=%d",
                removed,
                len(df_clean),
            )

        if df_clean.empty:
            raise ValueError(
                "Feature sanitization removed all rows. "
                "Check feature windows and data length."
            )

        return df_clean
    
    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute feature and target generation for all tickers in the run.
        """
        stage_name = "feature_pipeline"

        used_tickers = self._load_used_tickers()

        self.monitor.log_stage_start( stage_name, {"num_tickers": len(used_tickers)}, )

        self.logger.info(
            "Starting feature generation | run_id=%s | tickers=%d",
            self.run_id,
            len(used_tickers),
        )

        for idx, ticker in enumerate(used_tickers, start=1):
            clean_path = self.clean_root / f"{ticker}.csv"
            feature_path = self.feature_root / f"{ticker}.parquet"

            self.logger.info(
                "[FEAT] (%d/%d) Processing %s",
                idx,
                len(used_tickers),
                ticker,
            )

            if not clean_path.exists():
                self.logger.error(
                    "[FEAT] Cleaned file missing for %s: %s",
                    ticker,
                    clean_path,
                )
                continue

            #if self._is_up_to_date(clean_path, feature_path):
                self.logger.info( "[FEAT] Skipping %s (features up-to-date)", ticker, )
                continue

            try:
                df_clean = pd.read_csv(clean_path)

                builder = FeatureBuilder(equity=ticker)
                df_features = builder.build_all(df_clean)

                df_features = self._add_target(df_features)

                # remove rows with NaN or infinite values
                df_features = self._sanitize_features(df_features)
                
                # Add equity identifier column after NaN sanitization
                df_features["equity_id"] = ticker

                df_features.to_parquet(
                    feature_path,
                    index=False,
                    engine="pyarrow",
                )

                self.logger.info( "[FEAT] Saved features %s | rows=%d | path=%s", ticker, len(df_features), feature_path, )

            except Exception as exc:
                self.logger.error(
                    "[FEAT] Feature generation failed for %s | error=%s",
                    ticker,
                    str(exc),
                    exc_info=True,
                )

        self.monitor.log_stage_end( stage_name, {"status": "completed"}, )

        self.logger.info( "Feature generation pipeline completed | run_id=%s", self.run_id, )
