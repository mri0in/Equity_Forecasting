# src/pipeline/A_ingestion_pipeline.py
"""
Ingestion Pipeline
------------------
Downloads raw OHLCV data for a list of tickers (NSE + global) using yfinance,
but only when the raw file is missing or older than a configurable expiry
(e.g., 90 days). Designed to be called from the pipeline wrapper / DAG.

Expected config.yaml excerpt:
-----------------------------
ingestion:
  tickers: ["AAPL", "MSFT", "RELIANCE.NS"]
  start_date: "2013-01-01"
  end_date: null
  sleep_time: 0.25
  max_retries: 3
  save_path: "datalake/data/raw/"
  max_age_days: 90

Notes:
- This module performs only ingestion orchestration.
- Actual download/backoff logic is implemented here; saving is to the configured save_path.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import yfinance as yf

from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)
monitor = TrainingMonitor()


class IngestionPipeline:
    """
    Orchestrates downloading raw OHLCV CSVs for a universe of tickers.

    The pipeline is config-driven. It will *skip* downloading any ticker whose
    raw CSV file exists and whose modification time is within `max_age_days`.

    Attributes
    ----------
    config_path : str
        Path to the YAML configuration used to drive ingestion parameters.
    tickers : List[str]
        List of ticker symbols to ingest.
    start_date : str
        Start date for yfinance download.
    end_date : Optional[str]
        End date for yfinance download.
    sleep_time : float
        Sleep between tickers to avoid burst loads.
    max_retries : int
        Number of retries per ticker on failure.
    save_dir : str
        Directory to save raw CSV files.
    max_age_days : int
        Number of days after which a cached file is considered stale and will be refreshed.
    """

    def __init__(self, config_path: str) -> None:
        if not config_path:
            raise ValueError("config_path must be provided to IngestionPipeline")

        cfg = load_config(config_path)
        ingest_cfg = cfg.get("ingestion", {})

        # Input validation and defaults
        self.tickers: List[str] = ingest_cfg.get("tickers", [])
        if not isinstance(self.tickers, list):
            raise TypeError("ingestion.tickers must be a list of ticker strings")

        self.start_date: str = ingest_cfg.get("start_date", "2013-01-01")
        self.end_date: Optional[str] = ingest_cfg.get("end_date", None)
        self.sleep_time: float = float(ingest_cfg.get("sleep_time", 0.25))
        self.max_retries: int = int(ingest_cfg.get("max_retries", 3))
        self.save_dir: str = ingest_cfg.get("save_path", "datalake/data/raw/")
        self.max_age_days: int = int(ingest_cfg.get("max_age_days", 90))

        # Ensure save directory exists
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "IngestionPipeline initialized: %d tickers, save_dir=%s, max_age_days=%d",
            len(self.tickers), self.save_dir, self.max_age_days
        )

    # ------------------------------------------------------------------
    # Helper: file freshness check
    # ------------------------------------------------------------------
    def _raw_file_path(self, ticker: str) -> str:
        """Return platform-safe raw CSV path for a ticker."""
        safe_name = ticker.replace(".", "_").replace("/", "_")
        return os.path.join(self.save_dir, f"{safe_name}.csv")

    def needs_refresh(self, file_path: str) -> bool:
        """
        Returns True if the file does not exist or is older than max_age_days.

        Parameters
        ----------
        file_path : str
            Path to the raw CSV file.

        Returns
        -------
        bool
        """
        p = Path(file_path)
        if not p.exists():
            logger.info("Raw file missing: %s (will download)", file_path)
            return True

        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except Exception:
            # If we cannot stat the file for any reason, force refresh
            logger.warning("Unable to stat file %s — forcing refresh", file_path)
            return True

        age = (datetime.now() - mtime).days
        if age > self.max_age_days:
            logger.info("Raw file stale (%d days) → will refresh: %s", age, file_path)
            return True

        logger.info("Raw file fresh (%d days) → skipping download: %s", age, file_path)
        return False

    # ------------------------------------------------------------------
    # Internal: download single ticker with retry & backoff
    # ------------------------------------------------------------------
    def _download_ticker(self, ticker: str) -> None:
        """
        Download and save ticker CSV using yfinance with retry/backoff.
        This method assumes needs_refresh() already determined we need to fetch.
        """
        save_path = self._raw_file_path(ticker)

        backoff_base = 1.5
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("[%s] Download attempt %d/%d", ticker, attempt, self.max_retries)

                # Use period 'max' if start_date/end_date not set — but respect config
                df = yf.download(
                    tickers=ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                )

                if df is None or df.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")

                # Save CSV
                df.to_csv(save_path)
                logger.info("[%s] Saved raw CSV → %s", ticker, save_path)
                return

            except Exception as exc:
                logger.error("[%s] Error on attempt %d: %s", ticker, attempt, exc)
                if attempt < self.max_retries:
                    sleep_for = backoff_base ** attempt
                    logger.info("[%s] Backing off for %.1f sec before retry", ticker, sleep_for)
                    time.sleep(sleep_for)
                else:
                    logger.error("[%s] Exhausted retries — skipping ticker", ticker)

        # final inter-ticker sleep to avoid burst
        logger.debug("Sleeping %.3fs between tickers", self.sleep_time)
        time.sleep(self.sleep_time)

    # ------------------------------------------------------------------
    # Public: run the ingestion pipeline
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Run ingestion for all tickers defined in config.
        Downloads only missing or stale raw files and logs progress via the monitor.
        """
        stage_name = "ingestion_pipeline"
        monitor.log_stage_start(stage_name, {"num_tickers": len(self.tickers)})

        try:
            logger.info("Starting ingestion for %d tickers...", len(self.tickers))

            for ticker in self.tickers:
                file_path = self._raw_file_path(ticker)
                if self.needs_refresh(file_path):
                    # Attempt download (retries & backoff inside)
                    self._download_ticker(ticker)
                else:
                    logger.info("[%s] Skip download (fresh): %s", ticker, file_path)

            monitor.log_stage_end(stage_name, {"status": "completed"})
            logger.info("Ingestion pipeline completed successfully.")

        except Exception as exc:
            logger.exception("Ingestion pipeline failed: %s", exc)
            monitor.log_stage_end(stage_name, {"status": "failed"})
            raise
