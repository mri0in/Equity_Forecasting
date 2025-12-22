# src/pipeline/A_ingestion_pipeline.py
"""
Ingestion Pipeline
------------------
Downloads raw OHLCV data for a list of tickers using yfinance.

Design guarantees:
- One canonical CSV per ticker under datalake/data/raw/
- Each run checks freshness (max_age_days) before downloading
- Sequential, rate-limited fetching (Windows-safe)
- Retry + exponential backoff per ticker
- Full observability via logging + TrainingMonitor
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yfinance as yf

from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates ingestion of raw OHLCV CSVs for a list of tickers.
    """

    def __init__(self, config_path: str) -> None:
        if not config_path:
            raise ValueError("config_path must be provided")

        cfg = load_config(config_path)
        ingest_cfg = cfg.get("ingestion", {})

        # ------------------------------------------------------------------
        # Run metadata
        # ------------------------------------------------------------------
        run_id = f"INGESTION_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = f"datalake/runs/ingestion/{run_id}"

        # ------------------------------------------------------------------
        # Config parameters
        # ------------------------------------------------------------------
        self.tickers: List[str] = ingest_cfg.get("tickers", [])
        if not isinstance(self.tickers, list) or not self.tickers:
            raise ValueError("ingestion.tickers must be a non-empty list")

        self.start_date: str = ingest_cfg.get("start_date", "2013-01-01")
        self.end_date: Optional[str] = ingest_cfg.get("end_date", None)
        self.sleep_time: float = float(ingest_cfg.get("sleep_time", 0.5))
        self.max_retries: int = int(ingest_cfg.get("max_retries", 3))
        self.data_dir: str = ingest_cfg.get("save_path", "datalake/data/raw/")
        self.max_age_days: int = int(ingest_cfg.get("max_age_days", 90))

        # ------------------------------------------------------------------
        # Monitoring
        # ------------------------------------------------------------------
        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.run_dir,
            artifact_policy="metrics",
            enable_plots=False,
        )

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "IngestionPipeline initialized | tickers=%d | retries=%d | max_age_days=%d",
            len(self.tickers),
            self.max_retries,
            self.max_age_days,
        )

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _raw_file_path(self, ticker: str) -> str:
        """Generate a filesystem-safe canonical CSV path for a ticker."""
        safe = ticker.replace(".", "_").replace("/", "_")
        return os.path.join(self.data_dir, f"{safe}.csv")

    def _is_fresh(self, path: str) -> bool:
        """Return True if file exists and is younger than max_age_days."""
        if not os.path.exists(path):
            return False

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age_days = (datetime.now() - mtime).days

        logger.info(
            "[INGEST] Existing file %s age=%d days",
            os.path.basename(path),
            age_days,
        )
        return age_days <= self.max_age_days

    # ------------------------------------------------------------------
    # Yahoo Finance fetch (single-process, stable)
    # ------------------------------------------------------------------
    def _fetch_history(self, ticker: str):
        """
        Fetch OHLCV data from Yahoo Finance.

        IMPORTANT:
        - Runs in main process (no multiprocessing)
        - Proven stable on Windows
        """
        start_ts = time.time()
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                actions=False,
            )

            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")

            df = df.reset_index()

            logger.info(
                "[INGEST] Yahoo OK ticker=%s rows=%d elapsed=%.2fs",
                ticker,
                len(df),
                time.time() - start_ts,
            )
            return df

        except Exception as exc:
            logger.error(
                "[INGEST] Yahoo FAILED ticker=%s error=%s",
                ticker,
                str(exc),
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Single ticker ingestion
    # ------------------------------------------------------------------
    def _process_single_ticker(self, ticker: str) -> None:
        """Ingest one ticker with retry + exponential backoff."""
        save_path = self._raw_file_path(ticker)

        if self._is_fresh(save_path):
            logger.info("[INGEST] Skipping %s — fresh cache", ticker)
            return

        for attempt in range(1, self.max_retries + 1):
            logger.info("[INGEST] %s attempt %d/%d", ticker, attempt, self.max_retries)

            df = self._fetch_history(ticker)

            if df is not None and not df.empty:
                df.to_csv(save_path, index=False)
                logger.info(
                    "[INGEST] Saved %s rows=%d path=%s",
                    ticker,
                    len(df),
                    save_path,
                )
                return

            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.warning(
                    "[INGEST] %s retrying after %.1fs",
                    ticker,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(f"Failed to ingest {ticker} after retries")

    # ------------------------------------------------------------------
    # Yahoo warm-up
    # ------------------------------------------------------------------
    def _warmup_yahoo_session(self) -> None:
        """
        Prime Yahoo Finance session to avoid first-call TLS stalls.
        Uses first configured ticker (no hardcoding).
        """
        try:
            warmup_ticker = "AAPL" if not self.tickers else self.tickers[0] # Hardcoded AAPL ticker
            yf.Ticker(warmup_ticker).history(period="1d")
            logger.info("Yahoo Finance session warmed up successfully")
        except Exception as exc:
            logger.warning("Yahoo warm-up failed: %s", exc)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run ingestion for all configured tickers."""
        stage_name = "ingestion_pipeline"

        self.monitor.log_stage_start(
            stage_name,
            {"num_tickers": len(self.tickers)},
        )

        self._warmup_yahoo_session()

        logger.info("Starting ingestion loop")

        for idx, ticker in enumerate(self.tickers, start=1):
            logger.info("[INGEST] (%d/%d) Processing ticker=%s", idx, len(self.tickers), ticker)
            
            try:
                self._process_single_ticker(ticker)
            except Exception as exc:
                logger.error(
                    "[INGEST] FAILED ticker=%s error=%s",
                    ticker,
                    str(exc),
                    exc_info=True,
                )

            time.sleep(self.sleep_time)

        self.monitor.log_stage_end(stage_name, {"status": "completed"})
        logger.info("Ingestion pipeline completed")
