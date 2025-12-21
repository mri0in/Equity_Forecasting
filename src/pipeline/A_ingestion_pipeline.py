# src/pipeline/A_ingestion_pipeline.py
"""
Ingestion Pipeline
------------------
Downloads raw OHLCV data for a list of tickers (NSE + global) using yfinance,
but only when the raw file is missing or older than a configurable expiry
(e.g., 90 days). Designed to be called from the pipeline wrapper / DAG.

Key points:
- Each ticker gets a single canonical CSV in `data/raw/`
- Each batch checks every ticker for freshness (max_age_days)
- Child-process + hard timeout ensures no hang on Windows
- Retries & exponential backoff are applied per ticker
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from multiprocessing import Process, Queue

import yfinance as yf

from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Child-process Yahoo worker (TOP-LEVEL for Windows compatibility)
# ---------------------------------------------------------------------
def _yahoo_download_worker(ticker: str, start_date: str, end_date: Optional[str], q: Queue) -> None:
    """
    Fetch Yahoo Finance OHLCV data in a separate process.
    Returns either DataFrame or Exception via Queue.
    """
    try:
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        q.put(df)
    except Exception as exc:
        q.put(exc)


class IngestionPipeline:
    """
    Orchestrates ingestion of raw OHLCV CSVs for a list of tickers.
    Ensures canonical storage under `data/raw/` with freshness enforcement.
    """

    def __init__(self, config_path: str) -> None:
        if not config_path:
            raise ValueError("config_path must be provided")

        cfg = load_config(config_path)
        ingest_cfg = cfg.get("ingestion", {})

        run_id = f"INGESTION_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = f"datalake/runs/ingestion/{run_id}"

        # Config-driven parameters
        self.tickers: List[str] = ingest_cfg.get("tickers", [])
        if not isinstance(self.tickers, list):
            raise TypeError("ingestion.tickers must be a list")

        self.start_date: str = ingest_cfg.get("start_date", "2013-01-01")
        self.end_date: Optional[str] = ingest_cfg.get("end_date", None)
        self.sleep_time: float = float(ingest_cfg.get("sleep_time", 0.25))
        self.max_retries: int = int(ingest_cfg.get("max_retries", 3))
        self.data_dir: str = ingest_cfg.get("save_path", "datalake/data/raw/")
        self.max_age_days: int = int(ingest_cfg.get("max_age_days", 90))
        self.timeout_sec: int = int(ingest_cfg.get("timeout_sec", 20))

        # Monitoring
        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.run_dir,
            visualize=False,
            flush_every=1,
        )

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "IngestionPipeline initialized | tickers=%d | timeout=%ss | retries=%d",
            len(self.tickers),
            self.timeout_sec,
            self.max_retries,
        )

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _raw_file_path(self, ticker: str) -> str:
        """Generate a filesystem-safe canonical CSV path for a ticker."""
        safe = ticker.replace(".", "_").replace("/", "_")
        return os.path.join(self.data_dir, f"{safe}.csv")

    def _is_fresh(self, path: str) -> bool:
        """Return True if the file exists and is younger than max_age_days."""
        if not os.path.exists(path):
            return False

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age_days = (datetime.now() - mtime).days
        logger.info("[INGEST] Existing file age=%d days (%s)", age_days, os.path.basename(path))
        return age_days <= self.max_age_days

    # ------------------------------------------------------------------
    # Yahoo fetch with HARD timeout
    # ------------------------------------------------------------------
    def _fetch_with_timeout(self, ticker: str):
        """Fetch Yahoo Finance data in a child process with hard timeout."""
        q: Queue = Queue()
        p = Process(target=_yahoo_download_worker, args=(ticker, self.start_date, self.end_date, q), daemon=True)

        start_ts = time.time()
        p.start()
        p.join(timeout=self.timeout_sec)

        if p.is_alive():
            logger.error("[INGEST] TIMEOUT ticker=%s after %ss", ticker, self.timeout_sec)
            p.terminate()
            p.join()
            return None

        if q.empty():
            logger.error("[INGEST] EMPTY response ticker=%s", ticker)
            return None

        result = q.get()
        if isinstance(result, Exception):
            logger.exception("[INGEST] EXCEPTION ticker=%s", ticker, exc_info=result)
            return None

        logger.info("[INGEST] Yahoo OK ticker=%s rows=%d elapsed=%.2fs", ticker, len(result), time.time() - start_ts)
        return result

    # ------------------------------------------------------------------
    # Single-ticker ingestion
    # ------------------------------------------------------------------
    def _process_single_ticker(self, ticker: str) -> None:
        """Ingest a single ticker with retry & backoff."""
        save_path = self._raw_file_path(ticker)

        if self._is_fresh(save_path):
            logger.info("[INGEST] Skipping %s — fresh cache", ticker)
            return

        for attempt in range(1, self.max_retries + 1):
            logger.info("[INGEST] %s attempt %d/%d", ticker, attempt, self.max_retries)

            df = self._fetch_with_timeout(ticker)

            if df is not None and not df.empty:
                df.to_csv(save_path)
                logger.info("[INGEST] Saved %s rows=%d path=%s", ticker, len(df), save_path)
                return

            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.warning("[INGEST] %s retrying after %.1fs", ticker, backoff)
                time.sleep(backoff)

        raise RuntimeError(f"Failed to ingest {ticker} after retries")

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run ingestion for all configured tickers with freshness enforcement."""
        stage_name = "ingestion_pipeline"
        self.monitor.log_stage_start(stage_name, {"num_tickers": len(self.tickers)})

        logger.info("Starting ingestion loop")
        for idx, ticker in enumerate(self.tickers, start=1):
            logger.info("[INGEST] (%d/%d) Processing ticker=%s", idx, len(self.tickers), ticker)
            try:
                self._process_single_ticker(ticker)
            except Exception as exc:
                logger.error("[INGEST] FAILED ticker=%s error=%s", ticker, str(exc), exc_info=True)

            time.sleep(self.sleep_time)

        self.monitor.log_stage_end(stage_name, {"status": "completed"})
        logger.info("Ingestion pipeline completed")
