# src/pipeline/B_preprocessing_pipeline.py
"""
B. Preprocessing Pipeline
-------------------------

This pipeline performs deterministic, lightweight preprocessing on raw
market data produced by the ingestion pipeline (A).

Contract
--------
- B accepts a single runtime input: `run_id`
- From `run_id`, B resolves the canonical ingestion artifact:
      datalake/runs/{run_id}/ingestion/used_tickers.txt
- Only tickers listed in this file are processed
- No tickers are inferred from config, filesystem scans, or user input

Responsibilities
---------------
- Load the exact set of tickers used in ingestion for the given run
- Read raw OHLCV CSVs from datalake/data/raw/
- Apply basic cleaning:
    - Drop rows with missing values
    - Retain only required columns
- Write cleaned CSVs to datalake/data/cache/clean/
- Emit monitoring signals for observability

"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import json as jsonlib

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor


class PreprocessingPipeline:
    """
    Deterministic preprocessing pipeline driven by ingestion run artifacts.
    """

    def __init__(
        self,
        run_id: str,
        required_columns: List[str] | None = None,
        raw_root: str = "datalake/data/raw",
        clean_root: str = "datalake/data/cache/clean",
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id
        self.required_columns = required_columns or [
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"
        ]

        self.raw_root = Path(raw_root)
        self.clean_root = Path(clean_root)
        self.clean_root.mkdir(parents=True, exist_ok=True)

        # Resolve ingestion artifact path deterministically
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
            save_dir=Path(f"datalake/runs/{run_id}/preprocessing"),
            artifact_policy="none",
        )

        self.logger.info(
            "PreprocessingPipeline initialized | run_id=%s | used_tickers_path=%s",
            run_id,
            self.used_tickers_path,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_used_tickers(self) -> List[str]:
        """
        Load tickers used during ingestion for this run.

        Returns
        -------
        List[str]
            List of tickers to preprocess.

        Raises
        ------
        FileNotFoundError
            If the ingestion artifact is missing.
        ValueError
            If the artifact schema is invalid.
        """
        if not self.used_tickers_path.exists():
            raise FileNotFoundError(
                f"used_tickers.json not found for run_id={self.run_id} "
                f"path={self.used_tickers_path}"
            )

        with self.used_tickers_path.open("r", encoding="utf-8") as fh:
            payload = jsonlib.load(fh)

        # -------- schema validation --------
        if not isinstance(payload, dict):
            raise ValueError("used_tickers.json must contain a JSON object")

        tickers = payload.get("tickers")

        if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
            raise ValueError(
                "Invalid used_tickers.json schema: 'tickers' must be List[str]"
            )

        if not tickers:
            self.logger.warning(
                "[PRE] No tickers found in ingestion run_id=%s", self.run_id
            )

        self.logger.info(
            "[PRE] Loaded %d tickers from ingestion run_id=%s",
            len(tickers),
            self.run_id,
        )

        return tickers

    def _drop_missing(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Drop rows containing missing values."""
        before = df.shape
        df = df.dropna()
        after = df.shape

        self.logger.info(
            "[PRE] %s dropna %s -> %s",
            ticker,
            before,
            after,
        )
        return df

    def _select_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Retain only required columns."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            self.logger.warning(
                "[PRE] %s missing columns: %s",
                ticker,
                missing,
            )

        available = [c for c in self.required_columns if c in df.columns]
        df = df[available]

        self.logger.info(
            "[PRE] %s retained columns: %s",
            ticker,
            available,
        )
        return df

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute preprocessing for all tickers ingested in the given run.

        Returns
        -------
        str
            The run_id associated with this preprocessing run.
        """
        stage_name = "preprocessing_pipeline"

        used_tickers = self._load_used_tickers()

        self.monitor.log_stage_start(stage_name,{"num_tickers": len(used_tickers)},)

        self.logger.info(
            "[B] Starting preprocessing pipeline | run_id=%s | tickers=%d", self.run_id,len(used_tickers), )

        for idx, ticker in enumerate(used_tickers, start=1):
            raw_path = self.raw_root /f"{ticker}.csv"
            self.logger.info(
                "[PRE] (%d/%d) Processing %s",
                idx,
                len(used_tickers),
                ticker,
            )

            if not raw_path.exists():
                self.logger.error(
                    "[PRE] Raw file missing for %s: %s",
                    ticker,
                    raw_path,
                )
                continue

            try:
                df = pd.read_csv(raw_path)
                df = self._drop_missing(df, ticker)
                df = self._select_columns(df, ticker)

                out_path = self.clean_root / f"{ticker.replace('.', '_')}.csv"
                df.to_csv(out_path, index=False)

                self.logger.info(
                    "[PRE] Saved cleaned data %s rows=%d path=%s",
                    ticker,
                    len(df),
                    out_path,
                )

            except Exception as exc:
                self.logger.error(
                    "[PRE] Failed preprocessing %s error=%s",
                    ticker,
                    str(exc),
                    exc_info=True,
                )

        self.monitor.log_stage_end(stage_name,{"status": "completed"},)

        self.logger.info("[B] Preprocessing pipeline completed | run_id=%s",self.run_id,)
        
