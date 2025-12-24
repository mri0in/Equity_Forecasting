# src/pipeline/C_feature_pipeline.py
"""
C. Feature Generation Pipeline
------------------------------

This pipeline generates technical features from cleaned equity time-series
data produced by the preprocessing pipeline (B).

Contract
--------
- C accepts a single runtime input: `run_id`
- From `run_id`, C resolves the canonical ingestion artifact:
      datalake/runs/{run_id}/ingestion/used_tickers.txt
- Only tickers listed in this file are processed
- Cleaned inputs are read from datalake/data/cache/clean/
- Outputs are written as per-ticker Parquet files to
  datalake/data/cache/features/

Responsibilities
---------------
- Resolve the exact ticker universe used in the ingestion run
- Load cleaned OHLCV CSVs
- Generate technical indicators only (no labels, no joins)
- Cache per-ticker's OHLCV data enriched with features in Parquet format

"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.features.technical.build_features import FeatureBuilder
from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger
import json as jsonlib


class FeaturePipeline:
    """
    Feature generation pipeline driven strictly by ingestion run artifacts.
    """

    def __init__(
        self,
        run_id: str,
        clean_root: str = "datalake/data/cache/clean",
        feature_root: str = "datalake/data/cache/features",
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id
        self.clean_root = Path(clean_root)
        self.feature_root = Path(feature_root)

        self.feature_root.mkdir(parents=True, exist_ok=True)

        # Canonical ingestion artifact
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
            "FeaturePipeline initialized | run_id=%s | used_tickers_path=%s",
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
            Tickers eligible for feature generation.

        Raises
        ------
        FileNotFoundError
            If used_tickers.txt is missing.
        """
        if not self.used_tickers_path.exists():
            raise FileNotFoundError(
                f"used_tickers.txt not found for run_id={self.run_id}"
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

        self.logger.info(
            "[FEAT] Loaded %d tickers from ingestion run",
            len(tickers),
        )
        return tickers

    def _is_up_to_date(self, clean_path: Path, feature_path: Path) -> bool:
        """
        Check whether feature file is newer than its cleaned input.
        """
        if not feature_path.exists():
            return False

        return feature_path.stat().st_mtime >= clean_path.stat().st_mtime

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute feature generation for all tickers ingested in the given run.

        Returns
        -------
        str
            The run_id associated with this feature generation run.
        """
        stage_name = "feature_pipeline"

        used_tickers = self._load_used_tickers()

        self.monitor.log_stage_start(stage_name,{"num_tickers": len(used_tickers)},)

        self.logger.info(
            "Starting feature generation | run_id=%s | tickers=%d",
            self.run_id,
            len(used_tickers),
        )

        for idx, ticker in enumerate(used_tickers, start=1):
            clean_path = self.clean_root / f"{ticker}.csv"
            feature_path = self.feature_root / f"{ticker}.parquet"

            self.logger.info("[FEAT] (%d/%d) Processing %s", idx, len(used_tickers), ticker,)

            if not clean_path.exists():
                self.logger.error(
                    "[FEAT] Cleaned file missing for %s: %s",
                    ticker,
                    clean_path,
                )
                continue

            if self._is_up_to_date(clean_path, feature_path):
                self.logger.info(
                    "[FEAT] Skipping %s (features up-to-date)",
                    ticker,
                )
                continue

            try:
                df_clean = pd.read_csv(clean_path)

                builder = FeatureBuilder(equity=ticker)
                df_features = builder.build_all(df_clean)

                df_features.to_parquet(feature_path, index=False, engine="pyarrow")

                self.logger.info(
                    "[FEAT] Saved features %s rows=%d path=%s",
                    ticker,
                    len(df_features),
                    feature_path,
                )

            except Exception as exc:
                self.logger.error(
                    "[FEAT] Feature generation failed for %s error=%s",
                    ticker,
                    str(exc),
                    exc_info=True,
                )

        self.monitor.log_stage_end(stage_name, {"status": "completed"},)

        self.logger.info("Feature generation pipeline completed | run_id=%s", self.run_id,)
