# src/pipeline/B_preprocessing_pipeline.py
"""
B. Preprocessing Pipeline
-------------------------
Consumes ingestion artifacts, loads only the raw tickers ingested
in the current run, performs lightweight cleaning, and writes
cleaned data to cache/clean for downstream feature generation.

Responsibilities:
- Row cleaning (dropna)
- Column filtering (required_columns)
- Deterministic artifact-based processing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor


class PreprocessingPipeline:
    """
    Preprocessing pipeline operating strictly on ingestion artifacts.
    """

    def __init__(
        self,
        run_id: str,
        required_columns: List[str] | None = None,
        raw_root: str = "datalake/data/raw",
        clean_root: str = "datalake/data/cache/clean",
    ) -> None:
        """
        Parameters
        ----------
        run_id : str
            Ingestion run_id whose artifacts should be processed.
        required_columns : Optional[List[str]]
            Columns to retain after cleaning. Defaults to 7 canonical columns.
        raw_root : str
            Root directory containing raw CSVs.
        clean_root : str
            Output directory for cleaned CSVs.
        """
        self.run_id = run_id
        self.required_columns = required_columns or [
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"
        ]
        self.raw_root = Path(raw_root)
        self.clean_root = Path(clean_root)
        self.clean_root.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(self.__class__.__name__)
        self.monitor = TrainingMonitor(run_id=run_id, save_dir=Path(f"datalake/runs/{run_id}/preprocessing"), artifact_policy="none",)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_ingestion_artifacts(self) -> list[dict]:
        """
        Load ingestion artifacts.jsonl for the given run_id.
        Returns only tickers ingested in the current run.
        """
        artifacts_path = Path("datalake") / "runs" / self.run_id / "ingestion"  / "artifacts.jsonl"
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Ingestion artifacts not found for run_id={self.run_id}")

        artifacts: list[dict] = []
        with artifacts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                artifacts.append(json.loads(line))
        return artifacts

    def _drop_missing(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Drop rows with missing values.
        """
        before = df.shape
        df = df.dropna()
        after = df.shape
        self.logger.info(f"[PRE] {ticker} dropna {before} -> {after}")
        return df

    def _select_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Retain only required columns if specified.
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            self.logger.warning(f"[PRE] {ticker} missing columns: {missing}")

        available = [c for c in self.required_columns if c in df.columns]
        df = df[available]
        self.logger.info(f"[PRE] {ticker} selected columns: {available}")
        return df

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute preprocessing for all tickers present in ingestion artifacts.
        """
        self.monitor.log_stage_start("preprocessing_pipeline")
        self.logger.info("Starting preprocessing pipeline")

        artifacts = self._load_ingestion_artifacts()

        for idx, artifact in enumerate(artifacts, start=1):
            ticker = artifact["ticker"]
            raw_path = Path(artifact["raw_path"])
            self.logger.info(f"[PRE] ({idx}/{len(artifacts)}) Processing {ticker}")

            if not raw_path.exists():
                self.logger.error(f"[PRE] Raw file missing for {ticker}: {raw_path}")
                continue

            try:
                df = pd.read_csv(raw_path)
                df = self._drop_missing(df, ticker)
                df = self._select_columns(df, ticker)

                out_path = self.clean_root / f"{ticker}.csv"
                df.to_csv(out_path, index=False)
                self.logger.info(f"[PRE] Saved cleaned data {ticker} rows={len(df)} path={out_path}")

                self.monitor.log_artifact(
                    stage="preprocessing",
                    artifact_type="clean_csv",
                    ticker=ticker,
                    path=str(out_path),
                    rows=len(df),
                )

            except Exception as exc:
                self.logger.error(f"[PRE] Failed preprocessing {ticker}: {exc}", exc_info=True)

        self.monitor.log_stage_end("preprocessing_pipeline")
        self.logger.info("Preprocessing pipeline completed")
