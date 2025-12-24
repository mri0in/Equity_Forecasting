# src/pipeline/D_optimization_pipeline.py
"""
D. Optimization Pipeline
-----------------------
Consumes feature datasets generated in Pipeline C and performs
hyperparameter optimization in a run-aware, deterministic manner.

This pipeline is strictly scoped to *searching* optimal hyperparameters
for downstream training pipelines and does not train or persist final models.

Responsibilities:
- Resolve tickers used in the ingestion run
- Load per-ticker feature Parquet files
- Execute hyperparameter optimization (e.g., Optuna)
- Log trial parameters, metrics, and outcomes

Non-responsibilities:
- Feature generation
- Dataset consolidation
- Walk-forward validation
- Model training or persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.monitoring.monitor import TrainingMonitor
from src.optimizers import get_optimizer
from src.utils.config import load_config
from src.utils.logger import get_logger


class OptimizationPipeline:
    """
    Hyperparameter optimization pipeline operating strictly
    on feature artifacts derived from an ingestion run.
    """

    def __init__(
        self,
        run_id: str,
        config_path: str,
        feature_root: str = "datalake/data/cache/features",
    ) -> None:
        """
        Parameters
        ----------
        run_id : str
            Ingestion run identifier whose artifacts define the ticker universe.
        config_path : str
            Path to YAML configuration containing optimization settings.
        feature_root : str
            Directory containing per-ticker feature Parquet files.
        """
        self.run_id = run_id
        self.feature_root = Path(feature_root)
        self.config_path = config_path

        self.logger = get_logger(self.__class__.__name__)
        self.config: Dict = load_config(config_path)

        self.optim_cfg: Dict = self.config.get("optimization", {})
        self.optimizer_name: str = self.optim_cfg.get("backend", "optuna")

        self.used_tickers_path = (
            Path("datalake")
            / "runs"
            / run_id
            / "ingestion"
            / "used_tickers.json"
        )

        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=Path(f"datalake/runs/{run_id}/optimization"),
            artifact_policy="none",
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
            Deterministic list of tickers to optimize.
        """
        if not self.used_tickers_path.exists():
            raise FileNotFoundError(
                f"used_tickers.json not found for run_id={self.run_id}"
            )

        with self.used_tickers_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        tickers = payload["tickers"]

        self.logger.info(
            "[OPT] Loaded %d tickers from ingestion run",
            len(tickers),
        )
        return tickers

    def _load_features(self, ticker: str) -> pd.DataFrame:
        """
        Load feature dataset for a single ticker.
        """
        feature_path = self.feature_root / f"{ticker}.parquet"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file missing for ticker={ticker}: {feature_path}"
            )

        df = pd.read_parquet(feature_path)

        if df.empty:
            raise ValueError(
                f"Empty feature dataset for ticker={ticker}"
            )

        return df

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute hyperparameter optimization for all tickers
        present in the ingestion run.
        """
        stage = "optimization_pipeline"

        self.monitor.log_stage_start(
            stage,
            {
                "optimizer": self.optimizer_name,
                "run_id": self.run_id,
            },
        )

        self.logger.info("Starting optimization pipeline")

        tickers = self._load_used_tickers()
        optimizer_fn = get_optimizer(self.optimizer_name)
        n_trials = int(self.optim_cfg.get("n_trials", 50))

        for idx, ticker in enumerate(tickers, start=1):
            self.logger.info(
                "[OPT] (%d/%d) Optimizing ticker=%s",
                idx,
                len(tickers),
                ticker,
            )

            try:
                df = self._load_features(ticker)

                self.monitor.log_stage_start(
                    "hyperparameter_search",
                    {
                        "ticker": ticker,
                        "n_trials": n_trials,
                    },
                )

                optimizer_fn(
                    config=self.config,
                    df=df,
                    ticker=ticker,
                    n_trials=n_trials,
                    monitor=self.monitor,
                )

                self.monitor.log_stage_end(
                    "hyperparameter_search",
                    {
                        "ticker": ticker,
                        "status": "success",
                    },
                )

            except Exception as exc:
                self.logger.error(
                    "[OPT] Optimization failed for %s: %s",
                    ticker,
                    str(exc),
                    exc_info=True,
                )

                self.monitor.log_stage_end(
                    "hyperparameter_search",
                    {
                        "ticker": ticker,
                        "status": "failed",
                        "error": str(exc),
                    },
                )

        self.monitor.log_stage_end(stage, {"status": "completed"})
        self.logger.info("Optimization pipeline completed")
