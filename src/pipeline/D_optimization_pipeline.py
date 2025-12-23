"""
D. Optimization Pipeline
------------------------
Consumes feature Parquet files generated in Pipeline C and performs
hyperparameter optimization (e.g., Optuna) in a run-aware, reproducible manner.

Responsibilities:
- Load feature datasets for the current ingestion run
- Invoke optimizer backend (Optuna, etc.)
- Log trials, parameters, and metrics

Non-responsibilities:
- Feature generation
- Model training
- Walk-forward validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.monitoring.monitor import TrainingMonitor
from src.optimizers import get_optimizer


class OptimizationPipeline:
    """
    Hyperparameter optimization pipeline.
    """

    def __init__(
        self,
        run_id: str,
        config_path: str,
        feature_root: str = "datalake/data/cache/features",
        optimizer_name: str = "optuna",
    ) -> None:
        self.run_id = run_id
        self.config_path = config_path
        self.feature_root = Path(feature_root)
        self.optimizer_name = optimizer_name

        self.logger = get_logger(self.__class__.__name__)
        self.config: Dict = load_config(config_path)

        self.optim_cfg: Dict = self.config.get("optimization", {})
        self.train_cfg: Dict = self.config.get("training", {})

        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=f"datalake/runs/{run_id}/optimization",
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_feature_files(self) -> List[Path]:
        """
        Load feature Parquet files produced by Pipeline C for this run.
        """
        if not self.feature_root.exists():
            raise FileNotFoundError(
                f"Feature root not found: {self.feature_root}"
            )

        files = sorted(self.feature_root.glob("*.parquet"))

        if not files:
            raise RuntimeError(
                "No feature Parquet files found. "
                "Ensure Pipeline C has completed successfully."
            )

        self.logger.info(
            "[OPT] Found %d feature files", len(files)
        )
        return files

    def _load_features(self, path: Path) -> pd.DataFrame:
        """
        Load a single Parquet feature file.
        """
        df = pd.read_parquet(path)

        if df.empty:
            raise ValueError(f"Empty feature file: {path.name}")

        return df

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute hyperparameter optimization.
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

        feature_files = self._load_feature_files()
        optimizer_fn = get_optimizer(self.optimizer_name)

        n_trials = int(self.optim_cfg.get("n_trials", 50))

        for idx, feature_path in enumerate(feature_files, start=1):
            ticker = feature_path.stem

            self.logger.info(
                "[OPT] (%d/%d) Optimizing ticker=%s",
                idx,
                len(feature_files),
                ticker,
            )

            try:
                df = self._load_features(feature_path)

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
                    "[OPT] Failed optimization for %s: %s",
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
