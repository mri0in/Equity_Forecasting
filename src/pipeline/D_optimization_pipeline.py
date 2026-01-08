# src/pipeline/D_optimization_pipeline.py
"""
D. Optimization Pipeline
------------------------

Performs hyperparameter optimization over feature datasets produced
by the feature generation pipeline, scoped strictly to the ticker
universe defined by a single ingestion run.

This pipeline is responsible only for *searching* optimal model
hyperparameters and does not train, validate, or persist final models.

Contract
--------
- Runtime input: `run_id`
- Ticker universe resolved from:
      datalake/runs/{run_id}/ingestion/used_tickers.json
- Feature inputs loaded from:
      datalake/data/cache/features/{ticker}.parquet
- Feature datasets must already contain a valid `target` column
  (guaranteed upstream by Pipeline C)
- For LSTM, `model.lookback` defines the sequence length used to
  reshape X/y prior to training or optimization.

Responsibilities
----------------
- Resolve deterministic ticker universe for the run
- Load per-ticker feature datasets
- Split features (X) and target (y)
- Execute hyperparameter search using a pluggable optimizer
- Log optimization metrics and outcomes

Non-responsibilities
--------------------
- Feature generation
- Target engineering
- Model training
- Walk-forward validation
- Model persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.monitoring.monitor import TrainingMonitor
from src.optimizers.optuna_optimizer import OptunaOptimizer
from src.utils.config import load_config
from src.utils.logger import get_logger


class OptimizationPipeline:
    """
    Hyperparameter optimization pipeline operating on feature artifacts
    derived from a single ingestion run.
    """

    def __init__(
        self,
        run_id: str,
        config_path: str,
        feature_root: str = "datalake/data/cache/features",
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be provided")

        self.run_id = run_id
        self.feature_root = Path(feature_root)
        self.config_path = config_path

        self.logger = get_logger(self.__class__.__name__)
        self.config: Dict = load_config(config_path)

        self.optim_cfg: Dict = self.config.get("optimization", {})
        self.n_trials: int = int(self.optim_cfg.get("n_trials", 50))

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

        self.logger.info(
            "OptimizationPipeline initialized | run_id=%s | n_trials=%d",
            run_id,
            self.n_trials,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_used_tickers(self) -> List[str]:
        """Load the deterministic ticker universe for this run."""
        if not self.used_tickers_path.exists():
            raise FileNotFoundError(
                f"used_tickers.json not found for run_id={self.run_id}"
            )

        with self.used_tickers_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        tickers = payload["tickers"]

        if not isinstance(tickers, list) or not tickers:
            raise ValueError("Invalid or empty ticker list in used_tickers.json")

        self.logger.info("[OPT] Loaded %d tickers from ingestion run", len(tickers))
        return tickers

    def _load_features(self, ticker: str) -> pd.DataFrame:
        """Load feature dataset for a single ticker."""
        feature_path = self.feature_root / f"{ticker}.parquet"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file missing for ticker={ticker}: {feature_path}"
            )

        df = pd.read_parquet(feature_path)

        if df.empty:
            raise ValueError(f"Empty feature dataset for ticker={ticker}")

        return df

    def _split_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split feature matrix X and target vector y.

        Assumes upstream pipeline guarantees presence and correctness
        of the `target` column.
        """
        if "target" not in df.columns:
            raise ValueError("Feature dataset must contain 'target' column")

        drop_cols = {"target", "Date"}
        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X_df = X_df.select_dtypes(include=["number", "bool"])
        X = X_df.astype("float32").values
        y = df["target"].astype("float32").values

        if X.size == 0 or y.size == 0:
            raise ValueError("Invalid feature/target split")

        return X, y

def _build_sequences(
    self,
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat features into rolling sequences for sequence models.

    Parameters
    ----------
    X : np.ndarray
        Flat feature matrix (num_samples, num_features)
    y : np.ndarray
        Target vector (num_samples,)
    lookback : int
        Number of past timesteps to include in each sequence

    Returns
    -------
    X_seq : np.ndarray, shape (num_samples-lookback+1, lookback, num_features)
    y_seq : np.ndarray, shape (num_samples-lookback+1,)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    X_seq, y_seq = [], []
    for i in range(lookback - 1, len(X)):
        X_seq.append(X[i - lookback + 1 : i + 1])
        y_seq.append(y[i])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute hyperparameter optimization for all tickers
        defined in the ingestion run.
        """
        stage_name = "optimization_pipeline"
        self.monitor.log_stage_start(stage_name, {"run_id": self.run_id, "n_trials": self.n_trials})
        self.logger.info("Starting optimization pipeline")

        tickers = self._load_used_tickers()

        for idx, ticker in enumerate(tickers, start=1):
            self.logger.info("[OPT] (%d/%d) Optimizing ticker=%s", idx, len(tickers), ticker)
            try:
                df = self._load_features(ticker)
                X_train, y_train = self._split_xy(df)

                model_type = self.config.get("model", {}).get("type", "").lower()

                if model_type == "lstm":
                    lookback = int(self.config.get("model", {}).get("lookback", 1))

                    # reshape for LSTM sequences
                    X_train, y_train = self._build_sequences(X_train, y_train, lookback)

                    self.logger.info(
                        "[OPT] Reshaped data for LSTM | X=%s y=%s lookback=%d",
                        X_train.shape,
                        y_train.shape,
                        lookback,
                    )

                    if X_train.ndim != 3:
                        raise ValueError(f"LSTM requires 3D input, got shape {X_train.shape}")

                self.logger.info("[OPT] Final training tensor shapes | X=%s y=%s", X_train.shape, y_train.shape)

                optimizer = OptunaOptimizer(config=self.config, monitor=self.monitor, n_trials=self.n_trials)
                self.monitor.log_stage_start(
                    "hyperparameter_search",
                    {"ticker": ticker, "n_trials": self.n_trials},
                )

                result = optimizer.run(X_train=X_train, y_train=y_train)

                self.monitor.log_stage_end(
                    "hyperparameter_search",
                    {"ticker": ticker, "status": "success", **result},
                )

            except Exception as exc:
                self.logger.error("[OPT] Optimization failed for %s: %s", ticker, str(exc), exc_info=True)
                self.monitor.log_stage_end(
                    "hyperparameter_search",
                    {"ticker": ticker, "status": "failed", "error": str(exc)},
                )

        self.monitor.log_stage_end(stage_name, {"status": "completed"})
        self.logger.info("Optimization pipeline completed | run_id=%s", self.run_id)
