# src/pipeline/D_optimization_pipeline.py
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
    Pooled-equity hyperparameter optimization pipeline.

    Performs a single Optuna study over pooled feature datasets
    derived from a single ingestion run.

    IMPORTANT:
    - Pools data IN-MEMORY for optimization
    - Persists pooled dataset + best params for downstream pipelines
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
        self.config = load_config(config_path)

        self.logger = get_logger(self.__class__.__name__)

        self.optim_cfg: Dict = self.config.get("optimization", {})
        self.n_trials: int = int(self.optim_cfg.get("n_trials", 10))

        self.used_tickers_path = (
            Path("datalake")
            / "runs"
            / run_id
            / "ingestion"
            / "used_tickers.json"
        )

        self.optim_dir = Path(f"datalake/runs/{run_id}/optimization")
        self.dataset_dir = Path(f"datalake/runs/{run_id}/dataset")

        self.monitor = TrainingMonitor(
            run_id=run_id,
            save_dir=self.optim_dir,
            artifact_policy="none",
        )

        self.logger.info(
            "OptimizationPipeline initialized | run_id=%s | n_trials=%d",
            run_id,
            self.n_trials,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _load_used_tickers(self) -> List[str]:
        if not self.used_tickers_path.exists():
            raise FileNotFoundError("used_tickers.json not found")

        with self.used_tickers_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        tickers = payload.get("tickers")
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("Invalid ticker universe")

        return tickers

    def _load_features(self, ticker: str) -> pd.DataFrame:
        path = self.feature_root / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing features for {ticker}")

        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Empty feature set for {ticker}")

        return df

    def _split_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if "target" not in df.columns:
            raise ValueError("Missing target column")

        drop_cols = {"target", "Date"}
        X = (
            df.drop(columns=[c for c in drop_cols if c in df.columns])
            .select_dtypes(include=["number", "bool"])
            .astype("float32")
            .values
        )
        y = df["target"].astype("float32").values

        return X, y

    def _build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(lookback - 1, len(X)):
            X_seq.append(X[i - lookback + 1 : i + 1])
            y_seq.append(y[i])

        return (
            np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        stage = "optimization_pipeline"
        self.monitor.log_stage_start(
            stage, {"run_id": self.run_id, "n_trials": self.n_trials}
        )

        tickers = self._load_used_tickers()
        model_cfg = self.optim_cfg.get("model", {})
        model_type = model_cfg.get("type", "").lower()
        lookback = int(model_cfg.get("lookback", 1))

        # --------------------------------------------------
        # Deterministic equity_id mapping (artifact)
        # --------------------------------------------------
        equity_id_map = {t: i for i, t in enumerate(tickers)}
        self.optim_dir.mkdir(parents=True, exist_ok=True)

        map_path = self.optim_dir / "equity_id_map.json"
        map_path.write_text(json.dumps(equity_id_map, indent=2))
        self.logger.info(
            "[OPT] Saved equity_id_map (%d tickers) -> %s",
            len(tickers),
            map_path,
        )

        # --------------------------------------------------
        # Pool features IN-MEMORY
        # --------------------------------------------------
        X_all, y_all = [], []

        self.logger.info("[OPT] Pooling %d equities", len(tickers))
        start_time = pd.Timestamp.now()

        for ticker in tickers:
            df = self._load_features(ticker)
            X, y = self._split_xy(df)

            if model_type == "lstm":
                X, y = self._build_sequences(X, y, lookback)

            if len(X) == 0:
                continue

            X_all.append(X)
            y_all.append(y)

        if not X_all:
            raise RuntimeError("No valid training data after pooling")

        X_train = np.concatenate(X_all, axis=0)
        y_train = np.concatenate(y_all, axis=0)

        self.logger.info(
            "[OPT] Final pooled tensors | X=%s y=%s",
            X_train.shape,
            y_train.shape,
        )

        # --------------------------------------------------
        # Persist pooled dataset (NEW — downstream critical)
        # --------------------------------------------------
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        X_df = pd.DataFrame(X_train.reshape(len(X_train), -1))
        y_df = pd.DataFrame({"target": y_train})

        X_path = self.dataset_dir / "X_pooled.parquet"
        y_path = self.dataset_dir / "y_pooled.parquet"

        X_df.to_parquet(X_path)
        y_df.to_parquet(y_path)

        self.logger.info(
            "[OPT] Persisted pooled dataset -> %s , %s",
            X_path,
            y_path,
        )

        # --------------------------------------------------
        # Hyperparameter optimization (UNCHANGED)
        # --------------------------------------------------
        optimizer = OptunaOptimizer(
            config=self.optim_cfg,
            monitor=self.monitor,
            n_trials=self.n_trials,
        )

        self.monitor.log_stage_start(
            "hyperparameter_search",
            {"n_trials": self.n_trials, "num_equities": len(tickers)},
        )

        result = optimizer.run(X_train=X_train, y_train=y_train)

        # --------------------------------------------------
        # Persist best params for E pipeline
        # --------------------------------------------------
        best_params_path = self.optim_dir / "best_params.json"
        best_params_path.write_text(
            json.dumps(result.get("best_params", {}), indent=2)
        )

        summary_path = self.optim_dir / "study_summary.json"
        summary_path.write_text(json.dumps(result, indent=2))

        self.monitor.log_stage_end(
            "hyperparameter_search",
            {"status": "success", **result},
        )

        self.monitor.log_stage_end(stage, {"status": "completed"})

        self.logger.info(
            "Optimization pipeline completed | run_id=%s | total_time=%s",
            self.run_id,
            pd.Timestamp.now() - start_time,
        )
