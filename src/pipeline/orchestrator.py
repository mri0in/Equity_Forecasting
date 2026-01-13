# src/pipeline/orchestrator.py
"""
orchestrator.py

Master orchestrator for the Equity Forecasting project.
Coordinates all pipeline stages: training, optimization, validation,
ensembling, and prediction. Integrates with dashboard actions such as
sentiment and forecasting triggers.
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any

import joblib
import pandas as pd

from src.pipeline import (
    D_optimization_pipeline,
    E_ensemble_pipeline,
    E_modeltrainer_pipeline,
    G_wfv_pipeline,
    H_prediction_pipeline,
)
from src.utils.config_loader import load_typed_config, FullConfig

from src.monitoring.monitor import TrainingMonitor
monitor = TrainingMonitor()

# -------------------------------
# Completion markers for each pipeline task
# -------------------------------
TASK_MARKERS = {
    "train": "datalake/models/trained/.train_complete",
    "optimize": "datalake/experiments/optuna/.optimize_complete",
    "ensemble": "datalake/ensemble/.ensemble_complete",
    "predict": "datalake/predictions/.predict_complete",
    "walkforward": "datalake/wfv/.walkforward_complete",
}


class PipelineOrchestrator:
    """
    Orchestrates the full equity forecasting pipeline based on config.

    Handles conditional feature building and cached feature loading for
    dashboard-triggered forecasting. Ensures idempotent runs via task markers.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: FullConfig = load_typed_config(config_path)
        self.pipeline_cfg = self.config.pipeline
        self.logger = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------
    # Feature Preparation Stage (integrated for forecasting)
    # -----------------------------------------------------
    def prepare_features(self, ticker: str) -> str:
        """
        Ensure feature data is available for the given ticker.
        Builds and caches features only if not already cached.

        Args:
            ticker (str): Equity symbol to prepare data for.

        Returns:
            str: Path to the cached feature file (joblib).
        """
        # Prefer the canonical joblib features cache under datalake/cache/features/
        cache_dir = "datalake/cache/features"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{ticker}_features.joblib")

        if os.path.exists(cache_path):
            self.logger.info(f"Using cached features for {ticker}: {cache_path}")
            return cache_path

        # Fall back to older location if needed (datalake/features/)
        fallback_path = os.path.join("datalake", "features", f"{ticker}_features.joblib")
        if os.path.exists(fallback_path):
            self.logger.info(f"Using cached features (fallback) for {ticker}: {fallback_path}")
            return fallback_path

        # If we reach here we need to build features.
        self.logger.info(f"No cached features found for {ticker}. Attempting to build features...")
        monitor.log_event("Preparing features", {"equity": ticker})


        # Try to use the project's FeatureBuilder if available.
        # Since different FeatureBuilder implementations exist in repo variants,
        # we call the pipeline-level predict task which, by project design, will
        # ensure features are built when predict runs with a ticker context.
        # However to be explicit we attempt to call a known builder if importable.
        try:
            # Lazy import to avoid import-time dependency if module not present
            from src.features.technical.build_features import FeatureBuilder  # type: ignore

            # Some FeatureBuilder variants expect an equity string and internal cache handling,
            # others expose build_all(df). We attempt a conservative call pattern:
            builder = FeatureBuilder(ticker)  # try constructor with ticker
            # If builder has an explicit run() or build() method, prefer it
            if hasattr(builder, "run"):
                builder.run()
            elif hasattr(builder, "build_all"):
                # builder.build_all expects a DataFrame; try to fetch raw data via data loader if present
                try:
                    from src.data_processor.load_api_data import load_api_for_ticker  # type: ignore
                    raw_df = load_api_for_ticker(ticker)
                    builder.build_all(raw_df)
                except Exception:
                    # best-effort: call build_all with no args (some implementations may handle internal loading)
                    try:
                        builder.build_all()
                    except Exception:
                        self.logger.warning("FeatureBuilder.build_all required a DataFrame and loader failed.")
            else:
                self.logger.warning("FeatureBuilder found but has no recognized build method. Falling back to pipeline predict trigger.")
        except Exception as e:
            self.logger.debug(f"FeatureBuilder not used or failed to import: {e}. Will rely on pipeline predict task to build features.")

        # After attempting to build, check cache again
        if os.path.exists(cache_path):
            self.logger.info(f"Features successfully built and cached for {ticker}: {cache_path}")
            return cache_path

        if os.path.exists(fallback_path):
            self.logger.info(f"Features built and cached at fallback location for {ticker}: {fallback_path}")
            return fallback_path

        # As a last resort, return a non-existing path (caller should handle absence)
        self.logger.warning(f"Could not build or find cached features for {ticker}. Returning expected cache path: {cache_path}")
        return cache_path

    # -----------------------------------------------------
    # Core Pipeline Task Runner
    # -----------------------------------------------------

    def run_task(self, task: str, retries: int = 1, strict: bool = True) -> None:
        marker = TASK_MARKERS.get(task)

        # Already completed? No need to monitor.
        if marker and os.path.exists(marker):
            self.logger.info(f"Skipping task '{task}' — completion marker exists: {marker}")
            return

        attempt = 0
        while attempt < retries:
            try:
                self.logger.info(f"Running task: {task} (attempt {attempt + 1}/{retries})")

                # MONITORING: task start
                monitor.log_stage_start(task, {"attempt": attempt + 1})

                if task == "train":
                    E_modeltrainer_pipeline(self.config_path)
                elif task == "optimize":
                    D_optimization_pipeline(self.config_path)
                elif task == "ensemble":
                    E_ensemble_pipeline(self.config_path)
                elif task == "predict":
                    H_prediction_pipeline(self.config_path)
                elif task == "walkforward":
                    G_wfv_pipeline(self.config_path)
                else:
                    self.logger.warning(f"Unknown task '{task}' — skipping")

                # MONITORING: task end
                monitor.log_stage_end(task, {"status": "success"})
                return

            except Exception as e:
                attempt += 1
                self.logger.error(f"Task '{task}' failed on attempt {attempt}/{retries}: {e}")

                # MONITORING: task error
                monitor.error(task, str(e), {"attempt": attempt})

                if attempt >= retries:
                    if strict:
                        self.logger.critical(f"Task '{task}' failed after {retries} attempts. Aborting.")

                        # MONITORING: final fail end
                        monitor.log_stage_end(task, {"status": "failed"})
                        raise
                    else:
                        self.logger.warning(f"Skipping failed task '{task}' (strict={strict})")

                        # MONITORING: soft fail but still end
                        monitor.log_stage_end(task, {"status": "skipped"})
                        return

    # -----------------------------------------------------
    # Forecast wrapper used by API / Dashboard
    # -----------------------------------------------------
    def _candidate_prediction_paths(self, equity: str) -> List[str]:
        """
        Canonical list of joblib prediction artifact paths (preferred).
        Kept small and joblib-first per repo decision.
        """
        paths = [
            os.path.join("datalake", "predictions", f"{equity}_forecast.joblib"),
            os.path.join("datalake", "predictions", f"{equity}.joblib"),
            os.path.join("datalake", "cache", "forecasting", f"{equity}_forecast.joblib"),
            os.path.join("datalake", "cache", "forecasting", f"{equity}.joblib"),
        ]
        # keep pickle options as last-resort (not preferred)
        paths += [
            os.path.join("datalake", "cache", "forecasting", f"{equity}_forecast.pkl"),
            os.path.join("datalake", "predictions", f"{equity}_forecast.pkl"),
        ]
        return paths

    def _load_prediction_artifact(self, equity: str) -> Optional[Dict[str, Any]]:
        """
        Try to find and load a prediction artifact for given equity.
        Returns normalized dict or None.
        """
        for p in self._candidate_prediction_paths(equity):
            if not os.path.exists(p):
                continue
            try:
                self.logger.info(f"Loading prediction artifact for {equity} from {p}")
                obj = joblib.load(p)
                # Normalize common shapes
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, pd.DataFrame):
                    # Attempt best-effort mapping
                    d: Dict[str, Any] = {"equity": equity, "metadata": {"source": p}}
                    cols = [c.lower() for c in obj.columns]
                    if "date" in cols or "dates" in cols:
                        # pick first matching column name
                        for candidate in ("date", "dates"):
                            if candidate in obj.columns:
                                d["hist_dates"] = obj[candidate].astype(str).tolist()
                                break
                    if "hist_price" in cols or "hist_prices" in cols or "close" in cols:
                        for candidate in ("hist_price", "hist_prices", "close"):
                            if candidate in obj.columns:
                                d["hist_prices"] = obj[candidate].tolist()
                                break
                    for key in ("forecast_mean", "forecast_upper", "forecast_lower"):
                        if key in obj.columns:
                            d[key] = obj[key].tolist()
                    return d
                # numpy / list / other
                return {"equity": equity, "prediction": obj, "metadata": {"source": p}}
            except Exception as e:
                self.logger.error(f"Failed to load prediction artifact {p} for {equity}: {e}")
                continue
        return None

    def run_forecast_pipeline(
        self,
        equity: str,
        horizon: int = 7,
        force_rebuild_features: bool = False,
        allow_training: bool = False,
        retries: int = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Modular forecast entrypoint used by the dashboard / API.

        Behavior:
        1. Ensure features exist (checks cache; optionally attempts to build).
        2. Run the lightweight 'predict' pipeline task only (modular).
        3. Attempt to load the prediction artifact (joblib preferred).
        4. Return normalized dict expected by UI or None if no artifact found.

        Args:
            equity (str): Ticker symbol (e.g., 'TCS').
            horizon (int): Forecast horizon in days (UI may expect this).
            force_rebuild_features (bool): Force feature build even if cache exists.
            allow_training (bool): If True, orchestrator may trigger training when no model is found.
            retries (int, optional): Override configured retries for running tasks.

        Returns:
            Optional[Dict[str, Any]]: forecast dict or None
        """
        equity = equity.upper().strip()

        self.logger.info(f"run_forecast_pipeline start: equity={equity}, horizon={horizon}")
        monitor.log_stage_start("forecast_pipeline", {"equity": equity})


        # 1. Ensure features available
        feature_path = self.prepare_features(equity)
        if force_rebuild_features:
            # If caller forces rebuild, try to remove existing cache and rebuild
            try:
                if os.path.exists(feature_path):
                    os.remove(feature_path)
                    self.logger.info(f"Removed existing feature cache for {equity} at {feature_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove existing feature cache for {equity}: {e}")
            feature_path = self.prepare_features(equity)

        if not os.path.exists(feature_path):
            self.logger.warning(f"Feature file for {equity} not found at expected path: {feature_path}. Proceeding - predict task may build it.")

        # 2. Run only prediction task
        task_retries = retries if retries is not None else self.pipeline_cfg.retries
        try:
            self.logger.info(f"Invoking pipeline predict task for {equity} (retries={task_retries})")
            # The predict task (run_prediction) is expected to look for cached features and produce artifact(s).
            self.run_task("predict", retries=task_retries, strict=True)
        except Exception as e:
            self.logger.error(f"Prediction task failed for {equity}: {e}")
            # If prediction failed and allow_training is True, attempt to run training then predict
            if allow_training:
                try:
                    self.logger.info(f"Attempting training because allow_training=True for {equity}")
                    self.run_task("train", retries=task_retries, strict=True)
                    self.run_task("predict", retries=task_retries, strict=True)
                except Exception as e2:
                    self.logger.error(f"Training+Predict attempt failed for {equity}: {e2}")

        # 3. Try to load produced artifact(s)
        artifact = self._load_prediction_artifact(equity)
        if artifact is None:
            self.logger.warning(f"No prediction artifact found for {equity} after running predict task.")
            return None

        # 4. Normalize artifact to expected UI schema
        artifact.setdefault("equity", equity)
        metadata = artifact.setdefault("metadata", {})
        metadata.setdefault("source", metadata.get("source", "artifact"))
        # ensure forecast dates if missing (best-effort)
        if "forecast_dates" not in artifact:
            last_hist_date = pd.Timestamp.today()
            artifact["forecast_dates"] = (last_hist_date + pd.to_timedelta(range(1, horizon + 1), unit="D")).astype(str).tolist()
        # ensure forecast_mean present if only 'prediction' exists
        if "forecast_mean" not in artifact and "prediction" in artifact:
            try:
                artifact["forecast_mean"] = list(artifact["prediction"])
            except Exception:
                artifact["forecast_mean"] = [artifact["prediction"]]

        monitor.log_stage_end("forecast_pipeline", {"equity": equity})
        return artifact

    # -----------------------------------------------------
    # Pipeline Execution Logic
    # -----------------------------------------------------
    def run_pipeline(self, tasks: Optional[list] = None, ticker: Optional[str] = None) -> None:
        """
        Run the orchestrated pipeline with optional task list and ticker context.

        Args:
            tasks (list, optional): Tasks to execute (default from config if None).
            ticker (str, optional): Ticker symbol for dashboard-initiated forecasting.
        """
        tasks_to_run = tasks if tasks else self.pipeline_cfg.tasks
        retries = self.pipeline_cfg.retries
        strict = self.pipeline_cfg.strict

        self.logger.info(f"Starting orchestrated pipeline with tasks: {tasks_to_run}")

        # Dashboard forecast scenario
        if ticker and "predict" in tasks_to_run:
            feature_path = self.prepare_features(ticker)
            self.logger.info(f"Feature path prepared for forecasting: {feature_path}")

        for task in tasks_to_run:
            self.run_task(task, retries=retries, strict=strict)

        self.logger.info("Pipeline execution completed successfully.")

    # -----------------------------------------------------
    # Marker Reset Utility
    # -----------------------------------------------------
    def reset_task_markers(self, tasks: list = None) -> None:
        """
        Remove completion markers to force tasks to rerun.
        """
        tasks_to_reset = tasks if tasks else TASK_MARKERS.keys()
        for task in tasks_to_reset:
            marker = TASK_MARKERS.get(task)
            if marker and os.path.exists(marker):
                os.remove(marker)
                self.logger.info(f"Reset marker for task '{task}': {marker}")
            else:
                self.logger.debug(f"No marker found for task '{task}' — nothing to reset")
