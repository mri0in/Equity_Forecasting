# src/pipeline/H_prediction_pipeline.py

"""
Run prediction pipeline.

Loads cached feature files and runs inference using ModelPredictor.
Produces canonical joblib prediction artifacts under
datalake/predictions/{TICKER}_forecast.joblib.

⚠️ IMPORTANT WARNING:
Do NOT call these functions/classes directly.
Use the wrapper functions in src/pipeline/pipeline_wrapper.py
to enforce orchestration, logging, retries, and task markers.
"""

import os
import glob
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.predictor.predict import ModelPredictor
from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers (monitor passed explicitly)
# ---------------------------------------------------------------------
def _discover_feature_paths(
    monitor: TrainingMonitor,
    ticker: Optional[str] = None,
) -> List[str]:
    """
    Discover feature cache files to run prediction on.
    """
    monitor.log_stage_start("discover_feature_paths", {"ticker": ticker})

    candidates: List[str] = []
    cache_dirs = [
        os.path.join("datalake", "cache", "features"),
        os.path.join("datalake", "features"),
    ]

    if ticker:
        for d in cache_dirs:
            p = os.path.join(d, f"{ticker}_features.joblib")
            if os.path.exists(p):
                monitor.log_stage_end("discover_feature_paths")
                return [p]

        for d in cache_dirs:
            p_csv = os.path.join(d, f"{ticker}_features.csv")
            if os.path.exists(p_csv):
                monitor.log_stage_end("discover_feature_paths")
                return [p_csv]

        monitor.log_stage_end("discover_feature_paths", {"num_candidates": 0})
        return []

    for d in cache_dirs:
        if not os.path.exists(d):
            continue
        candidates.extend(glob.glob(os.path.join(d, "*_features.joblib")))
        candidates.extend(glob.glob(os.path.join(d, "*_features.csv")))

    candidates = sorted(list(dict.fromkeys(candidates)))
    monitor.log_stage_end(
        "discover_feature_paths",
        {"num_candidates": len(candidates)},
    )
    return candidates


def _load_feature_file(
    monitor: TrainingMonitor,
    path: str,
) -> pd.DataFrame:
    """
    Load features from a joblib or CSV file.
    """
    monitor.log_stage_start("load_feature_file", {"path": path})

    if not os.path.exists(path):
        monitor.log_stage_end(
            "load_feature_file",
            {"status": "file_not_found"},
        )
        raise FileNotFoundError(f"Feature file not found: {path}")

    if path.endswith(".joblib"):
        obj = joblib.load(path)

        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            if arr.ndim == 1:
                df = pd.DataFrame(arr, columns=["feature_0"])
            else:
                df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
        elif isinstance(obj, dict):
            for key in ("features", "X", "data"):
                if key in obj:
                    df = pd.DataFrame(obj[key])
                    break
            else:
                df = pd.DataFrame(obj)
        else:
            raise ValueError(f"Unsupported joblib content type: {type(obj)}")

    elif path.endswith(".csv"):
        df = pd.read_csv(path)

    else:
        raise ValueError(f"Unsupported feature file extension: {path}")

    monitor.log_stage_end(
        "load_feature_file",
        {"status": "success", "num_rows": len(df)},
    )
    return df


def _build_artifact(
    ticker: str,
    forecast_dates: List[str],
    forecast_mean: List[float],
    forecast_upper: Optional[List[float]] = None,
    forecast_lower: Optional[List[float]] = None,
    hist_dates: Optional[List[str]] = None,
    hist_prices: Optional[List[float]] = None,
    model_name: Optional[str] = None,
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build canonical forecast artifact dictionary.
    """
    return {
        "equity": ticker,
        "hist_dates": hist_dates or [],
        "hist_prices": hist_prices or [],
        "forecast_dates": forecast_dates,
        "forecast_mean": list(forecast_mean),
        "forecast_upper": list(forecast_upper) if forecast_upper else [],
        "forecast_lower": list(forecast_lower) if forecast_lower else [],
        "forecast_prices": list(forecast_mean),
        "metadata": {
            "source": "run_prediction",
            "model": model_name,
            "adapter": adapter_name,
        },
    }


# ---------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------
def run_prediction(
    config_path: str,
    ticker: Optional[str] = None,
) -> None:
    """
    Run prediction pipeline. Can be limited to a single ticker.
    """
    if not config_path:
        raise ValueError("config_path must be provided")

    # -------------------------------------------------
    # Run identity
    # -------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scope = ticker or "GLOBAL"
    run_id = f"{scope}_PREDICT_{timestamp}"
    run_dir = os.path.join("runs", "prediction", run_id)

    # -------------------------------------------------
    # Runtime monitor (CORRECT)
    # -------------------------------------------------
    monitor = TrainingMonitor(
        run_id=run_id,
        save_dir=run_dir,
        visualize=False,
        flush_every=1,
    )

    monitor.log_stage_start(
        "run_prediction",
        {"ticker": ticker, "config_path": config_path},
    )

    logger.info(
        "Starting prediction pipeline | run_id=%s | ticker=%s",
        run_id,
        ticker,
    )

    feature_paths = _discover_feature_paths(monitor, ticker)
    if not feature_paths:
        logger.warning("No feature files found. Prediction skipped.")
        monitor.log_stage_end(
            "run_prediction",
            {"status": "no_features"},
        )
        return

    predictor = ModelPredictor(config_path=config_path)

    out_dir = os.path.join("datalake", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    for feat_path in feature_paths:
        monitor.log_stage_start(
            "predict_single_ticker",
            {"feature_file": feat_path},
        )

        try:
            base = os.path.basename(feat_path)
            ticker_name = base.split("_features")[0]

            df_features = _load_feature_file(monitor, feat_path)
            preds = predictor.predict(df_features)

            if isinstance(preds, dict):
                forecast_mean = list(preds.get("mean", []))
                forecast_upper = list(preds.get("upper", []))
                forecast_lower = list(preds.get("lower", []))
            else:
                forecast_mean = list(preds)
                forecast_upper, forecast_lower = [], []

            horizon = len(forecast_mean)

            if pd.api.types.is_datetime64_any_dtype(df_features.index):
                last_ts = pd.to_datetime(df_features.index[-1])
                hist_dates = df_features.index.astype(str).tolist()
            elif "date" in df_features.columns:
                last_ts = pd.to_datetime(df_features["date"].iloc[-1])
                hist_dates = df_features["date"].astype(str).tolist()
            else:
                last_ts = pd.Timestamp.utcnow()
                hist_dates = []

            forecast_dates = (
                last_ts
                + pd.to_timedelta(range(1, horizon + 1), unit="D")
            ).astype(str).tolist()

            artifact = _build_artifact(
                ticker=ticker_name,
                forecast_dates=forecast_dates,
                forecast_mean=forecast_mean,
                forecast_upper=forecast_upper or None,
                forecast_lower=forecast_lower or None,
                hist_dates=hist_dates,
                hist_prices=(
                    df_features["close"].tolist()
                    if "close" in df_features.columns
                    else []
                ),
                model_name=getattr(predictor, "model_name", None),
                adapter_name=getattr(predictor, "adapter_name", None),
            )

            out_path = os.path.join(
                out_dir,
                f"{ticker_name}_forecast.joblib",
            )
            joblib.dump(artifact, out_path)

            logger.info(
                "Saved forecast artifact | ticker=%s | path=%s",
                ticker_name,
                out_path,
            )

            monitor.log_stage_end(
                "predict_single_ticker",
                {"ticker": ticker_name, "status": "success"},
            )

        except Exception as exc:
            logger.exception(
                "Prediction failed | feature_file=%s",
                feat_path,
            )
            monitor.log_stage_end(
                "predict_single_ticker",
                {
                    "feature_file": feat_path,
                    "status": "failure",
                    "error": str(exc),
                },
            )

    monitor.log_stage_end(
        "run_prediction",
        {"status": "completed"},
    )
