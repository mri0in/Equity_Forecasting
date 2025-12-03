# src/pipeline/run_prediction.py

"""
Run prediction pipeline.

Loads cached feature files and runs inference using ModelPredictor.
Produces canonical joblib prediction artifacts under
datalake/predictions/{TICKER}_forecast.joblib.

⚠️ IMPORTANT WARNING:
Do NOT call these functions/classes directly. Use the wrapper functions
in src/pipeline/pipeline_wrapper.py to enforce orchestration, logging,
retries, and task markers.
"""

import os
import glob
import logging
from typing import Optional, List, Any, Dict

import joblib
import numpy as np
import pandas as pd

from predictor.predict import ModelPredictor
from src.utils.logger import get_logger
from src.monitoring.monitor import TrainingMonitor

logger = get_logger(__name__)
monitor = TrainingMonitor()


def _discover_feature_paths(ticker: Optional[str] = None) -> List[str]:
    """
    Discover feature cache files to run prediction on.
    """
    monitor.log_stage_start("discover_feature_paths", {"ticker": ticker})
    candidates = []
    cache_dirs = [os.path.join("datalake", "cache", "features"), os.path.join("datalake", "features")]

    if ticker:
        for d in cache_dirs:
            p = os.path.join(d, f"{ticker}_features.joblib")
            if os.path.exists(p):
                candidates.append(p)
                monitor.log_stage_end("discover_feature_paths")
                return candidates
        for d in cache_dirs:
            p_csv = os.path.join(d, f"{ticker}_features.csv")
            if os.path.exists(p_csv):
                candidates.append(p_csv)
                monitor.log_stage_end("discover_feature_paths")
                return candidates
        monitor.log_stage_end("discover_feature_paths")
        return []

    for d in cache_dirs:
        if not os.path.exists(d):
            continue
        matches = glob.glob(os.path.join(d, "*_features.joblib"))
        candidates.extend(matches)
        csv_matches = glob.glob(os.path.join(d, "*_features.csv"))
        candidates.extend(csv_matches)

    candidates = sorted(list(dict.fromkeys(candidates)))
    monitor.log_stage_end("discover_feature_paths", {"num_candidates": len(candidates)})
    return candidates


def _load_feature_file(path: str) -> pd.DataFrame:
    """
    Load features from a joblib or CSV file and return as DataFrame.
    """
    monitor.log_stage_start("load_feature_file", {"path": path})
    if not os.path.exists(path):
        monitor.log_stage_end("load_feature_file", {"status": "file_not_found"})
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = None
    if path.endswith(".joblib"):
        obj = joblib.load(path)
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            if arr.ndim == 1:
                df = pd.DataFrame(arr, columns=["feature_0"])
            else:
                cols = [f"f{i}" for i in range(arr.shape[1])]
                df = pd.DataFrame(arr, columns=cols)
        elif isinstance(obj, dict):
            for key in ("features", "X", "data"):
                if key in obj:
                    df = pd.DataFrame(obj[key])
                    break
            if df is None:
                try:
                    df = pd.DataFrame(obj)
                except Exception:
                    raise ValueError(f"Unsupported joblib dict structure: {path}")
        else:
            raise ValueError(f"Unsupported joblib type {type(obj)} in {path}")
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported feature file extension for {path}")

    monitor.log_stage_end("load_feature_file", {"status": "success", "num_rows": len(df)})
    return df


def _build_artifact(ticker: str, forecast_dates: List[str], forecast_mean: List[float],
                    forecast_upper: Optional[List[float]] = None,
                    forecast_lower: Optional[List[float]] = None,
                    hist_dates: Optional[List[str]] = None,
                    hist_prices: Optional[List[float]] = None,
                    model_name: Optional[str] = None,
                    adapter_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Build canonical forecast artifact dictionary.
    """
    artifact = {
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
    return artifact


def run_prediction(config_path: str, ticker: Optional[str] = None) -> None:
    """
    Run prediction pipeline. Can be limited to a single ticker.
    """
    monitor.log_stage_start("run_prediction", {"ticker": ticker, "config": config_path})
    logger.info(f"run_prediction invoked (ticker={ticker}) using config: {config_path}")

    feature_paths = _discover_feature_paths(ticker)
    if not feature_paths:
        logger.warning("No feature files found to run prediction. Nothing to do.")
        monitor.log_stage_end("run_prediction", {"status": "no_features"})
        return

    predictor = ModelPredictor(config_path=config_path)
    out_dir = os.path.join("datalake", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    for feat_path in feature_paths:
        monitor.log_stage_start("predict_single_ticker", {"feature_file": feat_path})
        try:
            base = os.path.basename(feat_path)
            if base.endswith("_features.joblib") or base.endswith("_features.csv"):
                ticker_name = base.split("_features")[0]
            else:
                ticker_name = os.path.splitext(base)[0]

            df_features = _load_feature_file(feat_path)
            preds = predictor.predict(df_features)

            forecast_mean, forecast_upper, forecast_lower = [], [], []
            if isinstance(preds, dict):
                forecast_mean = list(preds.get("mean") or preds.get("forecast_mean") or [])
                forecast_upper = list(preds.get("upper") or [])
                forecast_lower = list(preds.get("lower") or [])
            else:
                forecast_mean = list(preds)

            forecast_horizon = len(forecast_mean)
            if pd.api.types.is_datetime64_any_dtype(df_features.index):
                last_ts = pd.to_datetime(df_features.index[-1])
            elif "date" in df_features.columns:
                last_ts = pd.to_datetime(df_features["date"].iloc[-1])
            else:
                last_ts = pd.Timestamp.today()
            forecast_dates = (last_ts + pd.to_timedelta(range(1, forecast_horizon + 1), unit="D")).astype(str).tolist()

            artifact = _build_artifact(
                ticker=ticker_name,
                forecast_dates=forecast_dates,
                forecast_mean=forecast_mean,
                forecast_upper=forecast_upper if forecast_upper else None,
                forecast_lower=forecast_lower if forecast_lower else None,
                hist_dates=(df_features.index.astype(str).tolist() if pd.api.types.is_datetime64_any_dtype(df_features.index) else []),
                hist_prices=(df_features["close"].tolist() if "close" in df_features.columns else []),
                model_name=getattr(predictor, "model_name", None),
                adapter_name=getattr(predictor, "adapter_name", None),
            )

            out_path = os.path.join(out_dir, f"{ticker_name}_forecast.joblib")
            joblib.dump(artifact, out_path)
            logger.info(f"Saved forecast artifact for {ticker_name} → {out_path}")
            monitor.log_stage_end("predict_single_ticker", {"ticker": ticker_name, "status": "success"})

        except Exception as e:
            logger.error(f"Prediction failed for feature file {feat_path}: {e}", exc_info=True)
            monitor.log_stage_end("predict_single_ticker", {"feature_file": feat_path, "status": "failure", "error": str(e)})
            continue

    marker = os.path.join(out_dir, ".predict_complete")
    try:
        with open(marker, "w") as fh:
            fh.write("complete\n")
        logger.info(f"Prediction completion marker written: {marker}")
    except Exception as e:
        logger.warning(f"Failed to write predict completion marker: {e}")

    monitor.log_stage_end("run_prediction", {"status": "completed"})
