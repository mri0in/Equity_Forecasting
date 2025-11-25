# src/pipeline/run_prediction.py

"""
Run prediction pipeline.

Loads cached feature files (joblib preferred) and runs inference using
the project's ModelPredictor. Produces canonical joblib prediction artifacts
under datalake/predictions/{TICKER}_forecast.joblib.

Behaviour:
 - If `ticker` argument provided: predict only for that ticker (if feature cache exists).
 - Otherwise: discover all feature caches in canonical cache dirs and predict for each.
 - Writes a completion marker at datalake/predictions/.predict_complete on success.


⚠️ IMPORTANT WARNING FOR USERS & DEVELOPERS
# For orchestration and end-user workflows, DO NOT call these classes
# directly. Instead, always use the wrapper functions in:
#
#     src/pipeline/pipeline_wrapper.py
#
# Example:
#     from src.pipeline.pipeline_wrapper import run_prediction
#     run_prediction("configs/predict_config.yaml")
#
# Reason:
# The wrappers provide a consistent interface for the orchestrator and enforce
# config-driven execution across the project. Direct class calls may bypass
# orchestration safeguards (retries, logging, markers).
# -------
"""

import os
import glob
import logging
from typing import Optional, List, Any, Dict

import joblib
import numpy as np
import pandas as pd

from predictor.predict import ModelPredictor  
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _discover_feature_paths(ticker: Optional[str] = None) -> List[str]:
    """
    Discover feature cache files to run prediction on.

    Preference order:
      1) datalake/cache/features/{TICKER}_features.joblib
      2) datalake/features/{TICKER}_features.joblib
    If ticker is None, returns all matching feature files in canonical dirs.
    """
    candidates = []
    cache_dirs = [os.path.join("datalake", "cache", "features"), os.path.join("datalake", "features")]

    if ticker:
        for d in cache_dirs:
            p = os.path.join(d, f"{ticker}_features.joblib")
            if os.path.exists(p):
                candidates.append(p)
                return candidates
        # allow csv fallback
        for d in cache_dirs:
            p_csv = os.path.join(d, f"{ticker}_features.csv")
            if os.path.exists(p_csv):
                candidates.append(p_csv)
                return candidates
        return []

    # no ticker: discover all joblib feature files
    for d in cache_dirs:
        if not os.path.exists(d):
            continue
        matches = glob.glob(os.path.join(d, "*_features.joblib"))
        candidates.extend(matches)
        # allow csvs as last-resort
        csv_matches = glob.glob(os.path.join(d, "*_features.csv"))
        candidates.extend(csv_matches)

    # deduplicate and sort
    candidates = sorted(list(dict.fromkeys(candidates)))
    return candidates


def _load_feature_file(path: str) -> pd.DataFrame:
    """
    Load features from a joblib or CSV file and return as a DataFrame.
    Raises FileNotFoundError or ValueError on invalid contents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    if path.endswith(".joblib"):
        obj = joblib.load(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        # If stored numpy array, convert to DataFrame with numeric columns
        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            # If 1D, convert to single-column frame
            if arr.ndim == 1:
                return pd.DataFrame(arr, columns=["feature_0"])
            # else create generic columns
            cols = [f"f{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
        # If a dict with 'features' or 'X' key
        if isinstance(obj, dict):
            for key in ("features", "X", "data"):
                if key in obj:
                    return pd.DataFrame(obj[key])
            # If dict is column-like, try to build DataFrame
            try:
                return pd.DataFrame(obj)
            except Exception:
                raise ValueError(f"Unsupported joblib object structure in {path}")
        raise ValueError(f"Unsupported joblib object type in {path}: {type(obj)}")

    # fallback: CSV
    if path.endswith(".csv"):
        return pd.read_csv(path)

    raise ValueError(f"Unsupported feature file extension for {path}")


def _build_artifact(ticker: str, forecast_dates: List[str], forecast_mean: List[float],
                    forecast_upper: Optional[List[float]] = None,
                    forecast_lower: Optional[List[float]] = None,
                    hist_dates: Optional[List[str]] = None,
                    hist_prices: Optional[List[float]] = None,
                    model_name: Optional[str] = None,
                    adapter_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Build canonical forecast artifact dictionary to be saved as joblib.
    """
    artifact = {
        "equity": ticker,
        "hist_dates": hist_dates or [],
        "hist_prices": hist_prices or [],
        "forecast_dates": forecast_dates,
        "forecast_mean": list(forecast_mean),
        "forecast_upper": list(forecast_upper) if forecast_upper is not None else [],
        "forecast_lower": list(forecast_lower) if forecast_lower is not None else [],
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
    Entry point used by orchestrator.run_task('predict', ...).

    Args:
        config_path: path to config yaml (passed to ModelPredictor)
        ticker: optional ticker string to limit prediction to a single equity.
    """
    logger.info(f"run_prediction invoked (ticker={ticker}) using config: {config_path}")

    # discover feature paths
    feature_paths = _discover_feature_paths(ticker)
    if not feature_paths:
        logger.warning("No feature files found to run prediction on. Nothing to do.")
        return

    # initialize predictor
    predictor = ModelPredictor(config_path=config_path)

    # ensure predictions dir exists
    out_dir = os.path.join("datalake", "predictions")
    os.makedirs(out_dir, exist_ok=True)

    # iterate feature files and run inference
    for feat_path in feature_paths:
        try:
            # derive ticker name from filename (assumes <TICKER>_features.*)
            base = os.path.basename(feat_path)
            if base.endswith("_features.joblib") or base.endswith("_features.csv"):
                ticker_name = base.split("_features")[0]
            else:
                # fallback to removing extension
                ticker_name = os.path.splitext(base)[0]

            logger.info(f"Running prediction for {ticker_name} using features at {feat_path}")
            df_features = _load_feature_file(feat_path)

            # ModelPredictor.predict should accept DataFrame or ndarray
            preds = predictor.predict(df_features)

            # Predictor may return different shapes:
            # - dict with keys 'mean', 'upper', 'lower' (preferred)
            # - 1D list/array of mean predictions
            model_name = getattr(predictor, "model_name", None)
            adapter_name = getattr(predictor, "adapter_name", None)

            forecast_mean = []
            forecast_upper = []
            forecast_lower = []

            if isinstance(preds, dict):
                # normalize expected keys
                if "mean" in preds:
                    forecast_mean = list(preds["mean"])
                elif "forecast_mean" in preds:
                    forecast_mean = list(preds["forecast_mean"])
                if "upper" in preds:
                    forecast_upper = list(preds["upper"])
                if "lower" in preds:
                    forecast_lower = list(preds["lower"])
            else:
                # treat as array-like of mean preds
                try:
                    forecast_mean = list(preds)
                except Exception:
                    forecast_mean = [preds]

            # Build forecast_dates from the last index of df_features if datetime-like, else use today
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
                model_name=model_name,
                adapter_name=adapter_name,
            )

            out_path = os.path.join(out_dir, f"{ticker_name}_forecast.joblib")
            joblib.dump(artifact, out_path)
            logger.info(f"Saved forecast artifact for {ticker_name} → {out_path}")

        except Exception as e:
            logger.error(f"Prediction failed for feature file {feat_path}: {e}", exc_info=True)
            continue

    # write completion marker
    marker = os.path.join("datalake", "predictions", ".predict_complete")
    try:
        with open(marker, "w") as fh:
            fh.write("complete\n")
        logger.info(f"Prediction completion marker written: {marker}")
    except Exception as e:
        logger.warning(f"Failed to write predict completion marker: {e}")
