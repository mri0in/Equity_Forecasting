"""
CLI runner for Pipeline D — Hyperparameter Optimization.

Usage
-----
python src/dag/runners/run_D_optimization.py \
    --run-id RUN_YYYYMMDD_HHMMSS \
    --config src/dag/runners/D_optimization_config.yaml

Purpose
-------
- Unit-test OptimizationPipeline independently
- Validate feature availability from Pipeline C
- Validate optimizer integration and trial logging

This runner does NOT:
- Train final models
- Persist model artifacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline.D_optimization_pipeline import OptimizationPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Pipeline D — Hyperparameter Optimization"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Ingestion run_id (produced by Pipeline A)",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to D_optimization_config.yaml",
    )

    parser.add_argument(
        "--feature-root",
        type=str,
        default="datalake/data/cache/features",
        help="Root directory containing feature Parquet files",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="optuna",
        help="Optimizer backend (default: optuna)",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    config_path = "src/dag/runners/D_optimization_config.yaml"
    if not Path(config_path).exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)


    try:
        pipeline = OptimizationPipeline(
            run_id=args.run_id,
            config_path=str(config_path),
            feature_root=args.feature_root,
            optimizer_name=args.optimizer,
        )

        pipeline.run()

    except Exception as exc:
        logger.critical(
            "Optimization pipeline failed | run_id=%s | error=%s",
            args.run_id,
            str(exc),
            exc_info=True,
        )
        sys.exit(2)

    logger.info(
        "Optimization pipeline completed successfully | run_id=%s",
        args.run_id,
    )


if __name__ == "__main__":
    main()
