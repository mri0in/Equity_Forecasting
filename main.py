"""
Central CLI entrypoint for the Equity Forecasting project.

This module parses command-line arguments to trigger various
functionalities through the Orchestrator: training, prediction,
optimization, ensembling, walk-forward validation, or starting the API.

Usage:
    python main.py <command> [--config CONFIG_PATH]

Supported commands:
    train           Train a model using a YAML config
    predict         Generate predictions using a trained model
    optimize        Run hyperparameter optimization
    ensemble        Run ensemble strategies (simple or meta-model based)
    walkforward     Run walk forward validator
    pipeline        Run the full pipeline as defined in config
    serve           Start the API server

Examples:
    python main.py train --config configs/train_config.yaml
    python main.py predict --config configs/predict_config.yaml
    python main.py optimize --config configs/optimize_config.yaml
    python main.py ensemble --config configs/ensemble_config.yaml --strategy meta
    python main.py walkforward --config configs/walkforward_config.yaml
    python main.py pipeline --config configs/full_config.yaml
    python main.py serve --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys

# Add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.orchestrator import PipelineOrchestrator
from api.main import start_api
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_config_path(config_path: str) -> None:
    """
    Validates whether the given config path exists and is a file.

    Args:
        config_path (str): Path to the config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")


def main():
    """
    Parse CLI arguments and dispatch commands to the Orchestrator or API.
    """
    parser = argparse.ArgumentParser(description="Equity Forecasting Project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", "-c", required=True, help="Path to training config YAML")

    # --- Predict ---
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--config", "-c", required=True, help="Path to prediction config YAML")

    # --- Optimize ---
    optimize_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    optimize_parser.add_argument("--config", "-c", required=True, help="Path to optimization config YAML")
    optimize_parser.add_argument(
        "--optimizer", choices=["optuna", "raytune", "hyperopt"], default="optuna",
        help="Select optimization backend"
    )

    # --- Ensemble ---
    ensemble_parser = subparsers.add_parser("ensemble", help="Run model ensembling")
    ensemble_parser.add_argument("--config", "-c", required=True, help="Path to ensembling config YAML")
    ensemble_parser.add_argument(
        "--strategy", choices=["simple", "meta"], default="meta",
        help="Ensemble strategy to use"
    )

    # --- Walkforward ---
    walk_parser = subparsers.add_parser("walkforward", help="Run walk-forward validation")
    walk_parser.add_argument("--config", "-c", required=True, help="Path to the WFV config YAML")

    # --- Pipeline ---
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full orchestrated pipeline")
    pipeline_parser.add_argument("--config", "-c", required=True, help="Path to pipeline config YAML")

    # --- Serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port for the API")

    args = parser.parse_args()

    try:
        if args.command == "serve":
            logger.info(f"Launching API server on {args.host}:{args.port}")
            start_api(host=args.host, port=args.port)
            return

        # --- Non-API commands use the orchestrator ---
        validate_config_path(getattr(args, "config", None))
        orchestrator = PipelineOrchestrator(config_path=args.config)

        if args.command == "pipeline":
            # Run the full pipeline as defined in YAML
            logger.info(f"Running full orchestrated pipeline from config: {args.config}")
            orchestrator.run_pipeline()


        if args.command == "train":
            validate_config_path(args.config)
            orchestrator.run_pipeline(["train"])

        elif args.command == "predict":
            validate_config_path(args.config)
            orchestrator.run_pipeline(["predict"])

        elif args.command == "optimize":
            validate_config_path(args.config)
            orchestrator.run_pipeline(["optimize"])

        elif args.command == "ensemble":
            validate_config_path(args.config)
            orchestrator.run_pipeline(["ensemble"])

        elif args.command == "walkforward":
            validate_config_path(args.config)
            orchestrator.run_pipeline(["walkforward"])

    except Exception as e:
        logger.exception(f"Fatal error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
