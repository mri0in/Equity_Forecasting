"""
CLI runner for Pipeline E Model Training.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline.E_modeltrainer_pipeline import ModelTrainerPipeline



def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Pipeline E — Model Training"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Ingestion run_id (produced by Pipeline A)",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint for Pipeline E.
    """
    args = parse_args()

    config_path = "src/dag/runners/E_modeltrainer_config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        sys.exit(1)

    pipeline = ModelTrainerPipeline(
        run_id=args.run_id,
        config_path=str(config_path),
    )

    pipeline.run()

    


if __name__ == "__main__":
    main()
