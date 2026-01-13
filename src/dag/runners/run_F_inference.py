from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline.F_inference_pipeline import InferencePipeline



def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Pipeline F — Inference"
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

    config_path = "src/dag/runners/F_inference_config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        sys.exit(1)

    pipeline = InferencePipeline(
        run_id=args.run_id,
        config_path=str(config_path),
    )

    pipeline.run()

    


if __name__ == "__main__":
    main()
