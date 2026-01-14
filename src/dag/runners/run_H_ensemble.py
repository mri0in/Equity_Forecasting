from __future__ import annotations
import argparse
from src.pipeline.H_ensemble_pipeline import EnsemblePipeline


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Pipeline H — Global Ensemble"
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

    pipeline = EnsemblePipeline(
        run_id=args.run_id,
    )

    pipeline.run()


if __name__ == "__main__":
    main()