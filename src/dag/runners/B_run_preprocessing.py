# src/dag/runners/B_run_preprocessing.py

import argparse
from src.pipeline.B_preprocessing_pipeline import PreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Ingestion run_id to preprocess",
    )
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        run_id=args.run_id,
        required_columns=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close",
        ],
    )

    pipeline.run()

if __name__ == "__main__":
    main()
