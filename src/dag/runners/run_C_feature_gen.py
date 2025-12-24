# src/dag/runners/run_C_feature_engineering.py
import argparse

from src.pipeline.C_feature_gen_pipeline import FeaturePipeline

def main():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    pipeline = FeaturePipeline(
        run_id=args.run_id
    )
    pipeline.run()

if __name__ == "__main__":
		main()