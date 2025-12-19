#src/dag/runners/A_run_ingestion.py
from src.pipeline.A_ingestion_pipeline import IngestionPipeline


if __name__ == "__main__":
    pipeline = IngestionPipeline(
        config_path="src/dag/runners/A_ingestion_config.yaml"
    )
    pipeline.run()

