#src/dag/runners/run_A_ingestion.py
from src.pipeline.A_ingestion_pipeline import IngestionPipeline
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    pipeline = IngestionPipeline(
        config_path="src/dag/runners/A_ingestion_config.yaml"
    )
    run_id = pipeline.run()
    print(f"[RUN ID] {run_id}")