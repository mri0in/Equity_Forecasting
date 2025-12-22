# src/pipeline/C_feature_pipeline.py
"""
C. Feature Generation Pipeline
------------------------------
Consumes preprocessing artifacts (clean CSVs) derived from an ingestion run,
generates technical features per ticker, and writes feature datasets to
Parquet format for downstream modeling.

Responsibilities:
- Ingestion-run–aware ticker selection
- Feature engineering (technical indicators only)
- Parquet-based caching of per-ticker feature sets
- Artifact logging for lineage and traceability

Non-responsibilities:
- Data ingestion
- Raw data cleaning
- Dataset consolidation / model training
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from src.features.technical.build_features import FeatureBuilder
from src.monitoring.monitor import TrainingMonitor
from src.utils.logger import get_logger


class FeaturePipeline:
    """
    Feature generation pipeline operating strictly on ingestion artifacts.
    """

    def __init__(
        self,
        run_id: str,
        clean_root: str = "datalake/data/cache/clean",
        feature_root: str = "datalake/data/cache/features",
    ) -> None:
        """
        Parameters
        ----------
        run_id : str
            Ingestion run_id whose outputs should be processed.
        clean_root : str
            Directory containing cleaned per-ticker CSVs.
        feature_root : str
            Output directory for per-ticker Parquet feature files.
        """
        self.run_id = run_id
        self.clean_root = Path(clean_root)
        self.feature_root = Path(feature_root)

        self.feature_root.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(self.__class__.__name__)
        self.monitor = TrainingMonitor(run_id=run_id)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _load_ingestion_artifacts(self) -> List[dict]:
        """
        Load ingestion artifacts.jsonl for the given run_id.
        """
        artifacts_path = (
            Path("datalake")
            / "runs"
            / "ingestion"
            / self.run_id
            / "artifacts.jsonl"
        )

        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Ingestion artifacts not found for run_id={self.run_id}"
            )

        artifacts: List[dict] = []
        with artifacts_path.open("r") as fh:
            for line in fh:
                artifacts.append(json.loads(line))

        return artifacts

    def _is_up_to_date(self, clean_path: Path, feature_path: Path) -> bool:
        """
        Determine whether feature file is already up-to-date
        relative to the cleaned input.
        """
        if not feature_path.exists():
            return False

        return feature_path.stat().st_mtime >= clean_path.stat().st_mtime

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute feature generation for all tickers present
        in ingestion artifacts.
        """
        self.monitor.log_stage_start("feature_pipeline")
        self.logger.info("Starting feature generation pipeline")

        artifacts = self._load_ingestion_artifacts()

        for idx, artifact in enumerate(artifacts, start=1):
            ticker = artifact["ticker"]
            clean_path = self.clean_root / f"{ticker}.csv"
            feature_path = self.feature_root / f"{ticker}.parquet"

            self.logger.info(
                f"[FEAT] ({idx}/{len(artifacts)}) Processing {ticker}"
            )

            if not clean_path.exists():
                self.logger.error(
                    f"[FEAT] Cleaned file missing for {ticker}: {clean_path}"
                )
                continue

            if self._is_up_to_date(clean_path, feature_path):
                self.logger.info(
                    f"[FEAT] Skipping {ticker} (features already up-to-date)"
                )
                continue

            try:
                df_clean = pd.read_csv(clean_path)

                feature_builder = FeatureBuilder(equity=ticker)
                df_features = feature_builder.build_all(df_clean)

                df_features.to_parquet(
                    feature_path,
                    index=False,
                    engine="pyarrow",
                )

                self.logger.info(
                    f"[FEAT] Saved features {ticker} rows={len(df_features)} "
                    f"path={feature_path}"
                )

                self.monitor.log_artifact(
                    stage="feature_generation",
                    artifact_type="feature_parquet",
                    ticker=ticker,
                    path=str(feature_path),
                    rows=len(df_features),
                )

            except Exception as exc:
                self.logger.error(
                    f"[FEAT] Feature generation failed for {ticker}: {exc}",
                    exc_info=True,
                )

        self.monitor.log_stage_end("feature_pipeline")
        self.logger.info("Feature generation pipeline completed")
