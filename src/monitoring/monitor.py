# src/monitoring/monitor.py
"""
Unified Training & Pipeline Monitor.

Responsibilities
----------------
- Append-only lifecycle + metric logging (metrics.jsonl)
- Append-only pipeline artifact logging (artifacts.jsonl)
- Optional CSV / plot persistence for training stages only

Artifact Policies
-----------------
- "none"     : metrics.jsonl only
- "metrics"  : metrics.jsonl + CSVs
- "training" : metrics.jsonl + CSVs + plots

IMPORTANT
---------
- artifacts.jsonl is ALWAYS written
- Artifact logging is pipeline-agnostic (A → H)
- No directory scanning is ever required downstream
"""

from __future__ import annotations

import csv
import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Literal

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------
# Immutable metric records (training-only)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    lr: Optional[float]
    early_stopped: bool
    timestamp: str


@dataclass(frozen=True)
class FoldRecord:
    fold_idx: int
    val_score: float
    timestamp: str


@dataclass(frozen=True)
class TrialRecord:
    trial_idx: int
    best_val: float
    params: Dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------
class TrainingMonitor:
    """
    Lightweight file-based monitor for all pipeline stages.

    This class NEVER infers semantics.
    Pipelines explicitly log what they produce.
    """

    def __init__(
        self,
        run_id: str,
        save_dir: str,
        artifact_policy: Literal["none", "metrics", "training"],
        enable_plots: bool = False,
    ) -> None:
        self.run_id = run_id
        self.save_dir = save_dir
        self.artifact_policy = artifact_policy
        self.enable_plots = enable_plots

        self._lock = Lock()

        # ---------------- Directories ----------------
        os.makedirs(self.save_dir, exist_ok=True)

        # ---------------- Core logs ----------------
        self.metrics_jsonl = os.path.join(self.save_dir, "metrics.jsonl")
        self.artifacts_jsonl = os.path.join(self.save_dir, "artifacts.jsonl")

        # ---------------- Training artifacts ----------------
        self.epoch_csv = os.path.join(self.save_dir, "epoch_metrics.csv")
        self.fold_csv = os.path.join(self.save_dir, "fold_metrics.csv")
        self.trial_csv = os.path.join(self.save_dir, "trial_metrics.csv")

        self.plots_dir = os.path.join(self.save_dir, "plots")
        if self.artifact_policy == "training":
            os.makedirs(self.plots_dir, exist_ok=True)

        # ---------------- Buffers ----------------
        self._epoch_records: List[EpochRecord] = []
        self._fold_records: List[FoldRecord] = []
        self._trial_records: List[TrialRecord] = []

        logger.info(
            "TrainingMonitor initialized | run_id=%s | policy=%s",
            self.run_id,
            self.artifact_policy,
        )

    # ------------------------------------------------------------------
    # Internal writers
    # ------------------------------------------------------------------
    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed writing %s", path, exc_info=exc)

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------
    def log_stage_start(self, stage: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "event": "stage_start",
            "stage": stage,
            "payload": payload or {},
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._append_jsonl(self.metrics_jsonl, event)
        logger.info("[MONITOR] Stage START: %s", stage)

    def log_stage_end(self, stage: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "event": "stage_end",
            "stage": stage,
            "payload": payload or {},
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._append_jsonl(self.metrics_jsonl, event)
        logger.info("[MONITOR] Stage END: %s", stage)

    # ------------------------------------------------------------------
    # Artifact logging (ALL PIPELINES)
    # ------------------------------------------------------------------
    def log_artifact(
        self,
        stage: str,
        artifact_type: str,
        ticker: Optional[str],
        path: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a concrete pipeline artifact.

        Examples
        --------
        - raw_csv
        - clean_csv
        - feature_parquet
        - model_file
        """
        record = {
            "run_id": self.run_id,
            "stage": stage,
            "artifact_type": artifact_type,
            "ticker": ticker,
            "path": path,
            "meta": meta or {},
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._append_jsonl(self.artifacts_jsonl, record)

    # ------------------------------------------------------------------
    # Training metrics (policy-gated)
    # ------------------------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        lr: Optional[float] = None,
        early_stop_triggered: bool = False,
    ) -> None:
        if self.artifact_policy == "none":
            return

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            early_stopped=early_stop_triggered,
            timestamp=_utcnow_iso(),
        )
        with self._lock:
            self._epoch_records.append(record)
            self._append_jsonl(
                self.metrics_jsonl,
                {"event": "epoch", "payload": asdict(record)},
            )

    def log_fold(self, fold_idx: int, val_score: float) -> None:
        if self.artifact_policy == "none":
            return

        record = FoldRecord(fold_idx, val_score, _utcnow_iso())
        with self._lock:
            self._fold_records.append(record)
            self._append_jsonl(
                self.metrics_jsonl,
                {"event": "fold", "payload": asdict(record)},
            )

    def log_trial(self, trial_idx: int, best_val: float, params: Dict[str, Any]) -> None:
        if self.artifact_policy != "training":
            return

        record = TrialRecord(trial_idx, best_val, params, _utcnow_iso())
        with self._lock:
            self._trial_records.append(record)
            self._append_jsonl(
                self.metrics_jsonl,
                {"event": "trial", "payload": asdict(record)},
            )

    # ------------------------------------------------------------------
    # Finalization (training-only)
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """
        Persist buffered CSVs / plots (training stages only).
        """
        if self.artifact_policy in ("metrics", "training"):
            if self._epoch_records:
                pd.DataFrame([asdict(r) for r in self._epoch_records]).to_csv(
                    self.epoch_csv, index=False
                )
            if self._fold_records:
                pd.DataFrame([asdict(r) for r in self._fold_records]).to_csv(
                    self.fold_csv, index=False
                )

        if self.artifact_policy == "training" and self._trial_records:
            pd.DataFrame([asdict(r) for r in self._trial_records]).to_csv(
                self.trial_csv, index=False
            )

        if self.artifact_policy == "training" and self.enable_plots:
            self._plot_loss_curve()

        logger.info("TrainingMonitor finalized | run_id=%s", self.run_id)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_loss_curve(self) -> None:
        if not self._epoch_records:
            return
        try:
            df = pd.DataFrame([asdict(r) for r in self._epoch_records])
            out = os.path.join(self.plots_dir, f"{self.run_id}_loss_curve.png")
            plt.figure(figsize=(10, 5))
            plt.plot(df["epoch"], df["train_loss"], label="train")
            plt.plot(df["epoch"], df["val_loss"], label="val")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
        except Exception as exc:
            logger.error("Loss curve plotting failed", exc_info=exc)
