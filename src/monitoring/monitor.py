# src/monitoring/monitor.py
"""
Training monitor module.

Provides TrainingMonitor: a lightweight file-based monitor for pipeline runs.
- Writes incremental JSONLines logs for near-real-time dashboard polling.
- Persists run-level artifacts in a deterministic artifacts.json file.
- Produces CSV summaries and PNG plots at finalize() (for training stages).

Artifact Policies:
------------------
- "none"     : metrics.jsonl only
- "metrics"  : metrics.jsonl + CSV artifacts
- "training" : metrics.jsonl + CSV + plots

"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Literal

import logging
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
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------
# Data Records (Immutable Contracts)
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
    Unified monitor for all pipeline stages.

    Parameters
    ----------
    run_id : str
        Globally unique run identifier.
    save_dir : str
        Root directory where all artifacts will be written.
    artifact_policy : {"none", "metrics", "training"}
        Controls which artifacts are persisted.
    enable_plots : bool
        Whether plotting is enabled (only relevant for training policy).

    Notes
    -----
    - metrics.jsonl is ALWAYS written (append-only)
    - CSVs and plots are written only when policy allows
    - No file is overwritten across stages
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

        # ------------------------------------------------------------------
        # Buffers (in-memory, flushed explicitly)
        # ------------------------------------------------------------------
        self._epoch_records: List[EpochRecord] = []
        self._fold_records: List[FoldRecord] = []
        self._trial_records: List[TrialRecord] = []
        self._events: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Artifact paths (stable contract)
        # ------------------------------------------------------------------
        os.makedirs(self.save_dir, exist_ok=True)

        self.metrics_jsonl: str = os.path.join(self.save_dir, "metrics.jsonl")
        self.epoch_csv: str = os.path.join(self.save_dir, "epoch_metrics.csv")
        self.fold_csv: str = os.path.join(self.save_dir, "fold_metrics.csv")
        self.trial_csv: str = os.path.join(self.save_dir, "trial_metrics.csv")

        self.plots_dir: str = os.path.join(self.save_dir, "plots")
        self.artifact_manifest: str = os.path.join(self.save_dir, "artifacts.json")

        if self.artifact_policy == "training":
            os.makedirs(self.plots_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Session metadata
        # ------------------------------------------------------------------
        self._session_meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "artifact_policy": self.artifact_policy,
            "started_at": _utcnow_iso(),
        }

        logger.info(
            "TrainingMonitor initialized | run_id=%s | policy=%s | plots=%s",
            self.run_id,
            self.artifact_policy,
            self.enable_plots,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        """Append a single JSON record to metrics.jsonl."""
        try:
            with open(self.metrics_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to append metrics.jsonl", exc_info=exc)

    # ------------------------------------------------------------------
    # Stage lifecycle logging
    # ------------------------------------------------------------------
    def log_stage_start(self, stage_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "event": "stage_start",
            "stage": stage_name,
            "payload": payload or {},
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._events.append(event)
            self._append_jsonl(event)
        logger.info("[MONITOR] Stage START: %s", stage_name)

    def log_stage_end(self, stage_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "event": "stage_end",
            "stage": stage_name,
            "payload": payload or {},
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._events.append(event)
            self._append_jsonl(event)
        logger.info("[MONITOR] Stage END: %s", stage_name)

    # ------------------------------------------------------------------
    # Metric logging
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
            self._append_jsonl({"event": "epoch", "payload": asdict(record)})

    def log_fold(self, fold_idx: int, val_score: float) -> None:
        if self.artifact_policy == "none":
            return

        record = FoldRecord(fold_idx, val_score, _utcnow_iso())

        with self._lock:
            self._fold_records.append(record)
            self._append_jsonl({"event": "fold", "payload": asdict(record)})

    def log_trial(self, trial_idx: int, best_val: float, params: Dict[str, Any]) -> None:
        if self.artifact_policy != "training":
            return

        record = TrialRecord(trial_idx, best_val, params, _utcnow_iso())

        with self._lock:
            self._trial_records.append(record)
            self._append_jsonl({"event": "trial", "payload": asdict(record)})

    # ------------------------------------------------------------------
    # Finalization & artifact persistence
    # ------------------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        """
        Flush all buffered artifacts and write artifact manifest.
        """
        logger.info("Finalizing TrainingMonitor | run_id=%s", self.run_id)

        artifacts: Dict[str, Any] = {
            "run_id": self.run_id,
            "artifact_policy": self.artifact_policy,
            "metrics_jsonl": self.metrics_jsonl,
            "epoch_csv": None,
            "fold_csv": None,
            "trial_csv": None,
            "plots": {},
        }

        # ---------------- CSV persistence ----------------
        if self.artifact_policy in ("metrics", "training"):
            if self._epoch_records:
                pd.DataFrame([asdict(r) for r in self._epoch_records]).to_csv(self.epoch_csv, index=False)
                artifacts["epoch_csv"] = self.epoch_csv

            if self._fold_records:
                pd.DataFrame([asdict(r) for r in self._fold_records]).to_csv(self.fold_csv, index=False)
                artifacts["fold_csv"] = self.fold_csv

        if self.artifact_policy == "training" and self._trial_records:
            pd.DataFrame([asdict(r) for r in self._trial_records]).to_csv(self.trial_csv, index=False)
            artifacts["trial_csv"] = self.trial_csv

        # ---------------- Plotting (explicit only) ----------------
        if self.artifact_policy == "training" and self.enable_plots:
            if self._epoch_records:
                loss_plot = os.path.join(self.plots_dir, f"{self.run_id}_loss_curve.png")
                self._plot_loss_curve(
                    pd.DataFrame([asdict(r) for r in self._epoch_records]),
                    loss_plot,
                )
                artifacts["plots"]["loss_curve"] = loss_plot

        # ---------------- Artifact manifest ----------------
        with open(self.artifact_manifest, "w", encoding="utf-8") as fh:
            json.dump(artifacts, fh, indent=2)

        logger.info("TrainingMonitor finalized successfully")
        return artifacts

    # ------------------------------------------------------------------
    # Plotting primitives (training-only)
    # ------------------------------------------------------------------
    def _plot_loss_curve(self, df: pd.DataFrame, out_path: str) -> None:
        """Plot training vs validation loss."""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df["epoch"], df["train_loss"], label="train_loss")
            plt.plot(df["epoch"], df["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception as exc:
            logger.error("Failed to plot loss curve", exc_info=exc)
