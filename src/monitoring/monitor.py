# src/monitor/monitor.py
"""
Training monitor module.

Provides TrainingMonitor: a lightweight file-based monitor for training runs.
- Writes incremental JSONLines logs for near-real-time dashboard polling.
- Produces CSV summaries and PNG plots at finalize().
- Tracks epoch-level, fold-level and trial-level metrics, plus arbitrary events.

Usage (example):
    monitor = TrainingMonitor(run_id="RELIANCE_20251125", save_dir="datalake/experiments/monitor/RELIANCE", visualize=False)
    monitor.start_session(model="LSTM_v1", config={"lr":1e-3})
    for epoch in range(epochs):
        monitor.log_epoch(epoch=epoch, train_loss=..., val_loss=..., lr=..., early_stop=False)
    monitor.log_fold(fold_idx=0, val_score=0.012)
    monitor.log_trial(trial_idx=0, best_val=0.011, params={"lr":1e-3})
    monitor.finalize()

Design notes:
- File outputs:
    <save_dir>/metrics.jsonl     (append-only stream)
    <save_dir>/epoch_metrics.csv
    <save_dir>/fold_metrics.csv
    <save_dir>/trial_metrics.csv
    <save_dir>/plots/<loss_curve>.png
    <save_dir>/training_summary.json
- All timestamps are UTC ISO8601.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    lr: Optional[float]
    early_stopped: bool
    timestamp: str


@dataclass
class FoldRecord:
    fold_idx: int
    val_score: float
    timestamp: str


@dataclass
class TrialRecord:
    trial_idx: int
    best_val: float
    params: Dict[str, Any]
    timestamp: str


class TrainingMonitor:
    """
    TrainingMonitor writes epoch/fold/trial events to disk and creates summary plots.

    Parameters
    ----------
    run_id : str
        Unique identifier for the training run (e.g. "RELIANCE_20251125_01").
    save_dir : str
        Directory where logs, CSVs, and plots will be written.
    visualize : bool
        If True, attempt to show plots via plt.show() when finalize() is called.
        (Default False — suitable for headless servers.)
    flush_every : int
        After how many epoch logs to flush the JSONL file. Default 1 (immediate).
    """

    def __init__(self, run_id: str, save_dir: str, visualize: bool = False, flush_every: int = 1) -> None:
        self.run_id = run_id
        self.save_dir = save_dir
        self.visualize = visualize
        self.flush_every = max(1, int(flush_every))
        self._lock = threading.Lock()

        # internal buffers
        self.epoch_records: List[EpochRecord] = []
        self.fold_records: List[FoldRecord] = []
        self.trial_records: List[TrialRecord] = []
        self.events: List[Dict[str, Any]] = []

        # file paths
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_jsonl = os.path.join(self.save_dir, "metrics.jsonl")
        self.epoch_csv = os.path.join(self.save_dir, "epoch_metrics.csv")
        self.fold_csv = os.path.join(self.save_dir, "fold_metrics.csv")
        self.trial_csv = os.path.join(self.save_dir, "trial_metrics.csv")
        self.plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.summary_json = os.path.join(self.save_dir, "training_summary.json")

        # counters
        self._epoch_log_count = 0

        # session meta
        self.session_info: Dict[str, Any] = {"run_id": self.run_id, "started_at": _utcnow_iso()}
        logger.info(f"TrainingMonitor initialized for run_id={self.run_id} at {self.save_dir}")

    # -------------------------
    # Session lifecycle
    # -------------------------
    def start_session(self, model: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Record session-level metadata.
        """
        with self._lock:
            self.session_info.update({"model": model, "config": config or {}, "session_started": _utcnow_iso()})
            self._append_jsonl({"event": "session_start", "payload": self.session_info})
            logger.info(f"Monitoring session started for {self.run_id} (model={model})")

    def end_session(self) -> None:
        """
        Finalize session (alias for finalize()).
        """
        self.session_info["session_ended"] = _utcnow_iso()
        self._append_jsonl({"event": "session_end", "payload": {"ended_at": self.session_info["session_ended"]}})
        logger.info(f"Monitoring session ended for {self.run_id}")

    # -------------------------
    # Logging helpers
    # -------------------------
    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        """
        Append a JSON line to the metrics.jsonl file in a thread-safe way.
        """
        try:
            with self._lock:
                with open(self.metrics_jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to metrics.jsonl: {e}")

    # -------------------------
    # Core logging methods
    # -------------------------
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        lr: Optional[float] = None,
        early_stop_triggered: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log epoch-level metrics.

        Parameters
        ----------
        epoch : int
        train_loss : float | None
        val_loss : float | None
        lr : float | None
        early_stop_triggered : bool
        extra : dict | None - additional arbitrary metrics to include
        """
        rec = EpochRecord(
            epoch=int(epoch),
            train_loss=None if train_loss is None else float(train_loss),
            val_loss=None if val_loss is None else float(val_loss),
            lr=None if lr is None else float(lr),
            early_stopped=bool(early_stop_triggered),
            timestamp=_utcnow_iso(),
        )
        with self._lock:
            self.epoch_records.append(rec)
            self._epoch_log_count += 1

            payload = {"event": "epoch", "payload": asdict(rec)}
            if extra:
                payload["payload"]["extra"] = extra
            self._append_jsonl(payload)

            # update epoch CSV on the fly
            try:
                need_header = not os.path.exists(self.epoch_csv)
                with open(self.epoch_csv, "a", newline="", encoding="utf-8") as csvf:
                    writer = csv.writer(csvf)
                    if need_header:
                        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "early_stopped", "timestamp"])
                    writer.writerow([rec.epoch, rec.train_loss, rec.val_loss, rec.lr, rec.early_stopped, rec.timestamp])
            except Exception as e:
                logger.error(f"Failed to write epoch CSV: {e}")

            # periodic flush (already written to disk each call, keeping for compliance)
            if self._epoch_log_count % self.flush_every == 0:
                # ensure file is synced; we opened+closed it so it's on disk; we still notify
                logger.debug(f"Epoch log flushed (count={self._epoch_log_count})")

    def log_fold(self, fold_idx: int, val_score: float, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log per-fold validation metric (used by Walk-Forward).
        """
        rec = FoldRecord(fold_idx=int(fold_idx), val_score=float(val_score), timestamp=_utcnow_iso())
        with self._lock:
            self.fold_records.append(rec)
            self._append_jsonl({"event": "fold", "payload": asdict(rec)})
            # append CSV
            try:
                need_header = not os.path.exists(self.fold_csv)
                with open(self.fold_csv, "a", newline="", encoding="utf-8") as csvf:
                    writer = csv.writer(csvf)
                    if need_header:
                        writer.writerow(["fold_idx", "val_score", "timestamp"])
                    writer.writerow([rec.fold_idx, rec.val_score, rec.timestamp])
            except Exception as e:
                logger.error(f"Failed to write fold CSV: {e}")

    def log_trial(self, trial_idx: int, best_val: float, params: Dict[str, Any]) -> None:
        """
        Log Optuna/Opt-like trial summary.
        """
        rec = TrialRecord(trial_idx=int(trial_idx), best_val=float(best_val), params=params or {}, timestamp=_utcnow_iso())
        with self._lock:
            self.trial_records.append(rec)
            self._append_jsonl({"event": "trial", "payload": asdict(rec)})
            # append CSV
            try:
                need_header = not os.path.exists(self.trial_csv)
                with open(self.trial_csv, "a", newline="", encoding="utf-8") as csvf:
                    writer = csv.writer(csvf)
                    if need_header:
                        writer.writerow(["trial_idx", "best_val", "params_json", "timestamp"])
                    writer.writerow([rec.trial_idx, rec.best_val, json.dumps(rec.params, default=str), rec.timestamp])
            except Exception as e:
                logger.error(f"Failed to write trial CSV: {e}")

    def log_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Generic event logger. Useful to record early-stop, saving checkpoints, etc.
        """
        event = {"event": name, "payload": payload or {}, "timestamp": _utcnow_iso()}
        with self._lock:
            self.events.append(event)
            self._append_jsonl(event)
            logger.info(f"Monitor event logged: {name}")

    # -------------------------
    # Finalize & plotting
    # -------------------------
    def finalize(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Finalize monitoring: create summary JSON, plots, and return a dict of artifact paths.

        Returns
        -------
        Dict[str, Any]
            {
                "metrics_jsonl": <path>,
                "epoch_csv": <path>,
                "fold_csv": <path>,
                "trial_csv": <path>,
                "plots": {"loss_curve": <path>, "fold_perf": <path>, "optuna_progress": <path>},
                "summary_json": <path>
            }
        """
        self.session_info["finalized_at"] = _utcnow_iso()
        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "session_info": self.session_info,
            "metrics_jsonl": self.metrics_jsonl,
            "epoch_csv": self.epoch_csv,
            "fold_csv": self.fold_csv,
            "trial_csv": self.trial_csv,
            "plots": {},
            "summary_json": self.summary_json,
        }

        # Generate DataFrames
        try:
            epoch_df = pd.DataFrame([asdict(r) for r in self.epoch_records]) if self.epoch_records else pd.DataFrame()
            fold_df = pd.DataFrame([asdict(r) for r in self.fold_records]) if self.fold_records else pd.DataFrame()
            trial_df = pd.DataFrame([asdict(r) for r in self.trial_records]) if self.trial_records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to assemble DataFrames from records: {e}")
            epoch_df = pd.DataFrame()
            fold_df = pd.DataFrame()
            trial_df = pd.DataFrame()

        # Save DataFrames as CSV (overwrite to ensure canonical)
        try:
            if not epoch_df.empty:
                epoch_df.to_csv(self.epoch_csv, index=False)
            if not fold_df.empty:
                fold_df.to_csv(self.fold_csv, index=False)
            if not trial_df.empty:
                trial_df.to_csv(self.trial_csv, index=False)
        except Exception as e:
            logger.error(f"Failed to write summary CSVs: {e}")

        # Create plots
        plots_out: Dict[str, str] = {}
        try:
            if not epoch_df.empty:
                p1 = os.path.join(self.plots_dir, f"{self.run_id}_loss_curve.png")
                self._plot_loss_curve(epoch_df, out_path=p1)
                plots_out["loss_curve"] = p1

            if not fold_df.empty:
                p2 = os.path.join(self.plots_dir, f"{self.run_id}_fold_perf.png")
                self._plot_fold_perf(fold_df, out_path=p2)
                plots_out["fold_perf"] = p2

            if not trial_df.empty:
                p3 = os.path.join(self.plots_dir, f"{self.run_id}_optuna_progress.png")
                self._plot_optuna_progress(trial_df, out_path=p3)
                plots_out["optuna_progress"] = p3
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

        summary["plots"] = plots_out

        # Write summary JSON
        try:
            with open(self.summary_json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write summary JSON: {e}")

        # Optionally show plots (useful during interactive runs)
        if self.visualize:
            try:
                for p in plots_out.values():
                    if os.path.exists(p):
                        img = plt.imread(p)
                        plt.figure(figsize=(10, 4))
                        plt.imshow(img)
                        plt.axis("off")
                        plt.show()
            except Exception as e:
                logger.error(f"Failed to display plots interactively: {e}")

        logger.info(f"TrainingMonitor finalized. Artifacts: {summary}")
        return summary

    # -------------------------
    # Plotting primitives
    # -------------------------
    def _plot_loss_curve(self, epoch_df: pd.DataFrame, out_path: str) -> None:
        """
        Plot train vs validation loss with early-stop markers.
        """
        try:
            plt.figure(figsize=(10, 5))
            if "epoch" in epoch_df.columns and "train_loss" in epoch_df.columns:
                plt.plot(epoch_df["epoch"], epoch_df["train_loss"], label="train_loss")
            if "epoch" in epoch_df.columns and "val_loss" in epoch_df.columns:
                plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="val_loss")
            # mark early stop epoch(s)
            early_epochs = epoch_df[epoch_df["early_stopped"] == True]["epoch"].tolist() if "early_stopped" in epoch_df.columns else []
            for e in early_epochs:
                plt.axvline(e, linestyle="--", linewidth=0.8)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{self.run_id} — Loss Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create loss curve plot: {e}")

    def _plot_fold_perf(self, fold_df: pd.DataFrame, out_path: str) -> None:
        """
        Bar chart of fold validation scores.
        """
        try:
            plt.figure(figsize=(8, 4))
            if "fold_idx" in fold_df.columns and "val_score" in fold_df.columns:
                plt.bar(fold_df["fold_idx"].astype(str), fold_df["val_score"])
                plt.xlabel("Fold")
                plt.ylabel("Validation Score")
                plt.title(f"{self.run_id} — Fold Validation Scores")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create fold perf plot: {e}")

    def _plot_optuna_progress(self, trial_df: pd.DataFrame, out_path: str) -> None:
        """
        Plot best_val per trial (trial_idx on x-axis).
        """
        try:
            plt.figure(figsize=(8, 4))
            if "trial_idx" in trial_df.columns and "best_val" in trial_df.columns:
                sorted_df = trial_df.sort_values("trial_idx")
                plt.plot(sorted_df["trial_idx"], sorted_df["best_val"], marker="o")
                plt.xlabel("Trial Index")
                plt.ylabel("Best Validation Score")
                plt.title(f"{self.run_id} — Optuna Trial Progress")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create optuna progress plot: {e}")
