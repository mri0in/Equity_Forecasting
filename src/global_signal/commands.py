"""
Global Signal Commands

These are human-invoked commands intended to be run from:
- Python REPL
- Notebook
- Internal admin runner
- CI / job orchestration

NOT from adapters or pipelines.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import json
import shutil


def promote_latest_global_signal(
    metric: str = "val_score",
    higher_is_better: bool = True,
    notes: Optional[str] = None,
) -> str:
    """
    Promote the best available global signal based on run metadata.

    Selection logic:
    - Scans completed runs
    - Selects best run using a metric
    - Promotes its global_signal.npy

    Returns
    -------
    str
        Promoted run_id
    """
    runs_dir = Path("c:/Users/Admin/Equity_Forecasting/datalake/runs") # Adjusted for local environment
    if not runs_dir.exists():
        raise RuntimeError("No runs directory found")

    candidates = []

    for run_dir in runs_dir.iterdir():
        metrics_path = run_dir / "metrics.json"
        signal_path = run_dir / "inference" / "global_signal.npy"

        if not metrics_path.exists() or not signal_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        if metric not in metrics:
            continue

        candidates.append(
            {
                "run_id": run_dir.name,
                "score": metrics[metric],
                "signal_path": signal_path,
            }
        )

    if not candidates:
        raise RuntimeError("No promotable global signals found")

    candidates.sort(
        key=lambda x: x["score"],
        reverse=higher_is_better,
    )

    best = candidates[0]

    _promote_signal(
        best["signal_path"],
        best["run_id"],
        notes or f"Auto-selected best run by {metric}",
    )

    return best["run_id"]


def _promote_signal(
    src_signal: Path,
    run_id: str,
    notes: str,
) -> None:
    dst_dir = Path("datalake") / "global_signal"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_signal = dst_dir / "active.npy"
    shutil.copy2(src_signal, dst_signal)

    metadata = {
        "run_id": run_id,
        "source_path": str(src_signal),
        "promoted_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "notes": notes,
    }

    with open(dst_dir / "active.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(dst_dir / "history.jsonl", "a") as f:
        f.write(json.dumps(metadata) + "\n")
