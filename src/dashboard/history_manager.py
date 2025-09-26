# src/dashboard/history_manager.py

import json
from pathlib import Path
from typing import List

class EquityHistory:
    def __init__(self, filepath: str = "datalake/data/raw/equity_history.json"):
        """
        Initialize the equity history manager.
        Loads existing history from JSON file if it exists.
        """
        self.filepath = Path(filepath)
        self.history: List[str] = []

        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.history = [str(e).upper() for e in data]
            except Exception as e:
                print(f"Warning: Could not load history file. {e}")

    def _save_history(self):
        """Save the current history to JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Warning: Could not save history file. {e}")

    def get_history(self) -> List[str]:
        """Return the list of historical equities."""
        return self.history

    def add_equity(self, equity: str) -> None:
        """Add a new equity to history if not already present."""
        equity = equity.upper().strip()
        if equity and equity not in self.history:
            self.history.append(equity)
            self._save_history()

    def clear_history(self) -> None:
        """Clear the equity history both in memory and file."""
        with open(self.filepath, "w") as f:
            json.dump([], f, indent=4)
