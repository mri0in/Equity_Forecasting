# src/dashboard/history_manager.py

import json
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class EquityHistory:
    def __init__(self, filepath: str = "datalake/data/raw"):
        """
        Initialize the equity history manager.
        Loads existing history from JSON file if it exists.
        """
        
        self.filepath = Path(filepath)
        self.history: List[str] = []
        self.pd = pd
        self.yf = yf

        if not self.filepath.exists():
            print(f"Warning: History path does not exist: {self.filepath}")

        elif self.filepath.is_dir():
            print(f"Warning: Expected history file but got directory: {self.filepath}")

        else:
            try:
                with self.filepath.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    self.history = [str(e).upper() for e in data]
                else:
                    print(
                        f"Warning: History file content invalid (expected list): {self.filepath}"
                    )

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in history file {self.filepath}: {e}")

            except Exception as e:
                print(f"Warning: Could not load history file {self.filepath}: {e}")

    def get_equity_data(self, equity: str) -> pd.DataFrame:
        """Fetch or retrieve equity data as a dataframe."""
        equity = equity.upper().strip()
        csv_file = self.filepath / f"{equity}.csv"
        
        # Check if file exists and is fresh (less than 1 day old)
        if csv_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
            if file_age < timedelta(days=1):
                return self.pd.read_csv(csv_file)
        
        # Download fresh data from yfinance
        try:
            df = self.yf.download(equity, period="1y", progress=False, timeout=120)
            df.to_csv(csv_file)
            return df
        except Exception as e:
            print(f"Error downloading {equity}: {e}")
            return csv_file

  
