# src/dag/state_manager.py
"""
State Manager for DAG Pipeline
------------------------------

Manages the execution state of each pipeline stage. 
Supports marking stages as RUNNING, SUCCESS, or FAILED,
checking stage status, and persisting state to a JSON file 
for checkpointing and resuming the pipeline.
"""

import json
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StateManager:
    """
    Tracks and persists the state of each DAG node.

    Attributes
    ----------
    state_file : str
        Path to the JSON file where state is stored.
    states : Dict[str, str]
        In-memory dictionary mapping stage names to their current status.
        Status can be 'RUNNING', 'SUCCESS', 'FAILED'.
    """

    VALID_STATES = {"RUNNING", "SUCCESS", "FAILED"}

    def __init__(self, state_file: str) -> None:
        """
        Initialize the StateManager with a JSON file for persistence.

        Parameters
        ----------
        state_file : str
            Path to the JSON file to save/load pipeline states.
        """
        self.state_file = state_file
        self.states: Dict[str, str] = {}

        self._load_state()

    # ----------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load state from the JSON file if it exists; otherwise initialize empty state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.states = json.load(f)
                logger.info(f"Loaded existing pipeline state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state file {self.state_file}: {e}")
                self.states = {}
        else:
            self.states = {}
            logger.info(f"No existing state file found. Initialized empty state.")

    def _save_state(self) -> None:
        """Persist the current in-memory state to the JSON file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.states, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    # ----------------------------------------------------------------------

    def mark_running(self, stage: str) -> None:
        """Mark a stage as RUNNING."""
        self.states[stage] = "RUNNING"
        self._save_state()
        logger.info(f"Stage '{stage}' marked as RUNNING.")

    def mark_success(self, stage: str) -> None:
        """Mark a stage as SUCCESS."""
        self.states[stage] = "SUCCESS"
        self._save_state()
        logger.info(f"Stage '{stage}' marked as SUCCESS.")

    def mark_failed(self, stage: str) -> None:
        """Mark a stage as FAILED."""
        self.states[stage] = "FAILED"
        self._save_state()
        logger.info(f"Stage '{stage}' marked as FAILED.")

    # ----------------------------------------------------------------------

    def is_success(self, stage: str) -> bool:
        """Return True if the stage has previously completed successfully."""
        return self.states.get(stage) == "SUCCESS"

    def is_running(self, stage: str) -> bool:
        """Return True if the stage is currently marked as running."""
        return self.states.get(stage) == "RUNNING"

    def is_failed(self, stage: str) -> bool:
        """Return True if the stage has previously failed."""
        return self.states.get(stage) == "FAILED"

    def reset_stage(self, stage: str) -> None:
        """Remove any previous state for the given stage."""
        if stage in self.states:
            del self.states[stage]
            self._save_state()
            logger.info(f"Stage '{stage}' state has been reset.")

    def reset_all(self) -> None:
        """Reset the states of all stages."""
        self.states.clear()
        self._save_state()
        logger.info("All stage states have been reset.")
