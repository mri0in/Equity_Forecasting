# src/dag/dag_errors.py
"""
Custom Exceptions for DAG Pipeline
----------------------------------

Defines exceptions specific to DAG execution for clear
error handling, retries, and logging.
"""


class DAGError(Exception):
    """
    Base exception for all DAG-related errors.
    """
    pass


class StageFailedError(DAGError):
    """
    Raised when a specific stage fails during execution.
    Allows DAG runner to catch and retry or mark failure.
    """

    def __init__(self, stage_name: str, message: str = ""):
        self.stage_name = stage_name
        self.message = message or f"Stage '{stage_name}' failed."
        super().__init__(self.message)


class DependencyError(DAGError):
    """
    Raised when a stage's dependencies are not met.
    """
    def __init__(self, stage_name: str, missing_dependencies: list[str]):
        self.stage_name = stage_name
        self.missing_dependencies = missing_dependencies
        message = (
            f"Stage '{stage_name}' cannot run. "
            f"Missing dependencies: {missing_dependencies}"
        )
        super().__init__(message)


class StateManagerError(DAGError):
    """
    Raised for issues related to the pipeline state manager,
    such as failure to read/write state file.
    """
    pass
