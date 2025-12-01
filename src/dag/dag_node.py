# src/dag/dag_node.py
"""
dag_node.py
Defines a single node (step) in the DAG execution graph.
"""

from __future__ import annotations
from typing import Callable, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DAGNode:
    """
    Represents a single step in the DAG.

    Each node wraps:
    - a unique name
    - a callable (function or method) implementing this step
    - dependencies (list of other node names)
    - status and execution output
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        dependencies: List[str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("DAGNode name must be a non-empty string.")

        if not callable(func):
            raise ValueError("func must be a callable.")

        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.output = None
        self.executed = False

        logger.debug(f"DAGNode initialized: {self.name} "
                     f"with deps={self.dependencies}")

    def execute(self, **kwargs) -> Any:
        """
        Executes the node's function with provided keyword arguments.

        The output is stored internally so downstream nodes may use it.
        """
        logger.info(f"Executing DAG step: {self.name}")

        try:
            self.output = self.func(**kwargs)
            self.executed = True
            logger.info(f"Step completed: {self.name}")
            return self.output

        except Exception as exc:
            logger.error(
                f"Execution failed for step '{self.name}': {exc}",
                exc_info=True
            )
            raise
