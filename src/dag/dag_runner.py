# src/dag/dag_runner.py
"""
DAG Runner Module
-----------------
Executes DAG nodes in correct topological order, applies retry logic,
updates state markers, integrates monitoring hooks, and checks
dependencies before running each stage.

This orchestrator is intentionally thin: it delegates business logic to 
src/dag/stages.py, and state handling to src/dag/state_manager.py.
"""

from typing import Callable, Dict, Any
import logging
import time

from .dag_graph import DAGGraph
from .state_manager import StateManager
from .dag_stages import PipelineStages
from .dag_errors import DependencyError, StageFailedError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DAGRunner:
    """
    Coordinates execution of an entire DAG-based pipeline.
    Handles:
        - Topological traversal
        - Retry logic
        - RUNNING / SUCCESS / FAILED state markers
        - Stage execution using PipelineStages
        - Monitoring hooks (if integrated externally)
    """

    def __init__(
        self,
        dag: DAGGraph,
        state_manager: StateManager,
        stages: PipelineStages,
        max_retries: int = 3,
        retry_delay_sec: int = 10
    ) -> None:
        """
        Parameters
        ----------
        dag : DAGGraph
            The DAG containing pipeline nodes & dependencies.

        state_manager : StateManager
            Tracks RUNNING, SUCCESS, FAILED markers.

        stages : PipelineStages
            Contains callable wrappers for each pipeline stage.

        max_retries : int
            How many times to retry a failing stage.

        retry_delay_sec : int
            Delay before retrying a failed stage.
        """
        self.dag = dag
        self.state = state_manager
        self.stages = stages

        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

    # ----------------------------------------------------------------------

    def run(self) -> None:
        """
        Executes the DAG in topological order, ensuring dependencies
        are satisfied before each stage, and applying retry logic.
        """
        logger.info("Starting DAG execution...")

        execution_order = self.dag.topological_sort()

        for node in execution_order:

            # Skip if already SUCCESS
            if self.state.is_success(node):
                logger.info(f"[SKIP] Stage already SUCCESS: {node}")
                continue

            # Dependency check
            missing = [
                dep for dep in self.dag.get_dependencies(node)
                if not self.state.is_success(dep)
            ]
            if missing:
                logger.error(
                    f"[DEPENDENCY ERROR] Stage `{node}` cannot run. "
                    f"Missing dependencies: {missing}"
                )
                raise DependencyError(node, missing)

            # Recover if previously RUNNING
            if self.state.is_running(node):
                logger.warning(
                    f"Stage {node} is marked RUNNING from previous crash; re-running..."
                )

            self.state.mark_running(node)

            # Execute stage
            success = self._execute_node(node)

            if success:
                self.state.mark_success(node)
                logger.info(f"[SUCCESS] Stage completed: {node}")
            else:
                self.state.mark_failed(node)
                logger.error(f"[FAILED] Stage failed permanently: {node}")
                break

        logger.info("DAG execution completed.")

    # ----------------------------------------------------------------------

    def _execute_node(self, node: str) -> bool:
        """
        Executes a single pipeline stage with retry logic.

        Parameters
        ----------
        node : str
            Name of the DAG node (stage).

        Returns
        -------
        bool
            True if stage succeeds, False otherwise.
        """
        logger.info(f"Executing stage: {node}")

        # Map node name → actual method in PipelineStages
        stage_fn = self._resolve_stage_function(node)

        for attempt in range(1, self.max_retries + 2):  # +1 final attempt
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries + 1} for stage: {node}")
                stage_fn()  # Execute actual stage logic
                return True

            except Exception as e:
                logger.exception(f"Error in stage `{node}` on attempt {attempt}: {e}")

                if attempt <= self.max_retries:
                    logger.info(
                        f"Retrying stage `{node}` after {self.retry_delay_sec} sec..."
                    )
                    time.sleep(self.retry_delay_sec)
                else:
                    logger.error(
                        f"Stage `{node}` exhausted all retries and will be marked FAILED."
                    )
                    return False

    # ----------------------------------------------------------------------

    def _resolve_stage_function(self, node: str) -> Callable[[], Any]:
        """
        Maps a DAG node name to its corresponding PipelineStages function.
        Ensures all nodes are explicitly registered.

        Raises KeyError if node is unknown.
        """
        stage_map: Dict[str, Callable[[], Any]] = {
            "ingestion": self.stages.run_ingestion,
            "preprocessing": self.stages.run_preprocessing,
            "feature_generation": self.stages.run_feature_generation,
            "training": self.stages.run_training,
            "optimization": self.stages.run_optimizer,
            "walkforward": self.stages.run_walkforward,
            "ensembling": self.stages.run_ensembling,
            "forecasting": self.stages.run_forecasting,
        }

        if node not in stage_map:
            raise KeyError(f"No stage function mapped for DAG node `{node}`")

        return stage_map[node]
