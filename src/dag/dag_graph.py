# src/dag/dag_graph.py
"""
dag_graph.py

DAG management utilities.

Provides:
- DAGGraph: class to register DAGNode instances (by name), validate the graph,
  detect cycles, and produce a topological ordering suitable for execution.

Design principles:
- Pure OOP: DAGGraph manages nodes and edges.
- Defensive input validation: clear errors for missing nodes / cycles.
- Logging at INFO/DEBUG levels to aid observability during pipeline runs.
- Small, well-documented public surface to be used by dag_runner.py.
"""

from __future__ import annotations
from typing import Dict, List, Set, Iterable, Optional
import logging

from src.dag.dag_node import DAGNode

logger = logging.getLogger(__name__)


class DAGError(Exception):
    """Base class for DAG related errors."""


class CycleDetectedError(DAGError):
    """Raised when a cycle is detected in the graph."""


class NodeNotFoundError(DAGError):
    """Raised when a referenced node name does not exist in the graph."""


class DAGGraph:
    """
    Directed Acyclic Graph container for DAGNode objects.

    Responsibilities:
    - Register DAGNode instances.
    - Validate that declared dependencies reference existing nodes.
    - Detect cycles and raise CycleDetectedError if found.
    - Produce a topological ordering of nodes for execution.

    Usage:
        dag = DAGGraph()
        dag.add_node(node)           # node is DAGNode
        order = dag.topological_sort()
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, DAGNode] = {}
        # adjacency: node -> set(children)
        self._adj: Dict[str, Set[str]] = {}
        # reverse adjacency: node -> set(parents)
        self._rev_adj: Dict[str, Set[str]] = {}
        logger.debug("Initialized empty DAGGraph")

    def add_node(self, node: DAGNode) -> None:
        """
        Add a DAGNode to the graph.

        Raises:
            ValueError: if node name already exists.
        """
        if node.name in self._nodes:
            msg = f"Node already registered: {node.name}"
            logger.error(msg)
            raise ValueError(msg)

        # register node
        self._nodes[node.name] = node
        self._adj.setdefault(node.name, set())
        self._rev_adj.setdefault(node.name, set())

        # ensure dependencies exist in structures (they may be added later)
        for dep in node.dependencies:
            self._adj.setdefault(dep, set())
            self._rev_adj.setdefault(dep, set())
            self._adj[dep].add(node.name)
            self._rev_adj[node.name].add(dep)

        logger.info("Added node to DAG: %s (deps=%s)", node.name, node.dependencies)

    def get_node(self, name: str) -> DAGNode:
        """Return the DAGNode for a given name, or raise NodeNotFoundError."""
        if name not in self._nodes:
            logger.error("Requested node not found: %s", name)
            raise NodeNotFoundError(f"Node not found: {name}")
        return self._nodes[name]

    def nodes(self) -> Iterable[DAGNode]:
        """Iterate over registered DAGNode objects (unspecified order)."""
        return iter(self._nodes.values())

    def children(self, name: str) -> List[str]:
        """Return list of direct children node names."""
        if name not in self._nodes:
            raise NodeNotFoundError(name)
        return list(self._adj.get(name, set()))

    def parents(self, name: str) -> List[str]:
        """Return list of direct parent node names."""
        if name not in self._nodes:
            raise NodeNotFoundError(name)
        return list(self._rev_adj.get(name, set()))

    def validate_dependencies(self) -> None:
        """
        Ensure that all dependency names referenced by nodes exist in the graph.

        Raises:
            NodeNotFoundError: if a dependency name is missing.
        """
        missing: List[str] = []
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    missing.append(f"{node.name} -> {dep}")
        if missing:
            msg = f"Missing dependency nodes: {missing}"
            logger.error(msg)
            raise NodeNotFoundError(msg)
        logger.debug("All dependencies validated successfully")

    def _compute_indegree(self) -> Dict[str, int]:
        """Return a mapping node_name -> indegree."""
        indegree: Dict[str, int] = {name: 0 for name in self._nodes}
        for src, children in self._adj.items():
            for child in children:
                if child in indegree:
                    indegree[child] += 1
        logger.debug("Computed indegree map: %s", indegree)
        return indegree

    def topological_sort(self) -> List[str]:
        """
        Return a topologically sorted list of node names using Kahn's algorithm.

        Raises:
            CycleDetectedError: if a cycle exists in the graph.
            NodeNotFoundError: if dependencies refer to non-existent nodes.
        """
        # Ensure references are valid first
        self.validate_dependencies()

        indegree = self._compute_indegree()
        # start with nodes that have indegree 0
        queue: List[str] = [n for n, deg in indegree.items() if deg == 0]
        order: List[str] = []
        logger.debug("Starting topological sort with initial queue: %s", queue)

        # Kahn's algorithm
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in list(self._adj.get(node, set())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        # If order doesn't contain all nodes, there is a cycle
        if len(order) != len(self._nodes):
            msg = "Cycle detected in DAG; topological sort incomplete"
            logger.error(msg)
            raise CycleDetectedError(msg)

        logger.info("Topological sort completed. Order: %s", order)
        return order

    def detect_cycle(self) -> bool:
        """
        Convenience method to check for cycles without raising.

        Returns:
            True if a cycle exists, False otherwise.
        """
        try:
            self.topological_sort()
            return False
        except CycleDetectedError:
            return True

    def export_adjacency(self) -> Dict[str, List[str]]:
        """Return a shallow copy of adjacency as a serializable dict."""
        return {k: list(v) for k, v in self._adj.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._nodes
