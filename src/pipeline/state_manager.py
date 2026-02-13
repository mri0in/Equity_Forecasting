"""Pipeline state manager for tracking task execution status."""

from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateManager:
    """Manage pipeline task execution state and transitions."""
    
    def __init__(self, state_dir: Optional[str] = None):
        """Initialize state manager.
        
        Args:
            state_dir: Directory for storing state files.
                      Defaults to datalake/pipeline_state/
        """
        self.state_dir = Path(state_dir or "datalake/cache/pipeline_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._load_state()
    
    def _get_state_file(self, task_name: str) -> Path:
        """Get state file path for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Path to state file
        """
        safe_name = task_name.replace("/", "_").replace(" ", "_")
        return self.state_dir / f"{safe_name}.json"
    
    def _load_state(self) -> None:
        """Load all existing state files into memory."""
        if not self.state_dir.exists():
            return
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    task_data = json.load(f)
                    task_name = task_data.get("task_name")
                    if task_name:
                        self.tasks[task_name] = task_data
                        logger.debug(f"Loaded state for task: {task_name}")
            except Exception as e:
                logger.warning(f"Failed to load state file {state_file}: {e}")
    
    def _persist_state(self, task_name: str) -> None:
        """Persist task state to disk.
        
        Args:
            task_name: Name of the task
        """
        if task_name not in self.tasks:
            logger.warning(f"Task {task_name} not in memory, skipping persist")
            return
        
        state_file = self._get_state_file(task_name)
        try:
            with open(state_file, 'w') as f:
                json.dump(self.tasks[task_name], f, indent=2, default=str)
                logger.debug(f"Persisted state for task: {task_name}")
        except Exception as e:
            logger.error(f"Failed to persist state for {task_name}: {e}")
    
    def mark_task_starting(
        self,
        task_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a task as starting/running.
        
        Args:
            task_name: Name of the task
            context: Optional context data about the task
        """
        self.tasks[task_name] = {
            "task_name": task_name,
            "status": TaskStatus.RUNNING.value,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "error": None,
            "context": context or {},
            "retries": 0,
        }
        self._persist_state(task_name)
        logger.info(f"Task {task_name} marked as RUNNING")
    
    def mark_task_completed(
        self,
        task_name: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a task as completed.
        
        Args:
            task_name: Name of the task
            result: Optional result data
        """
        if task_name not in self.tasks:
            logger.warning(f"Task {task_name} not initialized, creating entry")
            self.tasks[task_name] = {"task_name": task_name}
        
        self.tasks[task_name].update({
            "status": TaskStatus.COMPLETED.value,
            "ended_at": datetime.utcnow().isoformat(),
            "error": None,
            "result": result or {},
        })
        
        if "started_at" not in self.tasks[task_name]:
            self.tasks[task_name]["started_at"] = datetime.utcnow().isoformat()
        
        self._persist_state(task_name)
        logger.info(f"Task {task_name} marked as COMPLETED")
    
    def mark_task_failed(
        self,
        task_name: str,
        error: str,
        retry_count: int = 0
    ) -> None:
        """Mark a task as failed.
        
        Args:
            task_name: Name of the task
            error: Error message
            retry_count: Number of retry attempts
        """
        if task_name not in self.tasks:
            logger.warning(f"Task {task_name} not initialized, creating entry")
            self.tasks[task_name] = {"task_name": task_name}
        
        self.tasks[task_name].update({
            "status": TaskStatus.FAILED.value,
            "ended_at": datetime.utcnow().isoformat(),
            "error": str(error)[:500],  # Store first 500 chars of error
            "retries": retry_count,
        })
        
        if "started_at" not in self.tasks[task_name]:
            self.tasks[task_name]["started_at"] = datetime.utcnow().isoformat()
        
        self._persist_state(task_name)
        logger.error(f"Task {task_name} marked as FAILED: {error}")
    
    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task status dictionary or None if not found
        """
        return self.tasks.get(task_name)
    
    def clear_task(self, task_name: str) -> None:
        """Clear task state.
        
        Args:
            task_name: Name of the task
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            
            state_file = self._get_state_file(task_name)
            if state_file.exists():
                try:
                    state_file.unlink()
                    logger.info(f"Cleared state for task: {task_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete state file for {task_name}: {e}")
    
    def clear_all_tasks(self) -> None:
        """Clear all task states."""
        self.tasks.clear()
        try:
            for state_file in self.state_dir.glob("*.json"):
                state_file.unlink()
            logger.info("Cleared all task states")
        except Exception as e:
            logger.warning(f"Failed to clear all states: {e}")
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all task states.
        
        Returns:
            Dictionary of all task states
        """
        return self.tasks.copy()
    
    def get_failed_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all failed tasks.
        
        Returns:
            Dictionary of failed task states
        """
        return {
            name: state
            for name, state in self.tasks.items()
            if state.get("status") == TaskStatus.FAILED.value
        }
    
    def get_running_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all running tasks.
        
        Returns:
            Dictionary of running task states
        """
        return {
            name: state
            for name, state in self.tasks.items()
            if state.get("status") == TaskStatus.RUNNING.value
        }
    
    def get_completed_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all completed tasks.
        
        Returns:
            Dictionary of completed task states
        """
        return {
            name: state
            for name, state in self.tasks.items()
            if state.get("status") == TaskStatus.COMPLETED.value
        }
    
    def is_task_completed(self, task_name: str) -> bool:
        """Check if a task is completed.
        
        Args:
            task_name: Name of the task
            
        Returns:
            True if task is completed, False otherwise
        """
        status = self.get_task_status(task_name)
        return status is not None and status.get("status") == TaskStatus.COMPLETED.value
    
    def is_task_failed(self, task_name: str) -> bool:
        """Check if a task is failed.
        
        Args:
            task_name: Name of the task
            
        Returns:
            True if task is failed, False otherwise
        """
        status = self.get_task_status(task_name)
        return status is not None and status.get("status") == TaskStatus.FAILED.value
    
    def is_task_running(self, task_name: str) -> bool:
        """Check if a task is running.
        
        Args:
            task_name: Name of the task
            
        Returns:
            True if task is running, False otherwise
        """
        status = self.get_task_status(task_name)
        return status is not None and status.get("status") == TaskStatus.RUNNING.value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all tasks.
        
        Returns:
            Summary dictionary
        """
        total = len(self.tasks)
        completed = len(self.get_completed_tasks())
        failed = len(self.get_failed_tasks())
        running = len(self.get_running_tasks())
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": total - completed - failed - running,
            "success_rate": (completed / total * 100) if total > 0 else 0,
        }
    
    def log_summary(self) -> None:
        """Log summary statistics."""
        summary = self.get_summary()
        logger.info(
            "Task Summary: Total=%d, Completed=%d, Failed=%d, Running=%d, "
            "Success Rate=%.1f%%",
            summary["total_tasks"],
            summary["completed"],
            summary["failed"],
            summary["running"],
            summary["success_rate"],
        )
