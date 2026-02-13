"""Tests for pipeline state manager."""

import pytest
import tempfile
import json
from pathlib import Path
from src.pipeline.state_manager import StateManager, TaskStatus


class TestStateManager:
    """Test StateManager functionality."""
    
    @pytest.fixture
    def state_manager(self):
        """Create temporary state manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StateManager(state_dir=tmpdir)
    
    def test_mark_task_starting(self, state_manager):
        """Test marking task as starting."""
        state_manager.mark_task_starting("test_task", context={"run_id": "123"})
        
        status = state_manager.get_task_status("test_task")
        assert status is not None
        assert status["status"] == TaskStatus.RUNNING.value
        assert status["context"]["run_id"] == "123"
        assert status["started_at"] is not None
    
    def test_mark_task_completed(self, state_manager):
        """Test marking task as completed."""
        state_manager.mark_task_starting("test_task")
        state_manager.mark_task_completed("test_task", result={"rows": 100})
        
        status = state_manager.get_task_status("test_task")
        assert status["status"] == TaskStatus.COMPLETED.value
        assert status["error"] is None
        assert status["result"]["rows"] == 100
    
    def test_mark_task_failed(self, state_manager):
        """Test marking task as failed."""
        state_manager.mark_task_starting("test_task")
        error_msg = "Something went wrong"
        state_manager.mark_task_failed("test_task", error=error_msg)
        
        status = state_manager.get_task_status("test_task")
        assert status["status"] == TaskStatus.FAILED.value
        assert error_msg in status["error"]
    
    def test_task_status_queries(self, state_manager):
        """Test various task status queries."""
        state_manager.mark_task_starting("running_task")
        state_manager.mark_task_completed("completed_task")
        state_manager.mark_task_failed("failed_task", error="error")
        
        assert state_manager.is_task_running("running_task")
        assert state_manager.is_task_completed("completed_task")
        assert state_manager.is_task_failed("failed_task")
    
    def test_get_all_tasks(self, state_manager):
        """Test retrieving all tasks."""
        state_manager.mark_task_starting("task1")
        state_manager.mark_task_completed("task2")
        
        all_tasks = state_manager.get_all_tasks()
        assert len(all_tasks) == 2
        assert "task1" in all_tasks
        assert "task2" in all_tasks
    
    def test_get_failed_tasks(self, state_manager):
        """Test retrieving failed tasks."""
        state_manager.mark_task_completed("task1")
        state_manager.mark_task_failed("task2", error="error1")
        state_manager.mark_task_failed("task3", error="error2")
        
        failed = state_manager.get_failed_tasks()
        assert len(failed) == 2
        assert "task2" in failed
        assert "task3" in failed
    
    def test_get_completed_tasks(self, state_manager):
        """Test retrieving completed tasks."""
        state_manager.mark_task_completed("task1")
        state_manager.mark_task_completed("task2")
        state_manager.mark_task_failed("task3", error="error")
        
        completed = state_manager.get_completed_tasks()
        assert len(completed) == 2
        assert "task1" in completed
        assert "task2" in completed
    
    def test_get_running_tasks(self, state_manager):
        """Test retrieving running tasks."""
        state_manager.mark_task_starting("task1")
        state_manager.mark_task_starting("task2")
        state_manager.mark_task_completed("task3")
        
        running = state_manager.get_running_tasks()
        assert len(running) == 2
        assert "task1" in running
        assert "task2" in running
    
    def test_clear_task(self, state_manager):
        """Test clearing a single task."""
        state_manager.mark_task_starting("task1")
        state_manager.mark_task_starting("task2")
        
        state_manager.clear_task("task1")
        
        assert "task1" not in state_manager.get_all_tasks()
        assert "task2" in state_manager.get_all_tasks()
    
    def test_clear_all_tasks(self, state_manager):
        """Test clearing all tasks."""
        state_manager.mark_task_starting("task1")
        state_manager.mark_task_starting("task2")
        state_manager.mark_task_starting("task3")
        
        state_manager.clear_all_tasks()
        
        assert len(state_manager.get_all_tasks()) == 0
    
    def test_get_summary(self, state_manager):
        """Test getting summary statistics."""
        state_manager.mark_task_completed("task1")
        state_manager.mark_task_completed("task2")
        state_manager.mark_task_failed("task3", error="error")
        state_manager.mark_task_starting("task4")
        
        summary = state_manager.get_summary()
        
        assert summary["total_tasks"] == 4
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["running"] == 1
        assert summary["success_rate"] == 50.0
    
    def test_persistence(self):
        """Test state persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manager and add task
            manager1 = StateManager(state_dir=tmpdir)
            manager1.mark_task_starting("task1")
            manager1.mark_task_completed("task2")
            
            # Create new manager from same directory
            manager2 = StateManager(state_dir=tmpdir)
            
            # Should have loaded existing tasks
            assert manager2.is_task_completed("task2")
            assert manager2.is_task_running("task1")
    
    def test_task_context(self, state_manager):
        """Test passing context through task lifecycle."""
        context = {
            "run_id": "RUN_001",
            "tickers": ["AAPL", "MSFT"],
            "version": "1.0"
        }
        
        state_manager.mark_task_starting("etl_task", context=context)
        
        status = state_manager.get_task_status("etl_task")
        assert status["context"]["run_id"] == "RUN_001"
        assert len(status["context"]["tickers"]) == 2
    
    def test_error_truncation(self, state_manager):
        """Test that long error messages are truncated."""
        long_error = "x" * 1000  # 1000 character error
        state_manager.mark_task_failed("task1", error=long_error)
        
        status = state_manager.get_task_status("task1")
        # Should be truncated to 500 chars
        assert len(status["error"]) <= 500
    
    def test_retry_count(self, state_manager):
        """Test retry count tracking."""
        state_manager.mark_task_starting("task1")
        state_manager.mark_task_failed("task1", error="attempt 1", retry_count=1)
        
        status = state_manager.get_task_status("task1")
        assert status["retries"] == 1


class TestTaskStatus:
    """Test TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
