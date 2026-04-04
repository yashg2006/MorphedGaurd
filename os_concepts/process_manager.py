"""
Process Manager — demonstrates OS Process Management concepts.
Uses multiprocessing.Pool to run image analyses as separate processes.
"""
import multiprocessing
import os
import time
import logging
from typing import Callable, Any

import config

logger = logging.getLogger("ProcessManager")


class ProcessManager:
    """
    Manages a pool of worker processes for parallel image analysis.

    OS Concepts Demonstrated:
    - Process creation and lifecycle management
    - Process pools with configurable worker count
    - Async task submission and result collection
    - PID tracking for each worker
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or config.MAX_PROCESSES
        self.pool = None
        self.active_tasks = {}  # task_id -> AsyncResult
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "active_count": 0,
            "worker_pids": [],
            "start_time": time.time()
        }
        self._lock = multiprocessing.Lock()

    def start(self):
        """Initialize the process pool."""
        if self.pool is None:
            self.pool = multiprocessing.Pool(
                processes=self.max_workers,
                initializer=self._worker_init
            )
            logger.info(f"[ProcessManager] Started pool with {self.max_workers} workers "
                        f"(Parent PID: {os.getpid()})")

    def _worker_init(self):
        """Called when each worker process starts."""
        pid = os.getpid()
        logger.info(f"[ProcessManager] Worker process started (PID: {pid})")

    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit an analysis task to the process pool.

        Args:
            task_id: Unique identifier for this task
            func: The function to execute in a separate process
            *args, **kwargs: Arguments passed to the function

        Returns:
            task_id for status tracking
        """
        if self.pool is None:
            self.start()

        async_result = self.pool.apply_async(
            func, args=args, kwds=kwargs,
            callback=lambda r: self._on_complete(task_id),
            error_callback=lambda e: self._on_error(task_id, e)
        )

        self.active_tasks[task_id] = async_result
        self.stats["total_submitted"] += 1
        self.stats["active_count"] = len(
            [t for t in self.active_tasks.values() if not t.ready()]
        )

        logger.info(f"[ProcessManager] Task {task_id} submitted to process pool")
        return task_id

    def _on_complete(self, task_id: str):
        """Callback when a task completes successfully."""
        self.stats["total_completed"] += 1
        self.stats["active_count"] = max(0, self.stats["active_count"] - 1)
        logger.info(f"[ProcessManager] Task {task_id} completed")

    def _on_error(self, task_id: str, error: Exception):
        """Callback when a task fails."""
        self.stats["total_failed"] += 1
        self.stats["active_count"] = max(0, self.stats["active_count"] - 1)
        logger.error(f"[ProcessManager] Task {task_id} failed: {error}")

    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get the result of a submitted task."""
        if task_id not in self.active_tasks:
            return None
        try:
            return self.active_tasks[task_id].get(timeout=timeout)
        except multiprocessing.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"[ProcessManager] Error getting result for {task_id}: {e}")
            return None

    def is_complete(self, task_id: str) -> bool:
        """Check if a task has completed."""
        if task_id not in self.active_tasks:
            return False
        return self.active_tasks[task_id].ready()

    def get_stats(self) -> dict:
        """Get current process pool statistics."""
        self.stats["active_count"] = len(
            [t for t in self.active_tasks.values() if not t.ready()]
        )
        self.stats["uptime"] = round(time.time() - self.stats["start_time"], 1)
        self.stats["parent_pid"] = os.getpid()
        self.stats["max_workers"] = self.max_workers
        return self.stats.copy()

    def shutdown(self):
        """Gracefully shut down the process pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("[ProcessManager] Pool shutdown complete")
