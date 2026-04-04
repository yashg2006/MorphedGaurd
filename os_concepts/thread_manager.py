"""
Thread Manager — demonstrates OS Threading concepts.
Uses ThreadPoolExecutor for pipeline-stage parallelism.
"""
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any

import config

logger = logging.getLogger("ThreadManager")


class ThreadManager:
    """
    Manages thread pools for the three-stage detection pipeline.

    OS Concepts Demonstrated:
    - Thread creation and lifecycle
    - Thread pools for different pipeline stages
    - Thread IDs and naming
    - Concurrent execution of independent tasks
    """

    def __init__(self, max_threads: int = None):
        self.max_threads = max_threads or config.MAX_THREADS
        self.executors = {}
        self.stats = {
            "preprocessing": {"submitted": 0, "completed": 0, "active": 0},
            "extraction": {"submitted": 0, "completed": 0, "active": 0},
            "detection": {"submitted": 0, "completed": 0, "active": 0},
            "thread_ids": [],
            "total_threads_created": 0,
        }
        self._stats_lock = threading.Lock()

    def _get_executor(self, stage: str) -> ThreadPoolExecutor:
        """Get or create a thread pool for the given pipeline stage."""
        if stage not in self.executors:
            worker_count = max(2, self.max_threads // 3)
            self.executors[stage] = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"Stage-{stage}"
            )
            logger.info(f"[ThreadManager] Created thread pool for '{stage}' "
                        f"with {worker_count} workers")
        return self.executors[stage]

    def submit_preprocessing(self, func: Callable, *args) -> Any:
        """
        Submit a task to the preprocessing thread pool.
        Used for: image resizing, format conversion, normalization.
        """
        return self._submit("preprocessing", func, *args)

    def submit_extraction(self, func: Callable, *args) -> Any:
        """
        Submit a task to the feature extraction thread pool.
        Used for: ELA, noise analysis, EXIF parsing, copy-move detection.
        """
        return self._submit("extraction", func, *args)

    def submit_detection(self, func: Callable, *args) -> Any:
        """
        Submit a task to the detection thread pool.
        Used for: CNN classification, final scoring.
        """
        return self._submit("detection", func, *args)

    def _submit(self, stage: str, func: Callable, *args):
        """Submit a task to a specific stage's thread pool."""
        executor = self._get_executor(stage)

        with self._stats_lock:
            self.stats[stage]["submitted"] += 1
            self.stats[stage]["active"] += 1
            self.stats["total_threads_created"] += 1

        def wrapper(*a):
            tid = threading.current_thread().ident
            tname = threading.current_thread().name
            logger.info(f"[ThreadManager] {stage} task running on "
                        f"thread {tname} (ID: {tid})")

            with self._stats_lock:
                if tid not in self.stats["thread_ids"]:
                    self.stats["thread_ids"].append(tid)

            try:
                result = func(*a)
                return result
            finally:
                with self._stats_lock:
                    self.stats[stage]["completed"] += 1
                    self.stats[stage]["active"] = max(0,
                                                       self.stats[stage]["active"] - 1)

        future = executor.submit(wrapper, *args)
        return future

    def run_parallel_extraction(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        Run multiple extraction tasks in parallel and collect results.

        Args:
            tasks: List of dicts with 'name', 'func', 'args' keys

        Returns:
            Dict mapping task names to their results
        """
        futures = {}
        for task in tasks:
            future = self.submit_extraction(task["func"], *task["args"])
            futures[task["name"]] = future

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=60)
            except Exception as e:
                logger.error(f"[ThreadManager] Extraction task '{name}' failed: {e}")
                results[name] = {"error": str(e), "score": 0}

        return results

    def get_stats(self) -> dict:
        """Get thread pool statistics."""
        with self._stats_lock:
            stats = {
                "stages": {
                    stage: self.stats[stage].copy()
                    for stage in ["preprocessing", "extraction", "detection"]
                },
                "total_threads_created": self.stats["total_threads_created"],
                "active_thread_ids": self.stats["thread_ids"][-10:],  # Last 10
                "current_thread_count": threading.active_count(),
                "current_thread": threading.current_thread().name,
            }
        return stats

    def shutdown(self):
        """Shut down all thread pools."""
        for stage, executor in self.executors.items():
            executor.shutdown(wait=True)
            logger.info(f"[ThreadManager] Pool '{stage}' shut down")
        self.executors.clear()
