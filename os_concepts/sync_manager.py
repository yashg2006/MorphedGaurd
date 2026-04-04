"""
Synchronization Manager — demonstrates OS synchronization primitives.
Provides semaphores, mutexes/locks, and condition variables.
"""
import threading
import time
import logging

import config

logger = logging.getLogger("SyncManager")


class SyncManager:
    """
    Manages synchronization primitives for thread/process coordination.

    OS Concepts Demonstrated:
    - Semaphore: Limits concurrent image analyses
    - Mutex (Lock): Protects shared data structures
    - Condition Variable: Signals batch completion
    - Read-Write Lock: Multiple readers, exclusive writer
    """

    def __init__(self):
        # ── Semaphore: limit concurrent analyses ──────
        self.analysis_semaphore = threading.Semaphore(config.SEMAPHORE_LIMIT)
        self.semaphore_stats = {
            "max_permits": config.SEMAPHORE_LIMIT,
            "current_acquisitions": 0,
            "total_acquisitions": 0,
            "total_waits": 0
        }

        # ── Mutex/Lock: protect shared results dict ──
        self.results_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.lock_stats = {
            "results_lock_acquisitions": 0,
            "log_lock_acquisitions": 0,
            "contention_count": 0
        }

        # ── Condition Variable: batch completion ──────
        self.batch_condition = threading.Condition()
        self.batch_complete = False
        self.batch_stats = {
            "signals_sent": 0,
            "waits": 0
        }

        self._stats_lock = threading.Lock()

    # ── Semaphore Methods ─────────────────────────────

    def acquire_analysis_slot(self) -> bool:
        """
        Acquire a semaphore slot before starting an analysis.
        Blocks if maximum concurrent analyses are running.
        """
        logger.info(f"[SyncManager] Thread {threading.current_thread().name} "
                     f"waiting for analysis semaphore...")

        with self._stats_lock:
            self.semaphore_stats["total_waits"] += 1

        acquired = self.analysis_semaphore.acquire(timeout=config.LOCK_TIMEOUT)

        if acquired:
            with self._stats_lock:
                self.semaphore_stats["current_acquisitions"] += 1
                self.semaphore_stats["total_acquisitions"] += 1
            logger.info(f"[SyncManager] Thread {threading.current_thread().name} "
                         f"acquired analysis slot "
                         f"({self.semaphore_stats['current_acquisitions']}"
                         f"/{self.semaphore_stats['max_permits']})")
        else:
            logger.warning(f"[SyncManager] Semaphore acquisition timed out!")
        return acquired

    def release_analysis_slot(self):
        """Release a semaphore slot after analysis completes."""
        self.analysis_semaphore.release()
        with self._stats_lock:
            self.semaphore_stats["current_acquisitions"] = max(
                0, self.semaphore_stats["current_acquisitions"] - 1
            )
        logger.info(f"[SyncManager] Thread {threading.current_thread().name} "
                     f"released analysis slot")

    # ── Mutex/Lock Methods ────────────────────────────

    def acquire_results_lock(self) -> bool:
        """Acquire the mutex lock for writing to shared results."""
        acquired = self.results_lock.acquire(timeout=config.LOCK_TIMEOUT)
        if acquired:
            with self._stats_lock:
                self.lock_stats["results_lock_acquisitions"] += 1
        else:
            with self._stats_lock:
                self.lock_stats["contention_count"] += 1
            logger.warning("[SyncManager] Results lock acquisition timed out! "
                           "Possible deadlock detected.")
        return acquired

    def release_results_lock(self):
        """Release the results mutex lock."""
        try:
            self.results_lock.release()
        except RuntimeError:
            pass  # Lock was not acquired

    def acquire_log_lock(self) -> bool:
        """Acquire the mutex lock for writing to log files."""
        acquired = self.log_lock.acquire(timeout=config.LOCK_TIMEOUT)
        if acquired:
            with self._stats_lock:
                self.lock_stats["log_lock_acquisitions"] += 1
        return acquired

    def release_log_lock(self):
        """Release the log mutex lock."""
        try:
            self.log_lock.release()
        except RuntimeError:
            pass

    # ── Condition Variable Methods ────────────────────

    def wait_for_batch_complete(self, timeout: float = 120):
        """Wait until the batch processing is signaled as complete."""
        with self.batch_condition:
            with self._stats_lock:
                self.batch_stats["waits"] += 1
            logger.info("[SyncManager] Waiting for batch completion signal...")
            self.batch_condition.wait_for(
                lambda: self.batch_complete, timeout=timeout
            )

    def signal_batch_complete(self):
        """Signal that batch processing is complete."""
        with self.batch_condition:
            self.batch_complete = True
            self.batch_condition.notify_all()
            with self._stats_lock:
                self.batch_stats["signals_sent"] += 1
            logger.info("[SyncManager] Batch completion signaled!")

    def reset_batch(self):
        """Reset the batch completion flag for a new batch."""
        with self.batch_condition:
            self.batch_complete = False

    # ── Stats ─────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._stats_lock:
            return {
                "semaphore": self.semaphore_stats.copy(),
                "locks": self.lock_stats.copy(),
                "condition_variable": self.batch_stats.copy()
            }
