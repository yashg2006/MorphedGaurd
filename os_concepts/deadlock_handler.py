"""
Deadlock Handler — demonstrates OS Deadlock Handling concepts.
Provides resource ordering, timeout detection, and recovery.
"""
import threading
import time
import logging
from typing import List, Optional

import config

logger = logging.getLogger("DeadlockHandler")


class ResourceLock:
    """
    A lock wrapper that supports ordered acquisition and timeout detection.
    """

    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order  # Used for resource ordering protocol
        self.lock = threading.Lock()
        self.owner = None
        self.acquired_at = None

    def acquire(self, timeout: float = None) -> bool:
        timeout = timeout or config.LOCK_TIMEOUT
        acquired = self.lock.acquire(timeout=timeout)
        if acquired:
            self.owner = threading.current_thread().name
            self.acquired_at = time.time()
        return acquired

    def release(self):
        self.owner = None
        self.acquired_at = None
        try:
            self.lock.release()
        except RuntimeError:
            pass

    def __repr__(self):
        return f"ResourceLock({self.name}, order={self.order}, owner={self.owner})"


class DeadlockHandler:
    """
    Manages resource allocation to prevent and detect deadlocks.

    OS Concepts Demonstrated:
    - Resource Ordering Protocol: Always acquire locks in ascending order
    - Timeout-based Detection: Detect deadlocks via lock timeout
    - Recovery: Release all locks and retry with exponential backoff
    - Wait-for Graph: Track which threads hold/wait for which resources
    """

    def __init__(self):
        self.resources = {}
        self.wait_for_graph = {}  # thread_name -> list of resource names waiting for
        self.hold_graph = {}      # thread_name -> list of resource names held
        self.stats = {
            "deadlocks_detected": 0,
            "deadlocks_recovered": 0,
            "total_acquisitions": 0,
            "total_timeouts": 0,
            "retries": 0
        }
        self._meta_lock = threading.Lock()

    def register_resource(self, name: str, order: int) -> ResourceLock:
        """Register a new resource with a fixed ordering number."""
        resource = ResourceLock(name, order)
        self.resources[name] = resource
        logger.info(f"[DeadlockHandler] Registered resource '{name}' "
                    f"(order={order})")
        return resource

    def acquire_resources(self, resource_names: List[str],
                          max_retries: int = 3) -> bool:
        """
        Acquire multiple resources using resource ordering protocol.
        Resources are always acquired in ascending order to prevent deadlocks.

        Args:
            resource_names: Names of resources to acquire
            max_retries: Max retries on failure

        Returns:
            True if all resources acquired, False otherwise
        """
        thread_name = threading.current_thread().name

        # ── Sort by order (Resource Ordering Protocol) ─
        ordered = sorted(
            [self.resources[n] for n in resource_names if n in self.resources],
            key=lambda r: r.order
        )

        if not ordered:
            return True

        for attempt in range(max_retries):
            acquired = []
            success = True

            with self._meta_lock:
                self.wait_for_graph[thread_name] = [r.name for r in ordered]
                self.hold_graph.setdefault(thread_name, [])

            for resource in ordered:
                logger.debug(f"[DeadlockHandler] {thread_name} attempting to "
                             f"acquire '{resource.name}' (attempt {attempt + 1})")

                if resource.acquire(timeout=config.LOCK_TIMEOUT):
                    acquired.append(resource)
                    with self._meta_lock:
                        self.hold_graph[thread_name].append(resource.name)
                        if resource.name in self.wait_for_graph.get(thread_name, []):
                            self.wait_for_graph[thread_name].remove(resource.name)
                        self.stats["total_acquisitions"] += 1
                else:
                    # ── Timeout — potential deadlock! ──
                    self.stats["total_timeouts"] += 1
                    self.stats["deadlocks_detected"] += 1
                    logger.warning(
                        f"[DeadlockHandler] DEADLOCK DETECTED! "
                        f"{thread_name} timed out waiting for '{resource.name}' "
                        f"(held by {resource.owner})"
                    )

                    # ── Recovery: release all and retry ─
                    for r in reversed(acquired):
                        r.release()
                        with self._meta_lock:
                            if r.name in self.hold_graph.get(thread_name, []):
                                self.hold_graph[thread_name].remove(r.name)

                    self.stats["retries"] += 1
                    success = False

                    # Exponential backoff
                    backoff = 0.1 * (2 ** attempt)
                    logger.info(f"[DeadlockHandler] {thread_name} backing off "
                                f"for {backoff:.1f}s before retry")
                    time.sleep(backoff)
                    break

            if success:
                with self._meta_lock:
                    self.wait_for_graph.pop(thread_name, None)
                self.stats["deadlocks_recovered"] += (1 if attempt > 0 else 0)
                return True

        logger.error(f"[DeadlockHandler] {thread_name} failed to acquire "
                     f"resources after {max_retries} attempts")
        return False

    def release_resources(self, resource_names: List[str]):
        """Release multiple resources (in reverse order)."""
        thread_name = threading.current_thread().name
        ordered = sorted(
            [self.resources[n] for n in resource_names if n in self.resources],
            key=lambda r: r.order, reverse=True
        )

        for resource in ordered:
            resource.release()
            with self._meta_lock:
                if thread_name in self.hold_graph:
                    if resource.name in self.hold_graph[thread_name]:
                        self.hold_graph[thread_name].remove(resource.name)

        logger.debug(f"[DeadlockHandler] {thread_name} released "
                     f"{len(ordered)} resources")

    def get_wait_for_graph(self) -> dict:
        """Get the current wait-for graph (for deadlock visualization)."""
        with self._meta_lock:
            return {
                "waiting": {k: v[:] for k, v in self.wait_for_graph.items()},
                "holding": {k: v[:] for k, v in self.hold_graph.items()},
                "resources": {
                    name: {
                        "order": r.order,
                        "owner": r.owner,
                        "held_since": round(time.time() - r.acquired_at, 1)
                        if r.acquired_at else None
                    }
                    for name, r in self.resources.items()
                }
            }

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "registered_resources": len(self.resources),
            "wait_for_graph": self.get_wait_for_graph()
        }
