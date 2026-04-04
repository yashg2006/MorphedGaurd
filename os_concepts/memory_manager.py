"""
Memory Manager — demonstrates OS Memory Management concepts.
Provides LRU caching, buffer pooling, and memory tracking.
"""
import threading
import time
import sys
import logging
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import config

logger = logging.getLogger("MemoryManager")


class LRUCache:
    """
    Least Recently Used (LRU) Cache for analysis results.

    OS Concept: Page replacement algorithm — evicts least recently
    used entries when cache is full, similar to LRU page replacement.
    """

    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.CACHE_MAX_SIZE
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, promoting it to most-recently-used."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                logger.debug(f"[LRUCache] Cache HIT for '{key}'")
                return self.cache[key]
            else:
                self.stats["misses"] += 1
                logger.debug(f"[LRUCache] Cache MISS for '{key}'")
                return None

    def put(self, key: str, value: Any):
        """Insert or update a cache entry, evicting LRU if full."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.max_size:
                    evicted_key, _ = self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
                    logger.info(f"[LRUCache] Evicted '{evicted_key}' (LRU policy)")
                self.cache[key] = value
            self.stats["current_size"] = len(self.cache)

    def invalidate(self, key: str):
        """Remove a specific entry from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats["current_size"] = len(self.cache)

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats["current_size"] = 0

    def get_stats(self) -> dict:
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                **self.stats,
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 1)
            }


class BufferPool:
    """
    Pre-allocated Buffer Pool for image processing.

    OS Concept: Memory pooling — pre-allocates fixed-size buffers
    to avoid repeated allocation/deallocation overhead,
    similar to buffer pool management in OS kernels.
    """

    def __init__(self, pool_size: int = None, buffer_shape: tuple = (1024, 1024, 3)):
        self.pool_size = pool_size or config.BUFFER_POOL_SIZE
        self.buffer_shape = buffer_shape
        self.buffers = []
        self.available = []
        self.lock = threading.Lock()
        self.stats = {
            "total_buffers": 0,
            "available": 0,
            "allocations": 0,
            "releases": 0,
            "memory_bytes": 0
        }
        self._initialize_pool()

    def _initialize_pool(self):
        """Pre-allocate numpy arrays for the buffer pool."""
        for i in range(self.pool_size):
            buf = np.zeros(self.buffer_shape, dtype=np.uint8)
            self.buffers.append(buf)
            self.available.append(i)

        self.stats["total_buffers"] = self.pool_size
        self.stats["available"] = self.pool_size
        self.stats["memory_bytes"] = sum(b.nbytes for b in self.buffers)
        logger.info(f"[BufferPool] Initialized {self.pool_size} buffers "
                    f"({self.stats['memory_bytes'] / 1024 / 1024:.1f} MB)")

    def acquire(self) -> Optional[np.ndarray]:
        """Acquire a buffer from the pool."""
        with self.lock:
            if self.available:
                idx = self.available.pop()
                self.stats["allocations"] += 1
                self.stats["available"] = len(self.available)
                logger.debug(f"[BufferPool] Buffer {idx} acquired "
                             f"({self.stats['available']} remaining)")
                return self.buffers[idx]
            else:
                logger.warning("[BufferPool] No buffers available!")
                return None

    def release(self, buffer: np.ndarray):
        """Release a buffer back to the pool."""
        with self.lock:
            for i, buf in enumerate(self.buffers):
                if buf is buffer:
                    if i not in self.available:
                        self.available.append(i)
                        self.stats["releases"] += 1
                        self.stats["available"] = len(self.available)
                        buffer[:] = 0  # Zero out the buffer
                    break

    def get_stats(self) -> dict:
        with self.lock:
            return {
                **self.stats,
                "pool_size": self.pool_size,
                "buffer_shape": str(self.buffer_shape),
                "in_use": self.pool_size - len(self.available)
            }


class MemoryManager:
    """
    Unified memory manager combining cache and buffer pool.
    """

    def __init__(self):
        self.cache = LRUCache()
        self.buffer_pool = BufferPool()
        self.tracking = {
            "peak_memory_mb": 0,
            "allocations_total": 0
        }

    def get_stats(self) -> dict:
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
        except ImportError:
            rss_mb = 0

        return {
            "cache": self.cache.get_stats(),
            "buffer_pool": self.buffer_pool.get_stats(),
            "process_memory_mb": round(rss_mb, 1),
            "peak_memory_mb": self.tracking["peak_memory_mb"]
        }
