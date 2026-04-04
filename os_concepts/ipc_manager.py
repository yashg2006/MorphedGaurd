"""
IPC Manager — demonstrates Inter-Process Communication concepts.
Provides message queues, shared memory, and pipes.
"""
import multiprocessing
import time
import logging
from typing import Any, Optional

logger = logging.getLogger("IPCManager")


class MessageQueue:
    """
    Message Queue for task dispatch and result collection.

    OS Concept: Message passing between processes using FIFO queues.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.queue = multiprocessing.Queue()
        self.stats = {"sent": 0, "received": 0}

    def send(self, message: Any):
        """Send a message to the queue."""
        self.queue.put({
            "data": message,
            "timestamp": time.time(),
            "sender_pid": multiprocessing.current_process().pid
        })
        self.stats["sent"] += 1
        logger.info(f"[MQ-{self.name}] Message sent (total: {self.stats['sent']})")

    def receive(self, timeout: float = 5.0) -> Optional[Any]:
        """Receive a message from the queue."""
        try:
            msg = self.queue.get(timeout=timeout)
            self.stats["received"] += 1
            logger.info(f"[MQ-{self.name}] Message received "
                        f"(total: {self.stats['received']})")
            return msg["data"]
        except Exception:
            return None

    def is_empty(self) -> bool:
        return self.queue.empty()

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "sent": self.stats["sent"],
            "received": self.stats["received"],
            "pending": self.stats["sent"] - self.stats["received"]
        }


class SharedMemoryManager:
    """
    Shared Memory for progress tracking across processes.

    OS Concept: Shared memory segments for inter-process data sharing.
    """

    def __init__(self):
        self.progress = multiprocessing.Value('d', 0.0)    # Overall progress (0-100)
        self.active_count = multiprocessing.Value('i', 0)   # Active analyses
        self.completed_count = multiprocessing.Value('i', 0) # Completed analyses
        self.error_count = multiprocessing.Value('i', 0)     # Failed analyses

    def update_progress(self, value: float):
        """Update overall progress (thread-safe via Value's internal lock)."""
        with self.progress.get_lock():
            self.progress.value = value

    def increment_active(self):
        with self.active_count.get_lock():
            self.active_count.value += 1

    def decrement_active(self):
        with self.active_count.get_lock():
            self.active_count.value = max(0, self.active_count.value - 1)

    def increment_completed(self):
        with self.completed_count.get_lock():
            self.completed_count.value += 1

    def increment_errors(self):
        with self.error_count.get_lock():
            self.error_count.value += 1

    def get_state(self) -> dict:
        """Read current shared memory state."""
        return {
            "progress": round(self.progress.value, 1),
            "active": self.active_count.value,
            "completed": self.completed_count.value,
            "errors": self.error_count.value
        }


class PipeManager:
    """
    Named Pipes for direct process-to-process communication.

    OS Concept: Unidirectional/bidirectional pipes for data streaming.
    """

    def __init__(self):
        self.pipes = {}
        self.stats = {"pipes_created": 0, "messages_sent": 0}

    def create_pipe(self, name: str):
        """Create a named bidirectional pipe."""
        parent_conn, child_conn = multiprocessing.Pipe()
        self.pipes[name] = {
            "parent": parent_conn,
            "child": child_conn
        }
        self.stats["pipes_created"] += 1
        logger.info(f"[PipeManager] Created pipe '{name}'")
        return parent_conn, child_conn

    def send(self, pipe_name: str, data: Any, end: str = "parent"):
        """Send data through a named pipe."""
        if pipe_name in self.pipes:
            self.pipes[pipe_name][end].send(data)
            self.stats["messages_sent"] += 1

    def receive(self, pipe_name: str, end: str = "child", timeout: float = 5.0) -> Any:
        """Receive data from a named pipe."""
        if pipe_name in self.pipes:
            conn = self.pipes[pipe_name][end]
            if conn.poll(timeout):
                return conn.recv()
        return None

    def get_stats(self) -> dict:
        return self.stats.copy()


class IPCManager:
    """
    Unified IPC manager combining all IPC mechanisms.
    """

    def __init__(self):
        self.task_queue = MessageQueue("tasks")
        self.result_queue = MessageQueue("results")
        self.shared_memory = SharedMemoryManager()
        self.pipe_manager = PipeManager()

    def get_stats(self) -> dict:
        return {
            "message_queues": {
                "task_queue": self.task_queue.get_stats(),
                "result_queue": self.result_queue.get_stats()
            },
            "shared_memory": self.shared_memory.get_state(),
            "pipes": self.pipe_manager.get_stats()
        }
