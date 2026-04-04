"""
File System Manager — demonstrates OS File System Management.
Handles directory creation, file organization, and logging.
"""
import os
import time
import json
import shutil
import logging
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger("FileManager")


class FileManager:
    """
    Manages the file system for the detection system.

    OS Concepts Demonstrated:
    - Directory structure creation and management
    - File I/O operations
    - Structured logging per analysis
    - File metadata tracking
    """

    def __init__(self):
        self.directories = {
            "uploads": config.UPLOAD_DIR,
            "results": config.RESULTS_DIR,
            "logs": config.LOGS_DIR,
            "models": config.MODEL_DIR
        }
        self._initialize_directories()
        self.stats = {
            "files_processed": 0,
            "total_bytes_processed": 0,
            "logs_written": 0,
            "results_saved": 0
        }

    def _initialize_directories(self):
        """Create required directories if they don't exist."""
        for name, path in self.directories.items():
            os.makedirs(path, exist_ok=True)
            logger.info(f"[FileManager] Directory ready: {path}")

    def save_upload(self, file_storage, filename: str) -> str:
        """
        Save an uploaded file to the uploads directory.

        Returns:
            Full path to the saved file
        """
        # Sanitize filename
        safe_name = self._sanitize_filename(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{timestamp}_{safe_name}"
        filepath = os.path.join(config.UPLOAD_DIR, unique_name)

        file_storage.save(filepath)

        file_size = os.path.getsize(filepath)
        self.stats["files_processed"] += 1
        self.stats["total_bytes_processed"] += file_size

        logger.info(f"[FileManager] Saved upload: {unique_name} ({file_size} bytes)")
        return filepath

    def save_result_image(self, image_array, task_id: str,
                          suffix: str = "result") -> str:
        """Save a result image (numpy array) to the results directory."""
        import cv2
        filename = f"{task_id}_{suffix}.png"
        filepath = os.path.join(config.RESULTS_DIR, filename)
        cv2.imwrite(filepath, image_array)
        self.stats["results_saved"] += 1
        logger.info(f"[FileManager] Saved result: {filename}")
        return filepath

    def write_analysis_log(self, task_id: str, results: dict):
        """Write a structured JSON log for an analysis."""
        log_entry = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "results": self._make_serializable(results)
        }

        log_file = os.path.join(config.LOGS_DIR, f"{task_id}_log.json")
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        self.stats["logs_written"] += 1
        logger.info(f"[FileManager] Log written: {log_file}")
        return log_file

    def _make_serializable(self, obj):
        """Convert non-serializable objects for JSON logging."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def _sanitize_filename(self, filename: str) -> str:
        """Remove potentially dangerous characters from filenames."""
        keepcharacters = (' ', '.', '_', '-')
        return "".join(c for c in filename
                       if c.isalnum() or c in keepcharacters).strip()

    def is_allowed_file(self, filename: str) -> bool:
        """Check if a file has an allowed extension."""
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        return ext in config.ALLOWED_EXTENSIONS

    def get_directory_stats(self) -> dict:
        """Get statistics about managed directories."""
        dir_stats = {}
        for name, path in self.directories.items():
            if os.path.exists(path):
                files = os.listdir(path)
                total_size = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in files if os.path.isfile(os.path.join(path, f))
                )
                dir_stats[name] = {
                    "path": path,
                    "file_count": len(files),
                    "total_size_mb": round(total_size / 1024 / 1024, 2)
                }
        return dir_stats

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "directories": self.get_directory_stats(),
            "total_mb_processed": round(
                self.stats["total_bytes_processed"] / 1024 / 1024, 2
            )
        }


file_manager = FileManager()
