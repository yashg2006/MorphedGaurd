"""Database persistence for MorphGuard analysis tasks."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("Database")


class DatabaseManager:
    """Stores analysis status and results using parameterized SQL."""

    def __init__(self, database_url: str):
        self.database_url = self._normalize_url(database_url)
        connect_args = {}
        if self.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            future=True,
        )
        self.initialize()

    @staticmethod
    def _normalize_url(database_url: str) -> str:
        if database_url.startswith("postgres://"):
            return database_url.replace("postgres://", "postgresql+psycopg://", 1)
        if database_url.startswith("postgresql://"):
            return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return database_url

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def initialize(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    task_id VARCHAR(80) PRIMARY KEY,
                    batch_id VARCHAR(80),
                    filename VARCHAR(255),
                    status VARCHAR(32) NOT NULL,
                    verdict VARCHAR(32),
                    overall_score FLOAT,
                    confidence FLOAT,
                    elapsed_seconds FLOAT,
                    payload_json TEXT NOT NULL,
                    created_at VARCHAR(40) NOT NULL,
                    updated_at VARCHAR(40) NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_analysis_results_batch_id
                ON analysis_results (batch_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_analysis_results_status
                ON analysis_results (status)
            """))

    def save_initial(
        self,
        task_id: str,
        filename: str,
        batch_id: str | None = None,
    ) -> None:
        payload = {
            "task_id": task_id,
            "batch_id": batch_id,
            "filename": filename,
            "status": "processing",
            "timestamp": self._now(),
        }
        self.save_result(payload)

    def save_result(self, result: dict[str, Any]) -> None:
        now = self._now()
        payload_json = json.dumps(result, separators=(",", ":"), default=str)
        params = {
            "task_id": result["task_id"],
            "batch_id": result.get("batch_id"),
            "filename": result.get("filename"),
            "status": result.get("status", "unknown"),
            "verdict": result.get("verdict"),
            "overall_score": result.get("overall_score"),
            "confidence": result.get("confidence"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "payload_json": payload_json,
            "created_at": result.get("timestamp", now),
            "updated_at": now,
        }

        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO analysis_results (
                    task_id, batch_id, filename, status, verdict,
                    overall_score, confidence, elapsed_seconds,
                    payload_json, created_at, updated_at
                )
                VALUES (
                    :task_id, :batch_id, :filename, :status, :verdict,
                    :overall_score, :confidence, :elapsed_seconds,
                    :payload_json, :created_at, :updated_at
                )
                ON CONFLICT (task_id) DO UPDATE SET
                    batch_id = excluded.batch_id,
                    filename = excluded.filename,
                    status = excluded.status,
                    verdict = excluded.verdict,
                    overall_score = excluded.overall_score,
                    confidence = excluded.confidence,
                    elapsed_seconds = excluded.elapsed_seconds,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
            """), params)

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(text("""
                SELECT payload_json
                FROM analysis_results
                WHERE task_id = :task_id
            """), {"task_id": task_id}).mappings().first()

        if not row:
            return None
        return json.loads(row["payload_json"])

    def healthcheck(self) -> dict[str, Any]:
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            return {"ok": True}
        except SQLAlchemyError as exc:
            logger.exception("[Database] Healthcheck failed")
            return {"ok": False, "error": str(exc)}
