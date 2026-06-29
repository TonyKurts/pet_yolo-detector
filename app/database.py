import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_NEW = 0
STATUS_PROCESSING = 1
STATUS_COMPLETE = 2
STATUS_ERROR = 3


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    id: str
    created_at: str
    filename: str
    file_path: str | None
    object_count: int | None
    processing_time: float | None
    status: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ProcessingRecord":
        return cls(**dict(row))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "filename": self.filename,
            "file_path": self.file_path,
            "object_count": self.object_count,
            "processing_time": self.processing_time,
            "status": self.status,
        }


class ServerBusyError(RuntimeError):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_history (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    object_count INTEGER,
                    processing_time REAL,
                    status INTEGER NOT NULL CHECK (status BETWEEN 0 AND 3)
                )
                """
            )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_created_at "
                "ON processing_history(created_at DESC)"
            )

            connection.execute(
                "UPDATE processing_history SET status = ? WHERE status = ?",
                (STATUS_ERROR, STATUS_PROCESSING),
            )

            connection.commit()

    def create_processing(self, task_id: str, filename: str) -> ProcessingRecord:
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT 1 FROM processing_history WHERE status = ? LIMIT 1",
                (STATUS_PROCESSING,),
            ).fetchone()
            
            if active:
                connection.rollback()
                raise ServerBusyError("Another image is already being processed")

            connection.execute(
                """
                INSERT INTO processing_history
                    (id, created_at, filename, file_path, object_count,
                     processing_time, status)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (task_id, created_at, filename, STATUS_PROCESSING),
            )
            connection.commit()

        return ProcessingRecord(
            id=task_id,
            created_at=created_at,
            filename=filename,
            file_path=None,
            object_count=None,
            processing_time=None,
            status=STATUS_PROCESSING,
        )

    def finish_processing(
        self,
        task_id: str,
        file_path: str,
        object_count: int,
        processing_time: float,
    ) -> None:
        self._update_result(
            task_id,
            status=STATUS_COMPLETE,
            file_path=file_path,
            object_count=object_count,
            processing_time=processing_time,
        )

    def fail_processing(self, task_id: str, processing_time: float) -> None:
        self._update_result(
            task_id,
            status=STATUS_ERROR,
            file_path=None,
            object_count=None,
            processing_time=processing_time,
        )

    def _update_result(
        self,
        task_id: str,
        *,
        status: int,
        file_path: str | None,
        object_count: int | None,
        processing_time: float,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE processing_history
                SET file_path = ?, object_count = ?, processing_time = ?, status = ?
                WHERE id = ?
                """,
                (file_path, object_count, processing_time, status, task_id),
            )
            connection.commit()

    def get(self, task_id: str) -> ProcessingRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM processing_history WHERE id = ?", (task_id,)
            ).fetchone()
        return ProcessingRecord.from_row(row) if row else None

    def latest(self) -> ProcessingRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM processing_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return ProcessingRecord.from_row(row) if row else None

    def all(self) -> list[ProcessingRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM processing_history ORDER BY created_at DESC"
            ).fetchall()
        return [ProcessingRecord.from_row(row) for row in rows]
