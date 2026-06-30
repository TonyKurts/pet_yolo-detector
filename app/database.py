import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Новая запись создается сразу в статусе обработки, поэтому значение "new" зарезервировано.
STATUS_NEW = 0
# Задание активно и еще не завершено.
STATUS_PROCESSING = 1
# Обработка успешно завершена, а результат сохранен.
STATUS_COMPLETE = 2
# Во время обработки произошла ошибка.
STATUS_ERROR = 3


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    # Уникальныи идентификатор задания.
    id: str
    # Время создания записи в UTC.
    created_at: str
    # Оригинальное имя загруженного пользователем файла.
    filename: str
    # Относительныи путь до итогового файла, если он был успешно создан.
    file_path: str | None
    # Количество обнаруженных объектов на изображении.
    object_count: int | None
    # Продолжительность обработки в секундах.
    processing_time: float | None
    # Текущее состояние задания.
    status: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ProcessingRecord":
        # sqlite3.Row приводится к словарю, чтобы создать dataclass по именованным полям.
        return cls(**dict(row))

    def as_dict(self) -> dict[str, Any]:
        # Явное преобразование нужно для стабильного API-ответа без утечки внутреннего типа.
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
    # Исключение сигнализирует, что сервер уже занят другим заданием.
    pass


class Database:
    def __init__(self, path: Path) -> None:
        # Объект хранит только путь, а соединения создает по требованию.
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # Каждая операция работает через отдельное соединение, чтобы не делить его между потоками.
        connection = sqlite3.connect(self.path, timeout=30)
        # Возврат строк с доступом по имени упрощает маппинг в dataclass.
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            # Соединение закрывается всегда, даже если запрос завершился ошибкои.
            connection.close()

    def initialize(self) -> None:
        # Каталог под БД создается заранее, чтобы sqlite мог открыть или создать файл.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            # WAL улучшает конкурентныи доступ на чтение во время записи.
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

            # Индекс ускоряет выдачу последних записеи в истории.
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_created_at "
                "ON processing_history(created_at DESC)"
            )

            # После перезапуска приложения незавершенные задания помечаются как ошибочные.
            connection.execute(
                "UPDATE processing_history SET status = ? WHERE status = ?",
                (STATUS_ERROR, STATUS_PROCESSING),
            )

            # DDL и служебные обновления фиксируются однои транзакциеи.
            connection.commit()

    def create_processing(self, task_id: str, filename: str) -> ProcessingRecord:
        # Время сохраняется в UTC и с микросекундами для однозначнои сортировки.
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        
        with self.connect() as connection:
            # IMMEDIATE сразу берет блокировку на запись и не дает двум запросам стартовать параллельно.
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT 1 FROM processing_history WHERE status = ? LIMIT 1",
                (STATUS_PROCESSING,),
            ).fetchone()
            
            if active:
                # Если активная обработка уже есть, откатываем транзакцию и запрещаем новую.
                connection.rollback()
                raise ServerBusyError("Another image is already being processed")

            # Новая запись создается сразу как активная, чтобы статус был виден до старта фоновой задачи.
            connection.execute(
                """
                INSERT INTO processing_history
                    (id, created_at, filename, file_path, object_count,
                     processing_time, status)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (task_id, created_at, filename, STATUS_PROCESSING),
            )
            # Фиксируем запись только после успешной проверки на занятость.
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
        # Успешное завершение записывает путь к результату и статистику детекции.
        self._update_result(
            task_id,
            status=STATUS_COMPLETE,
            file_path=file_path,
            object_count=object_count,
            processing_time=processing_time,
        )

    def fail_processing(self, task_id: str, processing_time: float) -> None:
        # При ошибке сохраняется только время работы и финальныи статус ошибки.
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
            # Обновляется одна конкретная запись по идентификатору задания.
            connection.execute(
                """
                UPDATE processing_history
                SET file_path = ?, object_count = ?, processing_time = ?, status = ?
                WHERE id = ?
                """,
                (file_path, object_count, processing_time, status, task_id),
            )
            # Изменения нужно явно зафиксировать, иначе они пропадут после закрытия соединения.
            connection.commit()

    def get(self, task_id: str) -> ProcessingRecord | None:
        with self.connect() as connection:
            # Поиск по первичному ключу возвращает максимум одну запись.
            row = connection.execute(
                "SELECT * FROM processing_history WHERE id = ?", (task_id,)
            ).fetchone()
        return ProcessingRecord.from_row(row) if row else None

    def latest(self) -> ProcessingRecord | None:
        with self.connect() as connection:
            # Последняя запись определяется по времени создания, а не по порядку вставки.
            row = connection.execute(
                "SELECT * FROM processing_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return ProcessingRecord.from_row(row) if row else None

    def all(self) -> list[ProcessingRecord]:
        with self.connect() as connection:
            # История отдается в обратном хронологическом порядке для удобства интерфеиса.
            rows = connection.execute(
                "SELECT * FROM processing_history ORDER BY created_at DESC"
            ).fetchall()
        return [ProcessingRecord.from_row(row) for row in rows]
