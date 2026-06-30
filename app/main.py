import csv
import io
import mimetypes

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image, UnidentifiedImageError

from app.config import Settings, settings as default_settings
from app.database import STATUS_PROCESSING, Database, ServerBusyError
from app.services.detection import YoloPersonDetector, run_detection_in_background


# Разрешены только расширения, которые интерфеис и сервер поддерживают явно.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# PIL дополнительно проверяет реальныи формат изображения после сохранения на диск.
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}


def create_app(
    settings: Settings = default_settings,
    *,
    detector: Any | None = None,
    task_runner: Callable[..., Any] = run_detection_in_background,
) -> FastAPI:
    # БД и детектор создаются на уровне приложения и переиспользуются всеми запросами.
    database = Database(settings.database_path)
    person_detector = detector or YoloPersonDetector(settings.model_path)
    # Каталоги хранения создаются заранее, чтобы обработчики не падали на первом запросе.
    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Инициализация схемы БД выполняется при старте приложения.
        database.initialize()
        yield

    app = FastAPI(
        title="Store Visitor Counter",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Общие зависимости сохраняются в state, чтобы при необходимости их можно было достать из app.
    app.state.database = database
    app.state.detector = person_detector
    app.state.settings = settings

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        # Главная страница отдается как обычныи статическии фаил.
        return _small_file_response(settings.static_dir / "index.html", "text/html")

    @app.get("/static/{file_path:path}", include_in_schema=False)
    async def static_asset(file_path: str) -> Response:
        # Публичная раздача статических ресурсов идет только внутри разрешенного каталога.
        path = _resolve_public_file(settings.static_dir, file_path)
        return _small_file_response(path)

    @app.get("/storage/{file_path:path}", include_in_schema=False)
    async def stored_file(file_path: str) -> StreamingResponse:
        # Файлы результатов читаются потоково, чтобы не держать целиком в памяти крупные изображения.
        path = _resolve_public_file(settings.storage_dir, file_path)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return StreamingResponse(
            _file_chunks(path),
            media_type=media_type,
            headers={"Content-Length": str(path.stat().st_size)},
        )

    @app.post("/api/upload", status_code=202)
    async def upload_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict[str, str]:
        # Берем только безопасное имя без каталогов, даже если клиент прислал путь.
        original_name = Path(file.filename or "").name
        extension = Path(original_name).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            # Ограничение по расширению отсекает неподдерживаемые форматы еще до чтения содержимого.
            raise HTTPException(
                status_code=415,
                detail="Only .png, .jpg and .jpeg files are supported",
            )

        # Идентификатор задания одновременно используется в API и имени исходного файла.
        task_id = str(uuid4())
        source_path = settings.originals_dir / f"{task_id}{extension}"
        
        try:
            # Сначала файл сохраняется с ограничением по размеру, затем валидируется как изображение.
            await _save_upload(file, source_path, settings.max_upload_bytes)
            _verify_image(source_path)
        except HTTPException:
            # При любои ожидаемои ошибке временно сохраненныи файл удаляется.
            source_path.unlink(missing_ok=True)
            raise
        finally:
            # FastAPI-объект загрузки нужно закрыть в любом сценарии.
            await file.close()

        try:
            # Запись в БД резервирует право на единственную активную обработку.
            database.create_processing(task_id, original_name)
        except ServerBusyError as exc:
            # Если сервер занят, загруженныи исходник тоже удаляется как неиспользуемыи.
            source_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # Имя итогового изображения отделено от task_id, чтобы не связывать внешний URL и идентификатор задания.
        output_name = f"{uuid4()}.jpg"
        destination = settings.processed_dir / output_name
        # В БД сохраняется относительныи путь, пригодныи для последующеи раздачи через /storage.
        relative_destination = f"processed/{output_name}"
        # Тяжелая обработка ставится в фон и не задерживает HTTP-ответ клиенту.
        background_tasks.add_task(
            task_runner,
            task_id=task_id,
            source=source_path,
            destination=destination,
            relative_destination=relative_destination,
            database=database,
            detector=person_detector,
        )
        # Клиенту сразу возвращается идентификатор для опроса статуса.
        return {"task_id": task_id}

    @app.get("/api/status")
    async def task_status(task_id: str | None = None) -> dict[str, Any]:
        # Если task_id не передан, интерфеис получает статус последнего созданного задания.
        record = database.get(task_id) if task_id else database.latest()
        if task_id and record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            # Поле available показывает, можно ли отправлять новое изображение прямо сейчас.
            "available": record is None or record.status != STATUS_PROCESSING,
            "task": record.as_dict() if record else None,
        }

    @app.get("/api/history")
    async def history() -> list[dict[str, Any]]:
        # История полностью строится из сериализованных записеи БД.
        return [record.as_dict() for record in database.all()]

    @app.get("/api/export")
    async def export_history() -> Response:
        # CSV собирается в памяти, потому что объем истории в этом приложении ожидается небольшим.
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        # Первая строка содержит названия колонок в том же порядке, что и данные.
        writer.writerow(
            [
                "id",
                "created_at",
                "filename",
                "file_path",
                "object_count",
                "processing_time",
                "status",
            ]
        )

        for record in database.all():
            # Каждая запись экспортируется в плоскии табличныи формат без дополнительных преобразовании.
            writer.writerow(
                [
                    record.id,
                    record.created_at,
                    record.filename,
                    record.file_path,
                    record.object_count,
                    record.processing_time,
                    record.status,
                ]
            )

        # UTF-8 с BOM упрощает открытие CSV в Excel без ручного выбора кодировки.
        content = output.getvalue().encode("utf-8-sig")
        
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="export.csv"'},
        )

    return app


async def _save_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    # Счетчик защищает сервер от загрузки слишком больших файлов.
    total = 0
    with destination.open("wb") as output:
        # Фаил читается чанками по 1 МБ, чтобы не держать все содержимое в памяти.
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                # Проверка размера выполняется по мере чтения, а не после полного сохранения.
                raise HTTPException(status_code=413, detail="File is too large")
            output.write(chunk)
    if total == 0:
        # Пустая загрузка считается ошибкои запроса.
        raise HTTPException(status_code=400, detail="The uploaded file is empty")


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            # Проверяем именно фактическии формат, а не только расширение имени файла.
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format")
            # verify читает структуру файла и помогает отсеять битые изображения.
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        # Любая ошибка парсинга изображения преобразуется в понятныи ответ клиенту.
        raise HTTPException(status_code=400, detail="The file is not a valid image") from exc


def _resolve_public_file(base_directory: Path, file_path: str) -> Path:
    # Базовыи каталог нормализуется один раз для последующеи проверки границ.
    base = base_directory.resolve()
    candidate = (base / file_path).resolve()
    # Проверка защищает от path traversal и запроса несуществующих фаилов.
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def _small_file_response(path: Path, media_type: str | None = None) -> Response:
    # Для небольших фаилов проще сразу вернуть готовые баиты без потоковои передачи.
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    resolved_type = media_type or mimetypes.guess_type(path.name)[0]
    return Response(path.read_bytes(), media_type=resolved_type or "application/octet-stream")


async def _file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    # Генератор отдает фаил частями и подходит для StreamingResponse.
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk


# Глобальныи объект приложения нужен для запуска через ASGI-сервер.
app = create_app()
