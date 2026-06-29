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


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}


def create_app(
    settings: Settings = default_settings,
    *,
    detector: Any | None = None,
    task_runner: Callable[..., Any] = run_detection_in_background,
) -> FastAPI:
    database = Database(settings.database_path)
    person_detector = detector or YoloPersonDetector(settings.model_path)
    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        yield

    app = FastAPI(
        title="Store Visitor Counter",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.database = database
    app.state.detector = person_detector
    app.state.settings = settings

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        return _small_file_response(settings.static_dir / "index.html", "text/html")

    @app.get("/static/{file_path:path}", include_in_schema=False)
    async def static_asset(file_path: str) -> Response:
        path = _resolve_public_file(settings.static_dir, file_path)
        return _small_file_response(path)

    @app.get("/storage/{file_path:path}", include_in_schema=False)
    async def stored_file(file_path: str) -> StreamingResponse:
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
        original_name = Path(file.filename or "").name
        extension = Path(original_name).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Only .png, .jpg and .jpeg files are supported",
            )

        task_id = str(uuid4())
        source_path = settings.originals_dir / f"{task_id}{extension}"
        
        try:
            await _save_upload(file, source_path, settings.max_upload_bytes)
            _verify_image(source_path)
        except HTTPException:
            source_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        try:
            database.create_processing(task_id, original_name)
        except ServerBusyError as exc:
            source_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        output_name = f"{uuid4()}.jpg"
        destination = settings.processed_dir / output_name
        relative_destination = f"processed/{output_name}"
        background_tasks.add_task(
            task_runner,
            task_id=task_id,
            source=source_path,
            destination=destination,
            relative_destination=relative_destination,
            database=database,
            detector=person_detector,
        )
        return {"task_id": task_id}

    @app.get("/api/status")
    async def task_status(task_id: str | None = None) -> dict[str, Any]:
        record = database.get(task_id) if task_id else database.latest()
        if task_id and record is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            "available": record is None or record.status != STATUS_PROCESSING,
            "task": record.as_dict() if record else None,
        }

    @app.get("/api/history")
    async def history() -> list[dict[str, Any]]:
        return [record.as_dict() for record in database.all()]

    @app.get("/api/export")
    async def export_history() -> Response:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
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

        content = output.getvalue().encode("utf-8-sig")
        
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="export.csv"'},
        )

    return app


async def _save_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    total = 0
    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="File is too large")
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image") from exc


def _resolve_public_file(base_directory: Path, file_path: str) -> Path:
    base = base_directory.resolve()
    candidate = (base / file_path).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def _small_file_response(path: Path, media_type: str | None = None) -> Response:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    resolved_type = media_type or mimetypes.guess_type(path.name)[0]
    return Response(path.read_bytes(), media_type=resolved_type or "application/octet-stream")


async def _file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk


app = create_app()
