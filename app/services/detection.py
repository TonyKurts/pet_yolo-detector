import asyncio
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Protocol

from app.database import Database


logger = logging.getLogger(__name__)


class Detector(Protocol):
    def process(self, source: Path, destination: Path) -> int: ...


class YoloPersonDetector:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from ultralytics import YOLO

                    self._model = YOLO(self.model_path)
        return self._model

    def process(self, source: Path, destination: Path) -> int:
        model = self._get_model()
        # COCO class 0 is person. The filter is deliberately applied at inference.
        results = model.predict(source=str(source), classes=[0], verbose=False)
        if not results:
            raise RuntimeError("YOLO did not return an inference result")

        result = results[0]
        plotted = result.plot()
        destination.parent.mkdir(parents=True, exist_ok=True)

        import cv2

        if not cv2.imwrite(str(destination), plotted):
            raise RuntimeError("Could not save the processed image")
        return len(result.boxes)


def run_detection_task(
    *,
    task_id: str,
    source: Path,
    destination: Path,
    relative_destination: str,
    database: Database,
    detector: Detector,
) -> None:
    started_at = perf_counter()
    try:
        object_count = detector.process(source, destination)
        elapsed = round(perf_counter() - started_at, 3)
        database.finish_processing(
            task_id,
            file_path=relative_destination,
            object_count=object_count,
            processing_time=elapsed,
        )
    except Exception:
        elapsed = round(perf_counter() - started_at, 3)
        database.fail_processing(task_id, processing_time=elapsed)
        logger.exception("Image processing failed for task %s", task_id)


async def run_detection_in_background(**kwargs) -> None:
    await asyncio.to_thread(run_detection_task, **kwargs)
