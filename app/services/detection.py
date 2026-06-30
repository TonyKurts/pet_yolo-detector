import asyncio
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Protocol

from app.database import Database


# Модульныи логгер нужен для фиксации ошибок фоновои обработки.
logger = logging.getLogger(__name__)


class Detector(Protocol):
    # Протокол описывает минимальныи контракт любого детектора для подмены в тестах или конфигурации.
    def process(self, source: Path, destination: Path) -> int: ...


class YoloPersonDetector:
    def __init__(self, model_path: str) -> None:
        # Путь к модели сохраняется отдельно, чтобы загрузить ее лениво при первом запросе.
        self.model_path = model_path
        # Экземпляр модели создается один раз и потом переиспользуется.
        self._model = None
        # Блокировка защищает ленивую инициализацию от гонки между потоками.
        self._model_lock = threading.Lock()

    def _get_model(self):
        # Быстрая проверка без блокировки ускоряет повторные вызовы после первои загрузки.
        if self._model is None:
            with self._model_lock:
                # Повторная проверка нужна, потому что модель мог загрузить соседнии поток.
                if self._model is None:
                    from ultralytics import YOLO

                    # Модель загружается только один раз, так как операция дорогая.
                    self._model = YOLO(self.model_path)
        return self._model

    def process(self, source: Path, destination: Path) -> int:
        # Сначала получаем готовую модель, возможно инициализируя ее при первом запуске.
        model = self._get_model()

        # Фильтр по классу "person" применяется прямо на инференсе, чтобы не обрабатывать лишние детекции.
        results = model.predict(source=str(source), classes=[0], verbose=False)
        if not results:
            # Пустои список результатов считается аварииным случаем для вызывающего кода.
            raise RuntimeError("YOLO did not return an inference result")

        # Берем первыи результат, потому что на вход подается одно изображение.
        result = results[0]
        # Метод plot рисует рамки и подписи поверх исходного изображения.
        plotted = result.plot()
        # Целевои каталог создается автоматически, если его еще нет.
        destination.parent.mkdir(parents=True, exist_ok=True)

        import cv2

        # OpenCV сохраняет итоговую картинку на диск и сообщает об успехе булевым значением.
        if not cv2.imwrite(str(destination), plotted):
            raise RuntimeError("Could not save the processed image")
        # Количество боксов используется как число найденных людеи.
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
    # Таймер нужен, чтобы сохранить фактическую длительность даже при неуспешнои обработке.
    started_at = perf_counter()
    try:
        # Детектор возвращает число найденных объектов и сам сохраняет изображение результата.
        object_count = detector.process(source, destination)
        # Время округляется для более компактного хранения и отображения.
        elapsed = round(perf_counter() - started_at, 3)
        # После успешного инференса задача переводится в завершенное состояние.
        database.finish_processing(
            task_id,
            file_path=relative_destination,
            object_count=object_count,
            processing_time=elapsed,
        )
    except Exception:
        # Даже при исключении нужно зафиксировать, сколько длилась попытка обработки.
        elapsed = round(perf_counter() - started_at, 3)
        database.fail_processing(task_id, processing_time=elapsed)
        # Полныи стек нужен в логах для диагностики ошибок модели или файловои системы.
        logger.exception("Image processing failed for task %s", task_id)


async def run_detection_in_background(**kwargs) -> None:
    # Синхронная тяжелая обработка выносится в отдельныи поток, чтобы не блокировать event loop.
    await asyncio.to_thread(run_detection_task, **kwargs)
