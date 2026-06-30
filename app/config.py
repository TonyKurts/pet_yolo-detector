from dataclasses import dataclass
from pathlib import Path


# Базовая директория проекта нужна для построения остальных путей от корня репозитория.
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    # Путь к SQLite-файлу, в котором хранится история обработки.
    database_path: Path = BASE_DIR / "visitors.db"
    # Общая директория для исходных и обработанных изображении.
    storage_dir: Path = BASE_DIR / "storage"
    # Каталог со статическими фронтенд-ресурсами.
    static_dir: Path = BASE_DIR / "app" / "static"
    # Имя или путь до YOLO-модели, которую загрузит ultralytics.
    model_path: str = "yolo11n.pt"
    # Максимально допустимыи размер загружаемого файла в баитах.
    max_upload_bytes: int = 15 * 1024 * 1024

    @property
    def originals_dir(self) -> Path:
        # Исходные загруженные изображения сохраняются отдельно от результатов обработки.
        return self.storage_dir / "originals"

    @property
    def processed_dir(self) -> Path:
        # Сюда складываются изображения с отрисованными рамками после детекции.
        return self.storage_dir / "processed"


# Единыи экземпляр настроек используется приложением по умолчанию.
settings = Settings()
