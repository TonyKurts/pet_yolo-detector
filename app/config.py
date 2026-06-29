from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = BASE_DIR / "visitors.db"
    storage_dir: Path = BASE_DIR / "storage"
    static_dir: Path = BASE_DIR / "app" / "static"
    model_path: str = "yolo11n.pt"
    max_upload_bytes: int = 15 * 1024 * 1024

    @property
    def originals_dir(self) -> Path:
        return self.storage_dir / "originals"

    @property
    def processed_dir(self) -> Path:
        return self.storage_dir / "processed"


settings = Settings()
