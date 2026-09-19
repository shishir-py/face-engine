"""Configuration loader for FaceMatch AI."""

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class PathsConfig(BaseModel):
    image_repository: str = "data/people"
    faiss_index: str = "data/index"
    database: str = "data/db/facematch.db"
    upload_temp: str = "data/uploads"


class FaceDetectionConfig(BaseModel):
    model_pack: str = "buffalo_l"
    min_face_size: int = 20
    detection_threshold: float = 0.5
    max_faces_per_image: int = 0


class FaceRecognitionConfig(BaseModel):
    embedding_dim: int = 512
    align_faces: bool = True


class SearchConfig(BaseModel):
    default_top_k: int = 10
    max_top_k: int = 100
    index_type: str = "flat"
    ivf_clusters: int = 100
    ivf_nprobe: int = 10


class ThresholdsConfig(BaseModel):
    high_confidence: float = 0.55
    possible_match: float = 0.40
    low_confidence: float = 0.30


class EnrollmentConfig(BaseModel):
    batch_size: int = 32
    skip_existing: bool = True
    supported_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    paths: PathsConfig = PathsConfig()
    face_detection: FaceDetectionConfig = FaceDetectionConfig()
    face_recognition: FaceRecognitionConfig = FaceRecognitionConfig()
    search: SearchConfig = SearchConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    enrollment: EnrollmentConfig = EnrollmentConfig()
    logging: LoggingConfig = LoggingConfig()


_config: AppConfig | None = None


def load_config(config_path: str | None = None) -> AppConfig:
    """Load configuration from YAML file."""
    global _config

    if _config is not None:
        return _config

    if config_path is None:
        config_path = os.environ.get(
            "FACEMATCH_CONFIG",
            str(Path(__file__).parent.parent / "config.yaml"),
        )

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        _config = AppConfig(**data)
    else:
        _config = AppConfig()

    return _config


def get_config() -> AppConfig:
    """Get the current configuration (load if not yet loaded)."""
    if _config is None:
        return load_config()
    return _config
