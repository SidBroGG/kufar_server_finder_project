from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_ANALYSIS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SPECS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CHUNK_SIZE = 30
DEFAULT_SPECS_CHUNK_SIZE = 25
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_WORKER_COUNT = 3
DEFAULT_VISION_MAX_IMAGES = 5
DEFAULT_IMAGE_TIMEOUT = 20.0


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    api_key: str
    worker_count: int = DEFAULT_WORKER_COUNT
    base_url: str | None = None
    api_version: str | None = None
    analysis_model: str = DEFAULT_ANALYSIS_MODEL
    specs_model: str = DEFAULT_SPECS_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    specs_chunk_size: int = DEFAULT_SPECS_CHUNK_SIZE
    request_delay: float = DEFAULT_REQUEST_DELAY
    max_retries: int = DEFAULT_MAX_RETRIES
    vision_max_images: int = DEFAULT_VISION_MAX_IMAGES
    image_timeout: float = DEFAULT_IMAGE_TIMEOUT
    max_description_chars: int = 1_200

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("GEMINI_API_KEY не может быть пустым")
        if self.worker_count <= 0:
            raise ValueError("GEMINI_WORKER_COUNT должен быть больше нуля")

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Задайте GEMINI_API_KEY в файле .env.")

        return cls(
            api_key=api_key,
            worker_count=_positive_int(
                "GEMINI_WORKER_COUNT", DEFAULT_WORKER_COUNT
            ),
            base_url=_optional_string("GEMINI_BASE_URL"),
            api_version=_optional_string("GEMINI_API_VERSION"),
            analysis_model=os.getenv(
                "GEMINI_ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL
            ),
            specs_model=os.getenv("GEMINI_SPECS_MODEL", DEFAULT_SPECS_MODEL),
            vision_model=os.getenv("GEMINI_VISION_MODEL", DEFAULT_VISION_MODEL),
            chunk_size=_positive_int("GEMINI_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            specs_chunk_size=_positive_int(
                "GEMINI_SPECS_CHUNK_SIZE", DEFAULT_SPECS_CHUNK_SIZE
            ),
            request_delay=_non_negative_float(
                "GEMINI_REQUEST_DELAY", DEFAULT_REQUEST_DELAY
            ),
            max_retries=_positive_int(
                "GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES
            ),
            vision_max_images=_positive_int(
                "GEMINI_VISION_MAX_IMAGES", DEFAULT_VISION_MAX_IMAGES
            ),
            image_timeout=_positive_float(
                "GEMINI_IMAGE_TIMEOUT", DEFAULT_IMAGE_TIMEOUT
            ),
        )


@dataclass(frozen=True, slots=True)
class KufarConfig:
    region: str = "7"
    category_computers: str = "16020"
    category_laptops: str = "16040"
    page_size: int = 43
    request_timeout: float = 20.0
    page_delay: float = 1.0
    detail_delay: float = 1.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )


def _optional_string(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} должен быть больше нуля")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} должен быть больше нуля")
    return value


def _non_negative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} не может быть отрицательным")
    return value
