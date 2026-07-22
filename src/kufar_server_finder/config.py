from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_ANALYSIS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SPECS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CHUNK_SIZE = 30
DEFAULT_SPECS_CHUNK_SIZE = 25
DEFAULT_MAX_CHUNK_CHARS = 25_000
DEFAULT_SPECS_MAX_CHUNK_CHARS = 20_000
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_WORKER_COUNT = 3
DEFAULT_VISION_MAX_IMAGES = 5
DEFAULT_IMAGE_DOWNLOAD_WORKERS = 3
DEFAULT_IMAGE_TIMEOUT = 20.0
DEFAULT_KUFAR_REGION = "7"
DEFAULT_KUFAR_REQUEST_TIMEOUT = 20.0
DEFAULT_KUFAR_PAGE_DELAY = 1.0
DEFAULT_KUFAR_DETAIL_DELAY = 1.0
DEFAULT_KUFAR_DETAIL_WORKERS = 3
DEFAULT_KUFAR_DETAIL_MAX_RETRIES = 3
DEFAULT_KUFAR_RATE_LIMIT_THRESHOLD = 3


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
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    specs_max_chunk_chars: int = DEFAULT_SPECS_MAX_CHUNK_CHARS
    request_delay: float = DEFAULT_REQUEST_DELAY
    max_retries: int = DEFAULT_MAX_RETRIES
    vision_max_images: int = DEFAULT_VISION_MAX_IMAGES
    image_download_workers: int = DEFAULT_IMAGE_DOWNLOAD_WORKERS
    image_timeout: float = DEFAULT_IMAGE_TIMEOUT
    max_description_chars: int = 1_200

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("GEMINI_API_KEY не может быть пустым")
        _validate_positive("GEMINI_WORKER_COUNT", self.worker_count)
        _validate_positive("GEMINI_CHUNK_SIZE", self.chunk_size)
        _validate_positive("GEMINI_SPECS_CHUNK_SIZE", self.specs_chunk_size)
        _validate_positive("GEMINI_MAX_CHUNK_CHARS", self.max_chunk_chars)
        _validate_positive(
            "GEMINI_SPECS_MAX_CHUNK_CHARS", self.specs_max_chunk_chars
        )
        _validate_positive("GEMINI_MAX_RETRIES", self.max_retries)
        _validate_positive("GEMINI_VISION_MAX_IMAGES", self.vision_max_images)
        _validate_positive(
            "GEMINI_IMAGE_DOWNLOAD_WORKERS", self.image_download_workers
        )
        if self.request_delay < 0:
            raise ValueError("GEMINI_REQUEST_DELAY не может быть отрицательным")
        if self.image_timeout <= 0:
            raise ValueError("GEMINI_IMAGE_TIMEOUT должен быть больше нуля")

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
            max_chunk_chars=_positive_int(
                "GEMINI_MAX_CHUNK_CHARS", DEFAULT_MAX_CHUNK_CHARS
            ),
            specs_max_chunk_chars=_positive_int(
                "GEMINI_SPECS_MAX_CHUNK_CHARS",
                DEFAULT_SPECS_MAX_CHUNK_CHARS,
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
            image_download_workers=_positive_int(
                "GEMINI_IMAGE_DOWNLOAD_WORKERS",
                DEFAULT_IMAGE_DOWNLOAD_WORKERS,
            ),
            image_timeout=_positive_float(
                "GEMINI_IMAGE_TIMEOUT", DEFAULT_IMAGE_TIMEOUT
            ),
        )


@dataclass(frozen=True, slots=True)
class KufarConfig:
    region: str = DEFAULT_KUFAR_REGION
    category_computers: str = "16020"
    category_laptops: str = "16040"
    page_size: int = 43
    request_timeout: float = DEFAULT_KUFAR_REQUEST_TIMEOUT
    page_delay: float = DEFAULT_KUFAR_PAGE_DELAY
    detail_delay: float = DEFAULT_KUFAR_DETAIL_DELAY
    detail_workers: int = DEFAULT_KUFAR_DETAIL_WORKERS
    detail_max_retries: int = DEFAULT_KUFAR_DETAIL_MAX_RETRIES
    rate_limit_threshold: int = DEFAULT_KUFAR_RATE_LIMIT_THRESHOLD
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )

    def __post_init__(self) -> None:
        if not self.region.strip():
            raise ValueError("KUFAR_REGION не может быть пустым")
        if self.request_timeout <= 0:
            raise ValueError("KUFAR_TIMEOUT должен быть больше нуля")
        if self.page_delay < 0:
            raise ValueError("KUFAR_PAGE_DELAY не может быть отрицательным")
        if self.detail_workers <= 0:
            raise ValueError("KUFAR_DETAIL_WORKERS должен быть больше нуля")
        if self.detail_max_retries <= 0:
            raise ValueError("KUFAR_DETAIL_RETRIES должен быть больше нуля")
        if self.rate_limit_threshold <= 0:
            raise ValueError("KUFAR_RATE_LIMIT_THRESHOLD должен быть больше нуля")
        if self.detail_delay < 0:
            raise ValueError("KUFAR_DETAIL_DELAY не может быть отрицательным")

    @classmethod
    def from_env(cls) -> "KufarConfig":
        load_dotenv()
        return cls(
            region=_string("KUFAR_REGION", DEFAULT_KUFAR_REGION),
            request_timeout=_positive_float(
                "KUFAR_TIMEOUT", DEFAULT_KUFAR_REQUEST_TIMEOUT
            ),
            page_delay=_non_negative_float(
                "KUFAR_PAGE_DELAY", DEFAULT_KUFAR_PAGE_DELAY
            ),
            detail_delay=_non_negative_float(
                "KUFAR_DETAIL_DELAY", DEFAULT_KUFAR_DETAIL_DELAY
            ),
            detail_workers=_positive_int(
                "KUFAR_DETAIL_WORKERS", DEFAULT_KUFAR_DETAIL_WORKERS
            ),
            detail_max_retries=_positive_int(
                "KUFAR_DETAIL_RETRIES", DEFAULT_KUFAR_DETAIL_MAX_RETRIES
            ),
        )


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} должен быть больше нуля")


def _optional_string(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _string(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} не может быть пустым")
    return value


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



