from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_ANALYSIS_MODEL = "gemini-3.1-flash-lite"
DEFAULT_SPECS_MODEL = "gemini-3.1-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_CHUNK_SIZE = 30
DEFAULT_SPECS_CHUNK_SIZE = 25
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_VISION_MAX_IMAGES = 5
DEFAULT_IMAGE_TIMEOUT = 20.0
GEMINI_WORKER_COUNT = 3
GEMINI_KEYS_PER_WORKER = 3
GEMINI_API_KEY_COUNT = GEMINI_WORKER_COUNT * GEMINI_KEYS_PER_WORKER
GEMINI_API_KEY_ENV_NAMES = (
    "GEMINI_API_KEY",
    *(f"GEMINI_API_KEY_{index}" for index in range(2, GEMINI_API_KEY_COUNT + 1)),
)


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    api_key: str
    backup_api_keys: tuple[str, ...] = ()
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

    @property
    def api_keys(self) -> tuple[str, ...]:
        return (self.api_key, *self.backup_api_keys)

    @property
    def worker_api_key_groups(self) -> tuple[tuple[str, ...], ...]:
        keys = self.api_keys
        if len(keys) != GEMINI_API_KEY_COUNT:
            raise ValueError(
                "Для трёх Gemini worker требуется ровно 9 API-ключей: "
                "GEMINI_API_KEY и GEMINI_API_KEY_2 ... GEMINI_API_KEY_9."
            )
        if len(set(keys)) != len(keys):
            raise ValueError("Все 9 Gemini API-ключей должны быть уникальными.")

        return tuple(
            tuple(keys[start : start + GEMINI_KEYS_PER_WORKER])
            for start in range(0, GEMINI_API_KEY_COUNT, GEMINI_KEYS_PER_WORKER)
        )

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        load_dotenv()
        api_keys = tuple(
            os.getenv(name, "").strip() for name in GEMINI_API_KEY_ENV_NAMES
        )
        missing = [
            name for name, value in zip(GEMINI_API_KEY_ENV_NAMES, api_keys) if not value
        ]
        if missing:
            raise ValueError(
                "Для трёх Gemini worker задайте все 9 API-ключей. "
                f"Не заполнены: {', '.join(missing)}."
            )
        if len(set(api_keys)) != GEMINI_API_KEY_COUNT:
            raise ValueError("Все 9 Gemini API-ключей должны быть уникальными.")

        return cls(
            api_key=api_keys[0],
            backup_api_keys=api_keys[1:],
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
