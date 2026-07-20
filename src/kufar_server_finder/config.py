from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_ANALYSIS_MODEL = "gemini-3.1-flash-lite"
DEFAULT_SPECS_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SIZE = 30
DEFAULT_SPECS_CHUNK_SIZE = 25
DEFAULT_REQUEST_DELAY = 2.0
DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    api_key: str
    analysis_model: str = DEFAULT_ANALYSIS_MODEL
    specs_model: str = DEFAULT_SPECS_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    specs_chunk_size: int = DEFAULT_SPECS_CHUNK_SIZE
    request_delay: float = DEFAULT_REQUEST_DELAY
    max_retries: int = DEFAULT_MAX_RETRIES
    max_description_chars: int = 1_200

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Переменная GEMINI_API_KEY не задана. "
                "Скопируйте .env.example в .env и укажите ключ."
            )

        return cls(
            api_key=api_key,
            analysis_model=os.getenv(
                "GEMINI_ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL
            ),
            specs_model=os.getenv("GEMINI_SPECS_MODEL", DEFAULT_SPECS_MODEL),
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
        )


@dataclass(frozen=True, slots=True)
class KufarConfig:
    region: str = "7"
    category_computers: str = "16020"
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


def _non_negative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} не может быть отрицательным")
    return value
