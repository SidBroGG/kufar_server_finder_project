from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, TypeAdapter

from .config import GeminiConfig
from .models import AdAnalysis, PCComponentSpec, VisionComponentSpec
from .prompts import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    SPECS_SYSTEM_INSTRUCTION,
    VISION_SPECS_SYSTEM_INSTRUCTION,
)

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


class GeminiAnalyzer:
    def __init__(
        self,
        config: GeminiConfig,
        client: genai.Client | None = None,
        image_session: requests.Session | None = None,
    ) -> None:
        self.config = config

        if client is not None:
            self._clients = [client]
        else:
            self._clients = [
                genai.Client(api_key=api_key) for api_key in config.api_keys
            ]

        self._client_index = 0
        self.client = self._clients[0]
        self.image_session = image_session or requests.Session()
        self.image_session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                )
            }
        )

    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]:
        payload = [self._analysis_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.chunk_size,
            model=self.config.analysis_model,
            instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_model=AdAnalysis,
            prompt_prefix="Проанализируй объявления",
        )

    def extract_explicit_specs(
        self, ads: list[dict[str, Any]]
    ) -> list[PCComponentSpec]:
        payload = [self._specs_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
            model=self.config.specs_model,
            instruction=SPECS_SYSTEM_INSTRUCTION,
            response_model=PCComponentSpec,
            prompt_prefix="Извлеки только явно написанные характеристики",
        )

    def infer_specs(self, ads: list[dict[str, Any]]) -> list[PCComponentSpec]:
        """Обратная совместимость: теперь метод не делает предположений."""
        return self.extract_explicit_specs(ads)

    def infer_specs_from_images(
        self, ads: list[dict[str, Any]]
    ) -> list[VisionComponentSpec]:
        results: list[VisionComponentSpec] = []
        candidates = [ad for ad in ads if self._missing_spec_fields(ad)]
        total = len(candidates)

        for index, ad in enumerate(candidates, start=1):
            missing_fields = self._missing_spec_fields(ad)
            image_parts = self._download_image_parts(ad)
            if not image_parts:
                logger.info(
                    "Фото-анализ пропущен, изображения недоступны: %s",
                    ad.get("link"),
                )
                continue

            logger.info("Фото-анализ объявления %s из %s", index, total)
            prompt = json.dumps(
                {
                    "link": ad.get("link"),
                    "missing_fields": missing_fields,
                },
                ensure_ascii=False,
            )
            contents: list[Any] = [
                "Проанализируй фотографии. Входные данные: " + prompt,
                *image_parts,
            ]
            spec = self._generate_single_structured(
                model=self.config.vision_model,
                contents=contents,
                instruction=VISION_SPECS_SYSTEM_INSTRUCTION,
                response_model=VisionComponentSpec,
            )
            if spec is not None:
                results.append(spec)

            if index < total and self.config.request_delay:
                time.sleep(self.config.request_delay)

        return results

    def _process_chunks(
        self,
        *,
        payload: list[dict[str, Any]],
        chunk_size: int,
        model: str,
        instruction: str,
        response_model: type[ModelT],
        prompt_prefix: str,
    ) -> list[ModelT]:
        if not payload:
            return []

        result: list[ModelT] = []
        total = len(payload)
        for start in range(0, total, chunk_size):
            chunk = payload[start : start + chunk_size]
            logger.info(
                "AI-обработка объявлений %s–%s из %s",
                start + 1,
                start + len(chunk),
                total,
            )
            prompt = f"{prompt_prefix}:\n{json.dumps(chunk, ensure_ascii=False)}"
            result.extend(
                self._generate_structured_list(
                    model=model,
                    contents=prompt,
                    instruction=instruction,
                    response_model=response_model,
                )
            )
            if start + chunk_size < total and self.config.request_delay:
                time.sleep(self.config.request_delay)
        return result

    def _generate_structured_list(
        self,
        *,
        model: str,
        contents: Any,
        instruction: str,
        response_model: type[ModelT],
    ) -> list[ModelT]:
        adapter = TypeAdapter(list[response_model])
        value = self._request_json(
            model=model,
            contents=contents,
            instruction=instruction,
            response_schema=list[response_model],
            adapter=adapter,
        )
        return value if isinstance(value, list) else []

    def _generate_single_structured(
        self,
        *,
        model: str,
        contents: Any,
        instruction: str,
        response_model: type[ModelT],
    ) -> ModelT | None:
        adapter = TypeAdapter(response_model)
        value = self._request_json(
            model=model,
            contents=contents,
            instruction=instruction,
            response_schema=response_model,
            adapter=adapter,
        )
        return value if isinstance(value, response_model) else None

    def _request_json(
        self,
        *,
        model: str,
        contents: Any,
        instruction: str,
        response_schema: Any,
        adapter: TypeAdapter[Any],
    ) -> Any | None:
        last_error: Exception | None = None
        rate_limit_attempts = 0
        other_attempts = 0
        max_rate_limit_attempts = self.config.max_retries * len(self._clients)

        while True:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini вернул пустой ответ")
                return adapter.validate_python(json.loads(response.text))
            except Exception as exc:  # SDK выбрасывает несколько типов ошибок
                last_error = exc

                if self._is_rate_limit_error(exc):
                    rate_limit_attempts += 1
                    previous_key = self._client_index + 1
                    self._rotate_api_key()
                    logger.warning(
                        "Gemini 429 на ключе %s/%s; следующий ключ %s/%s; "
                        "попытка %s/%s",
                        previous_key,
                        len(self._clients),
                        self._client_index + 1,
                        len(self._clients),
                        rate_limit_attempts,
                        max_rate_limit_attempts,
                    )
                    if rate_limit_attempts < max_rate_limit_attempts:
                        continue
                    break

                other_attempts += 1
                logger.warning(
                    "Ошибка Gemini, попытка %s/%s: %s",
                    other_attempts,
                    self.config.max_retries,
                    exc,
                )
                if other_attempts >= self.config.max_retries:
                    break
                time.sleep(min(2 ** (other_attempts - 1), 8))

        logger.error("Запрос пропущен после повторных ошибок: %s", last_error)
        return None

    def _rotate_api_key(self) -> None:
        self._client_index = (self._client_index + 1) % len(self._clients)
        self.client = self._clients[self._client_index]

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        for attribute in ("code", "status_code"):
            value = getattr(exc, attribute, None)
            if callable(value):
                value = value()
            try:
                if int(value) == 429:
                    return True
            except (TypeError, ValueError):
                pass

        message = str(exc).upper()
        return "429" in message or "RESOURCE_EXHAUSTED" in message

    def _download_image_parts(self, ad: dict[str, Any]) -> list[types.Part]:
        parts: list[types.Part] = []
        image_urls = ad.get("images") or []

        for url in image_urls[: self.config.vision_max_images]:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            try:
                response = self.image_session.get(
                    url,
                    timeout=self.config.image_timeout,
                )
                response.raise_for_status()
                mime_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                mime_type = mime_type.strip().lower()
                if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
                    logger.warning(
                        "Неподдерживаемый формат изображения %s: %s",
                        url,
                        mime_type,
                    )
                    continue
                parts.append(
                    types.Part.from_bytes(
                        data=response.content,
                        mime_type=mime_type,
                    )
                )
            except requests.RequestException as exc:
                logger.warning("Не удалось загрузить фото %s: %s", url, exc)

        return parts

    @staticmethod
    def _missing_spec_fields(ad: dict[str, Any]) -> list[str]:
        return [
            field
            for field in (
                "cpu_model",
                "ram_type",
                "ram_gb",
                "product_type",
                "estimated_system_power_w",
            )
            if ad.get(field) in (None, "")
        ]

    def _analysis_payload(self, ad: dict[str, Any]) -> dict[str, Any]:
        return {
            "link": ad.get("link"),
            "title": ad.get("title", ""),
            "price": ad.get("price", 0),
            "description": self._trim_description(ad),
            "characteristics": ad.get("characteristics") or {},
        }

    def _specs_payload(self, ad: dict[str, Any]) -> dict[str, Any]:
        return {
            "link": ad.get("link"),
            "title": ad.get("title", ""),
            "description": self._trim_description(ad, limit=600),
            "characteristics": ad.get("characteristics") or {},
        }

    def _trim_description(self, ad: dict[str, Any], limit: int | None = None) -> str:
        description = str(ad.get("description") or "")
        return description[: limit or self.config.max_description_chars]
