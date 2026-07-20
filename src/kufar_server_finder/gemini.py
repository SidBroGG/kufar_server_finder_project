from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, TypeAdapter

from .config import GeminiConfig
from .models import AdAnalysis, PCComponentSpec
from .prompts import ANALYSIS_SYSTEM_INSTRUCTION, SPECS_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


class GeminiAnalyzer:
    def __init__(
        self,
        config: GeminiConfig,
        client: genai.Client | None = None,
    ) -> None:
        self.config = config
        self.client = client or genai.Client(api_key=config.api_key)

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

    def infer_specs(self, ads: list[dict[str, Any]]) -> list[PCComponentSpec]:
        payload = [self._specs_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
            model=self.config.specs_model,
            instruction=SPECS_SYSTEM_INSTRUCTION,
            response_model=PCComponentSpec,
            prompt_prefix="Определи характеристики",
        )

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
                self._generate_structured(
                    model=model,
                    prompt=prompt,
                    instruction=instruction,
                    response_model=response_model,
                )
            )
            if start + chunk_size < total and self.config.request_delay:
                time.sleep(self.config.request_delay)
        return result

    def _generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        instruction: str,
        response_model: type[ModelT],
    ) -> list[ModelT]:
        adapter = TypeAdapter(list[response_model])
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        response_mime_type="application/json",
                        response_schema=list[response_model],
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini вернул пустой ответ")
                return adapter.validate_python(json.loads(response.text))
            except Exception as exc:  # SDK выбрасывает несколько типов ошибок
                last_error = exc
                logger.warning(
                    "Ошибка Gemini, попытка %s/%s: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        logger.error("Чанк пропущен после повторных ошибок: %s", last_error)
        return []

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
