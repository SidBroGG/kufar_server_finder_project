"""Server-finder-specific Gemini operations built on kufar-finder-core."""
from __future__ import annotations

import json
import logging
from typing import Any

from kufar_finder_core import GeminiEngine

from .models import (
    AdAnalysis,
    CpuNameNormalization,
    VisionComponentSpec,
)
from .prompts import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    CPU_NAME_NORMALIZATION_SYSTEM_INSTRUCTION,
    VISION_SPECS_SYSTEM_INSTRUCTION,
)
from .visual_refinement import fields_needing_visual_analysis

logger = logging.getLogger(__name__)


class GeminiAnalyzer(GeminiEngine):
    """Adds the application's prompts and response models to GeminiEngine."""

    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]:
        payload = [self._analysis_payload(ad) for ad in ads]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.chunk_size,
            max_chunk_chars=self.config.max_chunk_chars,
            model=self.config.analysis_model,
            instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_model=AdAnalysis,
            prompt_prefix="Проанализируй объявления и извлеки характеристики",
        )

    def normalize_cpu_names(
        self,
        ads: list[dict[str, Any]],
    ) -> list[CpuNameNormalization]:
        payload = [
            {"link": ad.get("link"), "cpu_model": ad.get("cpu_model")}
            for ad in ads
        ]
        return self._process_chunks(
            payload=payload,
            chunk_size=self.config.specs_chunk_size,
            max_chunk_chars=self.config.specs_max_chunk_chars,
            model=self.config.specs_model,
            instruction=CPU_NAME_NORMALIZATION_SYSTEM_INSTRUCTION,
            response_model=CpuNameNormalization,
            prompt_prefix="Нормализуй названия процессоров перед поиском benchmark",
        )

    def infer_specs_from_images(
        self,
        ads: list[dict[str, Any]],
    ) -> list[VisionComponentSpec]:
        candidates = [ad for ad in ads if fields_needing_visual_analysis(ad)]
        total = len(candidates)
        tasks = list(enumerate(candidates, start=1))

        def process(worker: Any, task: tuple[int, dict[str, Any]]):
            index, ad = task
            requested_fields = fields_needing_visual_analysis(ad)
            image_parts = self._download_image_parts(
                ad,
                image_session=worker.image_session,
            )
            if not image_parts:
                logger.info(
                    "Фото-анализ пропущен, изображения недоступны: %s",
                    ad.get("link"),
                )
                return None

            logger.info(
                "Worker %s: фото-анализ объявления %s из %s",
                worker.number,
                index,
                total,
            )
            payload = json.dumps(
                {
                    "link": ad.get("link"),
                    "fields_to_analyze": requested_fields,
                    "existing_values": {
                        field: ad.get(field) for field in requested_fields
                    },
                },
                ensure_ascii=False,
            )
            return self._generate_single_structured(
                worker=worker,
                model=self.config.vision_model,
                contents=[
                    "Проанализируй фотографии. Входные данные: " + payload,
                    *image_parts,
                ],
                instruction=VISION_SPECS_SYSTEM_INSTRUCTION,
                response_model=VisionComponentSpec,
            )

        values = self._run_parallel(tasks, operation=process, fallback=lambda: None)
        return [value for value in values if value is not None]

    def _analysis_payload(self, ad: dict[str, Any]) -> dict[str, Any]:
        return {
            "link": ad.get("link"),
            "title": ad.get("title", ""),
            "price": ad.get("price", 0),
            "description": self._trim_description(ad),
            "characteristics": ad.get("characteristics") or {},
        }

    def _trim_description(
        self,
        ad: dict[str, Any],
        limit: int | None = None,
    ) -> str:
        description = str(ad.get("description") or "")
        return description[: limit or self.config.max_description_chars]
