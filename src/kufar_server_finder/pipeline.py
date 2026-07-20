from __future__ import annotations

import logging
from typing import Any, Protocol

from .models import AdAnalysis, PCComponentSpec, VisionComponentSpec

logger = logging.getLogger(__name__)


class AdsAnalyzer(Protocol):
    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]: ...

    def extract_explicit_specs(
        self, ads: list[dict[str, Any]]
    ) -> list[PCComponentSpec]: ...

    def infer_specs_from_images(
        self, ads: list[dict[str, Any]]
    ) -> list[VisionComponentSpec]: ...


class AdPipeline:
    def __init__(self, analyzer: AdsAnalyzer) -> None:
        self.analyzer = analyzer

    def filter_working_targets(
        self,
        ads: list[dict[str, Any]],
        *,
        extract_specs: bool = False,
    ) -> list[dict[str, Any]]:
        if not ads:
            return []

        analyses = self.analyzer.analyze_ads(ads)
        by_link = {item.link: item for item in analyses}

        filtered: list[dict[str, Any]] = []
        for ad in ads:
            link = ad.get("link")
            analysis = by_link.get(link)
            if not analysis or not analysis.is_target or not analysis.is_working:
                continue

            item = dict(ad)
            item["price"] = analysis.real_price
            filtered.append(item)

        if extract_specs and filtered:
            self._merge_explicit_specs(filtered)

        logger.info("После AI-фильтрации осталось объявлений: %s", len(filtered))
        return filtered

    def enrich_missing_specs_from_images(
        self,
        ads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = [dict(ad) for ad in ads]
        if result:
            self._merge_vision_specs(result)
        return result

    def _merge_explicit_specs(self, ads: list[dict[str, Any]]) -> None:
        specs_by_link = {
            item.link: item for item in self.analyzer.extract_explicit_specs(ads)
        }
        for ad in ads:
            spec = specs_by_link.get(ad.get("link"))
            if not spec:
                continue
            self._set_exact_value(ad, "cpu_model", spec.cpu_model)
            self._set_exact_value(ad, "ram_type", spec.ram_type)
            self._set_exact_value(ad, "ram_gb", spec.ram_gb)

    def _merge_vision_specs(self, ads: list[dict[str, Any]]) -> None:
        specs_by_link = {
            item.link: item for item in self.analyzer.infer_specs_from_images(ads)
        }
        for ad in ads:
            spec = specs_by_link.get(ad.get("link"))
            if not spec:
                continue
            self._set_vision_guess(
                ad,
                "cpu_model",
                spec.cpu_model,
                spec.cpu_model_confidence,
            )
            self._set_vision_guess(
                ad,
                "ram_type",
                spec.ram_type,
                spec.ram_type_confidence,
            )
            self._set_vision_guess(
                ad,
                "ram_gb",
                spec.ram_gb,
                spec.ram_gb_confidence,
            )

    @staticmethod
    def _set_exact_value(ad: dict[str, Any], field: str, value: Any) -> None:
        if value is None:
            return
        ad[field] = value
        ad[f"{field}_source"] = "text_exact"

    @staticmethod
    def _set_vision_guess(
        ad: dict[str, Any],
        field: str,
        value: Any,
        confidence: str | None,
    ) -> None:
        if ad.get(field) not in (None, "") or value is None:
            return
        ad[field] = value
        ad[f"{field}_source"] = "image_guess"
        if confidence is not None:
            ad[f"{field}_confidence"] = confidence
