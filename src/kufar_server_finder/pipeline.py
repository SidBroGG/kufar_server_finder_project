from __future__ import annotations

import logging
from typing import Any, Protocol

from .models import AdAnalysis, PCComponentSpec

logger = logging.getLogger(__name__)


class AdsAnalyzer(Protocol):
    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]: ...

    def infer_specs(self, ads: list[dict[str, Any]]) -> list[PCComponentSpec]: ...


class AdPipeline:
    def __init__(self, analyzer: AdsAnalyzer) -> None:
        self.analyzer = analyzer

    def filter_working_targets(
        self,
        ads: list[dict[str, Any]],
        *,
        infer_specs: bool = False,
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

        if infer_specs and filtered:
            self._merge_specs(filtered)

        logger.info("После AI-фильтрации осталось объявлений: %s", len(filtered))
        return filtered

    def _merge_specs(self, ads: list[dict[str, Any]]) -> None:
        specs_by_link = {
            item.link: item for item in self.analyzer.infer_specs(ads)
        }
        for ad in ads:
            spec = specs_by_link.get(ad.get("link"))
            if spec:
                ad["cpu_model"] = spec.cpu_model
                ad["ram_type"] = spec.ram_type
                ad["ram_gb"] = spec.ram_gb
