from __future__ import annotations

import logging
from typing import Any, Protocol

from .models import AdAnalysis, PCComponentSpec, VisionComponentSpec
from .socket_inference import infer_socket_from_cpu

logger = logging.getLogger(__name__)


class AdsAnalyzer(Protocol):
    def analyze_ads(self, ads: list[dict[str, Any]]) -> list[AdAnalysis]: ...

    def extract_explicit_specs(
        self,
        ads: list[dict[str, Any]],
    ) -> list[PCComponentSpec]: ...

    def infer_specs_from_images(
        self,
        ads: list[dict[str, Any]],
    ) -> list[VisionComponentSpec]: ...


class AdPipeline:
    def __init__(self, analyzer: AdsAnalyzer) -> None:
        self.analyzer = analyzer

    def filter_working_targets(
        self,
        ads: list[dict[str, Any]],
        *,
        extract_specs: bool = False,
        infer_specs: bool | None = None,
    ) -> list[dict[str, Any]]:
        # Совместимость со старым именем аргумента.
        if infer_specs is not None:
            extract_specs = infer_specs
        if not ads:
            return []

        analyses = self.analyzer.analyze_ads(ads)
        analyses_by_link = {item.link: item for item in analyses}
        filtered: list[dict[str, Any]] = []
        pending_count = 0

        for ad in ads:
            link = ad.get("link")
            analysis = analyses_by_link.get(link)
            description_failed = bool(ad.get("description_load_error")) or (
                ad.get("description_status") == "load_error"
            )

            if analysis is None:
                item = dict(ad)
                item["analysis_status"] = "pending"
                item["analysis_error"] = "Gemini не вернул результат для объявления"
                filtered.append(item)
                pending_count += 1
                continue

            # При сетевой ошибке описания отрицательный вывод ненадёжен:
            # сохраняем объявление для повторной обработки вместо удаления.
            if description_failed and not (analysis.is_target and analysis.is_working):
                item = dict(ad)
                item["analysis_status"] = "pending"
                item["analysis_error"] = (
                    "Описание не загрузилось; отрицательный AI-результат не применён"
                )
                filtered.append(item)
                pending_count += 1
                continue

            if not analysis.is_target or not analysis.is_working:
                continue

            item = dict(ad)
            if analysis.real_price > 0:
                item["price"] = analysis.real_price
            filtered.append(item)

        if extract_specs and filtered:
            self._merge_explicit_specs(filtered)
        self._infer_sockets_from_cpu_models(filtered)

        logger.info(
            "После AI-фильтрации осталось объявлений: %s; ожидают повтора: %s",
            len(filtered),
            pending_count,
        )
        return filtered

    def enrich_missing_specs_from_images(
        self,
        ads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = [dict(ad) for ad in ads]
        if result:
            # Сначала используем уже известную модель CPU, чтобы не отправлять
            # фотографии в Gemini только ради сокета.
            self._infer_sockets_from_cpu_models(result)
            self._merge_vision_specs(result)
            self._infer_sockets_from_cpu_models(result)
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
            self._set_text_socket(
                ad,
                spec.cpu_socket,
                spec.cpu_socket_source,
                spec.cpu_socket_confidence,
            )

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
            self._set_vision_guess(
                ad,
                "cpu_socket",
                spec.cpu_socket,
                spec.cpu_socket_confidence,
            )
            self._set_vision_guess(
                ad,
                "product_type",
                spec.product_type,
                spec.product_type_confidence,
            )
            self._set_vision_guess(
                ad,
                "estimated_system_power_w",
                spec.estimated_system_power_w,
                spec.estimated_system_power_w_confidence,
            )

    @staticmethod
    def _set_text_socket(
        ad: dict[str, Any],
        value: str | None,
        source: str | None,
        confidence: str | None,
    ) -> None:
        if ad.get("cpu_socket") not in (None, "") or value is None:
            return
        resolved_source = source or "description_guess"
        ad["cpu_socket"] = value
        ad["cpu_socket_source"] = resolved_source
        if resolved_source == "text_exact":
            ad.pop("cpu_socket_confidence", None)
        elif confidence is not None:
            ad["cpu_socket_confidence"] = confidence

    @staticmethod
    def _infer_sockets_from_cpu_models(ads: list[dict[str, Any]]) -> None:
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        for ad in ads:
            if ad.get("cpu_socket") not in (None, ""):
                continue
            guess = infer_socket_from_cpu(ad.get("cpu_model"))
            if guess is None:
                continue

            confidence = guess.confidence
            cpu_confidence = ad.get("cpu_model_confidence")
            if cpu_confidence in confidence_order:
                confidence = min(
                    confidence,
                    cpu_confidence,
                    key=confidence_order.__getitem__,
                )

            ad["cpu_socket"] = guess.socket
            ad["cpu_socket_source"] = "cpu_model_guess"
            ad["cpu_socket_confidence"] = confidence

    @staticmethod
    def _set_exact_value(ad: dict[str, Any], field: str, value: Any) -> None:
        if value is None:
            return
        ad[field] = value
        ad[f"{field}_source"] = "text_exact"
        ad.pop(f"{field}_confidence", None)

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
