from __future__ import annotations

import logging
from typing import Any, Protocol

from .models import AdAnalysis, PCComponentSpec, VisionComponentSpec
from .socket_inference import infer_socket_from_cpu
from .visual_refinement import should_replace_with_vision

logger = logging.getLogger(__name__)


_VISUAL_FALLBACKS: dict[str, tuple[str, str, int]] = {
    "desktop_pc": (
        "Intel Core / AMD Ryzen (примерное семейство)",
        "DDR3/DDR4 (примерно)",
        8,
    ),
    "laptop": (
        "Intel Core Mobile / AMD Ryzen Mobile (примерное семейство)",
        "DDR3L/DDR4 (примерно)",
        8,
    ),
    "mini_pc": (
        "Intel Celeron/Pentium/Core U / AMD Ryzen Embedded (примерное семейство)",
        "DDR3L/DDR4 (примерно)",
        8,
    ),
    "thin_client": (
        "Intel Atom/Celeron / AMD Embedded (примерное семейство)",
        "DDR3/DDR4 (примерно)",
        4,
    ),
    "server": (
        "Intel Xeon / AMD EPYC (примерное семейство)",
        "DDR3 ECC/DDR4 ECC (примерно)",
        16,
    ),
    "workstation": (
        "Intel Xeon/Core X / AMD Threadripper (примерное семейство)",
        "DDR4 ECC/DDR4 (примерно)",
        16,
    ),
    "all_in_one": (
        "Intel Core Mobile / AMD Ryzen Mobile (примерное семейство)",
        "DDR3L/DDR4 (примерно)",
        8,
    ),
    "motherboard_bundle": (
        "Intel Core / AMD Ryzen (примерное семейство)",
        "DDR3/DDR4 (примерно)",
        8,
    ),
    "other": (
        "Intel/AMD x86-64 (примерное семейство)",
        "DDR3/DDR4 (примерно)",
        8,
    ),
}

_SOCKET_CPU_FALLBACKS = {
    "LGA1155": "Intel Core 2nd/3rd gen (примерное семейство)",
    "LGA1150": "Intel Core 4th/5th gen (примерное семейство)",
    "LGA1151": "Intel Core 6th-9th gen (примерное семейство)",
    "LGA1200": "Intel Core 10th/11th gen (примерное семейство)",
    "LGA1700": "Intel Core 12th-14th gen (примерное семейство)",
    "LGA2011": "Intel Xeon E5 v1/v2 (примерное семейство)",
    "LGA2011-3": "Intel Xeon E5 v3/v4 (примерное семейство)",
    "AM3": "AMD Phenom II / Athlon II (примерное семейство)",
    "AM3+": "AMD FX (примерное семейство)",
    "AM4": "AMD Ryzen 1000-5000 (примерное семейство)",
    "AM5": "AMD Ryzen 7000-9000 (примерное семейство)",
    "BGA (SOLDERED)": "Intel/AMD Mobile или Embedded (примерное семейство)",
}

_SOCKET_RAM_FALLBACKS = {
    "LGA1155": "DDR3 (примерно)",
    "LGA1150": "DDR3 (примерно)",
    "LGA1151": "DDR4/DDR3L (примерно)",
    "LGA1200": "DDR4 (примерно)",
    "LGA1700": "DDR4/DDR5 (примерно)",
    "LGA2011": "DDR3 ECC/DDR3 (примерно)",
    "LGA2011-3": "DDR4 ECC/DDR4 (примерно)",
    "AM3": "DDR3 (примерно)",
    "AM3+": "DDR3 (примерно)",
    "AM4": "DDR4 (примерно)",
    "AM5": "DDR5 (примерно)",
}

_VISUAL_SOCKET_FALLBACKS = {
    "desktop_pc": "LGA115x/LGA1200/AM4 (примерно)",
    "laptop": "BGA (soldered, примерно)",
    "mini_pc": "BGA (soldered, примерно)",
    "thin_client": "BGA (soldered, примерно)",
    "server": "LGA2011/LGA3647/SP3 (примерно)",
    "workstation": "LGA2011-3/LGA2066/TR4 (примерно)",
    "all_in_one": "BGA (soldered, примерно)",
    "motherboard_bundle": "LGA115x/LGA1200/AM4 (примерно)",
    "other": "Неизвестный сокет (примерно)",
}

_VISUAL_POWER_FALLBACKS = {
    "desktop_pc": 150,
    "laptop": 65,
    "mini_pc": 35,
    "thin_client": 25,
    "server": 250,
    "workstation": 300,
    "all_in_one": 90,
    "motherboard_bundle": 100,
    "other": 100,
}


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
            self._fill_missing_vision_estimates(result)
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


    @classmethod
    def _fill_missing_vision_estimates(
        cls,
        ads: list[dict[str, Any]],
    ) -> None:
        """Гарантирует непустые visual-поля и их confidence."""
        for ad in ads:
            cls._set_visual_fallback(ad, "product_type", "other")
            product_type = str(ad.get("product_type") or "other")
            if product_type not in _VISUAL_FALLBACKS:
                product_type = "other"

            cpu_fallback, ram_type_fallback, ram_gb_fallback = (
                _VISUAL_FALLBACKS[product_type]
            )

            socket = str(ad.get("cpu_socket") or "").strip().upper()
            cpu_fallback = _SOCKET_CPU_FALLBACKS.get(socket, cpu_fallback)
            ram_type_fallback = _SOCKET_RAM_FALLBACKS.get(
                socket,
                ram_type_fallback,
            )

            cls._set_visual_fallback(ad, "cpu_model", cpu_fallback)
            cls._set_visual_fallback(ad, "ram_type", ram_type_fallback)
            cls._set_visual_fallback(ad, "ram_gb", ram_gb_fallback)
            cls._set_visual_fallback(
                ad,
                "cpu_socket",
                _VISUAL_SOCKET_FALLBACKS[product_type],
            )
            cls._set_visual_fallback(
                ad,
                "estimated_system_power_w",
                _VISUAL_POWER_FALLBACKS[product_type],
            )

            cls._ensure_confidence(ad, "cpu_model")
            cls._ensure_confidence(ad, "ram_type")
            cls._ensure_confidence(ad, "ram_gb")
            cls._ensure_confidence(ad, "cpu_socket")
            cls._ensure_confidence(ad, "product_type")
            cls._ensure_confidence(ad, "estimated_system_power_w")

    @staticmethod
    def _set_visual_fallback(
        ad: dict[str, Any],
        field: str,
        value: Any,
    ) -> None:
        current = ad.get(field)
        is_missing = current is None or (
            isinstance(current, str) and not current.strip()
        )
        if field in {"ram_gb", "estimated_system_power_w"}:
            try:
                is_missing = current is None or int(current) <= 0
            except (TypeError, ValueError):
                is_missing = True
        elif field == "product_type":
            is_missing = current not in _VISUAL_FALLBACKS

        if not is_missing:
            return

        ad[field] = value
        ad[f"{field}_source"] = "visual_fallback"
        ad[f"{field}_confidence"] = "low"

    @staticmethod
    def _ensure_confidence(ad: dict[str, Any], field: str) -> None:
        confidence_field = f"{field}_confidence"
        if ad.get(confidence_field) in {"low", "medium", "high"}:
            return

        source = ad.get(f"{field}_source")
        ad[confidence_field] = "high" if source == "text_exact" else "low"

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
        if not should_replace_with_vision(ad, field, value, confidence):
            return

        value_changed = ad.get(field) != value
        ad[field] = value
        ad[f"{field}_source"] = "image_guess"
        if confidence is not None:
            ad[f"{field}_confidence"] = confidence

        if field == "cpu_model" and value_changed:
            # Старый benchmark и вычисленный по старой модели сокет становятся
            # недостоверными после уточнения CPU по фотографии.
            ad.pop("cpu_mark", None)
            ad.pop("cpu_benchmark_name", None)
            ad.pop("cpu_benchmark_source", None)
            socket_source = ad.get("cpu_socket_source")
            socket_confidence = ad.get("cpu_socket_confidence")
            if socket_source in {"cpu_model_guess", "visual_fallback"} or (
                socket_source == "image_guess" and socket_confidence == "low"
            ):
                ad.pop("cpu_socket", None)
                ad.pop("cpu_socket_source", None)
                ad.pop("cpu_socket_confidence", None)
