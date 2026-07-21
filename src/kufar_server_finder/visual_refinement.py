from __future__ import annotations

import re
from typing import Any, Mapping

VISUAL_SPEC_FIELDS = (
    "cpu_model",
    "cpu_socket",
    "ram_type",
    "ram_gb",
    "product_type",
    "estimated_system_power_w",
)

_CONFIDENCE_RANK = {None: -1, "low": 0, "medium": 1, "high": 2}
_APPROXIMATE_MARKERS = (
    "пример",
    "неизвест",
    "unknown",
    "вероят",
    "семейств",
    "класс процессора",
)

_CPU_MODEL_PATTERNS = (
    re.compile(r"\b(?:core\s+)?i[3579][\s-]?\d{3,5}[a-z0-9]*\b", re.I),
    re.compile(r"\bcore\s+ultra\s+[3579]\s+\d{3}[a-z0-9]*\b", re.I),
    re.compile(
        r"\b(?:atom(?:\s+cpu)?\s+)?(?:x[357][\s-])?[a-z]\d{3,5}[a-z0-9-]*\b",
        re.I,
    ),
    re.compile(
        r"\b(?:celeron|pentium)(?:\s+(?:gold|silver))?"
        r"\s+[a-z]?\d{3,5}[a-z0-9-]*\b",
        re.I,
    ),
    re.compile(r"\bxeon\s+[a-z0-9]+(?:[\s-]+[a-z0-9]+)+\b", re.I),
    re.compile(
        r"\b(?:ryzen|athlon)\s+(?:pro\s+)?[3579]?\s*\d{4}[a-z0-9]*\b",
        re.I,
    ),
    re.compile(r"\b(?:fx|a[4689]|e)[\s-]?\d{3,4}[a-z0-9]*\b", re.I),
)


def fields_needing_visual_analysis(ad: Mapping[str, Any]) -> list[str]:
    """Возвращает пустые или недостаточно точные поля для фото-анализа."""
    result: list[str] = []
    for field in VISUAL_SPEC_FIELDS:
        value = ad.get(field)
        source = ad.get(f"{field}_source")
        confidence = ad.get(f"{field}_confidence")

        if _is_missing(value):
            result.append(field)
            continue
        if source == "visual_fallback":
            result.append(field)
            continue
        if confidence == "low" and source != "text_exact":
            result.append(field)
            continue

        if field == "cpu_model" and cpu_model_specificity(value) < 2:
            result.append(field)
        elif field == "ram_type" and _ram_type_specificity(value) < 2:
            result.append(field)
        elif field == "cpu_socket" and _socket_specificity(value) < 2:
            result.append(field)
        elif field == "product_type" and str(value).strip().casefold() == "other":
            result.append(field)

    return result


def should_replace_with_vision(
    ad: Mapping[str, Any],
    field: str,
    new_value: Any,
    new_confidence: str | None,
) -> bool:
    """Разрешает фото уточнить только отсутствующее или заведомо общее значение."""
    current_value = ad.get(field)
    if _is_missing(new_value):
        return False
    if _is_missing(current_value):
        return True
    if _normalized_value(current_value) == _normalized_value(new_value):
        return False

    source = ad.get(f"{field}_source")
    current_confidence = ad.get(f"{field}_confidence")
    new_rank = _CONFIDENCE_RANK.get(new_confidence, -1)
    current_rank = _CONFIDENCE_RANK.get(current_confidence, -1)
    quality_improved = _field_specificity(field, new_value) > _field_specificity(
        field, current_value
    )

    # Текстовое значение заменяется только прямой, хорошо читаемой маркировкой
    # на фото и только когда новое значение действительно точнее.
    if source == "text_exact":
        return new_confidence == "high" and quality_improved

    if source == "visual_fallback":
        return new_rank >= _CONFIDENCE_RANK["medium"]

    if field == "product_type" and str(current_value).casefold() == "other":
        return new_rank >= _CONFIDENCE_RANK["medium"]

    if quality_improved and new_rank >= _CONFIDENCE_RANK["medium"]:
        return True

    if source == "image_guess" or current_confidence in {"low", "medium", "high"}:
        return new_rank > current_rank and new_rank >= _CONFIDENCE_RANK["medium"]

    return False


def cpu_model_specificity(value: Any) -> int:
    """0 — пусто, 1 — семейство/пример, 2 — видна конкретная модель."""
    if _is_missing(value):
        return 0
    text = _normalize_text(value)
    if any(marker in text for marker in _APPROXIMATE_MARKERS):
        return 1
    if any(pattern.search(text) for pattern in _CPU_MODEL_PATTERNS):
        return 2
    return 1


def _field_specificity(field: str, value: Any) -> int:
    if field == "cpu_model":
        return cpu_model_specificity(value)
    if field == "ram_type":
        return _ram_type_specificity(value)
    if field == "cpu_socket":
        return _socket_specificity(value)
    if field == "product_type":
        return 1 if str(value).strip().casefold() == "other" else 2
    return 1


def _ram_type_specificity(value: Any) -> int:
    if _is_missing(value):
        return 0
    text = _normalize_text(value)
    if any(marker in text for marker in _APPROXIMATE_MARKERS) or "/" in text:
        return 1
    return 2 if re.search(r"\bddr[1-5](?:l|x)?\b", text, re.I) else 1


def _socket_specificity(value: Any) -> int:
    if _is_missing(value):
        return 0
    text = _normalize_text(value)
    if any(marker in text for marker in _APPROXIMATE_MARKERS) or "/" in text:
        return 1
    if re.search(r"\blga115x\b", text, re.I):
        return 1
    return 2


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_text(value: Any) -> str:
    return " ".join(
        str(value).casefold().replace("®", " ").replace("™", " ").split()
    )


def _normalized_value(value: Any) -> str:
    return _normalize_text(value)
