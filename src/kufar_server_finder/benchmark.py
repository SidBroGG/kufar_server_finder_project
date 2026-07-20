from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_AND_GENERIC_WORDS = {
    "intel",
    "amd",
    "cpu",
    "processor",
}

_RELAXED_WORDS = {
    "core",
    "cores",
    "thread",
    "threads",
    "apu",
}


@dataclass(frozen=True, slots=True)
class CpuBenchmark:
    name: str
    points: int


class CpuBenchmarkDataset:
    def __init__(
        self,
        benchmarks: dict[str, CpuBenchmark],
        *,
        source_name: str,
    ) -> None:
        self._benchmarks = benchmarks
        self._source_name = source_name

    @classmethod
    def from_csv(cls, path: str | Path) -> "CpuBenchmarkDataset":
        source = Path(path)
        aliases: dict[str, list[CpuBenchmark]] = {}

        with source.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"cpuName", "cpuMark"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{source}: нужны колонки cpuName и cpuMark"
                )

            for row in reader:
                name = (row.get("cpuName") or "").strip()
                raw_points = (row.get("cpuMark") or "").strip()
                if not name or not raw_points:
                    continue

                try:
                    benchmark = CpuBenchmark(name=name, points=int(float(raw_points)))
                except ValueError:
                    logger.warning(
                        "Пропущен некорректный cpuMark для %r: %r",
                        name,
                        raw_points,
                    )
                    continue

                for alias in _cpu_aliases(name):
                    aliases.setdefault(alias, []).append(benchmark)

        # Не используем неоднозначные алиасы: например одинаковую модель
        # с разной частотой или количеством ядер.
        unique = {
            alias: items[0]
            for alias, items in aliases.items()
            if len({(item.name, item.points) for item in items}) == 1
        }
        logger.info(
            "Загружено CPU benchmark-записей: %s",
            len(unique),
        )
        return cls(unique, source_name=source.name)

    def find(self, cpu_model: str | None) -> CpuBenchmark | None:
        if not cpu_model:
            return None

        for alias in _cpu_aliases(cpu_model):
            benchmark = self._benchmarks.get(alias)
            if benchmark is not None:
                return benchmark
        return None

    def add_points(
        self,
        ads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        matched = 0

        for ad in ads:
            item = dict(ad)
            benchmark = self.find(str(item.get("cpu_model") or ""))
            if benchmark is not None:
                item["cpu_benchmark_points"] = benchmark.points
                item["cpu_benchmark_name"] = benchmark.name
                item["cpu_benchmark_source"] = self._source_name
                matched += 1
            result.append(item)

        logger.info(
            "Benchmark найден для %s из %s объявлений",
            matched,
            len(result),
        )
        return result


def _cpu_aliases(value: str) -> tuple[str, ...]:
    strict = _normalize_cpu_name(value, relaxed=False)
    relaxed = _normalize_cpu_name(value, relaxed=True)
    return tuple(dict.fromkeys(alias for alias in (strict, relaxed) if alias))


def _normalize_cpu_name(value: str, *, relaxed: bool) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.replace("®", "").replace("™", "")
    text = re.sub(r"\((?:r|tm)\)", " ", text)

    # Частота после @ в PassMark не является частью модели.
    text = re.sub(
        r"@\s*\d+(?:[.,]\d+)?\s*(?:ghz|mhz)\b",
        " ",
        text,
    )

    if relaxed:
        text = re.sub(r"\b\d+\s*[- ]?(?:core|cores|thread|threads)\b", " ", text)

    tokens = re.findall(r"[a-z0-9]+", text)
    ignored = set(_VENDOR_AND_GENERIC_WORDS)
    if relaxed:
        ignored.update(_RELAXED_WORDS)

    return " ".join(token for token in tokens if token not in ignored)
