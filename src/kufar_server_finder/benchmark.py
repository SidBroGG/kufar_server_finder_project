from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_FREQUENCY_SUFFIX_RE = re.compile(
    r"\s*@\s*\d+(?:[.,]\d+)?\s*(?:ghz|mhz)\b.*$",
    flags=re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class CpuBenchmarkEntry:
    cpu_name: str
    cpu_mark: float
    row: Mapping[str, str]

    @property
    def name(self) -> str:
        return self.cpu_name

    @property
    def score(self) -> float:
        return self.cpu_mark


class CpuBenchmarkDataset:
    """Индекс CPU Benchmark с совместимым API для CLI и тестов."""

    def __init__(
        self,
        source: str | Path | Iterable[CpuBenchmarkEntry],
    ) -> None:
        if isinstance(source, (str, Path)):
            entries = self._read_csv(Path(source))
        else:
            entries = list(source)

        self.entries = tuple(entries)
        self._by_normalized_name: dict[str, CpuBenchmarkEntry] = {}
        for entry in self.entries:
            normalized = self.normalize_cpu_name(entry.cpu_name)
            if normalized:
                self._by_normalized_name.setdefault(normalized, entry)

    @classmethod
    def from_csv(cls, path: str | Path) -> CpuBenchmarkDataset:
        return cls(path)

    @classmethod
    def load(cls, path: str | Path) -> CpuBenchmarkDataset:
        return cls.from_csv(path)

    @staticmethod
    def normalize_cpu_name(value: str) -> str:
        text = _FREQUENCY_SUFFIX_RE.sub("", str(value).strip().casefold())
        text = (
            text.replace("®", " ")
            .replace("™", " ")
            .replace("(r)", " ")
            .replace("(tm)", " ")
        )
        return _NON_ALNUM_RE.sub(" ", text).strip()

    def find(self, cpu_model: str | None) -> CpuBenchmarkEntry | None:
        if not cpu_model:
            return None

        normalized = self.normalize_cpu_name(cpu_model)
        if not normalized:
            return None

        exact = self._by_normalized_name.get(normalized)
        if exact is not None:
            return exact

        # Разрешаем только достаточно длинное вхождение, чтобы не спутать,
        # например, i5-3470 и i5-3470T.
        candidates: list[tuple[int, CpuBenchmarkEntry]] = []
        for dataset_name, entry in self._by_normalized_name.items():
            if len(dataset_name) < 6:
                continue
            if dataset_name in normalized or normalized in dataset_name:
                candidates.append((len(dataset_name), entry))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    lookup = find
    match = find
    find_match = find

    def score(self, cpu_model: str | None) -> float | None:
        entry = self.find(cpu_model)
        return entry.cpu_mark if entry else None

    get_score = score
    get_cpu_mark = score

    def enrich_ad(self, ad: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(ad)
        entry = self.find(result.get("cpu_model"))
        if entry is None:
            return result

        result["cpu_mark"] = _clean_number(entry.cpu_mark)
        result["cpu_benchmark_name"] = entry.cpu_name
        result["cpu_benchmark_source"] = "dataset"
        return result

    def enrich_ads(self, ads: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.enrich_ad(ad) for ad in ads]

    apply = enrich_ads
    enrich = enrich_ads
    apply_to_ads = enrich_ads

    @staticmethod
    def _read_csv(path: Path) -> list[CpuBenchmarkEntry]:
        try:
            file = path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise OSError(f"Не удалось открыть датасет CPU {path}: {exc}") from exc

        with file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"{path}: CSV не содержит заголовок")
            required = {"cpuName", "cpuMark"}
            missing = required.difference(reader.fieldnames)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{path}: отсутствуют столбцы: {names}")

            entries: list[CpuBenchmarkEntry] = []
            for line_number, row in enumerate(reader, start=2):
                name = str(row.get("cpuName") or "").strip()
                mark = _parse_number(row.get("cpuMark"))
                if not name or mark is None:
                    continue
                entries.append(
                    CpuBenchmarkEntry(
                        cpu_name=name,
                        cpu_mark=mark,
                        row={key: value or "" for key, value in row.items() if key},
                    )
                )

        if not entries:
            raise ValueError(f"{path}: не найдено ни одной строки с cpuName/cpuMark")
        return entries


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    if text.count(",") == 1 and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value
