from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_FUZZY_THRESHOLD = 0.90

_FREQUENCY_RE = re.compile(
    r"(?:\s*@\s*|\s+)\d+(?:[.,]\d+)?\s*(?:ghz|mhz)\b",
    re.IGNORECASE,
)
_VENDOR_RE = re.compile(r"\b(?:intel|amd)\b", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-zа-яё0-9]+", re.IGNORECASE)
_MODEL_TOKEN_RE = re.compile(r"[a-z]*\d+[a-z0-9]*", re.IGNORECASE)
_VERSION_TOKEN_RE = re.compile(r"v\d+", re.IGNORECASE)

BenchmarkRow = dict[str, Any]


def normalize_cpu_name(value: str | None) -> str:
    """Нормализует название CPU перед сравнением с cpuName из CSV."""
    if not value:
        return ""

    normalized = str(value).casefold().replace("ё", "е")
    normalized = _FREQUENCY_RE.sub(" ", normalized)
    normalized = _VENDOR_RE.sub(" ", normalized)
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def cpu_name_similarity(left: str, right: str) -> float:
    """Возвращает коэффициент сходства нормализованных названий от 0 до 1."""
    if not left or not right:
        return 0.0

    normal_ratio = SequenceMatcher(None, left, right).ratio()
    compact_ratio = SequenceMatcher(
        None,
        left.replace(" ", ""),
        right.replace(" ", ""),
    ).ratio()
    return max(normal_ratio, compact_ratio)


def load_cpu_benchmark(csv_path: str | Path) -> list[BenchmarkRow]:
    """Загружает CSV и заранее нормализует cpuName для быстрого fuzzy search."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "cpuName" not in reader.fieldnames:
            raise ValueError(f"{path}: отсутствует колонка cpuName")
        if "cpuMark" not in reader.fieldnames:
            raise ValueError(f"{path}: отсутствует колонка cpuMark")

        rows: list[BenchmarkRow] = []
        for source_row in reader:
            cpu_name = str(source_row.get("cpuName") or "").strip()
            if not cpu_name:
                continue

            row: BenchmarkRow = dict(source_row)
            row["_normalized_cpu_name"] = normalize_cpu_name(cpu_name)
            rows.append(row)

    return rows


def find_best_cpu_match(
    cpu_model: str | None,
    benchmark_rows: Iterable[Mapping[str, Any]],
    *,
    min_score: float = DEFAULT_FUZZY_THRESHOLD,
) -> BenchmarkRow | None:
    """Находит наиболее похожий cpuName после нормализации обоих названий."""
    if not 0 <= min_score <= 1:
        raise ValueError("min_score должен быть в диапазоне от 0 до 1")

    normalized_cpu = normalize_cpu_name(cpu_model)
    if not normalized_cpu:
        return None

    query_signature = _model_signature(normalized_cpu)
    best_row: BenchmarkRow | None = None
    best_score = -1.0
    best_length_delta = 10**9

    for source_row in benchmark_rows:
        cpu_name = str(source_row.get("cpuName") or "").strip()
        if not cpu_name:
            continue

        normalized_candidate = str(
            source_row.get("_normalized_cpu_name")
            or normalize_cpu_name(cpu_name)
        )
        if not normalized_candidate:
            continue

        candidate_signature = _model_signature(normalized_candidate)
        if (
            query_signature
            and candidate_signature
            and query_signature != candidate_signature
        ):
            # Не даём fuzzy search спутать, например, 3470 с 3470T/3470S.
            continue

        score = cpu_name_similarity(normalized_cpu, normalized_candidate)
        length_delta = abs(len(normalized_cpu) - len(normalized_candidate))

        if score > best_score or (
            score == best_score and length_delta < best_length_delta
        ):
            best_score = score
            best_length_delta = length_delta
            best_row = dict(source_row)

    if best_row is None or best_score < min_score:
        return None

    best_row.pop("_normalized_cpu_name", None)
    return best_row


def get_cpu_mark(
    cpu_model: str | None,
    benchmark_rows: Iterable[Mapping[str, Any]],
    *,
    min_score: float = DEFAULT_FUZZY_THRESHOLD,
) -> int | None:
    """Возвращает cpuMark для лучшего fuzzy-совпадения."""
    match = find_best_cpu_match(
        cpu_model,
        benchmark_rows,
        min_score=min_score,
    )
    if match is None:
        return None
    return _parse_cpu_mark(match.get("cpuMark"))


def apply_cpu_benchmarks(
    ads: Sequence[Mapping[str, Any]],
    benchmark: str | Path | Iterable[Mapping[str, Any]],
    *,
    min_score: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Добавляет cpuMark объявлениям, для которых найдено надёжное совпадение."""
    rows = (
        load_cpu_benchmark(benchmark)
        if isinstance(benchmark, (str, Path))
        else list(benchmark)
    )

    result: list[dict[str, Any]] = []
    for ad in ads:
        item = dict(ad)
        cpu_mark = get_cpu_mark(
            item.get("cpu_model"),
            rows,
            min_score=min_score,
        )
        if cpu_mark is not None:
            item["cpuMark"] = cpu_mark
        result.append(item)

    return result


class CpuBenchmarkDataset:
    """Загруженный CSV-датасет с fuzzy-поиском процессоров."""

    def __init__(
        self,
        source: str | Path | Iterable[Mapping[str, Any]],
        *,
        min_score: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("min_score должен быть в диапазоне от 0 до 1")

        self.min_score = min_score
        self.rows = (
            load_cpu_benchmark(source)
            if isinstance(source, (str, Path))
            else [dict(row) for row in source]
        )
        # Совместимость с кодом, где записи назывались records/data.
        self.records = self.rows
        self.data = self.rows

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        min_score: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> "CpuBenchmarkDataset":
        return cls(csv_path, min_score=min_score)

    @classmethod
    def load(
        cls,
        csv_path: str | Path,
        *,
        min_score: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> "CpuBenchmarkDataset":
        return cls.from_csv(csv_path, min_score=min_score)

    def __len__(self) -> int:
        return len(self.rows)

    def find(self, cpu_model: str | None) -> BenchmarkRow | None:
        return find_best_cpu_match(
            cpu_model,
            self.rows,
            min_score=self.min_score,
        )

    def find_best_match(self, cpu_model: str | None) -> BenchmarkRow | None:
        return self.find(cpu_model)

    def match(self, cpu_model: str | None) -> BenchmarkRow | None:
        return self.find(cpu_model)

    def get_cpu_mark(self, cpu_model: str | None) -> int | None:
        return get_cpu_mark(
            cpu_model,
            self.rows,
            min_score=self.min_score,
        )

    def lookup(self, cpu_model: str | None) -> int | None:
        """Возвращает cpuMark; имя сохранено для старого CLI/pipeline."""
        return self.get_cpu_mark(cpu_model)

    def find_cpu_mark(self, cpu_model: str | None) -> int | None:
        return self.get_cpu_mark(cpu_model)

    def score_for(self, cpu_model: str | None) -> int | None:
        return self.get_cpu_mark(cpu_model)

    def enrich_ads(
        self,
        ads: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return apply_cpu_benchmarks(
            ads,
            self.rows,
            min_score=self.min_score,
        )

    def enrich(self, ads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self.enrich_ads(ads)

    def apply(self, ads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self.enrich_ads(ads)


# Новое имя оставлено как совместимый псевдоним.
CpuBenchmarkLookup = CpuBenchmarkDataset

def _model_signature(normalized_name: str) -> str | None:
    """Извлекает точную модельную часть, чтобы не смешивать суффиксы CPU."""
    tokens = _MODEL_TOKEN_RE.findall(normalized_name)
    if not tokens:
        return None

    last = tokens[-1].casefold()
    if _VERSION_TOKEN_RE.fullmatch(last) and len(tokens) >= 2:
        return tokens[-2].casefold() + last
    return last


def _parse_cpu_mark(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return None


# Совместимые имена для существующего кода.
find_cpu_benchmark = find_best_cpu_match
find_cpu_mark = get_cpu_mark
add_cpu_benchmarks = apply_cpu_benchmarks
enrich_ads_with_cpu_benchmark = apply_cpu_benchmarks
CPUBenchmark = CpuBenchmarkDataset
