from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


_FREQUENCY_SUFFIX_RE = re.compile(
    r"\s*@\s*\d+(?:[.,]\d+)?\s*(?:ghz|mhz)\b.*$",
    flags=re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_TOKEN_REPLACEMENTS = {
    "athlone": "athlon",
    "athlonne": "athlon",
    "atlon": "athlon",
    "celeronr": "celeron",
    "pentiumr": "pentium",
}
_GENERIC_TOKENS = {
    "cpu",
    "processor",
    "dual",
    "quad",
    "core",
    "cores",
}
_OPTIONAL_TOKENS = {"amd", "intel", "ii"}
_TOPOLOGY_MODEL_TOKENS = {"x2", "x3", "x4", "x6", "x8", "x12", "x16"}
_MIN_FUZZY_SCORE = 0.74
_MIN_SCORE_MARGIN = 0.035


@dataclass(frozen=True, slots=True)
class CpuBenchmarkEntry:
    cpu_name: str
    cpu_mark: float
    row: Mapping[str, str]



@dataclass(frozen=True, slots=True)
class _IndexedEntry:
    normalized_name: str
    tokens: frozenset[str]
    anchor_tokens: frozenset[str]
    entry: CpuBenchmarkEntry


class CpuBenchmarkDataset:
    """Индекс CPU Benchmark с устойчивым поиском по неточным названиям."""

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
        self._indexed_entries: list[_IndexedEntry] = []
        self._by_anchor_token: dict[str, list[_IndexedEntry]] = {}

        for entry in self.entries:
            normalized = self.normalize_cpu_name(entry.cpu_name)
            if not normalized:
                continue

            self._by_normalized_name.setdefault(normalized, entry)
            tokens = frozenset(normalized.split())
            indexed = _IndexedEntry(
                normalized_name=normalized,
                tokens=tokens,
                anchor_tokens=frozenset(_anchor_tokens(tokens)),
                entry=entry,
            )
            self._indexed_entries.append(indexed)
            for token in indexed.anchor_tokens:
                self._by_anchor_token.setdefault(token, []).append(indexed)

    @classmethod
    def from_csv(cls, path: str | Path) -> CpuBenchmarkDataset:
        return cls(path)


    @staticmethod
    def normalize_cpu_name(value: str) -> str:
        text = _FREQUENCY_SUFFIX_RE.sub("", str(value).strip().casefold())
        text = (
            text.replace("®", " ")
            .replace("™", " ")
            .replace("(r)", " ")
            .replace("(tm)", " ")
        )
        # Частые варианты слитного написания из объявлений и датасетов.
        text = re.sub(r"\bcore\s*2\b", "core 2", text)
        text = re.sub(r"\bdual\s*core\b", "dual core", text)
        text = re.sub(r"\bquad\s*core\b", "quad core", text)
        text = _NON_ALNUM_RE.sub(" ", text).strip()

        tokens = [_TOKEN_REPLACEMENTS.get(token, token) for token in text.split()]
        return " ".join(tokens)

    def find(self, cpu_model: str | None) -> CpuBenchmarkEntry | None:
        if not cpu_model:
            return None

        normalized = self.normalize_cpu_name(cpu_model)
        if not normalized:
            return None

        exact = self._by_normalized_name.get(normalized)
        if exact is not None:
            return exact

        query_tokens = frozenset(normalized.split())
        query_anchors = frozenset(_anchor_tokens(query_tokens))
        candidates = self._candidate_entries(query_anchors)
        if not candidates:
            return None

        ranked: list[tuple[float, int, _IndexedEntry]] = []
        for candidate in candidates:
            # Номер/суффикс модели обязан совпасть точно. Поэтому 3470 не
            # сопоставляется с 3470T, а Q6600 — с Q6700.
            if query_anchors and not query_anchors.issubset(candidate.anchor_tokens):
                continue

            score = _similarity_score(
                normalized,
                query_tokens,
                candidate.normalized_name,
                candidate.tokens,
            )
            ranked.append((score, len(candidate.normalized_name), candidate))

        if not ranked:
            return None

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, _, best = ranked[0]
        if best_score < _MIN_FUZZY_SCORE:
            return None

        if len(ranked) > 1:
            second_score = ranked[1][0]
            if best_score - second_score < _MIN_SCORE_MARGIN:
                return None

        return best.entry


    def score(self, cpu_model: str | None) -> float | None:
        entry = self.find(cpu_model)
        return entry.cpu_mark if entry else None


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


    def _candidate_entries(
        self,
        query_anchors: frozenset[str],
    ) -> list[_IndexedEntry]:
        if not query_anchors:
            return self._indexed_entries

        groups = [self._by_anchor_token.get(token, []) for token in query_anchors]
        if any(not group for group in groups):
            return []

        # Начинаем с самого редкого номера модели для быстрого сужения датасета.
        smallest = min(groups, key=len)
        return list(smallest)

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
            for row in reader:
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


def _anchor_tokens(tokens: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for token in tokens:
        if token in _TOPOLOGY_MODEL_TOKENS:
            continue
        digit_count = sum(character.isdigit() for character in token)
        if digit_count >= 3:
            result.add(token)
    return result


def _token_weight(token: str) -> float:
    if token in _GENERIC_TOKENS:
        return 0.35
    if token in _OPTIONAL_TOKENS:
        return 0.5
    if token in _TOPOLOGY_MODEL_TOKENS:
        return 1.5
    if sum(character.isdigit() for character in token) >= 3:
        return 5.0
    if any(character.isdigit() for character in token):
        return 2.5
    return 1.5


def _similarity_score(
    query_name: str,
    query_tokens: frozenset[str],
    candidate_name: str,
    candidate_tokens: frozenset[str],
) -> float:
    shared = query_tokens.intersection(candidate_tokens)
    shared_weight = sum(_token_weight(token) for token in shared)
    query_weight = sum(_token_weight(token) for token in query_tokens)
    candidate_weight = sum(_token_weight(token) for token in candidate_tokens)

    if not query_weight or not candidate_weight:
        return 0.0

    query_containment = shared_weight / query_weight
    dice = (2.0 * shared_weight) / (query_weight + candidate_weight)
    sequence = SequenceMatcher(None, query_name, candidate_name).ratio()
    return 0.52 * query_containment + 0.35 * dice + 0.13 * sequence


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
