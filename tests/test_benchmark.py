from kufar_server_finder.benchmark import CpuBenchmarkDataset


def test_dataset_loads_and_enriches(tmp_path):
    csv_file = tmp_path / "cpu.csv"
    csv_file.write_text(
        "cpuName,cpuMark\nIntel Core i5-3470 @ 3.20GHz,4665\n",
        encoding="utf-8",
    )
    dataset = CpuBenchmarkDataset.from_csv(csv_file)
    match = dataset.find("Intel Core i5-3470")
    assert match is not None
    assert match.cpu_mark == 4665

    result = dataset.enrich_ads([{"cpu_model": "Core i5-3470"}])
    assert result[0]["cpu_mark"] == 4665
    assert result[0]["cpu_benchmark_source"] == "dataset"


def test_fuzzy_matching_handles_common_cpu_name_variants(tmp_path):
    csv_file = tmp_path / "cpu.csv"
    csv_file.write_text(
        "cpuName,cpuMark\n"
        "AMD Athlon 64 X2 Dual Core 5200+,101\n"
        "Intel Core2 Quad Q6600,202\n"
        "Intel Core 2 Duo E8400,303\n"
        "AMD Athlon 64 X2 Dual Core 4600+,404\n"
        "AMD Athlon II X4 640,505\n",
        encoding="utf-8",
    )
    dataset = CpuBenchmarkDataset.from_csv(csv_file)

    cases = {
        "AMD Athlon 64 X2 5200+": "AMD Athlon 64 X2 Dual Core 5200+",
        "Intel core 2 quad q6600": "Intel Core2 Quad Q6600",
        "Intel Core2 Duo E8400": "Intel Core 2 Duo E8400",
        "AMD DualCore Athlon 64 X2 4600": (
            "AMD Athlon 64 X2 Dual Core 4600+"
        ),
        "athlone x4 640\n": "AMD Athlon II X4 640",
    }

    for query, expected in cases.items():
        match = dataset.find(query)
        assert match is not None
        assert match.cpu_name == expected


def test_fuzzy_matching_does_not_drop_cpu_suffix(tmp_path):
    csv_file = tmp_path / "cpu.csv"
    csv_file.write_text(
        "cpuName,cpuMark\n"
        "Intel Core i5-3470T,3000\n",
        encoding="utf-8",
    )
    dataset = CpuBenchmarkDataset.from_csv(csv_file)

    assert dataset.find("Intel Core i5-3470") is None

import pytest

from kufar_server_finder.benchmark import (
    CpuBenchmarkEntry,
    _clean_number,
    _parse_number,
    _similarity_score,
    _token_weight,
)


def test_entry_properties_aliases_and_iterable_source():
    entry = CpuBenchmarkEntry("CPU 1234", 12.5, {"x": "y"})
    dataset = CpuBenchmarkDataset([entry])

    assert entry.name == "CPU 1234"
    assert entry.score == 12.5
    assert dataset.load is not None
    assert dataset.lookup("CPU 1234") is entry
    assert dataset.match("CPU 1234") is entry
    assert dataset.find_match("CPU 1234") is entry
    assert dataset.score("CPU 1234") == 12.5
    assert dataset.get_score("missing") is None
    assert dataset.apply([{"cpu_model": "CPU 1234"}])[0]["cpu_mark"] == 12.5


def test_find_handles_empty_unknown_and_no_anchor_queries():
    dataset = CpuBenchmarkDataset(
        [
            CpuBenchmarkEntry("Intel CPU", 1, {}),
            CpuBenchmarkEntry("", 2, {}),
        ]
    )
    assert dataset.find(None) is None
    assert dataset.find("!!!") is None
    assert dataset.find("AMD 9999") is None
    assert dataset.find("Intel CPU") is not None
    assert dataset.enrich_ad({"cpu_model": "missing"}) == {"cpu_model": "missing"}


def test_csv_validation_errors(tmp_path):
    with pytest.raises(OSError, match="Не удалось открыть"):
        CpuBenchmarkDataset.from_csv(tmp_path / "missing.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="не содержит заголовок"):
        CpuBenchmarkDataset.from_csv(empty)

    missing_columns = tmp_path / "columns.csv"
    missing_columns.write_text("name,score\nCPU,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отсутствуют столбцы"):
        CpuBenchmarkDataset.from_csv(missing_columns)

    no_valid_rows = tmp_path / "rows.csv"
    no_valid_rows.write_text("cpuName,cpuMark\n,\nCPU,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не найдено"):
        CpuBenchmarkDataset.from_csv(no_valid_rows)


def test_number_parsing_weights_and_cleaning_helpers():
    assert _parse_number(None) is None
    assert _parse_number("") is None
    assert _parse_number("1 234") == 1234
    assert _parse_number("12,5") == 12.5
    assert _parse_number("1,234.5") == 1234.5
    assert _parse_number("bad") is None
    assert _clean_number(10.0) == 10
    assert _clean_number(10.5) == 10.5

    assert _token_weight("core") == 0.35
    assert _token_weight("intel") == 0.5
    assert _token_weight("x4") == 1.5
    assert _token_weight("1234") == 5.0
    assert _token_weight("i5") == 2.5
    assert _token_weight("ryzen") == 1.5
    assert _similarity_score("", frozenset(), "x", frozenset({"x"})) == 0
