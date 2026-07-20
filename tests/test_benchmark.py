from pathlib import Path

from kufar_server_finder.benchmark import (
    CpuBenchmarkDataset,
    apply_cpu_benchmarks,
    find_best_cpu_match,
    get_cpu_mark,
    load_cpu_benchmark,
    normalize_cpu_name,
)


def _write_csv(path: Path) -> None:
    path.write_text(
        "cpuName,price,cpuMark\n"
        "Intel Core2 Duo E6420 @ 2.13GHz,,714\n"
        "Intel Core i5-3470 @ 3.20GHz,,4662\n"
        "Intel Core i5-3470T @ 2.90GHz,,2958\n"
        "Intel Core i5-3470S @ 2.90GHz,,4177\n",
        encoding="utf-8",
    )


def test_normalization_is_preserved():
    assert normalize_cpu_name("Intel Core i5-3470 @ 3.20GHz") == "core i5 3470"
    assert normalize_cpu_name("AMD Ryzen 5 3600") == "ryzen 5 3600"


def test_fuzzy_search_matches_core_2_and_core2(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    _write_csv(csv_path)
    rows = load_cpu_benchmark(csv_path)

    match = find_best_cpu_match("Intel Core 2 Duo E6420", rows)

    assert match is not None
    assert match["cpuName"] == "Intel Core2 Duo E6420 @ 2.13GHz"
    assert get_cpu_mark("Intel Core 2 Duo E6420", rows) == 714


def test_fuzzy_search_does_not_mix_model_suffixes(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    _write_csv(csv_path)
    rows = load_cpu_benchmark(csv_path)

    assert get_cpu_mark("Intel Core i5 3470", rows) == 4662
    assert get_cpu_mark("Intel Core i5 3470T", rows) == 2958
    assert get_cpu_mark("Intel Core i5 3470S", rows) == 4177


def test_below_threshold_returns_none(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    _write_csv(csv_path)
    rows = load_cpu_benchmark(csv_path)

    assert get_cpu_mark("Intel Pentium G4560", rows) is None


def test_apply_cpu_benchmarks_does_not_mutate_source(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    _write_csv(csv_path)
    source = [{"cpu_model": "Intel Core 2 Duo E6420", "title": "PC"}]

    result = apply_cpu_benchmarks(source, csv_path)

    assert result[0]["cpuMark"] == 714
    assert "cpuMark" not in source[0]


def test_cpu_benchmark_dataset_compatibility(tmp_path):
    csv_path = tmp_path / "cpu.csv"
    _write_csv(csv_path)

    dataset = CpuBenchmarkDataset.from_csv(csv_path)

    assert dataset.lookup("Intel Core 2 Duo E6420") == 714
    assert dataset.get_cpu_mark("Intel Core i5-3470T") == 2958
    assert dataset.find("Intel Core i5-3470S")["cpuMark"] == "4177"
    assert dataset.enrich_ads([{"cpu_model": "Intel Core 2 Duo E6420"}])[0]["cpuMark"] == 714
