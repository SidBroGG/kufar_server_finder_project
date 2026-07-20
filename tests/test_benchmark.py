from pathlib import Path

from kufar_server_finder.benchmark import CpuBenchmarkDataset


def test_adds_cpu_benchmark_without_mutating_source(tmp_path: Path):
    csv_path = tmp_path / "cpu.csv"
    csv_path.write_text(
        "cpuName,cpuMark\n"
        "Intel Core i5-3470 @ 3.20GHz,4657\n"
        "Intel Core i5-3470S @ 2.90GHz,4353\n",
        encoding="utf-8",
    )
    source = [
        {"cpu_model": "Core i5-3470", "title": "PC"},
        {"cpu_model": "Unknown CPU", "title": "Other"},
    ]

    result = CpuBenchmarkDataset.from_csv(csv_path).add_points(source)

    assert result[0]["cpu_benchmark_points"] == 4657
    assert result[0]["cpu_benchmark_name"] == "Intel Core i5-3470 @ 3.20GHz"
    assert result[0]["cpu_benchmark_source"] == "cpu.csv"
    assert "cpu_benchmark_points" not in result[1]
    assert "cpu_benchmark_points" not in source[0]


def test_matches_common_cpu_format(tmp_path: Path):
    csv_path = tmp_path / "cpu.csv"
    csv_path.write_text(
        "cpuName,cpuMark\n"
        "AMD Ryzen 5 3600,17822\n",
        encoding="utf-8",
    )
    dataset = CpuBenchmarkDataset.from_csv(csv_path)

    benchmark = dataset.find("AMD Ryzen 5 3600 6-Core Processor")

    assert benchmark is not None
    assert benchmark.points == 17822
