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
