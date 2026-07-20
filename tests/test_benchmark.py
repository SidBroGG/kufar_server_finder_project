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
