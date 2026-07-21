from kufar_server_finder.cli import build_parser


def test_cli_module_imports_and_parser_builds():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--computers-only",
            "--max-price",
            "20",
            "--infer-specs",
            "--dataset",
            "CPU_benchmark_v4.csv",
        ]
    )
    assert args.command == "run"
    assert args.extract_specs is True
    assert args.dataset == "CPU_benchmark_v4.csv"
    assert args.excel_output == "output.xlsx"


def test_run_creates_excel_from_final_json(monkeypatch, tmp_path):
    from kufar_server_finder import cli

    raw_path = tmp_path / "raw.json"
    json_path = tmp_path / "output.json"
    excel_path = tmp_path / "output.xlsx"
    calls = []

    monkeypatch.setattr(cli, "_collect", lambda args: [{"link": "x"}])
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, extract_specs: [{"link": "x", "price": 10}],
    )
    monkeypatch.setattr(cli, "_vision", lambda ads: ads)
    monkeypatch.setattr(cli, "_apply_benchmark", lambda ads, dataset: ads)
    monkeypatch.setattr(
        cli,
        "export_ads_json_to_excel",
        lambda source, destination: calls.append((source, destination)),
    )

    result = cli.main(
        [
            "run",
            "--raw-output",
            str(raw_path),
            "--output",
            str(json_path),
            "--excel-output",
            str(excel_path),
            "--page-delay",
            "0",
            "--detail-delay",
            "0",
        ]
    )

    assert result == 0
    assert calls == [(str(json_path), str(excel_path))]
    assert raw_path.exists()
    assert json_path.exists()

def test_new_commands_parse():
    parser = build_parser()

    pipeline_args = parser.parse_args(
        [
            "pipeline",
            "--input",
            "raw.json",
            "--output",
            "final.json",
            "--excel-output",
            "final.xlsx",
            "--extract-specs",
            "--dataset",
            "cpu.csv",
        ]
    )
    assert pipeline_args.command == "pipeline"
    assert pipeline_args.extract_specs is True
    assert pipeline_args.dataset == "cpu.csv"

    benchmark_args = parser.parse_args(
        [
            "benchmark",
            "--input",
            "vision.json",
            "--output",
            "benchmarked.json",
            "--dataset",
            "cpu.csv",
        ]
    )
    assert benchmark_args.command == "benchmark"
    assert benchmark_args.dataset == "cpu.csv"

    excel_args = parser.parse_args(
        ["excel", "--input", "output.json", "--output", "output.xlsx"]
    )
    assert excel_args.command == "excel"


def test_pipeline_processes_existing_json_without_collect(monkeypatch, tmp_path):
    from kufar_server_finder import cli
    from kufar_server_finder.storage import save_ads

    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "final.json"
    excel_path = tmp_path / "final.xlsx"
    save_ads(input_path, [{"link": "x"}])
    calls = []

    monkeypatch.setattr(
        cli,
        "_collect",
        lambda args: (_ for _ in ()).throw(AssertionError("collect called")),
    )
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, extract_specs: [
            {"link": "x", "stage": "analyze", "extract_specs": extract_specs}
        ],
    )
    monkeypatch.setattr(
        cli,
        "_vision",
        lambda ads: [{**ads[0], "stage": "vision"}],
    )
    monkeypatch.setattr(
        cli,
        "_apply_benchmark",
        lambda ads, dataset: [{**ads[0], "dataset": dataset}],
    )
    monkeypatch.setattr(
        cli,
        "export_ads_json_to_excel",
        lambda source, destination: calls.append((source, destination)),
    )

    result = cli.main(
        [
            "pipeline",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--excel-output",
            str(excel_path),
            "--extract-specs",
            "--dataset",
            "cpu.csv",
        ]
    )

    assert result == 0
    assert calls == [(str(output_path), str(excel_path))]
    assert output_path.exists()


def test_benchmark_command_updates_json(monkeypatch, tmp_path):
    from kufar_server_finder import cli
    from kufar_server_finder.storage import load_ads, save_ads

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "benchmark.json"
    save_ads(input_path, [{"link": "x", "cpu_model": "CPU"}])

    monkeypatch.setattr(
        cli,
        "_apply_benchmark",
        lambda ads, dataset: [
            {**ads[0], "cpu_mark": 123, "used_dataset": dataset}
        ],
    )

    result = cli.main(
        [
            "benchmark",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dataset",
            "cpu.csv",
        ]
    )

    assert result == 0
    assert load_ads(output_path)[0]["cpu_mark"] == 123


def test_excel_command_exports_json(monkeypatch, tmp_path):
    from kufar_server_finder import cli

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.xlsx"
    calls = []
    monkeypatch.setattr(
        cli,
        "export_ads_json_to_excel",
        lambda source, destination: calls.append((source, destination)),
    )

    result = cli.main(
        [
            "excel",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert calls == [(str(input_path), str(output_path))]

import argparse
import pytest


def test_collect_analyze_and_vision_commands(monkeypatch, tmp_path):
    from kufar_server_finder import cli
    from kufar_server_finder.storage import load_ads, save_ads

    collect_output = tmp_path / "collect.json"
    monkeypatch.setattr(cli, "_collect", lambda args: [{"link": "collect"}])
    assert cli.main(["collect", "--output", str(collect_output)]) == 0
    assert load_ads(collect_output) == [{"link": "collect"}]

    input_path = tmp_path / "input.json"
    analyze_output = tmp_path / "analyze.json"
    vision_output = tmp_path / "vision.json"
    save_ads(input_path, [{"link": "x"}])
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, extract_specs: [{**ads[0], "extract": extract_specs}],
    )
    monkeypatch.setattr(
        cli,
        "_apply_benchmark",
        lambda ads, dataset: [{**ads[0], "dataset": dataset}],
    )
    assert cli.main(
        [
            "analyze",
            "--input",
            str(input_path),
            "--output",
            str(analyze_output),
            "--extract-specs",
            "--dataset",
            "cpu.csv",
        ]
    ) == 0
    assert load_ads(analyze_output)[0]["extract"] is True

    monkeypatch.setattr(cli, "_vision", lambda ads: [{**ads[0], "vision": True}])
    assert cli.main(
        ["vision", "--input", str(input_path), "--output", str(vision_output)]
    ) == 0
    assert load_ads(vision_output)[0]["vision"] is True


def test_main_returns_expected_error_codes(monkeypatch, tmp_path):
    from kufar_server_finder import cli

    monkeypatch.setattr(cli, "_collect", lambda args: (_ for _ in ()).throw(ValueError("bad")))
    assert cli.main(["collect", "--output", str(tmp_path / "x.json")]) == 2

    monkeypatch.setattr(cli, "_collect", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["collect", "--output", str(tmp_path / "x.json")]) == 1


def test_collect_builds_kufar_config_and_forwards_arguments(monkeypatch):
    from kufar_server_finder import cli

    captured = {}

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config

        def fetch_ads(self, **kwargs):
            captured["kwargs"] = kwargs
            return [{"link": "x"}]

    monkeypatch.setattr(cli, "KufarClient", FakeClient)
    args = argparse.Namespace(
        region="5",
        timeout=9,
        page_delay=-1,
        detail_delay=-2,
        query="pc",
        computers_only=True,
        max_price=20,
        no_descriptions=True,
    )

    assert cli._collect(args) == [{"link": "x"}]
    assert captured["config"].region == "5"
    assert captured["config"].page_delay == 0
    assert captured["config"].detail_delay == 0
    assert captured["kwargs"] == {
        "query": "pc",
        "computers_only": True,
        "max_price": 20,
        "load_descriptions": False,
    }


def test_pipeline_helpers_delegate(monkeypatch):
    from kufar_server_finder import cli

    class FakePipeline:
        def filter_working_targets(self, ads, extract_specs):
            return [{"stage": "analyze", "extract": extract_specs}]

        def enrich_missing_specs_from_images(self, ads):
            return [{"stage": "vision"}]

    monkeypatch.setattr(cli, "_build_pipeline", lambda: FakePipeline())
    assert cli._analyze([{}], extract_specs=True)[0]["extract"] is True
    assert cli._vision([{}]) == [{"stage": "vision"}]


def test_build_pipeline_creates_analyzer(monkeypatch):
    from kufar_server_finder import cli, gemini

    config = object()
    analyzer = object()
    monkeypatch.setattr(cli.GeminiConfig, "from_env", lambda: config)
    monkeypatch.setattr(gemini, "GeminiAnalyzer", lambda value: analyzer)

    pipeline = cli._build_pipeline()
    assert pipeline.analyzer is analyzer


def test_apply_benchmark_skips_or_enriches(monkeypatch):
    from kufar_server_finder import cli

    ads = [{"link": "x", "cpu_model": "CPU"}]
    assert cli._apply_benchmark(ads, None) is ads

    class FakeDataset:
        def enrich_ads(self, values):
            return [{**values[0], "cpu_mark": 10}]

    class FakePipeline:
        def normalize_cpu_models_for_benchmark(self, values):
            return [{**values[0], "cpu_model_normalized": True}]

    monkeypatch.setattr(
        cli.CpuBenchmarkDataset,
        "from_csv",
        lambda path: FakeDataset(),
    )
    monkeypatch.setattr(cli, "_build_pipeline", lambda: FakePipeline())

    assert cli._apply_benchmark(ads, "cpu.csv")[0]["cpu_mark"] == 10
