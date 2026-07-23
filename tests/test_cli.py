import argparse
import threading

import pytest

from kufar_server_finder.cli import build_parser


def test_cli_module_imports_and_parser_builds_without_removed_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--max-price",
            "20",
            "--dataset",
            "CPU_benchmark_v4.csv",
        ]
    )

    assert args.command == "run"
    assert args.dataset == "CPU_benchmark_v4.csv"
    assert args.excel_output == "output.xlsx"
    for removed in (
        "query",
        "computers_only",
        "no_descriptions",
        "region",
        "detail_delay",
        "detail_workers",
        "detail_retries",
        "timeout",
        "page_delay",
        "extract_specs",
    ):
        assert not hasattr(args, removed)


@pytest.mark.parametrize(
    "arguments",
    [
        ["collect", "--query", "pc"],
        ["collect", "--computers-only"],
        ["collect", "--no-descriptions"],
        ["collect", "--region", "5"],
        ["collect", "--detail-delay", "2"],
        ["collect", "--detail-workers", "2"],
        ["collect", "--detail-retries", "4"],
        ["collect", "--timeout", "10"],
        ["collect", "--page-delay", "0"],
        ["run", "--page-delay", "0"],
        ["analyze", "--extract-specs"],
        ["analyze", "--infer-specs"],
    ],
)
def test_removed_cli_flags_are_rejected(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_run_creates_excel_from_final_json(monkeypatch, tmp_path):
    from kufar_server_finder import cli

    raw_path = tmp_path / "raw.json"
    json_path = tmp_path / "output.json"
    excel_path = tmp_path / "output.xlsx"
    calls = []
    pipeline_builds = []

    class FakePipeline:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    shared_pipeline = FakePipeline()

    class FakeGeminiConfig:
        chunk_size = 2

    class FakeClient:
        def __init__(self, config):
            self.closed = False

        def iter_ads(self, **kwargs):
            assert kwargs == {"min_price": 0.0, "max_price": 100.0}
            yield {"link": "x", "price": 10}

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli.GeminiConfig, "from_env", lambda: FakeGeminiConfig())
    monkeypatch.setattr(cli.KufarConfig, "from_env", lambda **kwargs: object())
    monkeypatch.setattr(cli, "KufarClient", FakeClient)
    monkeypatch.setattr(
        cli,
        "_build_pipeline",
        lambda config=None: pipeline_builds.append(shared_pipeline) or shared_pipeline,
    )
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, pipeline=None: [{"link": "x", "price": 10}],
    )
    monkeypatch.setattr(cli, "_vision", lambda ads, *, pipeline=None: ads)
    monkeypatch.setattr(
        cli,
        "_apply_benchmark_dataset",
        lambda ads, dataset, *, pipeline=None: ads,
    )
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
        ]
    )

    assert result == 0
    assert calls == [(str(json_path), str(excel_path))]
    assert raw_path.exists()
    assert json_path.exists()
    assert pipeline_builds == [shared_pipeline]
    assert shared_pipeline.closed is True


def test_run_starts_pipeline_before_kufar_collection_finishes(
    monkeypatch,
    tmp_path,
):
    from kufar_server_finder import cli
    from kufar_finder_core import load_items

    first_batch_started = threading.Event()
    processed_batches = []

    class FakeGeminiConfig:
        chunk_size = 2

    class FakeClient:
        def __init__(self, config):
            pass

        def iter_ads(self, **kwargs):
            yield {"link": "1", "price": 3}
            yield {"link": "2", "price": 2}
            assert first_batch_started.wait(timeout=2)
            yield {"link": "3", "price": 1}

        def close(self):
            pass

    class FakePipeline:
        def close(self):
            pass

    def process_batch(ads, *, pipeline, benchmark_dataset):
        processed_batches.append([ad["link"] for ad in ads])
        first_batch_started.set()
        return [{**ad, "processed": True} for ad in ads]

    monkeypatch.setattr(cli.GeminiConfig, "from_env", lambda: FakeGeminiConfig())
    monkeypatch.setattr(cli.KufarConfig, "from_env", lambda **kwargs: object())
    monkeypatch.setattr(cli, "KufarClient", FakeClient)
    monkeypatch.setattr(cli, "_build_pipeline", lambda config=None: FakePipeline())
    monkeypatch.setattr(cli, "_process_run_batch", process_batch)
    monkeypatch.setattr(cli, "export_ads_json_to_excel", lambda *args: None)

    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "output.json"
    args = argparse.Namespace(
        min_price=0,
        max_price=100,
        raw_output=str(raw_path),
        output=str(output_path),
        excel_output=str(tmp_path / "output.xlsx"),
        dataset=None,
    )

    cli._run_streaming(args)

    assert processed_batches == [["1", "2"], ["3"]]
    assert [ad["link"] for ad in load_items(raw_path)] == ["3", "2", "1"]
    assert all(ad["processed"] for ad in load_items(output_path))


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
            "--dataset",
            "cpu.csv",
        ]
    )
    assert pipeline_args.command == "pipeline"
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
    from kufar_finder_core import save_items

    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "final.json"
    excel_path = tmp_path / "final.xlsx"
    save_items(input_path, [{"link": "x"}])
    calls = []
    shared_pipeline = object()
    monkeypatch.setattr(cli, "_build_pipeline", lambda: shared_pipeline)

    monkeypatch.setattr(
        cli,
        "_collect",
        lambda args: (_ for _ in ()).throw(AssertionError("collect called")),
    )
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, pipeline=None: [{"link": "x", "stage": "analyze"}],
    )
    monkeypatch.setattr(
        cli,
        "_vision",
        lambda ads, *, pipeline=None: [{**ads[0], "stage": "vision"}],
    )
    monkeypatch.setattr(
        cli,
        "_apply_benchmark",
        lambda ads, dataset, *, pipeline=None: [{**ads[0], "dataset": dataset}],
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
            "--dataset",
            "cpu.csv",
        ]
    )

    assert result == 0
    assert calls == [(str(output_path), str(excel_path))]
    assert output_path.exists()


def test_benchmark_command_updates_json(monkeypatch, tmp_path):
    from kufar_server_finder import cli
    from kufar_finder_core import load_items, save_items

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "benchmark.json"
    save_items(
        input_path,
        [
            {
                "link": "x",
                "cpu_model": "CPU",
                "minimum_configuration": "старый формат",
                "price_components": ["старый расчёт"],
            }
        ],
    )

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
    result = load_items(output_path)[0]
    assert result["cpu_mark"] == 123
    assert "minimum_configuration" not in result
    assert "price_components" not in result


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


def test_collect_analyze_and_vision_commands(monkeypatch, tmp_path):
    from kufar_server_finder import cli
    from kufar_finder_core import load_items, save_items

    collect_output = tmp_path / "collect.json"
    monkeypatch.setattr(cli, "_collect", lambda args: [{"link": "collect"}])
    assert cli.main(["collect", "--output", str(collect_output)]) == 0
    assert load_items(collect_output) == [{"link": "collect"}]

    input_path = tmp_path / "input.json"
    analyze_output = tmp_path / "analyze.json"
    vision_output = tmp_path / "vision.json"
    save_items(input_path, [{"link": "x"}])
    shared_pipeline = object()
    monkeypatch.setattr(cli, "_build_pipeline", lambda: shared_pipeline)
    monkeypatch.setattr(
        cli,
        "_analyze",
        lambda ads, *, pipeline=None: [{**ads[0], "specs": True}],
    )
    monkeypatch.setattr(
        cli,
        "_apply_benchmark",
        lambda ads, dataset, *, pipeline=None: [{**ads[0], "dataset": dataset}],
    )
    assert cli.main(
        [
            "analyze",
            "--input",
            str(input_path),
            "--output",
            str(analyze_output),
            "--dataset",
            "cpu.csv",
        ]
    ) == 0
    assert load_items(analyze_output)[0]["specs"] is True

    monkeypatch.setattr(
        cli,
        "_vision",
        lambda ads, *, pipeline=None: [{**ads[0], "vision": True}],
    )
    assert cli.main(
        ["vision", "--input", str(input_path), "--output", str(vision_output)]
    ) == 0
    assert load_items(vision_output)[0]["vision"] is True


def test_main_returns_expected_error_codes(monkeypatch, tmp_path):
    from kufar_server_finder import cli

    monkeypatch.setattr(
        cli,
        "_collect",
        lambda args: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert cli.main(["collect", "--output", str(tmp_path / "x.json")]) == 2

    monkeypatch.setattr(
        cli,
        "_collect",
        lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert cli.main(["collect", "--output", str(tmp_path / "x.json")]) == 1

    monkeypatch.setattr(
        cli,
        "_collect",
        lambda args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert cli.main(["collect", "--output", str(tmp_path / "x.json")]) == 130


def test_collect_reads_kufar_settings_from_env_and_uses_categories(monkeypatch):
    from kufar_server_finder import cli

    captured = {}

    class FakeClient:
        def __init__(self, config):
            captured["config"] = config
            captured["closed"] = False

        def fetch_ads(self, **kwargs):
            captured["kwargs"] = kwargs
            return [{"link": "x"}]

        def close(self):
            captured["closed"] = True

    monkeypatch.setenv("KUFAR_REGION", "5")
    monkeypatch.setenv("KUFAR_TIMEOUT", "9")
    monkeypatch.setenv("KUFAR_PAGE_DELAY", "0.5")
    monkeypatch.setenv("KUFAR_DETAIL_DELAY", "2")
    monkeypatch.setenv("KUFAR_DETAIL_WORKERS", "2")
    monkeypatch.setenv("KUFAR_DETAIL_RETRIES", "4")
    monkeypatch.setattr(cli, "KufarClient", FakeClient)
    args = argparse.Namespace(min_price=0, max_price=20)

    assert cli._collect(args) == [{"link": "x"}]
    assert captured["config"].region == "5"
    assert captured["config"].request_timeout == 9
    assert captured["config"].page_delay == 0.5
    assert captured["config"].detail_delay == 2
    assert captured["config"].detail_workers == 2
    assert captured["config"].detail_max_retries == 4
    assert captured["kwargs"] == {"min_price": 0, "max_price": 20}
    assert captured["closed"] is True


def test_pipeline_helpers_delegate_and_close_owned_pipeline(monkeypatch):
    from kufar_server_finder import cli

    class FakePipeline:
        def __init__(self):
            self.closed = False

        def filter_working_targets(self, ads):
            return [{"stage": "analyze", "specs": True}]

        def enrich_missing_specs_from_images(self, ads):
            return [{"stage": "vision"}]

        def close(self):
            self.closed = True

    pipelines = []

    def build():
        pipeline = FakePipeline()
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(cli, "_build_pipeline", build)
    assert cli._analyze([{}])[0]["specs"] is True
    assert pipelines[-1].closed is True
    assert cli._vision([{}]) == [{"stage": "vision"}]
    assert pipelines[-1].closed is True


def test_build_pipeline_creates_analyzer(monkeypatch):
    from kufar_server_finder import cli, gemini

    config = object()
    analyzer = object()
    monkeypatch.setattr(cli.GeminiConfig, "from_env", lambda: config)
    monkeypatch.setattr(gemini, "GeminiAnalyzer", lambda value: analyzer)

    pipeline = cli._build_pipeline()
    assert pipeline.analyzer is analyzer


def test_apply_benchmark_uses_local_match_before_ai(monkeypatch):
    from kufar_server_finder import cli

    ads = [{"link": "x", "cpu_model": "CPU"}]
    assert cli._apply_benchmark(ads, None) is ads

    class LocalDataset:
        def enrich_ads(self, values):
            return [{**value, "cpu_mark": 10} for value in values]

    monkeypatch.setattr(
        cli.CpuBenchmarkDataset,
        "from_csv",
        lambda path: LocalDataset(),
    )
    monkeypatch.setattr(
        cli,
        "_build_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("AI pipeline created")),
    )

    assert cli._apply_benchmark(ads, "cpu.csv")[0]["cpu_mark"] == 10


def test_apply_benchmark_normalizes_only_unmatched(monkeypatch):
    from kufar_server_finder import cli

    ads = [
        {"link": "matched", "cpu_model": "Known CPU"},
        {"link": "unmatched", "cpu_model": "Typo CPU"},
    ]

    class PartialDataset:
        def enrich_ads(self, values):
            result = []
            for value in values:
                item = dict(value)
                if item.get("cpu_model") in {"Known CPU", "Normalized CPU"}:
                    item["cpu_mark"] = 10
                result.append(item)
            return result

    normalized_inputs = []

    class FakePipeline:
        def normalize_cpu_models_for_benchmark(self, values):
            normalized_inputs.extend(values)
            return [
                {
                    **values[0],
                    "cpu_model": "Normalized CPU",
                    "cpu_model_normalized": True,
                }
            ]

    monkeypatch.setattr(
        cli.CpuBenchmarkDataset,
        "from_csv",
        lambda path: PartialDataset(),
    )

    result = cli._apply_benchmark(
        ads,
        "cpu.csv",
        pipeline=FakePipeline(),
    )

    assert [item["link"] for item in normalized_inputs] == ["unmatched"]
    assert [item["cpu_mark"] for item in result] == [10, 10]
