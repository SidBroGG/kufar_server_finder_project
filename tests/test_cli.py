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

