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
