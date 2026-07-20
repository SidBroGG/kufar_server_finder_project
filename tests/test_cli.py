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
