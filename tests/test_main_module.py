import runpy

import pytest


def test_package_main_uses_cli_exit_code(monkeypatch):
    from kufar_server_finder import cli

    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("kufar_server_finder.__main__", run_name="__main__")

    assert exc_info.value.code == 7
