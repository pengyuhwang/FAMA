"""CLI 框架的冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from fama import cli
from fama.utils.io import write_yaml, read_yaml


def test_cli_main_entrypoint_exists():
    """确认 CLI 入口可被导入。"""

    assert callable(cli.main)


def test_cli_subcommand_handlers_exist(tmp_path, monkeypatch, capsys):
    """在合成数据上运行 mine 子命令以验证全链路。"""

    config = read_yaml("fama/config/defaults.yaml")
    config["paths"]["market_data"] = str(tmp_path / "market.parquet")
    config["paths"]["factor_cache"] = str(tmp_path / "factors.yaml")
    config_path = tmp_path / "config.yaml"
    write_yaml(str(config_path), config)

    parser = cli._build_parser()
    args = parser.parse_args(["mine", "--config", str(config_path), "--skip-coe"])
    args.func(args)
    captured = capsys.readouterr()
    assert "Generated expressions" in captured.out
