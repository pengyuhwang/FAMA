"""FAMA 脚手架的冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from fama.data.dataloader import load_market_data
from fama.data.factor_space import Factor, FactorSet, deserialize_factor_set, serialize_factor_set
from fama.mining.orchestrator import PromptOrchestrator
from fama.utils.io import read_yaml


def test_data_dataloader_signature(tmp_path):
    """验证在文件缺失时会生成模拟数据。"""

    df = load_market_data(str(tmp_path / "missing.parquet"))
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_factor_set_serialization_signature(tmp_path):
    """FactorSet 的保存与恢复应保持一致。"""

    factors = FactorSet([Factor(name="alpha1", expression="RANK(CLOSE - OPEN)")])
    target = tmp_path / "factors.yaml"
    serialize_factor_set(factors, str(target))
    restored = deserialize_factor_set(str(target))
    assert restored.factors[0].expression == "RANK(CLOSE - OPEN)"


def test_prompt_orchestrator_run_signature(tmp_path):
    """确保 PromptOrchestrator.run 能返回生成的表达式。"""

    config = read_yaml("fama/config/defaults.yaml")
    config["paths"]["market_data"] = str(tmp_path / "market.parquet")
    config["paths"]["factor_cache"] = str(tmp_path / "factors.yaml")
    orchestrator = PromptOrchestrator(config)
    expressions = orchestrator.run(use_css=True, use_coe=True)
    assert expressions, "At least one factor should be generated."
