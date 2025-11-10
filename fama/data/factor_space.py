"""供 CSS、CoE 及主流程复用的因子容器定义。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import yaml

from fama.utils.io import ensure_dir


@dataclass
class Factor:
    """对挖掘出的 Alpha 表达式的轻量封装。"""

    name: str
    expression: str


@dataclass
class FactorSet:
    """FAMA 流程中使用的 Factor 列表容器。"""

    factors: list["Factor"] = field(default_factory=list)


def serialize_factor_set(fs: "FactorSet", path: str) -> None:
    """根据 README 中的“单次流程”说明，将 FactorSet 序列化到磁盘。

    Args:
        fs: 待序列化的 FactorSet。
        path: 目标输出路径，通常来自 ``paths.output_dir``。
    """

    payload = [asdict(factor) for factor in fs.factors]
    factor_path = Path(path)
    ensure_dir(str(factor_path.parent))
    with factor_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def deserialize_factor_set(path: str) -> "FactorSet":
    """读取序列化的 FactorSet（README “单次流程”章节）。

    Args:
        path: 已保存因子元数据的路径。

    Returns:
        反序列化后的 FactorSet，可直接供后续阶段使用。
    """

    factor_path = Path(path)
    if not factor_path.exists():
        return FactorSet([])
    with factor_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or []
    factors = [Factor(**item) for item in payload]
    return FactorSet(factors)
