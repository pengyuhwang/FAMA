"""Utilities to convert KunQuant Alpha158 operators into symbolic expressions."""

from __future__ import annotations

from pathlib import Path

import yaml

from fama.utils.io import ensure_dir

try:  # KunQuant is an optional dependency during development
    from KunQuant.Op import Builder, Input
    from KunQuant.predefined.Alpha158 import AllData
except Exception:  # pragma: no cover
    Builder = None
    Input = None
    AllData = None

# Reuse the Alpha101 expression helpers to avoid duplicating operator mapping.
from fama.factors.alpha101_extractor import AlphaExpression, _expr_for_op, _require_kunquant


def _default_alpha158_config(pack: AllData) -> dict:
    """Match the config used in alphatest/ExtFunction for Alpha158."""

    return {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": [
                ("OPEN", pack.open),
                ("HIGH", pack.high),
                ("LOW", pack.low),
                ("VWAP", pack.vwap),
            ],
        },
        "rolling": {
            "windows": [5, 10, 20, 30, 60],
        },
    }


def extract_alpha158_expressions(
    *,
    prefix: str = "alpha158_",
    config_factory=_default_alpha158_config,
) -> list[AlphaExpression]:
    """Convert KunQuant Alpha158 definitions into symbolic expressions.

    Args:
        prefix: Optional prefix to avoid name collision. Defaults to 'alpha158_'.
        config_factory: Callable that receives the `AllData` pack and returns the build config.
    """

    _require_kunquant()
    expressions: list[AlphaExpression] = []
    placeholders = ["open", "high", "low", "close", "volume", "amount"]

    with Builder():
        inputs = {field: Input(field) for field in placeholders}
        pack = AllData(**inputs)
        fields, names = pack.build(config_factory(pack))
        for raw_name, node in zip(names, fields):
            name = f"{prefix}{raw_name.lower()}" if prefix else raw_name.lower()
            expr = _expr_for_op(node, {})
            expressions.append(AlphaExpression(name=name, expression=expr, explanation=None))
    return expressions


def dump_alpha158_yaml(output_path: str | Path, *, prefix: str = "alpha158_") -> Path:
    """Extract Alpha158 expressions and write them to a YAML file."""

    expressions = extract_alpha158_expressions(prefix=prefix)
    payload = [
        {"name": expr.name, "expression": expr.expression, "explanation": expr.explanation}
        for expr in expressions
    ]
    output = Path(output_path)
    ensure_dir(str(output.parent))
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return output


__all__ = ["AlphaExpression", "extract_alpha158_expressions", "dump_alpha158_yaml"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export KunQuant Alpha158 factors to YAML.")
    parser.add_argument(
        "--output",
        default=Path(__file__).resolve().parents[2] / "data" / "factor_cache_new" / "alpha158_factors.yaml",
        help="YAML 输出路径（默认写入 data/factor_cache_new/alpha158_factors.yaml）",
    )
    parser.add_argument(
        "--prefix",
        default="alpha158_",
        help="因子名前缀，避免与现有因子重名（默认 alpha158_，设为空字符串可保留原名）。",
    )
    args = parser.parse_args()

    path = dump_alpha158_yaml(args.output, prefix=args.prefix)
    print(f"Wrote Alpha158 factors to {path}")
