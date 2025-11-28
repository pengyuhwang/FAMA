"""Utilities to convert KunQuant Alpha101 operators into symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List

from fama.utils.io import ensure_dir

try:  # KunQuant is an optional dependency during development
    from KunQuant.Op import Builder, Input
    from KunQuant.predefined.Alpha101 import AllData
except Exception:  # pragma: no cover
    Builder = None
    Input = None
    AllData = None


@dataclass
class AlphaExpression:
    name: str
    expression: str
    explanation: str | None = None


def _require_kunquant() -> None:
    if Builder is None or Input is None or AllData is None:
        raise ImportError("KunQuant with predefined Alpha101 is required for extraction.")


def _format_constant(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if int(value) == value:
        return str(int(value))
    return f"{value:.10g}"


def _expr_for_op(op, memo: Dict[int, str]) -> str:
    key = id(op)
    if key in memo:
        return memo[key]
    cls = op.__class__.__name__
    inputs = getattr(op, "inputs", [])
    attrs = getattr(op, "attrs", {})

    def child(idx: int) -> str:
        return _expr_for_op(inputs[idx], memo)

    if cls == "Input":
        expr = attrs["name"].upper()
    elif cls == "ConstantOp":
        expr = _format_constant(attrs["value"])
    elif cls == "Add":
        expr = f"({child(0)} + {child(1)})"
    elif cls == "Sub":
        expr = f"({child(0)} - {child(1)})"
    elif cls == "Mul":
        expr = f"({child(0)} * {child(1)})"
    elif cls == "Div":
        expr = f"({child(0)} / {child(1)})"
    elif cls == "MulConst":
        expr = f"({child(0)} * {_format_constant(attrs['value'])})"
    elif cls == "AddConst":
        expr = f"({child(0)} + {_format_constant(attrs['value'])})"
    elif cls == "SubConst":
        expr = f"({child(0)} - {_format_constant(attrs['value'])})"
    elif cls == "DivConst":
        expr = f"({child(0)} / {_format_constant(attrs['value'])})"
    elif cls == "Clip":
        expr = f"CLIP({child(0)}, {_format_constant(attrs.get('value', 0.0))})"
    elif cls == "Rank":
        expr = f"RANK({child(0)})"
    elif cls == "WindowedAvg":
        expr = f"TS_MEAN({child(0)}, {attrs['window']})"
    elif cls == "WindowedStddev":
        expr = f"TS_STDDEV({child(0)}, {attrs['window']})"
    elif cls == "WindowedCorrelation":
        expr = f"CORREL({child(0)}, {child(1)}, {attrs['window']})"
    elif cls == "WindowedCovariance":
        expr = f"COVAR({child(0)}, {child(1)}, {attrs['window']})"
    elif cls == "WindowedQuantile":
        expr = f"TS_QUANTILE({child(0)}, {attrs['window']}, {_format_constant(attrs['q'])})"
    elif cls == "WindowedLinearRegressionRSqaure":
        expr = f"TS_LINEAR_REGRESSION_R2({child(0)}, {attrs['window']})"
    elif cls == "WindowedLinearRegressionResi":
        expr = f"TS_LINEAR_REGRESSION_RESI({child(0)}, {attrs['window']})"
    elif cls == "WindowedLinearRegressionSlope":
        expr = f"TS_LINEAR_REGRESSION_SLOPE({child(0)}, {attrs['window']})"
    elif cls == "TsRank":
        expr = f"TS_RANK({child(0)}, {attrs['window']})"
    elif cls == "WindowedSum":
        expr = f"TS_SUM({child(0)}, {attrs['window']})"
    elif cls == "WindowedMin":
        expr = f"TS_MIN({child(0)}, {attrs['window']})"
    elif cls == "WindowedMax":
        expr = f"TS_MAX({child(0)}, {attrs['window']})"
    elif cls == "WindowedProduct":
        expr = f"TS_PRODUCT({child(0)}, {attrs['window']})"
    elif cls == "TsArgMax":
        expr = f"TS_ARGMAX({child(0)}, {attrs['window']})"
    elif cls == "TsArgMin":
        expr = f"TS_ARGMIN({child(0)}, {attrs['window']})"
    elif cls == "BackRef":
        window = attrs.get("window", 1)
        expr = f"DELAY({child(0)}, {window})"
    elif cls == "DecayLinear":
        expr = f"DECAY_LINEAR({child(0)}, {attrs['window']})"
    elif cls == "Scale":
        expr = f"SCALE({child(0)})"
    elif cls == "SetInfOrNanToValue":
        expr = f"REPLACE_NAN_INF({child(0)}, {_format_constant(attrs.get('value', 0.0))})"
    elif cls == "Select":
        expr = f"IF({child(0)}, {child(1)}, {child(2)})"
    elif cls == "Abs":
        expr = f"ABS({child(0)})"
    elif cls == "Sign":
        expr = f"SIGN({child(0)})"
    elif cls == "Pow":
        expr = f"({child(0)} ** {child(1)})"
    elif cls == "Log":
        expr = f"LOG({child(0)})"
    elif cls == "Exp":
        expr = f"EXP({child(0)})"
    elif cls == "Max":
        expr = f"MAX({child(0)}, {child(1)})"
    elif cls == "Min":
        expr = f"MIN({child(0)}, {child(1)})"
    elif cls == "GreaterThan":
        expr = f"GT({child(0)}, {child(1)})"
    elif cls == "GreaterEqual":
        expr = f"GE({child(0)}, {child(1)})"
    elif cls == "LessThan":
        expr = f"LT({child(0)}, {child(1)})"
    elif cls == "LessEqual":
        expr = f"LE({child(0)}, {child(1)})"
    elif cls == "LessThanConst":
        expr = f"LT({child(0)}, {_format_constant(attrs['value'])})"
    elif cls == "Equals":
        expr = f"EQ({child(0)}, {child(1)})"
    elif cls == "Or":
        expr = f"OR({child(0)}, {child(1)})"
    elif cls == "And":
        expr = f"AND({child(0)}, {child(1)})"
    elif cls == "Not":
        expr = f"NOT({child(0)})"
    else:
        raise NotImplementedError(f"Unsupported KunQuant op: {cls}")

    memo[key] = expr
    return expr


def extract_alpha101_expressions() -> list[AlphaExpression]:
    """Convert KunQuant Alpha101 definitions into symbolic expressions."""

    _require_kunquant()
    import KunQuant.predefined.Alpha101 as alpha_mod  # type: ignore

    factor_names = sorted(
        name for name in dir(alpha_mod) if name.startswith("alpha") and name[5:].isdigit()
    )
    expressions: list[AlphaExpression] = []
    placeholders = ["open", "high", "low", "close", "volume", "amount"]

    for name in factor_names:
        func = getattr(alpha_mod, name)
        with Builder():
            inputs = {field: Input(field) for field in placeholders}
            data = AllData(**inputs)
            node = func(data)
            expr = _expr_for_op(node, {})
            expressions.append(AlphaExpression(name=name.lower(), expression=expr, explanation=None))
    return expressions


__all__ = ["AlphaExpression", "extract_alpha101_expressions"]
